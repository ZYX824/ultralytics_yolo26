"""Small, disk-backed demo batch adapter: URLs -> sequential inference -> optional callbacks."""

import copy
import json
import logging
import os
import re
import secrets
import tempfile
import threading
import time
import uuid
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime, timedelta, timezone
from pathlib import Path
from urllib.parse import unquote, urljoin, urlsplit

import requests
from fastapi import APIRouter, HTTPException, Path as ApiPath, Request
from pydantic import BaseModel, ConfigDict, Field, HttpUrl, StrictInt, field_validator, model_validator

from .errors import GPUOutOfMemory, ServiceBusy

MAX_DOWNLOAD_BYTES = 32 * 1024 * 1024
# Class IDs verified against the deployed VisDrone checkpoint: car, van, truck, bus.
VEHICLE_CLASS_IDS = (3, 4, 5, 8)
TERMINAL_STATES = {"completed", "partial_failed", "failed", "interrupted"}
logger = logging.getLogger(__name__)


class Picture(BaseModel):
    """待检测图片及可选拍摄参数。"""

    model_config = ConfigDict(extra="forbid", title="待检测图片")
    pictureId: str = Field(min_length=1, max_length=256, title="图片唯一标识",
                          description="后端媒体表中的原始图片 ID；同一批次内必须唯一，结果和保存的上报报文原样带回")
    picturePath: str = Field(min_length=1, title="原图地址或对象路径",
                            description="完整 HTTP/HTTPS 原图下载地址，支持预签名 URL；相对对象路径需配置 YOLO_SERVICE_IMAGE_BASE_URL")
    info: dict = Field(default_factory=dict, title="补充几何参数",
                       description="可选，默认自动读取原图 EXIF/DJI XMP；缺失时可补充测距、姿态和相机内参，最大 64 KB")

    @field_validator("picturePath")
    @classmethod
    def validate_path(cls, value):
        parsed = urlsplit(value)
        if parsed.scheme:
            return str(HttpUrl(value))
        if (not parsed.path.strip("/") or parsed.netloc or
                any(part == ".." for part in unquote(parsed.path).replace("\\", "/").split("/"))):
            raise ValueError("picturePath must be an HTTP(S) URL or an object path without parent traversal")
        return value

    @field_validator("info")
    @classmethod
    def limit_info(cls, value):
        if len(json.dumps(value, allow_nan=False).encode("utf-8")) > 64 * 1024:
            raise ValueError("info exceeds 64 KB")
        return value


class BatchRequest(BaseModel):
    """批量图片检测请求，与后端文档中的字段保持兼容。"""

    model_config = ConfigDict(extra="forbid", title="批量检测请求", json_schema_extra={"examples": [{
        "pictures": [{
            "pictureId": "demo-001",
            "picturePath": (
                "http://113.249.105.17:9000/dbxq/DJ-MEDIA-OSS/63ab78dc-cce2-4ee0-8cb5-e68557f62c72/"
                "DJI_202609241910_008_63ab78dc-cce2-4ee0-8cb5-e68557f62c72/DJI_20260924191238_0005_V.jpeg"
            ),
        }], "types": [12001],
    }]})
    pictures: list[Picture | HttpUrl] = Field(min_length=1, max_length=50, title="待检测图片列表",
                                            description="一次提交 1–50 张图片；推荐使用包含 pictureId 和 picturePath 的对象，也支持完整 URL 字符串")
    types: list[StrictInt] = Field(..., min_length=1, max_length=50, title="检测算法类型编码",
                                  description="必填，由调用方指定平台算法 ID，应用于本次所有图片；当前支持 12001（车辆违停），重复编码不会重复上报")
    with_sam: bool = Field(False, title="启用分割与测量",
                          description="默认 false，仅检测目标框；显式传 true 才运行 SAM2 分割和近似尺寸测量")
    callbackUrl: HttpUrl | None = Field(None, title="可选回调地址",
                                       description="覆盖服务配置的回调地址；默认 report 模式向该地址逐目标 POST 告警报文，未配置地址时仅保存")

    @field_validator("pictures", mode="before")
    @classmethod
    def accept_url_list(cls, value):
        if isinstance(value, list):
            return [
                {"pictureId": f"image-{index + 1:03d}", "picturePath": str(picture)}
                if isinstance(picture, (str, HttpUrl)) else picture
                for index, picture in enumerate(value)
            ]
        return value

    @model_validator(mode="after")
    def unique_ids(self):
        ids = [picture.pictureId for picture in self.pictures]
        if len(ids) != len(set(ids)):
            raise ValueError("pictureId must be unique within a batch")
        return self


def now():
    return datetime.now(timezone.utc).isoformat()


def artifact_urls(report, public_base=""):
    """Add browser/download links without changing the saved pipeline report."""
    report = copy.deepcopy(report)
    base = public_base.rstrip("/") + f"/files/{report['image_id']}/"
    report["artifact_urls"] = {key: base + name for key, name in report["artifacts"].items()}
    for detection in report["detections"]:
        if detection.get("mask_file"):
            detection["mask_url"] = base + detection["mask_file"]
    return report


def demo_result(report, public_base):
    """Keep batches/callbacks small; the report URL exposes the full contours and geometry."""
    report = artifact_urls(report, public_base)
    measurement_keys = ("status", "reason", "area_m2", "extent_long_m", "extent_short_m", "touches_image_border")

    def measurement(value):
        return {key: value.get(key) for key in measurement_keys} if value is not None else None

    detections = []
    for detection in report["detections"]:
        target = {key: detection[key] for key in (
            "instance_id", "model_id", "class_id", "class_name", "confidence", "box_xyxy",
            "sam_score", "mask_pixels", "mask_url",
        ) if key in detection}
        target["measurement"] = measurement(detection.get("measurement"))
        detections.append(target)
    return {
        "image_id": report["image_id"], "width": report["width"], "height": report["height"],
        "with_sam": report["with_sam"], "measurement_status": report["measurement_status"],
        "detections": detections, "union_measurement": measurement(report["union_measurement"]),
        "artifact_urls": report["artifact_urls"],
        "result_url": public_base.rstrip("/") + f"/results/{report['image_id']}",
    }


def download_image(url, destination):
    """Download original bytes (including EXIF/XMP) with bounded size and time."""
    started = time.monotonic()
    with requests.get(url, stream=True, timeout=(10, 30)) as response:
        response.raise_for_status()
        declared = response.headers.get("Content-Length")
        if declared and int(declared) > MAX_DOWNLOAD_BYTES:
            raise ValueError("Image exceeds 32 MB")
        size = 0
        with destination.open("wb") as handle:
            for chunk in response.iter_content(chunk_size=64 * 1024):
                if time.monotonic() - started > 120:
                    raise ValueError("Image download exceeded 120 seconds")
                size += len(chunk)
                if size > MAX_DOWNLOAD_BYTES:
                    raise ValueError("Image exceeds 32 MB")
                handle.write(chunk)
        if not size:
            raise ValueError("Empty image download")


def send_callback(url, payload, token=""):
    """Confirm endpoint acceptance only; platform persistence happens asynchronously."""
    headers = {"Authorization": f"Bearer {token}"} if token else {}
    with requests.post(url, json=payload, headers=headers, timeout=(10, 30), allow_redirects=False) as response:
        if not 200 <= response.status_code < 300:
            raise ValueError(f"Callback returned HTTP {response.status_code}")
        # Also recognize application-level failure in a BladeX-style response.
        if response.content:
            try:
                body = response.json()
            except ValueError:
                body = None
            if isinstance(body, dict) and (body.get("success") is False or body.get("code", 200) != 200):
                raise ValueError("Callback returned application-level failure")


class BatchManager:
    def __init__(self, pipeline, *, max_batches=2, callback_url=None, callback_token="", public_base="",
                 image_base="", type_map=None, callback_format="report"):
        if max_batches < 1:
            raise ValueError("max_batches must be positive")
        self.pipeline = pipeline
        self.folder = pipeline.config.output / "batches"
        self.folder.mkdir(parents=True, exist_ok=True)
        self.callback_url = str(HttpUrl(callback_url)) if callback_url else None
        self.callback_token = callback_token
        self.public_base = public_base.rstrip("/")
        if self.public_base:
            HttpUrl(self.public_base)
        self.image_base = str(HttpUrl(image_base)).rstrip("/") + "/" if image_base else ""
        if callback_format not in {"report", "demo"}:
            raise ValueError("callback_format must be report or demo")
        self.callback_format = callback_format
        self.type_map = ({"12001": {"model_id": "visdrone2019", "class_ids": list(VEHICLE_CLASS_IDS),
                               "typeName": "车辆违停"}} if type_map is None else type_map)
        if not isinstance(self.type_map, dict):
            raise ValueError("type_map must be an object keyed by platform type code")
        self.type_map = {str(code): rule for code, rule in self.type_map.items()}
        for code, rule in self.type_map.items():
            if not re.fullmatch(r"-?\d+", str(code)) or not isinstance(rule, dict):
                raise ValueError("type_map keys must be integer type codes and values must be objects")
            if rule.get("model_id") not in {"uav_pdd2023", "visdrone2019", "*"}:
                raise ValueError("type_map model_id must be uav_pdd2023, visdrone2019 or *")
            if "class_ids" in rule and (not isinstance(rule["class_ids"], list) or
                    any(type(value) is not int or value < 0 for value in rule["class_ids"])):
                raise ValueError("type_map class_ids must be a list of nonnegative integers")
            if "typeName" in rule and not isinstance(rule["typeName"], str):
                raise ValueError("type_map typeName must be a string")
        self.lock = threading.RLock()
        self.slots = threading.BoundedSemaphore(max_batches)
        self.executor = ThreadPoolExecutor(max_workers=1, thread_name_prefix="demo-batch")
        self.closed = False

    def _save(self, job):
        job["updated_at"] = now()
        target = self.folder / f"{job['task_id']}.json"
        temporary = target.with_suffix(".tmp")
        with self.lock:
            temporary.write_text(json.dumps(job, ensure_ascii=False, indent=2, allow_nan=False), encoding="utf-8")
            os.replace(temporary, target)

    def recover(self):
        """Keep finished results; explicitly mark abandoned work after a process restart."""
        for path in self.folder.glob("*.json"):
            job = json.loads(path.read_text(encoding="utf-8"))
            changed = False
            if job["status"] not in TERMINAL_STATES:
                job["status"] = "interrupted"
                for item in job["pictures"]:
                    if item["status"] in {"pending", "running"}:
                        item.update(status="failed", error={"code": "interrupted", "message": "Service restarted"})
                self._counts(job)
                changed = True
            for item in job["pictures"]:
                callback = item["callback"]
                if callback["status"] in {"pending", "sending"}:
                    callback.update(status="failed", error="Service restarted before callback confirmation")
                    changed = True
                for report in item.get("reports", []):
                    if report["callback"]["status"] in {"pending", "sending"}:
                        report["callback"].update(status="failed", error="Service restarted before callback confirmation")
                        changed = True
            if changed:
                self._save(job)

    def close(self):
        with self.lock:
            self.closed = True
        self.executor.shutdown(wait=True)

    def get(self, task_id):
        if not re.fullmatch(r"[0-9a-f]{32}", task_id):
            raise ValueError("task_id must be a 32-character lowercase hexadecimal ID")
        with self.lock:
            return json.loads((self.folder / f"{task_id}.json").read_text(encoding="utf-8"))

    def submit(self, request, request_base):
        for picture in request.pictures:
            if not urlsplit(picture.picturePath).scheme and not self.image_base:
                raise ValueError("Relative picturePath requires YOLO_SERVICE_IMAGE_BASE_URL")
        configured_callback = str(request.callbackUrl) if request.callbackUrl else self.callback_url
        callback_url = configured_callback
        unknown_types = [code for code in request.types if str(code) not in self.type_map]
        if unknown_types:
            raise ValueError(f"Unsupported algorithm IDs: {unknown_types}; supported IDs: {list(self.type_map)}")
        if not self.slots.acquire(blocking=False):
            raise ServiceBusy("Batch queue full; retry later")
        try:
            job = {
                "schema_version": 1, "task_id": uuid.uuid4().hex, "status": "queued", "created_at": now(),
                "total": len(request.pictures), "processed": 0, "succeeded": 0, "failed": 0,
                "with_sam": request.with_sam, "callback_url": callback_url,
                "configured_callback_url": configured_callback,
                "types": request.types,
                "callback_format": self.callback_format,
                "public_base": self.public_base or request_base.rstrip("/"),
                "pictures": [{
                    "pictureId": picture.pictureId, "picturePath": str(picture.picturePath),
                    "download_url": (picture.picturePath if urlsplit(picture.picturePath).scheme else
                                     urljoin(self.image_base, picture.picturePath.lstrip("/"))),
                    "info": picture.info, "status": "pending", "result": None, "error": None,
                    "reports": [], "report_status": "pending",
                    "callback": {"status": "pending" if callback_url else "disabled", "attempts": 0, "error": None},
                } for picture in request.pictures],
            }
            with self.lock:
                if self.closed:
                    raise ServiceBusy("Service is shutting down")
                self._save(job)
                self.executor.submit(self._run, job)
            return job["task_id"]
        except Exception:
            self.slots.release()
            raise

    @staticmethod
    def _counts(job):
        job["succeeded"] = sum(item["status"] == "completed" for item in job["pictures"])
        job["failed"] = sum(item["status"] == "failed" for item in job["pictures"])
        job["processed"] = job["succeeded"] + job["failed"]

    def _reports(self, job, item):
        """Prepare the document's one-target payload without inventing platform type codes."""
        if item["status"] != "completed":
            item["report_status"] = "image_failed"
            return
        if not job["types"] or not self.type_map:
            item["report_status"] = "type_mapping_required"
            return
        for detection in item["result"]["detections"]:
            for code in dict.fromkeys(job["types"]):
                rule = self.type_map[str(code)]
                if rule["model_id"] != "*" and detection["model_id"] != rule["model_id"]:
                    continue
                if "class_ids" in rule and detection["class_id"] not in rule["class_ids"]:
                    continue
                target = {
                    "type": str(code), "typeName": rule.get("typeName", detection["class_name"]),
                    "bbox": [float(value) for value in detection["box_xyxy"]],
                    "confidence": float(detection["confidence"]),
                }
                payload = {
                    "source": "offline", "isReal": 0, "pictureId": item["pictureId"],
                    "eventId": f"{secrets.randbelow(10**16):016d}",
                    "timestamp": datetime.now(timezone(timedelta(hours=8))).strftime("%Y-%m-%d %H:%M:%S"),
                    "result": target,
                }
                item["reports"].append({"payload": payload, "callback": {
                    "status": "pending" if job["callback_url"] else "disabled", "attempts": 0, "error": None,
                }})
        item["report_status"] = "ready" if item["reports"] else "no_matching_targets"

    def _deliver(self, job, item, payload, callback):
        # The alarm platform has no eventId deduplication. A timeout can follow acceptance,
        # so do not automatically resend report events and risk duplicate alarm records.
        max_attempts = 1 if job["callback_format"] == "report" else 3
        for attempt in range(1, max_attempts + 1):
            callback.update(status="sending", attempts=attempt)
            self._save(job)
            try:
                token = self.callback_token if job["callback_url"] == self.callback_url else ""
                send_callback(job["callback_url"], payload, token)
                callback.update(status="delivered", error=None)
                self._save(job)
                logger.info("Callback accepted for task %s picture %s event %s", job["task_id"],
                            item["pictureId"], payload["eventId"])
                return
            except (requests.RequestException, ValueError) as exc:
                # Request exceptions may contain credentials in URLs; keep only safe HTTP errors.
                callback["error"] = (str(exc) if isinstance(exc, ValueError) else
                                     "Callback network failure; check receiving endpoint")
                logger.warning("Callback attempt %s failed for task %s picture %s: %s", attempt,
                               job["task_id"], item["pictureId"], callback["error"])
                if attempt < max_attempts:
                    time.sleep(attempt)
        callback["status"] = "failed"
        self._save(job)

    def _callback(self, job, item):
        if not job["callback_url"]:
            return
        if job["callback_format"] == "report":
            if not item["reports"]:
                item["callback"].update(status="skipped", error=None)
                self._save(job)
                return
            for report in item["reports"]:
                self._deliver(job, item, report["payload"], report["callback"])
            callbacks = [report["callback"] for report in item["reports"]]
            errors = [callback["error"] for callback in callbacks if callback["status"] == "failed"]
            item["callback"].update(status="failed" if errors else "delivered",
                                    attempts=sum(callback["attempts"] for callback in callbacks),
                                    error=errors[0] if errors else None)
            self._save(job)
            return
        payload = {
            "eventId": f"{job['task_id']}:{item['pictureId']}", "taskId": job["task_id"],
            "pictureId": item["pictureId"], "picturePath": item["picturePath"],
            "status": item["status"], "result": item["result"], "error": item["error"],
        }
        self._deliver(job, item, payload, item["callback"])

    def _run(self, job):
        try:
            job["status"] = "running"
            self._save(job)
            for item in job["pictures"]:
                item["status"] = "running"
                self._save(job)
                try:
                    if self.pipeline.oom:
                        raise GPUOutOfMemory("Inference stopped after GPU OOM")
                    with tempfile.TemporaryDirectory(prefix="demo-image-") as folder:
                        source = Path(folder) / "input.image"
                        download_image(item["download_url"], source)
                        report = self.pipeline.process(source, item["info"], with_sam=job["with_sam"])
                    item.update(status="completed", result=demo_result(report, job["public_base"]))
                except Exception as exc:
                    if isinstance(exc, GPUOutOfMemory):
                        code = "gpu_oom"
                    elif isinstance(exc, requests.RequestException):
                        code = "download_failed"
                    elif isinstance(exc, ServiceBusy):
                        code = "service_busy"
                    elif isinstance(exc, (ValueError, OSError)):
                        code = "invalid_image_or_model_files"
                    else:
                        code = "inference_failed"
                    # URLs may contain signing credentials: do not expose request exceptions.
                    message = str(exc) if isinstance(exc, (GPUOutOfMemory, ServiceBusy, ValueError)) else code
                    item.update(status="failed", error={"code": code, "message": message})
                    logger.warning("Task %s picture %s failed (%s)", job["task_id"], item["pictureId"], code)
                    if code == "inference_failed":
                        logger.exception("Inference failure for task %s picture %s", job["task_id"], item["pictureId"])
                self._counts(job)
                self._reports(job, item)
                self._save(job)
                self._callback(job, item)
            job["status"] = ("completed" if not job["failed"] else
                             "failed" if not job["succeeded"] else "partial_failed")
            self._save(job)
        except Exception:
            logger.exception("Batch worker failed for task %s", job["task_id"])
            job["status"] = "interrupted"
            for item in job["pictures"]:
                if item["status"] in {"pending", "running"}:
                    item.update(status="failed", error={"code": "worker_failed", "message": "See service log"})
            self._counts(job)
            self._save(job)
        finally:
            self.slots.release()


def batch_router(manager):
    router = APIRouter()
    responses = {
        200: {"description": "任务已受理，尚未完成推理；使用返回的 task_id 查询进度", "content": {
            "application/json": {"example": {
                "code": 200, "success": True, "msg": "操作成功", "data": True,
                "task_id": "0123456789abcdef0123456789abcdef", "status": "queued", "total": 1,
                "status_url": "/tasks/0123456789abcdef0123456789abcdef",
            }}}},
        422: {"description": "请求参数不合法：图片数量、ID、地址或类型编码不符合要求"},
        503: {"description": "批次队列已满或服务正在退出，可按 Retry-After 提示稍后重试"},
    }
    description = (
        "提交 1–50 张原图，立即返回任务 ID，后台逐张下载、检测、分割和测量。"
        "`code=200`、`data=true` 只表示受理成功；随后使用 `/tasks/{task_id}` 查询结果。"
        "`types` 必填，为调用方选择的平台算法 ID，应用于所有图片。当前支持 `12001`（车辆违停）；"
        "默认 `with_sam=false`，仅执行目标检测。汽车、面包车、卡车或公交车"
        "通过检测置信度阈值即触发上报；配置回调地址后逐车辆 POST，不判断停留时间或禁停区域。"
    )

    @router.post("/detect", tags=["批量检测"], summary="提交批量原图，执行目标检测、分割和近似测量",
                 description=description, responses=responses)
    @router.post("/alarm/record/detection2", tags=["批量检测"], summary="按后端文档格式提交批量检测（Demo 兼容入口）",
                 description="本服务的 Demo 兼容入口，与 /detect 执行相同处理；正式链路中该路径属于业务后端。" + description,
                 responses=responses)
    def detect(body: BatchRequest, request: Request):
        try:
            task_id = manager.submit(body, str(request.base_url))
        except ServiceBusy as exc:
            raise HTTPException(503, str(exc), headers={"Retry-After": "2"}) from exc
        except ValueError as exc:
            raise HTTPException(422, str(exc)) from exc
        return {"code": 200, "success": True, "msg": "操作成功", "data": True,
            "task_id": task_id, "status": "queued", "total": len(body.pictures),
            "status_url": f"/tasks/{task_id}",
        }

    @router.get(
        "/tasks/{task_id}", tags=["结果查询"], summary="查询批次进度、逐图结果及保存的上报报文",
        description=(
            "建议每 2 秒查询一次。`queued` 为排队中，`running` 为处理中；终态包括 `completed`（全部成功）、"
            "`partial_failed`（部分失败）、`failed`（全部失败）、`interrupted`（处理被中断）。\n\n"
            "`pictures[i].result` 包含原图像素框、掩码链接、近似尺寸和面积。面积单位为平方米，长短边单位为米。"
            "只有测量状态为 `estimated_ground_plane` 时才展示估算值；无目标也是处理成功。\n\n"
            "`pictures[i].reports[j].payload` 是单目标上报报文；对应 `callback.status` 记录发送状态。"
            "未配置回调地址为 `disabled`；`delivered` 表示平台已受理，不代表异步落库成功。"
            "report 模式仅尝试一次，发送失败为 `failed`；平台未按 eventId 去重，不自动重试。"
            "report 模式下无匹配目标或推理失败时不发送告警，逐图回调状态为 `skipped`。"
        ),
        responses={200: {"description": "返回任务进度、成功/失败数量和每张图片的结果"},
                   404: {"description": "未找到该任务"}, 422: {"description": "任务 ID 格式不正确"}},
    )
    def task(task_id: str = ApiPath(..., title="批次任务 ID", description="提交检测接口返回的 task_id，32 位小写十六进制字符串")):
        try:
            return manager.get(task_id)
        except ValueError as exc:
            raise HTTPException(422, str(exc)) from exc
        except FileNotFoundError as exc:
            raise HTTPException(404, "Task not found") from exc

    return router
