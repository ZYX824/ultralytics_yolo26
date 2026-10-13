"""HTTP upload interface. Run one worker so models occupy the GPU only once."""

import json
import os
import re
import logging
import tempfile
from contextlib import asynccontextmanager
from pathlib import Path

from fastapi import FastAPI, File, Form, HTTPException, Path as ApiPath, UploadFile
from fastapi.staticfiles import StaticFiles
from fastapi.middleware.cors import CORSMiddleware
from PIL import UnidentifiedImageError

from .batch import BatchManager, artifact_urls, batch_router
from .pipeline import Config, GPUOutOfMemory, Pipeline, ServiceBusy

MAX_UPLOAD_BYTES = 32 * 1024 * 1024


def environment_config():
    options = {}
    for key in ("device", "weights_uav", "weights_visdrone", "output", "sam_size", "sam_checkpoint"):
        value = os.environ.get(f"YOLO_SERVICE_{key.upper()}")
        if value:
            options[key] = value
    for key in ("imgsz", "max_det_per_model"):
        value = os.environ.get(f"YOLO_SERVICE_{key.upper()}")
        if value:
            options[key] = int(value)
    for key in ("conf", "iou"):
        value = os.environ.get(f"YOLO_SERVICE_{key.upper()}")
        if value:
            options[key] = float(value)
    return Config(**options)


def create_app(pipeline=None):
    pipeline = pipeline or Pipeline(environment_config())
    batches = BatchManager(
        pipeline, max_batches=int(os.environ.get("YOLO_SERVICE_MAX_BATCHES", "2")),
        callback_url=os.environ.get("YOLO_SERVICE_CALLBACK_URL"),
        callback_token=os.environ.get("YOLO_SERVICE_CALLBACK_TOKEN", ""),
        public_base=os.environ.get("YOLO_SERVICE_PUBLIC_BASE_URL", ""),
        image_base=os.environ.get("YOLO_SERVICE_IMAGE_BASE_URL", ""),
        type_map=json.loads(os.environ["YOLO_SERVICE_TYPE_MAP"]) if os.environ.get("YOLO_SERVICE_TYPE_MAP") else None,
        callback_format=os.environ.get("YOLO_SERVICE_CALLBACK_FORMAT", "report"),
    )

    @asynccontextmanager
    async def lifespan(app):
        batches.recover()
        try:
            yield
        finally:
            batches.close()

    app = FastAPI(
        title="无人机图像检测、分割与近似尺寸测量服务", version="2.1", lifespan=lifespan,
        description=(
            "默认执行双 YOLO 目标检测；显式启用 with_sam 后可执行 SAM2 分割和水平地面投影测量。\n\n"
            "**后端联调流程：**通过 `/detect` 或 `/alarm/record/detection2` 提交 1–50 张原图地址，"
            "取得 `task_id` 后查询 `/tasks/{task_id}`，获取预测框、掩码及近似尺寸和面积。\n\n"
            "原图中的 EXIF/DJI XMP 参数会自动提取。尺寸单位为米、面积单位为平方米；"
            "缺少有效测距、姿态或内参时，仍返回检测和分割结果，并说明无法测量的原因。"
            "调用方通过必填的 types 选择平台算法 ID；当前支持 12001（车辆违停）：检测到汽车、面包车、卡车或公交车即触发，"
            "配置回调地址后主动向业务后端 POST 告警报文。"
        ),
        openapi_tags=[
            {"name": "批量检测", "description": "提交图片地址列表，后台逐张执行检测、分割和近似尺寸测量。"},
            {"name": "结果查询", "description": "查询任务进度和结果；返回的文件链接可直接用于展示预测框和掩码。"},
            {"name": "单图检测", "description": "直接上传一张原图，同步返回完整结果。"},
            {"name": "服务状态", "description": "查看运行设备、模型加载情况和显存异常状态。"},
        ],
    )
    app.state.pipeline = pipeline
    app.state.batches = batches
    origins = [origin.strip() for origin in os.environ.get("YOLO_SERVICE_CORS_ORIGINS", "").split(",")
               if origin.strip()]
    if origins:
        app.add_middleware(CORSMiddleware, allow_origins=origins, allow_methods=["GET", "POST"],
                           allow_headers=["Content-Type", "Authorization"])
    app.include_router(batch_router(batches))
    pipeline.config.output.mkdir(parents=True, exist_ok=True)
    app.mount("/files", StaticFiles(directory=str(pipeline.config.output)), name="files")

    @app.get(
        "/health", tags=["服务状态"], summary="检查服务状态与模型加载情况",
        description=("`status=ready` 表示服务未因显存不足停止，不代表模型已加载或已完成推理验证。"
                     "模型在首次推理时加载；`device` 为运行设备，`models_loaded` 为已加载检测模型，"
                     "`sam_loaded` 表示分割模型是否已加载。"),
        responses={200: {"description": "返回服务运行状态"}},
    )
    def health():
        return {"status": "gpu_oom_stopped" if pipeline.oom else "ready",
                "device": str(pipeline.device), "models_loaded": list(pipeline.models),
                "sam_loaded": pipeline.predictor is not None}

    def add_urls(report):
        return artifact_urls(report)

    @app.get(
        "/results/{image_id}", tags=["结果查询"], summary="查询单张图片的完整检测与测量结果",
        description="读取已保存结果，不重复推理。包含原图坐标检测框、实例掩码链接、像素和地面轮廓、几何参数及测量状态。",
        responses={200: {"description": "返回完整检测报告及文件链接"},
                   404: {"description": "未找到该图片结果"}, 422: {"description": "图片结果 ID 格式不正确"}},
    )
    def result(image_id: str = ApiPath(..., title="图片结果 ID", description="AI 生成的 32 位小写十六进制 image_id；不是后端 pictureId")):
        if not re.fullmatch(r"[0-9a-f]{32}", image_id):
            raise HTTPException(422, "image_id must be a 32-character lowercase hexadecimal ID")
        path = pipeline.config.output / image_id / "result.json"
        if not path.is_file():
            raise HTTPException(404, "Result not found")
        return add_urls(json.loads(path.read_text(encoding="utf-8")))

    @app.post(
        "/predict", tags=["单图检测"], summary="上传单张原图，检测目标并生成掩码和近似尺寸",
        description=("使用 multipart/form-data 上传原图，同步等待处理结束。自动提取拍摄参数；"
                     "`info` 用于补充或覆盖原图几何输入。近似长短边是分割区域在水平目标平面上的覆盖范围。"
                     "HTTP 200 表示图片处理成功，不保证每个目标均具备测量条件。"),
        responses={200: {"description": "返回检测框、分割掩码、近似尺寸、面积和结果文件链接"},
                   413: {"description": "上传文件超过 32 MB"},
                   422: {"description": "图片、几何参数或 JSON 格式不合法"},
                   503: {"description": "推理队列已满、模型文件缺失或 GPU 显存不足"},
                   500: {"description": "推理异常，详情见服务日志"}},
    )
    def predict(
        image: UploadFile = File(..., title="原始图片", description="保留 EXIF/XMP 的原图，最大 32 MB、2500 万像素"),
        info: str = Form("{}", title="几何参数", description="此图的几何参数 JSON 字符串；通常使用 {} 自动读取原图，可补充测距、云台姿态和 calibrated_intrinsics"),
        with_sam: bool = Form(False, title="启用分割与测量", description="默认 false，仅检测目标框；true：生成 SAM 掩码、近似尺寸和面积"),
    ):
        try:
            if len(info) > 64 * 1024:
                raise ValueError("info exceeds 64 KB")
            supplied = json.loads(info)
            if not isinstance(supplied, dict):
                raise ValueError("info must be one image's geometry JSON object")
        except (ValueError, TypeError) as exc:
            raise HTTPException(422, str(exc)) from exc
        try:
            # No user-controlled path: original bytes retain EXIF/XMP for extraction.
            with tempfile.TemporaryDirectory(prefix="dual-yolo-") as folder:
                source = Path(folder) / "input.image"
                size = 0
                with source.open("wb") as handle:
                    while chunk := image.file.read(1024 * 1024):
                        size += len(chunk)
                        if size > MAX_UPLOAD_BYTES:
                            raise HTTPException(413, "Image exceeds 32 MB")
                        handle.write(chunk)
                if not size:
                    raise HTTPException(422, "Empty image")
                report = pipeline.process(source, supplied, with_sam)
        except GPUOutOfMemory as exc:
            raise HTTPException(503, {"code": "gpu_oom", "message": str(exc),
                                      "next_step": "run_this_service_on_server"}) from exc
        except ServiceBusy as exc:
            raise HTTPException(503, str(exc), headers={"Retry-After": "2"}) from exc
        except FileNotFoundError as exc:
            raise HTTPException(503, {"code": "missing_model_files", "message": str(exc)}) from exc
        except (UnidentifiedImageError, ValueError, OSError) as exc:
            raise HTTPException(422, str(exc)) from exc
        except RuntimeError as exc:
            logging.exception("Inference failed")
            raise HTTPException(500, {"code": "inference_failed", "message": "See service log for details"}) from exc
        finally:
            image.file.close()
        return add_urls(report)

    return app


app = create_app()
