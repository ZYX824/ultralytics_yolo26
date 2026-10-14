from pathlib import Path  # 用来定位文件、创建文件夹、读写文件
from urllib.parse import urljoin, urlsplit # 处理网络地址

import requests  # 下载图片
from PIL import Image as PILImage, ImageDraw, ImageFont # 用来打开、解码图片，读取宽高等信息

from ultralytics import YOLO # 负责推理
from threading import Lock # 让多次请求依次使用同一个模型

from datetime import datetime
from uuid import uuid4

import sys
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

import numpy as np
import cv2
import torch
from PIL import Image as PILImage

from ours.extraction.extract_gsd import (
    extract_image,
    format_gsd_message,
)

from measurement import measure_mask


# ========== SAM2 分割模型（模块加载时创建一次） ==========
SAM_ROOT = Path(__file__).resolve().parents[1] / "sam2"
SAM_CONFIG = "configs/sam2.1/sam2.1_hiera_s.yaml"
SAM_CHECKPOINT = SAM_ROOT / "checkpoints" / "sam2.1_hiera_small.pt"

sys.path.insert(0, str(SAM_ROOT))
from sam2.build_sam import build_sam2
from sam2.sam2_image_predictor import SAM2ImagePredictor

_sam_device = "cuda:0" if torch.cuda.is_available() else "cpu"
sam_predictor = SAM2ImagePredictor(
    build_sam2(SAM_CONFIG, str(SAM_CHECKPOINT), device=_sam_device)
)
sam_lock = Lock()

# 图片存储服务地址，沿用现有服务的配置。
# http://113.249.105.17:9000/dbxq/DJ-MEDIA-OSS/fbc168a2-5754-454f-99c6-f53390ad431d/Remote-Control/DJI_20261015104605_0006_V.jpeg
IMAGE_BASE_URL = "http://113.249.105.17:9000/"

# 设置图片下载后的保存目录
DOWNLOAD_DIR = Path(__file__).resolve().parent / "downloads"

# 设置权重文件的目录
WEIGHTS_PATH = (
    Path(__file__).resolve().parents[1]
    / "runs" / "UAV_m" / "best.pt"
)

# 模型在模块加载时创建一次，后续请求复用。
model = YOLO(str(WEIGHTS_PATH))
model_lock = Lock()

# 参与检测的模型类别，排除 4：Repair。
DETECT_CLASSES = [0, 1, 2, 3, 5]

# 模型类别编号 → 后端警告编号。K可以是整数
CLASS_TO_TYPE = {
    0: 2003,  # Alligator crack：龟裂
    1: 2000,  # Longitudinal crack：纵裂
    2: 2002,  # Oblique crack：斜裂
    3: 2004,  # Pothole：坑槽
    5: 2001,  # Transverse crack：横裂
}
# 模型类别编号对应的中文名称。
CLASS_TO_NAMES = {
    0: "道路龟裂",
    1: "道路纵裂",
    2: "道路斜裂",
    3: "道路坑槽",
    5: "道路横裂",
}


def download_picture(picture_path: str) -> Path:
    """根据后端提供的图片路径，下载原图并返回本地路径。"""

    # 1. 得到完整的下载地址。
    if picture_path.startswith(("http://", "https://")):
        url = picture_path
    else:
        url = urljoin(IMAGE_BASE_URL, picture_path.lstrip("/"))


    # 2. 创建下载目录。
    DOWNLOAD_DIR.mkdir(parents=True, exist_ok=True)

    # 3.从下载地址中提取原始文件名。
    filename = Path(urlsplit(url).path).name
    # 指定本地保存路径。
    local_path = DOWNLOAD_DIR / filename

    try:
        # 4. 请求图片地址，图片分块写入本地文件。
        with requests.get(url, stream=True, timeout=(5, 30)) as response:
            if response.history:
                print("重定向后的下载地址：", response.url)
            response.raise_for_status()

            with local_path.open("wb") as file:
                for chunk in response.iter_content(chunk_size=1024 * 1024):
                    if chunk:
                        file.write(chunk)

        # 5. 尝试解码图片，检查文件能否正常读取。
        with PILImage.open(local_path) as image:
            image.load()
            print("图片尺寸：", image.width, image.height)

    except Exception:
        # 下载或解码失败，删除残留文件，并继续抛出异常。
        local_path.unlink(missing_ok=True)
        raise

    return local_path


def process_picture(
    picture_id: str,
    picture_path: str,
    types: list[int],
) -> list[dict]:
    """下载一张原图，执行 YOLO 检测、GSD 提取、SAM 分割和地面测量。"""

    # 1. 下载原图。
    local_path = download_picture(picture_path)
    print("原图已保存：", local_path)

    # 2. 读取图片为 RGB 数组（供 SAM 使用）。
    rgb = np.array(PILImage.open(local_path).convert("RGB"))

    # 3. 提取几何元数据，区分正下视 GSD 与倾斜投影。
    try:
        geometry = extract_image(str(local_path))
        print(format_gsd_message(geometry))

    except Exception as exc:
        print(
            f"警告：几何元数据处理失败，"
            f"跳过地面测量，原因：{exc}"
        )
        geometry = None

    # 4. 执行 YOLO 推理，同一时间只允许一个请求使用模型。
    with model_lock:
        result = model.predict(
            source=str(local_path),
            classes=DETECT_CLASSES,
            imgsz=1280,
            conf=0.11,
            verbose=False, # 关闭 YOLO 每次推理时的常规详细日志输出。
        )[0]

    # 5. 提取检测结果。
    detections = []

    for box in result.boxes:
        # 模型类别，例如 1 表示纵裂。
        class_id = int(box.cls.item())

        # 转成后端警告编号，例如 1 → 2000。
        alarm_type = CLASS_TO_TYPE[class_id]

        # 跳过修补警告
        if alarm_type not in types:
            continue

        detections.append({
            "class_id": class_id,
            "type": alarm_type,
            "class_name": result.names[class_id],
            "confidence": float(box.conf.item()),
            "bbox": box.xyxy[0].cpu().tolist(),
        })
    print("筛选后的检测结果数量：", len(detections))

    print("开始去重检测结果...")



    # 6. SAM 分割 + 掩码地面测量。
    if detections and geometry is not None:
        try:
            with sam_lock:
                sam_predictor.set_image(rgb)
                for det in detections:
                    try:
                        masks, scores, _ = sam_predictor.predict(
                            box=np.asarray(det["bbox"], dtype=np.float32),
                            multimask_output=False,
                        )
                        mask = masks[0].astype(bool)

                        contours, hierarchy = cv2.findContours(
                            mask.astype(np.uint8), cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE,
                        )
                        det["mask_pixels"] = int(mask.sum())
                        det["sam_score"] = float(scores[0])
                        det["contours"] = [
                            {"points_xy": c.reshape(-1, 2).tolist(), "parent": int(hierarchy[0, i, 3])}
                            for i, c in enumerate(contours)
                        ]
                        det["measurement"] = measure_mask(mask, geometry, det["contours"])
                    except Exception:
                        print("警告：掩码/测量失败 class_id=%s", det["class_id"])
                sam_predictor.reset_predictor()
        except Exception:
            print("警告：SAM 分割失败，仅返回检测框结果")

    # 8. 组装报文。
    return build_report(picture_id=picture_id, detections=detections)





def build_report(
    picture_id: str,
    detections: list[dict]) -> list[dict]:
    """将多个预测框（含掩码测量）整理成告警报文。"""

    reports = []

    for detection in detections:
        report = {
            "source": "offline",
            "isReal": 0,
            "pictureId": picture_id,
            "eventId": f"ai-{uuid4().hex[:12]}",
            "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            "result": {
                "type": str(detection["type"]),
                "typeName": CLASS_TO_NAMES[detection["class_id"]],
                "bbox": detection["bbox"],
                "confidence": detection["confidence"],
            },
        }
        # # 附加掩码测量结果（如果有）
        # measurement = detection.get("measurement")
        # if measurement and measurement.get("area_m2") is not None:
        #     # 将共享测量模块的成功状态转换为本接口的 success。
        #     measurement_status = measurement["status"]
        #     if measurement_status == "estimated_ground_plane":
        #         measurement_status = "success"
        #     report["result"].update({
        #         "area_m2": measurement["area_m2"],
        #         "extent_long_m": measurement["extent_long_m"],
        #         "extent_short_m": measurement["extent_short_m"],
        #         "measurement_status": measurement_status,
        #     })
        reports.append(report)

    return reports
