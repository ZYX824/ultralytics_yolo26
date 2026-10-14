from datetime import datetime
from uuid import uuid4

import requests

from config import CALLBACK_ENABLED, CALLBACK_URL, CLASS_TO_NAMES


def build_report(picture_id: str, detections: list[dict]) -> list[dict]:
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


def send_report(report: dict):
    """向后端上报一个目标的检测结果。"""

    if not CALLBACK_ENABLED:
        print("回传已关闭，跳过上报：", report)
        return

    with requests.post(
        CALLBACK_URL,
        json=report,
        timeout=(5, 30),
    ) as response:
        response.raise_for_status()

        print("告警编号：", report["eventId"], "受理成功，等待平台异步处理")
