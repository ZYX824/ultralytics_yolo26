from fastapi import FastAPI, Request, BackgroundTasks
from fastapi.exceptions import RequestValidationError
from fastapi.responses import JSONResponse

import logging

import numpy as np
from PIL import Image as PILImage
from pydantic import BaseModel, Field, StrictInt, field_validator
import uvicorn

from down_picture import download_picture
from yolo_predict import predict
from mask_dedup import segment_and_merge
from measurement import measure_merged, read_geometry
from callback import build_report, send_report
from draw_result import draw_reports

# 允许调用方选择的警告编号（白名单）。
ALLOWED_TYPES = {
    10002, 10003, 10004, 10005, 10006, 10007, 10008,
    12001, 12002, 12003, 12005, 12006, 12008, 12009, 12007,
    12010, 12011, 12012, 12013, 12014, 12015, 12016, 12017, 12018, 12019,
    12020, 12021, 12022, 12023, 12024, 12025, 12026, 12027, 12028, 12029,
    12030, 12031, 12032, 12033, 12034, 12035, 12036, 12037, 12038, 12039,
    12040, 12041, 12042,
    1, 2000, 2001, 2002, 2003, 2004,
}


class Picture(BaseModel):
    pictureId: str = Field(..., min_length=1, description="对方提供的图片 ID")
    # 可以是对象存储路径，不要求是完整 URL；实际下载时需要配置存储地址。
    picturePath: str = Field(..., min_length=1, description="图片对象路径或完整下载地址")


class DetectRequest(BaseModel):
    pictures: list[Picture]

    types: list[StrictInt] = Field(
        ..., min_length=1, max_length=54,
        description="选择 1–54 个警告编号，每个编号必须是白名单中的整数",
    )


    @field_validator("types")
    @classmethod # 类方法不需要创建实例对象，就可以调用方法
    def validate_types(cls, value):
        # 类型和长度校验通过后，再逐个检查编号是否在白名单中。
        invalid_types = [code for code in value if code not in ALLOWED_TYPES]
        if invalid_types:
            raise ValueError(f"不支持的警告编号：{invalid_types}")
        return value


# 创建一个接口应用
app = FastAPI()


# 请求校验在 detect() 执行前发生，失败时由这个处理器统一返回。
@app.exception_handler(RequestValidationError)
async def validation_exception_handler(request: Request, exc: RequestValidationError):
    reasons = []
    for error in exc.errors():
        # 例如 ("body", "pictures", 0, "picturePath") 转成 pictures.0.picturePath。
        field = ".".join(str(part) for part in error["loc"] if part != "body")
        reasons.append(f"{field or '请求体'}: {error['msg']}")

    return JSONResponse(
        status_code=422,
        content={
            "code": 422,
            "msg": "请求参数不合法：" + "; ".join(reasons),
        },
    )

logger = logging.getLogger(__name__)


def process_picture(picture_id: str, picture_path: str, types: list[int]) -> list[dict]:
    """下载 → YOLO 检测 → SAM 分割并合并相连掩码 → 地面测量 → 组装报文。"""

    # 1. 下载原图。
    local_path = download_picture(picture_path)
    print("原图已保存：", local_path)

    # 2. YOLO 推理。
    detections = predict(local_path, types)

    # 3. SAM 分割，合并相连掩码；失败时退回原始预测框。
    if detections:
        try:
            with PILImage.open(local_path) as image:
                rgb = np.array(image.convert("RGB"))
            merged = segment_and_merge(rgb, detections)
            print("合并后的检测结果数量：", len(merged))

            # 4. 计算每个合并区域的地面面积。
            geometry = read_geometry(local_path)
            if geometry is not None:
                measure_merged(merged, rgb.shape[:2], geometry)
            detections = merged
        except Exception:
            logger.exception("SAM 分割失败，仅返回检测框结果：%s", picture_id)

    # 5. 组装报文。
    reports = build_report(picture_id=picture_id, detections=detections)

    # 6. 保存结果图，不受回传开关影响；绘制失败不影响上报。
    try:
        draw_reports(local_path, reports)
    except Exception:
        logger.exception("结果图绘制失败：%s", picture_id)

    return reports


def process_pictures(data: DetectRequest):
    """依次处理图片，并逐条上报告警。"""
    for picture in data.pictures:
        try:
            reports = process_picture(
                picture_id=picture.pictureId,
                picture_path=picture.picturePath,
                types=data.types,
            )
        except Exception:
            logger.exception("图片处理失败：%s", picture.pictureId)
            continue

        for report in reports:
            try:
                send_report(report)
            except Exception:
                logger.exception("告警上报失败：%s", report["eventId"])


# 别人向 /detect 发送 POST 请求时，就会执行这个函数
@app.post("/detect")
def detect(data: DetectRequest, background_tasks: BackgroundTasks):
    # 先安排任务，响应发送后再执行下载、推理和上报。
    background_tasks.add_task(process_pictures, data)

    return {
        "code": 200,
        "msg": "图片已接收，等待后台处理",
        "data": True
    }


# 直接运行这个文件时，启动服务
if __name__ == "__main__":
    uvicorn.run(app, host="127.0.0.1", port=8888)
