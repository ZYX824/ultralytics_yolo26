from pathlib import Path
from threading import Lock  # 让多次请求依次使用同一个模型

from ultralytics import YOLO  # 负责推理

from config import CLASS_TO_TYPE, DETECT_CLASSES, YOLO_CONF, YOLO_IMGSZ, YOLO_WEIGHTS

# 模型在模块加载时创建一次，后续请求复用。
model = YOLO(str(YOLO_WEIGHTS))
model_lock = Lock()


def predict(image_path: Path, types: list[int]) -> list[dict]:
    """执行 YOLO 推理，返回调用方选择的警告类型对应的预测框。"""

    # 同一时间只允许一个请求使用模型。
    with model_lock:
        result = model.predict(
            source=str(image_path),
            classes=DETECT_CLASSES,
            imgsz=YOLO_IMGSZ,
            conf=YOLO_CONF,
            verbose=False,  # 关闭 YOLO 每次推理时的常规详细日志输出。
        )[0]

    detections = []
    for box in result.boxes:
        # 模型类别，例如 1 表示纵裂。
        class_id = int(box.cls.item())

        # 转成后端警告编号，例如 1 → 2000；跳过调用方未选择的类型。
        alarm_type = CLASS_TO_TYPE[class_id]
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
    return detections
