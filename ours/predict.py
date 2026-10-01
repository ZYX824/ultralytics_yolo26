"""使用本地 YOLO26 权重推理仓库自带图片，结果保存在 ours/runs 下。"""

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))  # 直接运行脚本时使用当前仓库的 ultralytics。

from ultralytics import YOLO  # noqa: E402


if __name__ == "__main__":
    # 可替换为 yolo26s.pt 或 yolo26x.pt。
    model = YOLO(ROOT / "ours" / "weights" / "yolo26n.pt")
    model.predict(
        source=[str(ROOT / "ultralytics" / "assets" / name) for name in ("bus.jpg", "zidane.jpg")],
        imgsz=640,
        conf=0.25,
        save=True,
        project=str(ROOT / "ours" / "runs"),
        name="predict",
    )
