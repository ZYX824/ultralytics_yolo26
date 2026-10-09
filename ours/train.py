import sys
from pathlib import Path
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
# 直接运行脚本时使用当前仓库的 ultralytics。
from ultralytics import YOLO  # noqa: E402

if __name__ == "__main__":
    # 模型yaml路径
    model_yaml_path = ROOT / "ultralytics" / "cfg" / "models" / "26" / "yolo26.yaml"
    weight_path = ROOT / "ours" / "weights" / "yolo26n.pt"

    # 从yaml构建模型结构，然后加载预训练权重
    model = YOLO(model_yaml_path)
    model.load(weight_path)

    # 提取yaml文件名，去掉后缀，拼接前缀
    yaml_stem = model_yaml_path.stem
    run_name = f"UAV_{yaml_stem}"

    model.train(
        data="/root/UAV-PDD2023/data.yaml",
        epochs=200,
        patience=20,
        batch=16,
        imgsz=640,
        device=0,
        workers=4,  # Linux下workers=4没问题，数据加载报错就改成0
        save=True,
        project=str(ROOT / "ours" / "runs"),
        name=run_name,
    )
    print(f"本次训练run名称自动生成：{run_name}")
