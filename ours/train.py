import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

# 直接运行脚本时使用当前仓库的 ultralytics。
from ultralytics import YOLO  # noqa: E402

if __name__ == "__main__":
    # 模型yaml路径
    model_yaml_path = ROOT / "ultralytics" / "cfg" / "models" / "26" / "yolo26-GCConv.yaml"
    # 提取yaml文件名，去掉后缀，拼接前缀 bee_
    yaml_stem = model_yaml_path.stem
    run_name = f"bee_{yaml_stem}"

    model = YOLO(model_yaml_path).load(
        ROOT / "ours" / "weights" / "yolo26n.pt"
    )
    model.train(
        data=str(ROOT / "ours" / "datasets" / "Bee" / "data.yaml"),
        epochs=5,
        batch=8,
        imgsz=640,
        device=0,
        workers=0,  # Windows 下先使用主进程加载数据。
        save=True,
        project=str(ROOT / "ours" / "runs"),
        name=run_name,
    )
    print(f"本次训练run名称自动生成：{run_name}")