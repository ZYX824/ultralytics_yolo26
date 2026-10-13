import gc
import sys
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
# 直接运行脚本时使用当前仓库的 ultralytics。
from ultralytics import YOLO  # noqa: E402


def main():
    datasets_root = Path(r"H:\datasets")
    weight_path = ROOT / "ours" / "weights" / "yolo26n.pt"
    project = ROOT / "ours" / "runs"

    if not weight_path.is_file():
        raise FileNotFoundError(f"未找到官方预训练权重：{weight_path}")

    dataset_dirs = sorted(path for path in datasets_root.iterdir() if path.is_dir())
    if len(dataset_dirs) != 6:
        raise ValueError(f"预期有 6 个数据集，实际找到 {len(dataset_dirs)} 个：{datasets_root}")

    # 先检查全部配置；滑坡检测数据位于 _yolo/detect 下。
    datasets = []
    for dataset_dir in dataset_dirs:
        yolo_dir = dataset_dir / "_yolo"
        candidates = [yolo_dir / "data.yaml", yolo_dir / "dataset.yaml", yolo_dir / "detect" / "data.yaml"]
        data_yaml = next((path for path in candidates if path.is_file()), None)
        if data_yaml is None:
            raise FileNotFoundError(f"未找到数据集的 YOLO 检测配置：{yolo_dir}")
        datasets.append((dataset_dir.name, data_yaml))

    for index, (dataset_name, data_yaml) in enumerate(datasets, start=1):
        run_name = f"{dataset_name}_{weight_path.stem}"
        print(f"\n[{index}/{len(datasets)}] 开始训练：{dataset_name}\n配置：{data_yaml}")

        # 每个数据集重新加载官方权重，独立训练。
        model = YOLO(str(weight_path))
        try:
            model.train(
                data=str(data_yaml),
                epochs=200,
                patience=10,
                batch=16,
                imgsz=640,
                device=0,
                workers=4,  # Windows 下保留 main 保护；数据加载报错可改成 0。
                save=True,
                project=str(project),
                name=run_name,
                exist_ok=False,  # 重复运行时自动增加后缀，保留之前的结果。
            )
            print(f"[{index}/{len(datasets)}] 训练完成，结果保存至：{model.trainer.save_dir}")
        finally:
            # 释放本轮模型及显存，再开始下一个数据集。
            del model
            gc.collect()
            if torch.cuda.is_available():
                torch.cuda.empty_cache()


if __name__ == "__main__":
    main()
