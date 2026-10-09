"""Detect lr* images with YOLO26, then segment each detection with SAM2.

Run with defaults: python ours/yolo_sam2.py
Select a model: python ours/yolo_sam2.py --sam-size large
"""

import argparse
import json
import sys
from contextlib import nullcontext
from pathlib import Path

OURS = Path(__file__).resolve().parent
ROOT = OURS.parent
SAM_ROOT = OURS / "sam2"
# Prefer local sources and avoid importing the outer sam2 repository as a package.
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(SAM_ROOT))

import cv2
import numpy as np
import torch
from sam2.build_sam import build_sam2
from sam2.sam2_image_predictor import SAM2ImagePredictor
from ultralytics import YOLO
from ultralytics.utils.files import increment_path

SAM_MODELS = {
    "tiny": ("sam2.1_hiera_tiny.pt", "configs/sam2.1/sam2.1_hiera_t.yaml"),
    "small": ("sam2.1_hiera_small.pt", "configs/sam2.1/sam2.1_hiera_s.yaml"),
    "base_plus": ("sam2.1_hiera_base_plus.pt", "configs/sam2.1/sam2.1_hiera_b+.yaml"),
    "large": ("sam2.1_hiera_large.pt", "configs/sam2.1/sam2.1_hiera_l.yaml"),
}
IMAGE_SUFFIXES = {".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff", ".webp"}
COLORS = [(40, 200, 40), (0, 180, 255), (255, 100, 50), (180, 60, 220)]


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", type=Path, default=SAM_ROOT / "assets", help="Image or image directory")
    parser.add_argument("--prefix", default="lr", help="Filename prefix when source is a directory")
    parser.add_argument("--weights", type=Path, default=OURS / "weights" / "best.pt")
    parser.add_argument("--sam-size", choices=SAM_MODELS, default="small")
    parser.add_argument("--sam-checkpoint", type=Path, help="Override checkpoint for the selected SAM size")
    parser.add_argument("--output", type=Path, default=OURS / "runs" / "yolo_sam2")
    parser.add_argument("--conf", type=float, default=0.25, help="YOLO confidence threshold")
    parser.add_argument("--iou", type=float, default=0.7, help="YOLO NMS IoU threshold")
    parser.add_argument("--imgsz", type=int, default=640, help="YOLO input size")
    parser.add_argument("--classes", nargs="+", type=int, help="Optional YOLO class IDs to segment")
    parser.add_argument("--device", default="auto", help="auto, cpu, cuda, cuda:0, or GPU index")
    args = parser.parse_args()
    if not 0 <= args.conf <= 1 or not 0 <= args.iou <= 1:
        parser.error("--conf and --iou must be in [0, 1]")
    if args.imgsz <= 0:
        parser.error("--imgsz must be positive")
    return args


def find_images(source, prefix):
    if source.is_file():
        if source.suffix.lower() not in IMAGE_SUFFIXES:
            raise ValueError(f"Unsupported image: {source}")
        return [source]
    if not source.is_dir():
        raise FileNotFoundError(f"Source does not exist: {source}")
    images = sorted(
        path for path in source.iterdir()
        if path.is_file() and path.name.startswith(prefix) and path.suffix.lower() in IMAGE_SUFFIXES
    )
    if not images:
        raise FileNotFoundError(f"No {prefix}* images found in {source}")
    return images


def choose_device(value):
    if value == "auto":
        value = "cuda:0" if torch.cuda.is_available() else "cpu"
    elif value.isdigit():
        value = f"cuda:{value}"
    device = torch.device(value)
    if device.type not in {"cpu", "cuda"}:
        raise ValueError("Only CPU and CUDA devices are supported")
    if device.type == "cuda":
        if not torch.cuda.is_available():
            raise RuntimeError("CUDA is unavailable; use --device cpu")
        torch.cuda.set_device(device)
    return device


def save_image(path, image):
    # imencode + tofile also supports Windows paths containing non-ASCII characters.
    ok, encoded = cv2.imencode(path.suffix, image)
    if not ok:
        raise RuntimeError(f"Failed to encode image: {path}")
    encoded.tofile(str(path))


def process_image(image_path, detector, predictor, device, args, output):
    image = cv2.imdecode(np.fromfile(str(image_path), dtype=np.uint8), cv2.IMREAD_COLOR)
    if image is None:
        raise ValueError(f"Unable to read image: {image_path}")
    result = detector.predict(
        source=image, conf=args.conf, iou=args.iou, imgsz=args.imgsz,
        classes=args.classes, device=str(device), verbose=False,
    )[0]
    boxes = result.boxes.xyxy.cpu().numpy()
    confidences = result.boxes.conf.cpu().numpy()
    class_ids = result.boxes.cls.cpu().numpy().astype(int)
    height, width = image.shape[:2]
    union_mask = np.zeros((height, width), dtype=bool)
    overlay = image.copy()
    records = []
    image_output = output / image_path.name
    image_output.mkdir()
    (image_output / "masks").mkdir()

    precision = nullcontext()
    if device.type == "cuda":
        dtype = torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float16
        precision = torch.autocast("cuda", dtype=dtype)
    with torch.inference_mode(), precision:
        if len(boxes):
            predictor.set_image(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
        for index, (box, confidence, class_id) in enumerate(zip(boxes, confidences, class_ids), start=1):
            # Boxes are XYXY pixel coordinates on the original, unresized image.
            masks, scores, _ = predictor.predict(box=box, multimask_output=False)
            mask = masks[0].astype(bool)
            if mask.shape != (height, width):
                raise RuntimeError(f"Unexpected SAM2 mask shape: {mask.shape}")
            color = np.asarray(COLORS[(index - 1) % len(COLORS)], dtype=np.float32)
            overlay[mask] = (0.55 * overlay[mask] + 0.45 * color).astype(np.uint8)
            union_mask |= mask
            mask_name = f"masks/{index:03d}.png"
            save_image(image_output / mask_name, mask.astype(np.uint8) * 255)
            records.append({
                "instance_id": index,
                "class_id": int(class_id),
                "class_name": result.names[int(class_id)],
                "yolo_confidence": float(confidence),
                "box_xyxy": box.tolist(),
                "sam2_score": float(scores[0]),
                "mask_pixels": int(mask.sum()),
                "mask_file": mask_name,
            })
    predictor.reset_predictor()

    for record in records:
        color = COLORS[(record["instance_id"] - 1) % len(COLORS)]
        x1, y1, x2, y2 = map(round, record["box_xyxy"])
        cv2.rectangle(overlay, (x1, y1), (x2, y2), color, 2)
        label = f'{record["instance_id"]}: {record["class_name"]} {record["yolo_confidence"]:.2f}'
        cv2.putText(overlay, label, (x1, max(15, y1 - 6)), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 1)

    save_image(image_output / "yolo.jpg", result.plot())
    save_image(image_output / "overlay.jpg", overlay)
    save_image(image_output / "mask.png", union_mask.astype(np.uint8) * 255)
    metadata = {
        "image": str(image_path.resolve()),
        "width": width, "height": height,
        "yolo_weights": str(args.weights.resolve()),
        "sam_checkpoint": str(args.sam_checkpoint.resolve()),
        "sam_config": SAM_MODELS[args.sam_size][1],
        "conf": args.conf, "iou": args.iou, "imgsz": args.imgsz,
        "classes": args.classes, "detections": records,
    }
    (image_output / "detections.json").write_text(json.dumps(metadata, indent=2, ensure_ascii=False), encoding="utf-8")
    print(f"{image_path.name}: {len(records)} detections, {int(union_mask.sum())} mask pixels")


def main():
    args = parse_args()
    checkpoint_name, config = SAM_MODELS[args.sam_size]
    args.sam_checkpoint = args.sam_checkpoint or SAM_ROOT / "checkpoints" / checkpoint_name
    for path in (args.weights, args.sam_checkpoint):
        if not path.is_file():
            raise FileNotFoundError(f"Missing weights: {path}")
    images = find_images(args.source, args.prefix)
    device = choose_device(args.device)
    print(f"Device: {device}; SAM2.1: {args.sam_size}; images: {len(images)}")
    detector = YOLO(str(args.weights))
    predictor = SAM2ImagePredictor(build_sam2(config, str(args.sam_checkpoint), device=str(device)))
    output = increment_path(args.output, mkdir=True)
    print(f"Output: {output.resolve()}")
    for image_path in images:
        process_image(image_path, detector, predictor, device, args, output)
    print(f"Done. Results saved to: {output.resolve()}")


if __name__ == "__main__":
    main()
