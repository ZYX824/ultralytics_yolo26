"""Detect lr* images with YOLO26, then segment each detection with SAM2.

Run with defaults: python ours/yolo_sam2.py
Select a model: python ours/yolo_sam2.py --sam-size large
Set a calibrated scale: python ours/yolo_sam2.py --gsd-m-per-pixel 0.0035

Sizes are estimates on the input-image pixel grid. The default UAV-PDD2023
nominal scale assumes an 18 m footprint across 5184 pixels (not a 2592 crop).
Outputs include report.jpg, report.txt, detections.json and a run-wide summary.csv.
"""

# 这个脚本仅用于测试，统一的GSD
import argparse
import csv
import json
import math
import os
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
from PIL import Image, ImageDraw, ImageFont
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
DEFAULT_GSD = 18.0 / 5184
CLASS_NAMES_ZH = {
    "alligator crack": "龟裂",
    "longitudinal crack": "纵向裂缝",
    "oblique crack": "斜向裂缝",
    "pothole": "坑槽",
    "repair": "修补",
    "patching": "修补",
    "transverse crack": "横向裂缝",
}
CSV_FIELDS = [
    "image", "instance_id", "class_name", "class_name_zh", "yolo_confidence", "sam2_score",
    "mask_pixels", "estimated_area_m2", "estimated_extent_long_m", "estimated_extent_short_m",
    "measurement_status", "touches_image_border", "gsd_m_per_pixel", "gsd_source", "mask_file",
]


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
    parser.add_argument("--gsd-m-per-pixel", type=float, default=DEFAULT_GSD,
                        help="Meters per INPUT-image pixel; default: nominal UAV-PDD2023 18/5184")
    parser.add_argument("--gsd-source", help="Scale provenance, e.g. measured reference; recorded in reports")
    parser.add_argument("--font", type=Path, help="Optional Chinese TTF/TTC font for report.jpg")
    args = parser.parse_args()
    if not 0 <= args.conf <= 1 or not 0 <= args.iou <= 1:
        parser.error("--conf and --iou must be in [0, 1]")
    if args.imgsz <= 0:
        parser.error("--imgsz must be positive")
    if not math.isfinite(args.gsd_m_per_pixel) or args.gsd_m_per_pixel <= 0:
        parser.error("--gsd-m-per-pixel must be finite and positive")
    if args.font is not None and not args.font.is_file():
        parser.error(f"Font does not exist: {args.font}")
    if args.gsd_source is None:
        args.gsd_source = ("UAV-PDD2023 paper nominal: 18 m / 5184 px; uncalibrated"
                           if args.gsd_m_per_pixel == DEFAULT_GSD else "User-supplied uniform GSD")
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


def measure_mask(mask, gsd):
    """Measure foreground area and rotated enclosing extent, not crack width.

    Include full pixel cells rather than just their centers: this makes a solid
    W x H mask measure W x H pixels, and keeps single-pixel masks measurable.
    All components contribute to area and to the enclosing rectangle.
    """
    pixels = int(mask.sum())
    touches_border = bool(mask[0].any() or mask[-1].any() or mask[:, 0].any() or mask[:, -1].any())
    measurement = {
        "measurement_status": "estimated" if pixels else "empty_mask",
        "estimated_area_m2": pixels * gsd ** 2 if pixels else None,
        "estimated_extent_long_m": None,
        "estimated_extent_short_m": None,
        "touches_image_border": touches_border,
    }
    if not pixels:
        return measurement
    contours, _ = cv2.findContours(mask.astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    hull = cv2.convexHull(np.concatenate(contours)).reshape(-1, 2).astype(np.float32)
    offsets = np.array([[-0.5, -0.5], [-0.5, 0.5], [0.5, -0.5], [0.5, 0.5]], dtype=np.float32)
    corners = (hull[:, None, :] + offsets).reshape(-1, 2)
    _, sides, _ = cv2.minAreaRect(corners)
    long_side, short_side = sorted(sides, reverse=True)
    measurement.update(estimated_extent_long_m=float(long_side) * gsd,
                       estimated_extent_short_m=float(short_side) * gsd)
    return measurement


def describe_detection(record, chinese=True):
    name = record["class_name_zh"] if chinese else record["class_name"]
    prefix = f'{record["instance_id"]}号：{name}' if chinese else f'#{record["instance_id"]} {name}'
    if record["measurement_status"] == "empty_mask":
        return prefix + ("；掩码为空，无法测量" if chinese else "; empty mask, size unavailable")
    area = record["estimated_area_m2"]
    long_side, short_side = record["estimated_extent_long_m"], record["estimated_extent_short_m"]
    size = (f"；掩码面积约 {area:.4f} m²；覆盖范围约 {long_side:.2f} × {short_side:.2f} m"
            if chinese else f"; mask area ~{area:.4f} m2; extent ~{long_side:.2f} x {short_side:.2f} m")
    if record["touches_image_border"]:
        size += "；可能被图像边缘截断" if chinese else "; possibly cropped at image edge"
    return prefix + size


def load_report_font(font_path):
    candidates = ([font_path] if font_path else []) + [
        Path(os.environ.get("WINDIR", "C:/Windows")) / "Fonts/msyh.ttc",
        Path("/usr/share/fonts/opentype/noto/NotoSansCJK-Regular.ttc"),
        Path("/usr/share/fonts/truetype/wqy/wqy-zenhei.ttc"),
    ]
    for candidate in candidates:
        if candidate.is_file():
            return ImageFont.truetype(str(candidate), 24), True
    # Keep reports readable on machines without Chinese fonts.
    try:
        return ImageFont.truetype("DejaVuSans.ttf", 24), False
    except OSError:
        return ImageFont.load_default(), False


def save_report(image_output, overlay, records, args, union_pixels):
    """Keep numbered masks on the left and readable measurements on the right."""
    font, chinese = load_report_font(args.font)
    picture = Image.fromarray(cv2.cvtColor(overlay, cv2.COLOR_BGR2RGB))
    picture.thumbnail((1600, 1400))
    panel_width, header_height, row_height = 900, 140, 140
    canvas_height = header_height + max(picture.height, row_height * max(1, len(records)))
    canvas = Image.new("RGB", (picture.width + panel_width, canvas_height), "#f5f7fa")
    canvas.paste(picture, (0, header_height))
    draw = ImageDraw.Draw(canvas)
    title = f"{image_output.name} | " + (
        f"{len(records)} 个缺陷（估计尺寸）" if chinese else f"{len(records)} defects (estimated sizes)"
    )
    draw.text((20, 15), title, font=font, fill="#172334")
    draw.text((20, 53), f"GSD: {args.gsd_m_per_pixel * 1000:.4f} mm/px", font=font, fill="#172334")
    note = ("覆盖范围 = 掩码旋转外接矩形长边 × 短边；不是裂缝宽度" if chinese
            else "Extent = rotated mask bounding rectangle; not crack width")
    draw.text((20, 91), note, font=font, fill="#526071")
    if not records:
        draw.text((picture.width + 24, header_height + 20),
                  "未检测到缺陷" if chinese else "No defects detected", font=font, fill="#172334")
    for index, record in enumerate(records):
        x, y = picture.width + 24, header_height + row_height * index + 12
        color = tuple(reversed(COLORS[index % len(COLORS)]))
        draw.rectangle((x, y + 5, x + 10, y + 115), fill=color)
        x += 24
        name = record["class_name_zh"] if chinese else record["class_name"]
        heading = (f'#{record["instance_id"]} {name} | '
                   f'YOLO {record["yolo_confidence"]:.2f} / SAM2 {record["sam2_score"]:.2f}')
        draw.text((x, y), heading, font=font, fill="#172334")
        if record["measurement_status"] == "empty_mask":
            lines = ["掩码为空，无法测量" if chinese else "Empty mask; size unavailable"]
        else:
            lines = [
                ("掩码面积约 " if chinese else "Mask area ~ ") + f'{record["estimated_area_m2"]:.4f} m²',
                ("覆盖范围约 " if chinese else "Extent ~ ") + f'{record["estimated_extent_long_m"]:.2f} × {record["estimated_extent_short_m"]:.2f} m',
            ]
            if record["touches_image_border"]:
                lines.append("可能被图像边缘截断" if chinese else "Possibly cropped at image edge")
        for line_index, line in enumerate(lines, 1):
            draw.text((x, y + 30 * line_index), line, font=font, fill="#526071")
    canvas.save(image_output / "report.jpg", quality=95)
    lines = [
        f"图片：{image_output.name}；检测到 {len(records)} 个缺陷",
        f"尺度：{args.gsd_m_per_pixel:.10f} m/像素；来源：{args.gsd_source}",
        "以下为输入图像内的估计尺寸；覆盖范围为旋转外接矩形，不代表裂缝真实长度或宽度。",
        "面积表示掩码选中区域；龟裂等类别可能选中整个病害区域。",
        *[describe_detection(record) for record in records],
        f"合并掩码面积约 {union_pixels * args.gsd_m_per_pixel ** 2:.4f} m²（重叠像素只计一次）",
    ]
    if not records:
        lines.append("未检测到缺陷。")
    (image_output / "report.txt").write_text("\n".join(lines) + "\n", encoding="utf-8")


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
                "class_name_zh": CLASS_NAMES_ZH.get(
                    result.names[int(class_id)].strip().lower(), result.names[int(class_id)]
                ),
                "yolo_confidence": float(confidence),
                "box_xyxy": box.tolist(),
                "sam2_score": float(scores[0]),
                "mask_pixels": int(mask.sum()),
                "mask_file": mask_name,
                **measure_mask(mask, args.gsd_m_per_pixel),
            })
    predictor.reset_predictor()

    for record in records:
        color = COLORS[(record["instance_id"] - 1) % len(COLORS)]
        x1, y1, x2, y2 = map(round, record["box_xyxy"])
        cv2.rectangle(overlay, (x1, y1), (x2, y2), color, 2)
        # The detailed legend lives in report.jpg; compact ID badges avoid long overlapping labels.
        label = f'#{record["instance_id"]}'
        badge_x = min(max(0, x1), max(0, width - 95))
        badge_y = min(max(40, y1), height - 1)
        cv2.rectangle(overlay, (badge_x, badge_y - 40), (badge_x + 95, badge_y), color, -1)
        cv2.putText(overlay, label, (badge_x + 5, badge_y - 8), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 0), 2)

    save_image(image_output / "yolo.jpg", result.plot())
    save_image(image_output / "overlay.jpg", overlay)
    save_image(image_output / "mask.png", union_mask.astype(np.uint8) * 255)
    union_pixels = int(union_mask.sum())
    save_report(image_output, overlay, records, args, union_pixels)
    metadata = {
        "image": str(image_path.resolve()),
        "width": width, "height": height,
        "scale": {"gsd_m_per_pixel": args.gsd_m_per_pixel, "source": args.gsd_source,
                  "size_status": "estimated", "pixel_grid": "input_image"},
        "measurement_method": {
            "area": "foreground pixel count * GSD^2; selected region, not necessarily crack-only area",
            "extent": "minimum-area rotated rectangle enclosing all foreground pixel cells; not crack length/width",
            "border": "foreground touches input-image border; full defect size may be larger",
        },
        "union_mask_pixels": union_pixels,
        "estimated_union_area_m2": union_pixels * args.gsd_m_per_pixel ** 2,
        "yolo_weights": str(args.weights.resolve()),
        "sam_checkpoint": str(args.sam_checkpoint.resolve()),
        "sam_config": SAM_MODELS[args.sam_size][1],
        "conf": args.conf, "iou": args.iou, "imgsz": args.imgsz,
        "classes": args.classes, "detections": records,
    }
    (image_output / "detections.json").write_text(json.dumps(metadata, indent=2, ensure_ascii=False), encoding="utf-8")
    # Superscript 2 cannot be encoded by legacy Windows GBK terminals.
    print(f"{image_path.name}: {len(records)} 个缺陷，合并掩码面积约 {union_pixels * args.gsd_m_per_pixel ** 2:.4f} m^2")
    for record in records:
        print("  " + describe_detection(record).replace("m²", "m^2"))
    return records


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
    print(f"Estimated sizes using GSD {args.gsd_m_per_pixel * 1000:.4f} mm/px ({args.gsd_source})")
    with (output / "summary.csv").open("w", newline="", encoding="utf-8-sig") as stream:
        writer = csv.DictWriter(stream, fieldnames=CSV_FIELDS)
        writer.writeheader()
        for image_path in images:
            records = process_image(image_path, detector, predictor, device, args, output)
            for record in records:
                row = {key: record.get(key) for key in CSV_FIELDS}
                row.update(image=image_path.name, gsd_m_per_pixel=args.gsd_m_per_pixel,
                           gsd_source=args.gsd_source, mask_file=f'{image_path.name}/{record["mask_file"]}')
                writer.writerow(row)
            stream.flush()
    print(f"Done. Results saved to: {output.resolve()}")


if __name__ == "__main__":
    main()
