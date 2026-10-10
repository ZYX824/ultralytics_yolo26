"""Reusable single-GPU pipeline; model instances and SAM state are serialized."""

import json
import math
import shutil
import sys
import threading
import time
import uuid
from contextlib import nullcontext
from dataclasses import dataclass
from pathlib import Path

import cv2
import numpy as np
import torch
from PIL import Image

from ours.extraction.extract_gsd import CSV_FIELDS, estimate_gsd, read_metadata
from .measurement import measure_mask, measurement_method

ROOT = Path(__file__).resolve().parents[2]
OURS = ROOT / "ours"
SAM_ROOT = OURS / "sam2"
SAM_MODELS = {
    "tiny": ("sam2.1_hiera_tiny.pt", "configs/sam2.1/sam2.1_hiera_t.yaml"),
    "small": ("sam2.1_hiera_small.pt", "configs/sam2.1/sam2.1_hiera_s.yaml"),
    "base_plus": ("sam2.1_hiera_base_plus.pt", "configs/sam2.1/sam2.1_hiera_b+.yaml"),
    "large": ("sam2.1_hiera_large.pt", "configs/sam2.1/sam2.1_hiera_l.yaml"),
}
COLORS = {"uav_pdd2023": (40, 200, 40), "visdrone2019": (255, 160, 20)}  # RGB
DERIVED_FIELDS = {
    "image", "width_px", "height_px", "orientation", "height_agl_m", "height_source",
    "horizontal_distance_to_center_m", "focal_x_px", "focal_y_px", "principal_x_px", "principal_y_px",
    "intrinsics_source", "center_gsd_cross_track_m_per_pixel", "center_gsd_along_track_m_per_pixel",
    "gsd_m_per_pixel", "status", "warnings", "error",
}
INFO_FIELDS = set(CSV_FIELDS) - DERIVED_FIELDS


class GPUOutOfMemory(RuntimeError):
    """Stop local inference after OOM; use the same configuration on a server."""


class ServiceBusy(RuntimeError):
    """Bound the active + waiting pipeline requests."""


@dataclass
class Config:
    weights_uav: Path = OURS / "weights" / "UAV_PDD2023" / "best.pt"
    weights_visdrone: Path = OURS / "weights" / "visdrone2019" / "best.pt"
    output: Path = OURS / "service" / "runs"
    device: str = "auto"
    imgsz: int = 640
    conf: float = 0.25
    iou: float = 0.7
    sam_size: str = "small"
    sam_checkpoint: Path = None
    max_det_per_model: int = 100
    max_pixels: int = 25_000_000

    def __post_init__(self):
        if self.sam_size not in SAM_MODELS:
            raise ValueError("Invalid SAM size")
        if not 0 <= self.conf <= 1 or not 0 <= self.iou <= 1:
            raise ValueError("conf and iou must be in [0, 1]")
        if self.imgsz <= 0 or self.max_det_per_model <= 0 or self.max_pixels <= 0:
            raise ValueError("Image and detection limits must be positive")
        self.weights_uav = Path(self.weights_uav)
        self.weights_visdrone = Path(self.weights_visdrone)
        self.output = Path(self.output)
        self.sam_checkpoint = Path(self.sam_checkpoint or SAM_ROOT / "checkpoints" / SAM_MODELS[self.sam_size][0])


def image_geometry(path, supplied=None):
    """Merge only geometric inputs, recomputing height from the current range/pitch."""
    supplied = supplied or {}
    if not isinstance(supplied, dict):
        raise ValueError("info must be a JSON object containing one image's geometry")
    unknown = set(supplied) - set(CSV_FIELDS) - {"digital_zoom_ratio", "calibrated_intrinsics"}
    if unknown:
        raise ValueError(f"Unknown geometry fields: {', '.join(sorted(unknown))}")
    metadata = read_metadata(path)
    overridden = []
    for key, value in supplied.items():
        if key not in INFO_FIELDS or value is None:
            continue  # IDs, dimensions, derived height and GSD are always computed server-side.
        if key == "range_status":
            if not isinstance(value, str):
                raise ValueError("range_status must be a string")
        elif isinstance(value, bool) or not isinstance(value, (float, int)) or not math.isfinite(value):
            raise ValueError(f"{key} must be a finite number")
        if "latitude" in key and not -90 <= value <= 90:
            raise ValueError(f"{key} must be in [-90, 90]")
        if "longitude" in key and not -180 <= value <= 180:
            raise ValueError(f"{key} must be in [-180, 180]")
        metadata[key] = value
        overridden.append(key)
    for dimension in ("width_px", "height_px"):
        if dimension in supplied and supplied[dimension] != metadata[dimension]:
            raise ValueError(f"info {dimension} does not match the uploaded image")
    metadata.update(estimate_gsd(metadata))
    calibration = supplied.get("calibrated_intrinsics")
    if calibration is not None:
        required = {"fx_px", "fy_px", "cx_px", "cy_px"}
        if not isinstance(calibration, dict) or set(calibration) != required:
            raise ValueError("calibrated_intrinsics must contain fx_px, fy_px, cx_px, cy_px")
        for key, value in calibration.items():
            if isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(value):
                raise ValueError(f"calibrated_intrinsics.{key} must be finite")
        if calibration["fx_px"] <= 0 or calibration["fy_px"] <= 0:
            raise ValueError("Calibrated focal lengths must be positive")
        if not 0 <= calibration["cx_px"] < metadata["width_px"] or not (
            0 <= calibration["cy_px"] < metadata["height_px"]
        ):
            raise ValueError("Calibrated principal point must lie within the original image")
        metadata["calibrated_intrinsics"] = calibration
    metadata["supplied_fields"] = sorted(overridden)
    return metadata


def save_image(path, rgb):
    Image.fromarray(rgb).save(path)


def draw_boxes(rgb, records):
    """Draw both model namespaces together without suppressing unrelated classes."""
    canvas = np.ascontiguousarray(rgb.copy())
    height, width = canvas.shape[:2]
    for record in records:
        x1, y1, x2, y2 = (int(round(v)) for v in record["box_xyxy"])
        color = COLORS[record["model_id"]]
        cv2.rectangle(canvas, (x1, y1), (x2, y2), color, 2)
        label = f"#{record['instance_id']} {record['model_id']}:{record['class_name']} {record['confidence']:.2f}"
        measurement = record.get("measurement", {})
        if measurement.get("area_m2") is not None:
            label += (f" {measurement['extent_long_m']:.2f}x{measurement['extent_short_m']:.2f}m"
                      f" {measurement['area_m2']:.2f}m2")
        scale = max(0.4, min(0.8, width / 1800))
        (tw, th), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, scale, 1)
        tx = max(0, min(x1, width - tw - 4))
        ty = max(th + 4, min(y1, height - 4))
        cv2.rectangle(canvas, (tx, ty - th - 4), (tx + tw + 4, ty + 3), color, -1)
        cv2.putText(canvas, label, (tx + 2, ty), cv2.FONT_HERSHEY_SIMPLEX, scale, (0, 0, 0), 1)
    return canvas


class Pipeline:
    def __init__(self, config=None):
        self.config = config or Config()
        self.lock = threading.Lock()
        self.slots = threading.BoundedSemaphore(2)  # one active image + one waiting
        self.models = {}
        self.predictor = None
        self.oom = False
        value = self.config.device
        if value == "auto":
            value = "cuda:0" if torch.cuda.is_available() else "cpu"
        elif value.isdigit():
            value = f"cuda:{value}"
        self.device = torch.device(value)
        if self.device.type not in {"cpu", "cuda"}:
            raise ValueError("Only CPU/CUDA are supported")
        if self.device.type == "cuda":
            if not torch.cuda.is_available():
                raise RuntimeError("CUDA unavailable")
            torch.cuda.set_device(self.device)

    def _load(self, with_sam):
        from ultralytics import YOLO

        if not self.models:
            models = {}
            for name, path in (("uav_pdd2023", self.config.weights_uav),
                               ("visdrone2019", self.config.weights_visdrone)):
                if not path.is_file():
                    raise FileNotFoundError(f"Missing YOLO weights: {path}")
                models[name] = YOLO(str(path)).to(self.device)
            self.models = models
        if with_sam and self.predictor is None:
            if not self.config.sam_checkpoint.is_file():
                raise FileNotFoundError(f"Missing SAM checkpoint: {self.config.sam_checkpoint}")
            sys.path.insert(0, str(SAM_ROOT))
            from sam2.build_sam import build_sam2
            from sam2.sam2_image_predictor import SAM2ImagePredictor

            self.predictor = SAM2ImagePredictor(build_sam2(
                SAM_MODELS[self.config.sam_size][1], str(self.config.sam_checkpoint), device=str(self.device)
            ))

    def process(self, source, supplied_info=None, with_sam=True):
        if not self.slots.acquire(blocking=False):
            raise ServiceBusy("GPU queue full; retry later")
        try:
            with self.lock:
                if self.oom:
                    raise GPUOutOfMemory("Local inference stopped after OOM; move this service to the server")
                try:
                    return self._process(Path(source), supplied_info, with_sam)
                except torch.OutOfMemoryError as exc:
                    self.oom = True
                    raise GPUOutOfMemory("GPU OOM; stop local attempts and run this service on the server") from exc
                finally:
                    if self.predictor is not None:
                        self.predictor.reset_predictor()
        finally:
            self.slots.release()

    def _process(self, source, supplied_info, with_sam):
        config = self.config
        with Image.open(source) as image:
            if image.width * image.height > config.max_pixels:
                raise ValueError(f"Image exceeds {config.max_pixels} pixels")
            # Keep stored pixel orientation so EXIF geometry, boxes and masks share a grid.
            rgb = np.array(image.convert("RGB"))
            source_suffix = ".jpg" if image.format == "JPEG" else "." + image.format.lower()
        geometry = image_geometry(source, supplied_info)
        image_id = uuid.uuid4().hex
        output = config.output / image_id
        output.mkdir(parents=True)
        original_name = "original" + source_suffix
        shutil.copyfile(source, output / original_name)
        artifacts = {"original": original_name}
        started = time.perf_counter()
        if self.device.type == "cuda":
            torch.cuda.reset_peak_memory_stats(self.device)
        self._load(with_sam)
        loaded = time.perf_counter()
        bgr = np.ascontiguousarray(rgb[:, :, ::-1])
        records, model_results = [], []
        for name, model in self.models.items():
            t = time.perf_counter()
            result = model.predict(
                source=bgr, device=str(self.device), imgsz=config.imgsz, conf=config.conf,
                iou=config.iou, max_det=config.max_det_per_model, verbose=False,
            )[0]
            boxes = result.boxes.xyxy.cpu().numpy()
            classes = result.boxes.cls.cpu().numpy().astype(int)
            scores = result.boxes.conf.cpu().numpy()
            height, width = rgb.shape[:2]
            for box, class_id, confidence in zip(boxes, classes, scores):
                box = np.clip(box, [0, 0, 0, 0], [width, height, width, height])
                if box[2] <= box[0] or box[3] <= box[1]:
                    continue
                records.append({
                    "instance_id": len(records) + 1, "model_id": name,
                    "class_id": int(class_id), "class_name": result.names[int(class_id)],
                    "confidence": float(confidence), "box_xyxy": box.tolist(),
                })
            if self.device.type == "cuda":
                torch.cuda.synchronize(self.device)
            model_results.append({"model_id": name, "classes": result.names,
                                  "detections": sum(r["model_id"] == name for r in records),
                                  "elapsed_s": time.perf_counter() - t})
        save_image(output / "combined_yolo.jpg", draw_boxes(rgb, records))
        artifacts["combined_yolo"] = "combined_yolo.jpg"
        sam_started = time.perf_counter()
        union_measurement = None
        measurement_seconds = 0.0
        if with_sam:
            (output / "masks").mkdir()
            union = np.zeros(rgb.shape[:2], dtype=bool)
            overlay = rgb.copy()
            precision = nullcontext()
            if self.device.type == "cuda":
                dtype = torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float16
                precision = torch.autocast("cuda", dtype=dtype)
            with torch.inference_mode(), precision:
                if records:
                    self.predictor.set_image(rgb)  # original RGB, NEVER combined_yolo.jpg
                for record in records:
                    masks, scores, _ = self.predictor.predict(
                        box=np.asarray(record["box_xyxy"], dtype=np.float32), multimask_output=False
                    )
                    mask = masks[0].astype(bool)
                    if mask.shape != rgb.shape[:2]:
                        raise RuntimeError(f"Unexpected SAM mask shape: {mask.shape}")
                    union |= mask
                    color = np.asarray(COLORS[record["model_id"]], dtype=np.float32)
                    overlay[mask] = (overlay[mask] * 0.6 + color * 0.4).astype(np.uint8)
                    name = f"masks/{record['instance_id']:04d}.png"
                    save_image(output / name, mask.astype(np.uint8) * 255)
                    contours, hierarchy = cv2.findContours(
                        mask.astype(np.uint8), cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE
                    )
                    record.update(
                        sam_score=float(scores[0]), mask_file=name, mask_pixels=int(mask.sum()),
                        contours=[{"points_xy": c.reshape(-1, 2).tolist(),
                                   "parent": int(hierarchy[0, i, 3])} for i, c in enumerate(contours)],
                    )
                    measurement_start = time.perf_counter()
                    record["measurement"] = measure_mask(mask, geometry, record["contours"])
                    measurement_seconds += time.perf_counter() - measurement_start
            measurement_start = time.perf_counter()
            union_measurement = measure_mask(union, geometry)
            measurement_seconds += time.perf_counter() - measurement_start
            save_image(output / "union_mask.png", union.astype(np.uint8) * 255)
            save_image(output / "sam_overlay.jpg", draw_boxes(overlay, records))
            artifacts.update(union_mask="union_mask.png", sam_overlay="sam_overlay.jpg")
        if self.device.type == "cuda":
            torch.cuda.synchronize(self.device)
        timings = {"load_s": loaded - started,
                   "sam_s": time.perf_counter() - sam_started - measurement_seconds,
                   "measurement_s": measurement_seconds,
                   "total_s": time.perf_counter() - started}
        report = {
            "schema_version": 2,
            "image_id": image_id, "source_name": source.name, "width": rgb.shape[1], "height": rgb.shape[0],
            "geometry": geometry, "device": str(self.device), "with_sam": with_sam,
            "coordinate_grid": "original_stored_image_pixels", "models": model_results,
            "merge_policy": "concatenate_models_preserving_model_class_namespaces",
            "detections": records, "artifacts": {**artifacts, "report": "result.json"}, "timings": timings,
            "measurement_status": (union_measurement["status"] if union_measurement else "sam_disabled"),
            "union_measurement": union_measurement, "measurement_method": measurement_method(geometry),
        }
        if self.device.type == "cuda":
            report["gpu"] = {
                "name": torch.cuda.get_device_name(self.device),
                "peak_allocated_mb": torch.cuda.max_memory_allocated(self.device) / 1024**2,
                "peak_reserved_mb": torch.cuda.max_memory_reserved(self.device) / 1024**2,
            }
        geometry["image"] = source.name
        (output / "result.json").write_text(
            json.dumps(report, ensure_ascii=False, indent=2, allow_nan=False), encoding="utf-8"
        )
        return report
