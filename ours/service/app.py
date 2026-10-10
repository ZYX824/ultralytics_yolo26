"""HTTP upload interface. Run one worker so models occupy the GPU only once."""

import json
import os
import re
import logging
import tempfile
from pathlib import Path

from fastapi import FastAPI, File, Form, HTTPException, UploadFile
from fastapi.staticfiles import StaticFiles
from PIL import UnidentifiedImageError

from .pipeline import Config, GPUOutOfMemory, Pipeline, ServiceBusy

MAX_UPLOAD_BYTES = 32 * 1024 * 1024


def environment_config():
    options = {}
    for key in ("device", "weights_uav", "weights_visdrone", "output", "sam_size", "sam_checkpoint"):
        value = os.environ.get(f"YOLO_SERVICE_{key.upper()}")
        if value:
            options[key] = value
    for key in ("imgsz", "max_det_per_model"):
        value = os.environ.get(f"YOLO_SERVICE_{key.upper()}")
        if value:
            options[key] = int(value)
    for key in ("conf", "iou"):
        value = os.environ.get(f"YOLO_SERVICE_{key.upper()}")
        if value:
            options[key] = float(value)
    return Config(**options)


def create_app(pipeline=None):
    pipeline = pipeline or Pipeline(environment_config())
    app = FastAPI(title="Dual YOLO + SAM2 Ground Measurement", version="2.0",
                  description="Upload image + geometry; detect, segment and measure on the laser-target ground plane.")
    app.state.pipeline = pipeline
    pipeline.config.output.mkdir(parents=True, exist_ok=True)
    app.mount("/files", StaticFiles(directory=str(pipeline.config.output)), name="files")

    @app.get("/health")
    def health():
        return {"status": "gpu_oom_stopped" if pipeline.oom else "ready",
                "device": str(pipeline.device), "models_loaded": list(pipeline.models),
                "sam_loaded": pipeline.predictor is not None}

    def add_urls(report):
        base = f"/files/{report['image_id']}/"
        report["artifact_urls"] = {key: base + name for key, name in report["artifacts"].items()}
        for detection in report["detections"]:
            if detection.get("mask_file"):
                detection["mask_url"] = base + detection["mask_file"]
        return report

    @app.get("/results/{image_id}", summary="Retrieve a saved prediction and measurements")
    def result(image_id: str):
        if not re.fullmatch(r"[0-9a-f]{32}", image_id):
            raise HTTPException(422, "image_id must be a 32-character lowercase hexadecimal ID")
        path = pipeline.config.output / image_id / "result.json"
        if not path.is_file():
            raise HTTPException(404, "Result not found")
        return add_urls(json.loads(path.read_text(encoding="utf-8")))

    @app.post("/predict", summary="Dual detection, SAM masks and real-size ground-plane estimates")
    def predict(
        image: UploadFile = File(..., description="Original image, including EXIF/XMP if available"),
        info: str = Form("{}", description="One image's geometry JSON; optional calibrated_intrinsics"),
        with_sam: bool = Form(True, description="Enable SAM and physical measurements"),
    ):
        try:
            if len(info) > 64 * 1024:
                raise ValueError("info exceeds 64 KB")
            supplied = json.loads(info)
            if not isinstance(supplied, dict):
                raise ValueError("info must be one image's geometry JSON object")
        except (ValueError, TypeError) as exc:
            raise HTTPException(422, str(exc)) from exc
        try:
            # No user-controlled path: original bytes retain EXIF/XMP for extraction.
            with tempfile.TemporaryDirectory(prefix="dual-yolo-") as folder:
                source = Path(folder) / "input.image"
                size = 0
                with source.open("wb") as handle:
                    while chunk := image.file.read(1024 * 1024):
                        size += len(chunk)
                        if size > MAX_UPLOAD_BYTES:
                            raise HTTPException(413, "Image exceeds 32 MB")
                        handle.write(chunk)
                if not size:
                    raise HTTPException(422, "Empty image")
                report = pipeline.process(source, supplied, with_sam)
        except GPUOutOfMemory as exc:
            raise HTTPException(503, {"code": "gpu_oom", "message": str(exc),
                                      "next_step": "run_this_service_on_server"}) from exc
        except ServiceBusy as exc:
            raise HTTPException(503, str(exc), headers={"Retry-After": "2"}) from exc
        except FileNotFoundError as exc:
            raise HTTPException(503, {"code": "missing_model_files", "message": str(exc)}) from exc
        except (UnidentifiedImageError, ValueError, OSError) as exc:
            raise HTTPException(422, str(exc)) from exc
        except RuntimeError as exc:
            logging.exception("Inference failed")
            raise HTTPException(500, {"code": "inference_failed", "message": "See service log for details"}) from exc
        finally:
            image.file.close()
        return add_urls(report)

    return app


app = create_app()
