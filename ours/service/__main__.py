"""Run the same service pipeline locally without starting an HTTP server."""

import argparse
import json
from pathlib import Path

from .pipeline import Config, GPUOutOfMemory, Pipeline


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", type=Path, required=True)
    parser.add_argument("--info", type=Path, help="One geometry object or single-image extraction report")
    parser.add_argument("--device", default="auto")
    parser.add_argument("--imgsz", type=int, default=640)
    parser.add_argument("--conf", type=float, default=0.25)
    parser.add_argument("--max-det", type=int, default=100, help="Maximum detections from each model")
    parser.add_argument("--yolo-only", action="store_true")
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    info = None
    if args.info:
        info = json.loads(args.info.read_text(encoding="utf-8"))
        if isinstance(info, dict) and "images" in info:
            if len(info["images"]) != 1:
                parser.error("--info report must contain exactly one image")
            info = info["images"][0]
    options = {"device": args.device, "imgsz": args.imgsz, "conf": args.conf, "max_det_per_model": args.max_det}
    if args.output:
        options["output"] = args.output
    pipeline = Pipeline(Config(**options))
    try:
        report = pipeline.process(args.source, info, with_sam=not args.yolo_only)
    except GPUOutOfMemory as exc:
        raise SystemExit(str(exc))
    print(json.dumps({"image_id": report["image_id"], "models": report["models"],
                      "measurement_status": report["measurement_status"],
                      "union_area_m2": (report["union_measurement"] or {}).get("area_m2"),
                      "gpu": report.get("gpu"), "timings": report["timings"],
                      "output": str((pipeline.config.output / report["image_id"]).resolve())}, indent=2))


if __name__ == "__main__":
    main()
