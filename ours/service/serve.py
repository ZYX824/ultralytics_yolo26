"""Start the backend-facing service with one GPU worker."""

import argparse
import os

import uvicorn


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--host", default="0.0.0.0")
    parser.add_argument("--port", type=int, default=9000)
    parser.add_argument("--device", default="auto")
    parser.add_argument("--imgsz", type=int, default=1280)
    parser.add_argument("--conf", type=float, default=0.25)
    parser.add_argument("--callback-url", help="POST each detected target to this alarm report endpoint")
    args = parser.parse_args()
    if not 1 <= args.port <= 65535 or args.imgsz <= 0 or not 0 <= args.conf <= 1:
        parser.error("Invalid port, image size or confidence")
    for key in ("device", "imgsz", "conf"):
        os.environ[f"YOLO_SERVICE_{key.upper()}"] = str(getattr(args, key))
    if args.callback_url is not None:
        os.environ["YOLO_SERVICE_CALLBACK_URL"] = args.callback_url
    uvicorn.run("ours.service.app:app", host=args.host, port=args.port, workers=1)


if __name__ == "__main__":
    main()
