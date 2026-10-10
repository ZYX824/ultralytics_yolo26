"""Extract only geometry needed to map image contours to real ground dimensions.

python ours/extraction/extract_gsd.py --source image.jpeg

Per-image height above the center target plane is estimated from DJI laser
range and gimbal pitch: H = R * sin(abs(pitch)). RelativeAltitude is never used
as a substitute. The laser target is assumed to coincide with the image center.
"""

import argparse
import csv
import json
import math
import sys
import xml.etree.ElementTree as ET
from pathlib import Path

from PIL import Image

IMAGE_SUFFIXES = {".jpg", ".jpeg", ".png", ".tif", ".tiff", ".webp", ".bmp"}
DJI_NS = "http://www.dji.com/drone-dji/1.0/"
FULL_FRAME_DIAGONAL_MM = math.hypot(36, 24)
CSV_FIELDS = (
    "image", "width_px", "height_px", "orientation",
    "uav_latitude_deg", "uav_longitude_deg", "uav_absolute_altitude_m",
    "center_target_latitude_deg", "center_target_longitude_deg", "center_target_absolute_altitude_m",
    "center_range_m", "range_status", "gimbal_pitch_deg", "gimbal_roll_deg", "gimbal_yaw_deg",
    "focal_length_mm", "focal_length_35mm_mm", "digital_zoom_ratio",
    "height_agl_m", "height_source", "horizontal_distance_to_center_m",
    "focal_x_px", "focal_y_px", "principal_x_px", "principal_y_px", "intrinsics_source",
    "center_gsd_cross_track_m_per_pixel", "center_gsd_along_track_m_per_pixel",
    "gsd_m_per_pixel", "status", "warnings", "error",
)


def number(value):
    """Read finite EXIF rationals and XMP numbers."""
    try:
        result = float(value)
        return result if math.isfinite(result) else None
    except (TypeError, ValueError, ZeroDivisionError, OverflowError):
        return None


def positive(value):
    result = number(value)
    return result if result is not None and result > 0 else None


def coordinate(values, ref):
    if not isinstance(values, (tuple, list)) or len(values) != 3:
        return None
    parts = [number(v) for v in values]
    if any(v is None for v in parts) or ref not in {"N", "S", "E", "W"}:
        return None
    result = parts[0] + parts[1] / 60 + parts[2] / 3600
    return -result if ref in {"S", "W"} else result


def xmp_fields(packet):
    if not packet:
        return {}
    root = ET.fromstring(packet)
    fields = {}
    for element in root.iter():
        fields.update(element.attrib)
        if element.text and element.text.strip():
            fields[element.tag] = element.text.strip()
    return fields


def read_metadata(path):
    """Read geometric fields only, without decoding/resizing the input pixels."""
    warnings = []
    with Image.open(path) as image:
        width, height = image.size
        exif = image.getexif()
        try:
            optics = exif.get_ifd(34665)
            gps = exif.get_ifd(34853)
        except (OSError, ValueError, SyntaxError) as exc:
            optics, gps = {}, {}
            warnings.append(f"EXIF geometry unreadable: {exc}")
        packet = image.info.get("xmp") or image.info.get("XML:com.adobe.xmp")
        if not packet:
            header = b"http://ns.adobe.com/xap/1.0/\x00"
            for marker, payload in getattr(image, "applist", []):
                if marker == "APP1" and payload.startswith(header):
                    packet = payload[len(header):]
                    break
        if not packet and hasattr(image, "tag_v2"):
            packet = image.tag_v2.get(700)
        try:
            xmp = xmp_fields(packet)
        except (ET.ParseError, TypeError, ValueError) as exc:
            xmp = {}
            warnings.append(f"XMP geometry unreadable: {exc}")

    def dji(name):
        return xmp.get(f"{{{DJI_NS}}}{name}")

    latitude = number(dji("GpsLatitude"))
    longitude = number(dji("GpsLongitude"))
    altitude = number(dji("AbsoluteAltitude"))
    if altitude is None:
        altitude = number(gps.get(6))
        if gps.get(5) in (1, b"\x01") and altitude is not None:
            altitude = -altitude
    return {
        "image": str(Path(path).resolve()), "width_px": width, "height_px": height,
        "orientation": exif.get(274, 1),
        "uav_latitude_deg": latitude if latitude is not None else coordinate(gps.get(2), gps.get(1)),
        "uav_longitude_deg": longitude if longitude is not None else coordinate(gps.get(4), gps.get(3)),
        "uav_absolute_altitude_m": altitude,
        # User's center-target assumption: preserve the DJI LRF point rather than inventing a point.
        "center_target_latitude_deg": number(dji("LRFTargetLat")),
        "center_target_longitude_deg": number(dji("LRFTargetLon")),
        "center_target_absolute_altitude_m": number(dji("LRFTargetAbsAlt")),
        "center_range_m": number(dji("LRFTargetDistance")),
        "range_status": dji("LRFStatus"),
        "gimbal_pitch_deg": number(dji("GimbalPitchDegree")),
        "gimbal_roll_deg": number(dji("GimbalRollDegree")),
        "gimbal_yaw_deg": number(dji("GimbalYawDegree")),
        "focal_length_mm": number(optics.get(37386)),
        "focal_length_35mm_mm": number(optics.get(41989)),
        "digital_zoom_ratio": number(optics.get(41988)),
        "warnings": warnings,
    }


def estimate_gsd(metadata, *, sensor_width_mm=None, sensor_height_mm=None, focal_length_mm=None):
    """Derive per-image height and center-local scale from laser range and pitch.

    Center scales are in ground cross-track / along-track directions. They are
    image x/y scales only when camera roll is zero, and never a whole-image GSD
    for an oblique image. Ground is approximated by a horizontal target plane.
    """
    m = metadata
    warnings = list(m.get("warnings", []))
    result = {
        "height_agl_m": None, "height_source": None, "horizontal_distance_to_center_m": None,
        "focal_x_px": None, "focal_y_px": None,
        "principal_x_px": (m["width_px"] - 1) / 2, "principal_y_px": (m["height_px"] - 1) / 2,
        "intrinsics_source": None,
        "center_gsd_cross_track_m_per_pixel": None, "center_gsd_along_track_m_per_pixel": None,
        "gsd_m_per_pixel": None, "status": "missing_center_range", "warnings": warnings,
    }
    if (sensor_width_mm is None) != (sensor_height_mm is None):
        raise ValueError("Supply both sensor_width_mm and sensor_height_mm")
    for name, value in (("sensor_width_mm", sensor_width_mm), ("sensor_height_mm", sensor_height_mm),
                        ("focal_length_mm", focal_length_mm)):
        if value is not None and positive(value) is None:
            raise ValueError(f"{name} must be finite and positive")
    if focal_length_mm is not None and sensor_width_mm is None:
        raise ValueError("Actual focal length override requires active sensor dimensions")

    # Intrinsics are retained even when range/attitude is missing.
    focal = positive(focal_length_mm if focal_length_mm is not None else m.get("focal_length_mm"))
    equivalent = positive(m.get("focal_length_35mm_mm"))
    width, image_height = m["width_px"], m["height_px"]
    if sensor_width_mm is not None and focal is not None:
        result.update(focal_x_px=focal * width / sensor_width_mm,
                      focal_y_px=focal * image_height / sensor_height_mm,
                      intrinsics_source="user_active_sensor_dimensions_and_actual_focal_length")
    elif sensor_width_mm is None and equivalent is not None:
        fx = equivalent * math.hypot(width, image_height) / FULL_FRAME_DIAGONAL_MM
        result.update(focal_x_px=fx, focal_y_px=fx,
                      intrinsics_source="approximate_diagonal_35mm_equivalent_square_pixels")
        warnings.append("35mm equivalent focal length is approximate; calibrated intrinsics are preferable.")

    distance = positive(m.get("center_range_m"))
    if distance is None:
        return result
    range_status = m.get("range_status")
    if range_status is None or str(range_status).strip().lower() != "normal":
        result["status"] = "unverified_center_range"
        warnings.append("Laser range is not marked Normal; do not use it for height estimation.")
        return result
    pitch = number(m.get("gimbal_pitch_deg"))
    if pitch is None or not -90 <= pitch < 0:
        result["status"] = "missing_or_invalid_downward_pitch"
        return result
    depression = math.radians(-pitch)
    height = distance * math.sin(depression)
    result.update(height_agl_m=height, height_source="center_laser_range_times_sin_depression",
                  horizontal_distance_to_center_m=distance * math.cos(depression))
    warnings.append("Assumes laser target at image center, aligned laser/camera axes and horizontal target plane.")

    fx, fy = result["focal_x_px"], result["focal_y_px"]
    if fx is None or fy is None:
        result["status"] = "missing_camera_intrinsics"
        return result
    if m.get("orientation", 1) != 1:
        result["status"] = "orientation_requires_normalization"
        return result
    zoom = number(m.get("digital_zoom_ratio"))
    if zoom is not None and zoom > 1.001:
        result["status"] = "digital_zoom_requires_calibration"
        return result
    roll = number(m.get("gimbal_roll_deg"))
    yaw = number(m.get("gimbal_yaw_deg"))
    if roll is None or yaw is None:
        result["status"] = "missing_projection_attitude"
        return result

    # With roll, image x/y rotate relative to cross-/along-track. For square
    # pixels fx==fy this rotation leaves these principal local scales unchanged.
    if abs(fx - fy) / max(fx, fy) <= 1e-9 or abs(roll) <= 1e-9:
        result.update(center_gsd_cross_track_m_per_pixel=distance / fx,
                      center_gsd_along_track_m_per_pixel=distance / (fy * math.sin(depression)))
    if abs(pitch + 90) <= 1e-9 and abs(fx - fy) / max(fx, fy) <= 1e-9:
        result.update(gsd_m_per_pixel=height / fx, status="estimated_nadir")
    else:
        result["status"] = "oblique_requires_projection"
    warnings.append("Use contour ground-plane projection for oblique views; center scales are local only.")
    return result


def extract_image(path, **options):
    """Extract mapping geometry and derive height independently for this image."""
    metadata = read_metadata(path)
    metadata.update(estimate_gsd(metadata, **options))
    return metadata


def parse_args(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", type=Path, required=True, help="Original image or directory")
    parser.add_argument("--output", type=Path, default=Path(__file__).resolve().parent / "runs" / "latest")
    parser.add_argument("--recursive", action="store_true")
    parser.add_argument("--sensor-width-mm", type=float, help="Active sensor width for the current FOV")
    parser.add_argument("--sensor-height-mm", type=float, help="Active sensor height for the current FOV")
    parser.add_argument("--focal-length-mm", type=float, help="Actual focal length, not 35mm equivalent")
    args = parser.parse_args(argv)
    for key in ("sensor_width_mm", "sensor_height_mm", "focal_length_mm"):
        value = getattr(args, key)
        if value is not None and positive(value) is None:
            parser.error(f"--{key.replace('_', '-')} must be finite and positive")
    if (args.sensor_width_mm is None) != (args.sensor_height_mm is None):
        parser.error("Supply both --sensor-width-mm and --sensor-height-mm")
    if args.focal_length_mm is not None and args.sensor_width_mm is None:
        parser.error("--focal-length-mm requires active sensor dimensions")
    return args


def main(argv=None):
    args = parse_args(argv)
    if args.source.is_file() and args.source.suffix.lower() in IMAGE_SUFFIXES:
        paths = [args.source]
    elif args.source.is_dir():
        candidates = args.source.rglob("*") if args.recursive else args.source.iterdir()
        paths = sorted(p for p in candidates if p.is_file() and p.suffix.lower() in IMAGE_SUFFIXES)
    else:
        raise SystemExit(f"Image/directory not found or unsupported: {args.source}")
    if not paths:
        raise SystemExit(f"No supported images in {args.source}")
    options = {key: getattr(args, key) for key in ("sensor_width_mm", "sensor_height_mm", "focal_length_mm")}
    records = []
    for path in paths:
        try:
            record = extract_image(path, **options)
        except (OSError, ValueError, SyntaxError) as exc:
            record = {"image": str(path.resolve()), "status": "error", "error": str(exc), "warnings": []}
        records.append(record)
        print(f"{path.name}: height={record.get('height_agl_m')} m; {record['status']}")
    args.output.mkdir(parents=True, exist_ok=True)
    report = {
        "schema_version": 2, "coordinate_grid": "original_stored_image_pixels",
        "height_formula": "center_range_m * sin(abs(gimbal_pitch_deg))",
        "units": {"distance": "m", "gsd": "m/pixel", "focal_length": "mm", "angles": "degree"},
        "images": records,
    }
    (args.output / "metadata.json").write_text(
        json.dumps(report, ensure_ascii=False, indent=2, allow_nan=False), encoding="utf-8"
    )
    with (args.output / "summary.csv").open("w", newline="", encoding="utf-8-sig") as handle:
        writer = csv.DictWriter(handle, fieldnames=CSV_FIELDS, extrasaction="ignore")
        writer.writeheader()
        for record in records:
            writer.writerow({**record, "warnings": "; ".join(record["warnings"])})
    print(f"Saved {len(records)} geometry records to {args.output.resolve()}")
    return 1 if any(r["status"] == "error" for r in records) else 0


if __name__ == "__main__":
    sys.exit(main())
