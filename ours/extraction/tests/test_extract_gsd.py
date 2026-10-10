"""Verify per-image range geometry and the extraction boundary."""

import json
import math

import pytest
from PIL import Image, TiffImagePlugin

from ours.extraction.extract_gsd import FULL_FRAME_DIAGONAL_MM, estimate_gsd, extract_image, main, parse_args


def metadata(**changes):
    return {
        "width_px": 4000, "height_px": 3000, "focal_length_mm": 8,
        "focal_length_35mm_mm": 24, "gimbal_pitch_deg": -45, "gimbal_roll_deg": 0, "gimbal_yaw_deg": 76.7,
        "center_range_m": 177.947, "range_status": "Normal", "relative_altitude_m": 111.121,
        "warnings": [], **changes,
    }


def test_range_height_and_oblique_local_scale():
    scale = estimate_gsd(metadata(), sensor_width_mm=8, sensor_height_mm=6)
    assert scale["height_agl_m"] == pytest.approx(177.947 / math.sqrt(2))
    assert scale["horizontal_distance_to_center_m"] == pytest.approx(177.947 / math.sqrt(2))
    assert scale["height_agl_m"] != 111.121
    assert scale["center_gsd_cross_track_m_per_pixel"] == pytest.approx(177.947 / 4000)
    assert scale["center_gsd_along_track_m_per_pixel"] == pytest.approx(177.947 * math.sqrt(2) / 4000)
    assert scale["gsd_m_per_pixel"] is None
    assert scale["status"] == "oblique_requires_projection"


def test_each_image_uses_its_own_distance_and_angle():
    first = estimate_gsd(metadata(center_range_m=100, gimbal_pitch_deg=-30))
    second = estimate_gsd(metadata(center_range_m=200, gimbal_pitch_deg=-60))
    assert first["height_agl_m"] == pytest.approx(50)
    assert second["height_agl_m"] == pytest.approx(100 * math.sqrt(3))
    assert second["height_agl_m"] != first["height_agl_m"]


def test_nadir_scale_and_equivalent_diagonal_intrinsics():
    scale = estimate_gsd(metadata(center_range_m=100, gimbal_pitch_deg=-90))
    assert scale["height_agl_m"] == 100
    assert scale["gsd_m_per_pixel"] == pytest.approx(100 * FULL_FRAME_DIAGONAL_MM / (24 * 5000))
    assert scale["status"] == "estimated_nadir"


@pytest.mark.parametrize("changes,status", [
    ({"center_range_m": None}, "missing_center_range"),
    ({"center_range_m": 0}, "missing_center_range"),
    ({"center_range_m": float("nan")}, "missing_center_range"),
    ({"range_status": "Abnormal"}, "unverified_center_range"),
    ({"range_status": None}, "unverified_center_range"),
    ({"gimbal_pitch_deg": None}, "missing_or_invalid_downward_pitch"),
    ({"gimbal_pitch_deg": 0}, "missing_or_invalid_downward_pitch"),
    ({"gimbal_pitch_deg": 10}, "missing_or_invalid_downward_pitch"),
    ({"gimbal_pitch_deg": -100}, "missing_or_invalid_downward_pitch"),
])
def test_no_takeoff_height_fallback(changes, status):
    scale = estimate_gsd(metadata(**changes))
    assert scale["height_agl_m"] is None
    assert scale["gsd_m_per_pixel"] is None
    assert scale["status"] == status


@pytest.mark.parametrize("changes,status", [
    ({"focal_length_35mm_mm": None}, "missing_camera_intrinsics"),
    ({"gimbal_roll_deg": None}, "missing_projection_attitude"),
    ({"gimbal_yaw_deg": None}, "missing_projection_attitude"),
    ({"orientation": 6}, "orientation_requires_normalization"),
    ({"digital_zoom_ratio": 2}, "digital_zoom_requires_calibration"),
])
def test_height_remains_available_when_projection_parameters_missing(changes, status):
    scale = estimate_gsd(metadata(**changes))
    assert scale["height_agl_m"] == pytest.approx(177.947 / math.sqrt(2))
    assert scale["gsd_m_per_pixel"] is None
    assert scale["status"] == status


def write_image(path, distance, pitch):
    exif = Image.Exif()
    exif[271] = "DJI"
    exif[272] = "NotNeededInReport"
    exif[306] = "2026:07:30 15:15:14"
    exif[34665] = {37386: TiffImagePlugin.IFDRational(6.72), 41989: 24, 34855: 150}
    Image.new("RGB", (40, 30)).save(path, exif=exif)
    packet = f'''<x:xmpmeta xmlns:x="adobe:ns:meta/" xmlns:d="http://www.dji.com/drone-dji/1.0/">
    <x:item d:RelativeAltitude="111.121" d:GimbalPitchDegree="{pitch}" d:GimbalRollDegree="0"
    d:GimbalYawDegree="76.7" d:LRFTargetDistance="{distance}" d:LRFStatus="Normal"
    d:GpsLatitude="30.540428408" d:GpsLongitude="104.508034722"
    d:LRFTargetLat="30.5406860" d:LRFTargetLon="104.5093054" d:LRFTargetAbsAlt="373"/>
    <d:AbsoluteAltitude>498.721</d:AbsoluteAltitude></x:xmpmeta>'''.encode()
    payload = b"http://ns.adobe.com/xap/1.0/\x00" + packet
    original = path.read_bytes()
    path.write_bytes(original[:2] + b"\xff\xe1" + (len(payload) + 2).to_bytes(2, "big") + payload + original[2:])


@pytest.mark.parametrize("legacy_pillow", [False, True])
def test_jpeg_geometry_only_and_legacy_xmp(tmp_path, monkeypatch, legacy_pillow):
    path = tmp_path / "sample.jpeg"
    write_image(path, 177.947, -45)
    if legacy_pillow:
        original_open = Image.open

        def open_without_xmp(*args, **kwargs):
            image = original_open(*args, **kwargs)
            image.info.pop("xmp", None)
            return image

        monkeypatch.setattr(Image, "open", open_without_xmp)
    info = extract_image(path)
    assert info["uav_latitude_deg"] == 30.540428408
    assert info["uav_longitude_deg"] == 104.508034722
    assert info["center_target_latitude_deg"] == 30.5406860
    assert info["center_target_longitude_deg"] == 104.5093054
    assert info["center_target_absolute_altitude_m"] == 373
    assert info["focal_length_mm"] == pytest.approx(6.72)
    assert info["height_agl_m"] == pytest.approx(125.82753, abs=0.00001)
    assert not {"raw_exif", "raw_xmp", "camera_model", "captured_at", "relative_altitude_m"} & info.keys()
    json.dumps(info, allow_nan=False)


def test_batch_reports_each_height_and_handles_corrupt_image(tmp_path):
    images = tmp_path / "images"
    images.mkdir()
    write_image(images / "a.jpeg", 100, -30)
    write_image(images / "b.jpeg", 200, -60)
    Image.new("RGB", (40, 30)).save(images / "stripped.png")
    (images / "broken.jpeg").write_bytes(b"not an image")
    output = tmp_path / "reports"
    assert main(["--source", str(images), "--output", str(output)]) == 1
    report = json.loads((output / "metadata.json").read_text(encoding="utf-8"))
    records = report["images"]
    assert records[0]["height_agl_m"] == pytest.approx(50)
    assert records[1]["height_agl_m"] == pytest.approx(100 * math.sqrt(3))
    assert {r["status"] for r in records} == {"oblique_requires_projection", "missing_center_range", "error"}
    assert report["schema_version"] == 2
    assert (output / "summary.csv").read_bytes().startswith(b"\xef\xbb\xbf")


@pytest.mark.parametrize("value", ["nan", "inf", "0", "-1"])
def test_cli_rejects_invalid_sensor_dimensions(value):
    with pytest.raises(SystemExit):
        parse_args(["--source", "image.jpg", "--sensor-width-mm", value, "--sensor-height-mm", "6"])
