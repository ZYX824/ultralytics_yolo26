"""Contract checks for image/info pairing, merged boxes, masks and GPU failures."""

import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import numpy as np
import torch
from fastapi.testclient import TestClient
from PIL import Image

from ours.service.app import create_app
from ours.service.pipeline import Config, GPUOutOfMemory, Pipeline, ServiceBusy, image_geometry


class Boxes:
    def __init__(self, box, class_id):
        self.xyxy = torch.tensor([box], dtype=torch.float32)
        self.cls = torch.tensor([class_id])
        self.conf = torch.tensor([0.9])


class Detector:
    def __init__(self, box, name):
        self.box, self.name = box, name

    def predict(self, source, **kwargs):
        self.source = source.copy()
        result = type("Result", (), {"boxes": Boxes(self.box, 0), "names": {0: self.name}})()
        return [result]


class Predictor:
    def set_image(self, rgb):
        self.input = rgb.copy()

    def predict(self, box, **kwargs):
        mask = np.zeros(self.input.shape[:2], dtype=bool)
        x1, y1, x2, y2 = map(int, box)
        mask[y1:y2, x1:x2] = True
        return np.asarray([mask]), np.asarray([0.8]), None

    def reset_predictor(self):
        self.reset = True


class ServiceTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.folder = Path(self.temp.name)
        self.source = self.folder / "original.png"
        self.rgb = np.full((30, 40, 3), (12, 30, 70), dtype=np.uint8)
        Image.fromarray(self.rgb).save(self.source)
        self.config = Config(device="cpu", output=self.folder / "results")

    def tearDown(self):
        self.temp.cleanup()

    def fake_pipeline(self):
        pipeline = Pipeline(self.config)
        pipeline.models = {
            "uav_pdd2023": Detector([2, 3, 10, 12], "crack"),
            "visdrone2019": Detector([15, 10, 25, 20], "vehicle"),
        }
        pipeline.predictor = Predictor()
        pipeline._load = lambda with_sam: None
        return pipeline

    def test_geometry_recomputed_from_matching_image_and_supplied_inputs(self):
        info = image_geometry(self.source, {
            "center_range_m": 100, "gimbal_pitch_deg": -30, "range_status": "Normal",
            "height_agl_m": 999, "width_px": 40, "height_px": 30,
        })
        self.assertAlmostEqual(info["height_agl_m"], 50)
        self.assertEqual(info["width_px"], 40)
        with self.assertRaises(ValueError):
            image_geometry(self.source, {"width_px": 20})
        with self.assertRaises(ValueError):
            image_geometry(self.source, {"center_range_m": float("nan")})
        with self.assertRaises(ValueError):
            image_geometry(self.source, {"uav_latitude_deg": 100})
        with self.assertRaises(ValueError):
            image_geometry(self.source, {"calibrated_intrinsics": {"fx_px": 1000}})

    def test_dual_classes_preserved_and_sam_gets_unannotated_original(self):
        pipeline = self.fake_pipeline()
        report = pipeline.process(self.source)
        self.assertEqual([r["class_id"] for r in report["detections"]], [0, 0])
        self.assertEqual([r["class_name"] for r in report["detections"]], ["crack", "vehicle"])
        np.testing.assert_array_equal(pipeline.predictor.input, self.rgb)
        self.assertTrue(pipeline.predictor.reset)
        output = self.config.output / report["image_id"]
        self.assertTrue((output / "combined_yolo.jpg").is_file())
        with Image.open(output / "union_mask.png") as image:
            union = np.asarray(image)
        self.assertEqual(int((union > 0).sum()), 8 * 9 + 10 * 10)
        for record in report["detections"]:
            with Image.open(output / record["mask_file"]) as image:
                self.assertEqual(image.size, (40, 30))
            self.assertTrue(record["contours"])
        saved = json.loads((output / "result.json").read_text(encoding="utf-8"))
        self.assertEqual(saved["image_id"], report["image_id"])
        self.assertEqual((output / report["artifacts"]["original"]).read_bytes(), self.source.read_bytes())

    def test_oom_stops_retries_and_queue_overflow_is_rejected(self):
        pipeline = self.fake_pipeline()
        with patch.object(pipeline, "_process", side_effect=torch.OutOfMemoryError("OOM")) as process:
            with self.assertRaises(GPUOutOfMemory):
                pipeline.process(self.source)
            with self.assertRaises(GPUOutOfMemory):
                pipeline.process(self.source)
            self.assertEqual(process.call_count, 1)
        pipeline = self.fake_pipeline()
        pipeline.slots.acquire()
        pipeline.slots.acquire()
        with self.assertRaises(ServiceBusy):
            pipeline.process(self.source)

    def test_http_image_plus_geometry_and_artifact_download(self):
        pipeline = self.fake_pipeline()
        client = TestClient(create_app(pipeline))
        response = client.post("/predict", files={"image": ("image.png", self.source.read_bytes(), "image/png")},
                               data={"with_sam": "true", "info": json.dumps({"center_range_m": 100, "gimbal_pitch_deg": -30,
                                                         "gimbal_roll_deg": 0, "gimbal_yaw_deg": 0,
                                                         "calibrated_intrinsics": {"fx_px": 1000, "fy_px": 1000,
                                                                                   "cx_px": 19.5, "cy_px": 14.5},
                                                         "range_status": "Normal"})})
        self.assertEqual(response.status_code, 200, response.text)
        report = response.json()
        self.assertAlmostEqual(report["geometry"]["height_agl_m"], 50)
        self.assertEqual(len(report["detections"]), 2)
        self.assertEqual(report["measurement_status"], "estimated_ground_plane")
        self.assertGreater(report["union_measurement"]["area_m2"], 0)
        self.assertGreater(report["detections"][0]["measurement"]["extent_long_m"], 0)
        saved = client.get(f"/results/{report['image_id']}")
        self.assertEqual(saved.status_code, 200)
        self.assertEqual(saved.json()["union_measurement"], report["union_measurement"])
        self.assertEqual(client.get("/results/" + "0" * 32).status_code, 404)
        self.assertEqual(client.get("/results/not-an-id").status_code, 422)
        self.assertEqual(client.get(report["artifact_urls"]["combined_yolo"]).status_code, 200)
        self.assertEqual(client.get(report["detections"][0]["mask_url"]).status_code, 200)
        invalid = client.post("/predict", files={"image": ("x.png", self.source.read_bytes())},
                              data={"info": "[]"})
        self.assertEqual(invalid.status_code, 422)
        broken = client.post("/predict", files={"image": ("bad.png", b"broken")})
        self.assertEqual(broken.status_code, 422)
        with patch.object(pipeline, "process", side_effect=GPUOutOfMemory("stop")):
            oom = client.post("/predict", files={"image": ("x.png", self.source.read_bytes())})
            self.assertEqual(oom.status_code, 503)
            self.assertEqual(oom.json()["detail"]["code"], "gpu_oom")

    def test_http_defaults_to_detection_without_loading_sam(self):
        pipeline = self.fake_pipeline()
        pipeline.predictor = None
        with TestClient(create_app(pipeline)) as client, patch.object(pipeline, "_load") as load:
            response = client.post("/predict", files={"image": ("image.png", self.source.read_bytes())})
            self.assertEqual(response.status_code, 200, response.text)
            load.assert_called_once_with(False)
            report = response.json()
            self.assertFalse(report["with_sam"])
            self.assertEqual(len(report["detections"]), 2)
            self.assertEqual(report["measurement_status"], "sam_disabled")
            self.assertIsNone(report["union_measurement"])
            self.assertNotIn("sam_overlay", report["artifact_urls"])
            self.assertTrue(all("mask_url" not in detection for detection in report["detections"]))
            self.assertEqual(client.get(report["artifact_urls"]["combined_yolo"]).status_code, 200)


if __name__ == "__main__":
    unittest.main()
