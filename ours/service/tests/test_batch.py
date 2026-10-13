"""Batch HTTP/network contract tests that do not require GPU/model dependencies."""

import json
import shutil
import tempfile
import threading
import time
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

import requests
from fastapi import FastAPI
from fastapi.staticfiles import StaticFiles
from fastapi.testclient import TestClient
from PIL import Image

from ours.extraction.extract_gsd import read_metadata
from ours.service.batch import (
    MAX_DOWNLOAD_BYTES,
    BatchManager,
    BatchRequest,
    batch_router,
    download_image,
    send_callback,
)
from ours.service.errors import GPUOutOfMemory


class FakePipeline:
    def __init__(self, output):
        self.config = SimpleNamespace(output=output)
        self.oom = False
        self.inputs = []

    def process(self, source, info, with_sam=True):
        self.inputs.append((source.read_bytes(), info, with_sam, read_metadata(source)))
        image_id = f"{len(self.inputs):032x}"
        output = self.config.output / image_id
        output.mkdir()
        shutil.copyfile(source, output / "original.png")
        Image.new("RGB", (40, 30)).save(output / "combined_yolo.jpg")
        measured = {"status": "estimated_ground_plane", "reason": None, "area_m2": 2.0,
                    "extent_long_m": 2.0, "extent_short_m": 1.0, "touches_image_border": False}
        report = {
            "image_id": image_id, "width": 40, "height": 30, "with_sam": with_sam,
            "measurement_status": "estimated_ground_plane" if with_sam else "sam_disabled",
            "union_measurement": measured if with_sam else None,
            "artifacts": {"original": "original.png", "combined_yolo": "combined_yolo.jpg",
                          "report": "result.json"},
            "detections": [{"instance_id": 1, "model_id": "visdrone2019", "class_id": 3,
                            "class_name": "car", "confidence": 0.9, "box_xyxy": [2, 3, 10, 12],
                            "measurement": measured if with_sam else None}],
        }
        if with_sam:
            Image.new("RGB", (40, 30)).save(output / "sam_overlay.jpg")
            Image.new("L", (40, 30)).save(output / "mask.png")
            report["artifacts"]["sam_overlay"] = "sam_overlay.jpg"
            report["detections"][0]["mask_file"] = "mask.png"
        (output / "result.json").write_text(json.dumps(report), encoding="utf-8")
        return report


class Response:
    def __init__(self, content=b"", status=200, headers=None, body=None, chunks=None):
        self.content, self.status_code = content, status
        self.headers, self.body, self.chunks = headers or {}, body, chunks

    def __enter__(self):
        return self

    def __exit__(self, *args):
        pass

    def raise_for_status(self):
        if self.status_code >= 400:
            raise requests.HTTPError("download failed")

    def iter_content(self, chunk_size):
        return iter(self.chunks if self.chunks is not None else [self.content])

    def json(self):
        return self.body


class BatchTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.root = Path(self.temp.name)
        source = self.root / "source.png"
        exif = Image.Exif()
        exif[274] = 1
        Image.new("RGB", (40, 30), (12, 30, 70)).save(source, exif=exif)
        self.image = source.read_bytes()
        self.pipeline = FakePipeline(self.root / "results")
        self.manager = BatchManager(self.pipeline, public_base="http://ai.example:9000", callback_format="demo")
        self.app = FastAPI()
        self.app.include_router(batch_router(self.manager))
        self.app.mount("/files", StaticFiles(directory=str(self.pipeline.config.output)), name="files")
        self.client = TestClient(self.app)

    def tearDown(self):
        self.manager.close()
        self.temp.cleanup()

    def wait_task(self, task_id):
        deadline = time.monotonic() + 10
        while time.monotonic() < deadline:
            response = self.client.get(f"/tasks/{task_id}")
            self.assertEqual(response.status_code, 200)
            job = response.json()
            if job["status"] in {"completed", "partial_failed", "failed", "interrupted"}:
                return job
            time.sleep(0.01)
        self.fail("Task did not complete")

    def test_50_urls_produce_linked_results_and_preserve_original_bytes(self):
        urls = [f"http://minio.example:9000/dbxq/image-{i}.jpeg" for i in range(50)]
        with patch("ours.service.batch.requests.get", return_value=Response(self.image)) as get:
            accepted = self.client.post("/detect", json={"types": [12001], "pictures": urls, "with_sam": True})
            self.assertEqual(accepted.status_code, 200, accepted.text)
            self.assertEqual(accepted.json()["code"], 200)
            job = self.wait_task(accepted.json()["task_id"])
        self.assertEqual((job["status"], job["processed"], job["succeeded"], job["failed"]),
                         ("completed", 50, 50, 0))
        self.assertEqual(get.call_count, 50)
        self.assertEqual([item["picturePath"] for item in job["pictures"]], urls)
        self.assertEqual(job["pictures"][0]["pictureId"], "image-001")
        self.assertTrue(all(data == self.image and sam for data, info, sam, metadata in self.pipeline.inputs))
        self.assertEqual(self.pipeline.inputs[0][3]["orientation"], 1)
        result = job["pictures"][0]["result"]
        self.assertEqual(result["detections"][0]["box_xyxy"], [2, 3, 10, 12])
        self.assertEqual(result["detections"][0]["measurement"]["area_m2"], 2.0)
        self.assertTrue(result["detections"][0]["mask_url"].startswith("http://ai.example:9000/files/"))
        for url in [*result["artifact_urls"].values(), result["detections"][0]["mask_url"]]:
            self.assertEqual(self.client.get(url).status_code, 200)
        self.manager.close()
        recovered = BatchManager(self.pipeline)
        try:
            recovered.recover()
            self.assertEqual(recovered.get(job["task_id"])["status"], "completed")
        finally:
            recovered.close()

    def test_failed_download_does_not_stop_batch_and_callbacks_include_failures(self):
        pictures = [{"pictureId": f"p-{i}", "picturePath": f"http://minio.example/{i}.jpeg",
                     "info": {"center_range_m": 100}} for i in range(3)]
        responses = [Response(self.image), Response(status=404), Response(self.image)]
        with patch("ours.service.batch.requests.get", side_effect=responses), \
                patch("ours.service.batch.requests.post", return_value=Response()) as post:
            accepted = self.client.post("/detect", json={"types": [12001], "pictures": pictures,
                                                          "callbackUrl": "http://backend.example/demo/results"})
            job = self.wait_task(accepted.json()["task_id"])
        self.assertEqual((job["status"], job["succeeded"], job["failed"]), ("partial_failed", 2, 1))
        self.assertEqual(post.call_count, 3)
        callbacks = [call.kwargs["json"] for call in post.call_args_list]
        self.assertEqual([payload["status"] for payload in callbacks], ["completed", "failed", "completed"])
        self.assertEqual(callbacks[1]["error"]["code"], "download_failed")
        self.assertEqual(self.pipeline.inputs[0][1], {"center_range_m": 100})
        self.assertTrue(all(item["callback"]["status"] == "delivered" for item in job["pictures"]))

    def test_callback_retry_preserves_event_and_inference_result(self):
        with patch("ours.service.batch.requests.get", return_value=Response(self.image)), \
                patch("ours.service.batch.requests.post", side_effect=[
                    Response(status=503), Response(b"json", body={"code": 500}), Response(),
                ]) as post, patch("ours.service.batch.time.sleep"):
            accepted = self.client.post("/detect", json={"types": [12001], "pictures": ["http://minio.example/image.jpeg"],
                                                          "callbackUrl": "http://backend.example/demo/results"})
            job = self.wait_task(accepted.json()["task_id"])
        self.assertEqual(job["pictures"][0]["callback"], {"status": "delivered", "attempts": 3, "error": None})
        self.assertEqual(len({call.kwargs["json"]["eventId"] for call in post.call_args_list}), 1)

    def test_failed_callback_keeps_results_and_only_fixed_endpoint_gets_token(self):
        self.manager.callback_url = "http://backend.example/configured"
        self.manager.callback_token = "test-secret"
        with patch("ours.service.batch.requests.get", return_value=Response(self.image)), \
                patch("ours.service.batch.requests.post", return_value=Response(status=500)) as post, \
                patch("ours.service.batch.time.sleep"):
            accepted = self.client.post("/detect", json={"types": [12001], "pictures": ["http://minio.example/image.jpeg"],
                                                          "callbackUrl": "http://other.example/results"})
            job = self.wait_task(accepted.json()["task_id"])
        item = job["pictures"][0]
        self.assertEqual((job["status"], item["status"], item["callback"]["status"]),
                         ("completed", "completed", "failed"))
        self.assertIsNotNone(item["result"])
        self.assertEqual(post.call_args.kwargs["headers"], {})
        with patch("ours.service.batch.requests.get", return_value=Response(self.image)), \
                patch("ours.service.batch.requests.post", return_value=Response()) as post:
            accepted = self.client.post("/detect", json={"types": [12001], "pictures": ["http://minio.example/image.jpeg"]})
            self.wait_task(accepted.json()["task_id"])
        self.assertEqual(post.call_args.kwargs["headers"], {"Authorization": "Bearer test-secret"})

    def test_empty_detection_is_completed_and_oom_skips_remaining_downloads(self):
        original = self.pipeline.process

        def empty(*args, **kwargs):
            result = original(*args, **kwargs)
            result["detections"] = []
            result["measurement_status"] = "empty_mask"
            result["union_measurement"] = {"status": "empty_mask", "area_m2": 0}
            return result

        with patch("ours.service.batch.requests.get", return_value=Response(self.image)), \
                patch.object(self.pipeline, "process", side_effect=empty):
            accepted = self.client.post("/detect", json={"types": [12001], "pictures": ["http://minio.example/empty.jpeg"]})
            job = self.wait_task(accepted.json()["task_id"])
        self.assertEqual(job["pictures"][0]["result"]["detections"], [])
        self.assertEqual(job["status"], "completed")

        def oom(*args, **kwargs):
            self.pipeline.oom = True
            raise GPUOutOfMemory("OOM")

        with patch("ours.service.batch.requests.get", return_value=Response(self.image)) as get, \
                patch.object(self.pipeline, "process", side_effect=oom):
            accepted = self.client.post("/detect", json={"types": [12001], "pictures": ["http://minio.example/1.jpeg",
                                                                       "http://minio.example/2.jpeg"]})
            job = self.wait_task(accepted.json()["task_id"])
        self.assertEqual(get.call_count, 1)
        self.assertTrue(all(item["error"]["code"] == "gpu_oom" for item in job["pictures"]))

    def test_validation_and_bounded_queue(self):
        for body in [
            {"types": [12001], "pictures": []}, {"types": [12001], "pictures": ["http://minio.example/x.jpeg"] * 51},
            {"types": [12001], "pictures": ["file:///tmp/image.jpeg"]},
            {"types": [12001], "pictures": [{"pictureId": "same", "picturePath": "http://minio.example/x"}] * 2},
            {"pictures": ["http://minio.example/x"], "types": ["1"]},
        ]:
            self.assertEqual(self.client.post("/detect", json=body).status_code, 422)
        self.assertEqual(self.client.get("/tasks/not-an-id").status_code, 422)
        self.assertEqual(self.client.get("/tasks/" + "0" * 32).status_code, 404)
        started, release = threading.Event(), threading.Event()

        def blocked(url, destination):
            started.set()
            release.wait(5)
            destination.write_bytes(self.image)

        with patch("ours.service.batch.download_image", side_effect=blocked):
            first = self.client.post("/detect", json={"types": [12001], "pictures": ["http://minio.example/1"]})
            self.assertTrue(started.wait(2))
            second = self.client.post("/detect", json={"types": [12001], "pictures": ["http://minio.example/2"]})
            rejected = self.client.post("/detect", json={"types": [12001], "pictures": ["http://minio.example/3"]})
            self.assertEqual(rejected.status_code, 503)
            self.assertEqual(rejected.headers["retry-after"], "2")
            release.set()
            self.wait_task(first.json()["task_id"])
            self.wait_task(second.json()["task_id"])

    def test_algorithm_ids_are_required_and_unsupported_selection_rejects_whole_request(self):
        picture = {"pictureId": "media-id", "picturePath": "http://minio.example/image.jpeg"}
        with patch.object(self.pipeline, "process") as process:
            for body in [{"pictures": [picture]}, {"pictures": [picture], "types": []},
                         {"pictures": [picture], "types": ["12001"]}]:
                self.assertEqual(self.client.post("/detect", json=body).status_code, 422)
            for types in [[1], [10007], [12002], [12001, 12002]]:
                response = self.client.post("/detect", json={"pictures": [picture], "types": types})
                self.assertEqual(response.status_code, 422)
                self.assertIn("Unsupported algorithm IDs", response.json()["detail"])
                self.assertIn("12001", response.json()["detail"])
        process.assert_not_called()

    def test_restart_marks_abandoned_work_and_uncertain_callbacks(self):
        self.manager.executor.shutdown(wait=True)
        with patch.object(self.manager.executor, "submit"):
            task_id = self.manager.submit(BatchRequest(types=[12001], pictures=["http://minio.example/x"]), "http://ai.example")
        job = self.manager.get(task_id)
        job["pictures"][0]["callback"]["status"] = "sending"
        self.manager._save(job)
        self.manager.recover()
        restored = self.manager.get(task_id)
        self.assertEqual((restored["status"], restored["failed"]), ("interrupted", 1))
        self.assertEqual(restored["pictures"][0]["callback"]["status"], "failed")

    def test_stream_size_limit_and_empty_or_failed_download(self):
        target = self.root / "download.image"
        for response in [Response(headers={"Content-Length": str(MAX_DOWNLOAD_BYTES + 1)}),
                         Response(chunks=[b"x" * (MAX_DOWNLOAD_BYTES + 1)]), Response()]:
            with patch("ours.service.batch.requests.get", return_value=response), self.assertRaises(ValueError):
                download_image("http://minio.example/x", target)
        with patch("ours.service.batch.requests.get", return_value=Response(status=404)), \
                self.assertRaises(requests.HTTPError):
            download_image("http://minio.example/x", target)
        with patch("ours.service.batch.requests.post", return_value=Response(status=302)), \
                self.assertRaises(ValueError):
            send_callback("http://backend.example/results", {})

    def test_documented_request_routes_send_only_vehicles_to_report_endpoint(self):
        self.manager.callback_format = "report"
        self.manager.callback_url = "http://113.249.105.17:28096/api/alarm/record/report"
        original = self.pipeline.process

        def dual(*args, **kwargs):
            result = original(*args, **kwargs)
            vehicle = result["detections"][0]
            result["detections"].extend([
                dict(vehicle, instance_id=2, class_id=4, class_name="van"),
                dict(vehicle, instance_id=3, model_id="uav_pdd2023", class_name="crack"),
                dict(vehicle, instance_id=4, class_id=0, class_name="pedestrian"),
                dict(vehicle, instance_id=5, class_id=2, class_name="bicycle"),
            ])
            return result

        for path, with_sam in [("/detect", False), ("/alarm/record/detection2", True)]:
            body = {"pictures": [{"pictureId": "DJI_20261010103000_0001_V",
                                  "picturePath": "http://minio.example/image.jpeg"}],
                    "types": [12001, 12001], "with_sam": with_sam}
            with patch("ours.service.batch.requests.get", return_value=Response(self.image)), \
                    patch("ours.service.batch.requests.post", return_value=Response()) as post, \
                    patch.object(self.pipeline, "process", side_effect=dual):
                accepted = self.client.post(path, json=body)
                self.assertEqual(accepted.status_code, 200, accepted.text)
                self.assertEqual({key: accepted.json()[key] for key in ["code", "success", "msg", "data"]},
                                 {"code": 200, "success": True, "msg": "操作成功", "data": True})
                job = self.wait_task(accepted.json()["task_id"])
            self.assertEqual(post.call_count, 2)
            self.assertTrue(all(call.args[0] == self.manager.callback_url for call in post.call_args_list))
            self.assertTrue(all(call.kwargs["headers"] == {} for call in post.call_args_list))
            self.assertEqual(job["types"], [12001, 12001])
            item = job["pictures"][0]
            self.assertEqual((item["report_status"], item["callback"]["status"]), ("ready", "delivered"))
            self.assertEqual(len(item["reports"]), 2)
            self.assertEqual(len(item["result"]["detections"]), 5)
            self.assertEqual([detection["class_name"] for detection in item["result"]["detections"][:2]],
                             ["car", "van"])
            self.assertIn("combined_yolo", item["result"]["artifact_urls"])
            event_ids = set()
            for report in item["reports"]:
                payload = report["payload"]
                event_ids.add(payload["eventId"])
                self.assertRegex(payload["eventId"], r"^\d{16}$")
                self.assertEqual(set(payload), {"source", "isReal", "pictureId", "eventId", "timestamp", "result"})
                self.assertRegex(payload["timestamp"], r"^\d{4}-\d{2}-\d{2} \d{2}:\d{2}:\d{2}$")
                self.assertEqual((payload["source"], payload["isReal"], payload["pictureId"]),
                                 ("offline", 0, body["pictures"][0]["pictureId"]))
                self.assertEqual(payload["result"]["type"], "12001")
                self.assertEqual(payload["result"]["typeName"], "车辆违停")
                self.assertEqual(payload["result"], {"type": "12001", "typeName": "车辆违停",
                                                     "bbox": [2.0, 3.0, 10.0, 12.0], "confidence": 0.9})
                self.assertTrue(all(type(value) is float for value in payload["result"]["bbox"]))
                self.assertEqual(report["callback"]["status"], "delivered")
            self.assertEqual([call.kwargs["json"] for call in post.call_args_list],
                             [report["payload"] for report in item["reports"]])
            self.assertEqual(len(event_ids), 2)
        self.assertEqual(BatchRequest(types=[12001], pictures=["http://minio.example/image.jpeg"]).types, [12001])
        self.assertFalse(BatchRequest(types=[12001], pictures=["http://minio.example/image.jpeg"]).with_sam)

    def test_all_four_vehicle_classes_report_without_sam_by_default(self):
        self.manager.callback_format = "report"
        self.manager.callback_url = "http://backend.example/api/alarm/record/report"
        original = self.pipeline.process

        def vehicles(*args, **kwargs):
            result = original(*args, **kwargs)
            vehicle = result["detections"][0]
            result["detections"] = [dict(vehicle, instance_id=index, class_id=class_id, class_name=name)
                                    for index, (class_id, name) in enumerate(
                                        [(3, "car"), (4, "van"), (5, "truck"), (8, "bus")], 1)]
            return result

        with patch("ours.service.batch.requests.get", return_value=Response(self.image)), \
                patch("ours.service.batch.requests.post", return_value=Response()) as post, \
                patch.object(self.pipeline, "process", side_effect=vehicles):
            accepted = self.client.post("/detect", json={"types": [12001], "pictures": ["http://minio.example/image"]})
            job = self.wait_task(accepted.json()["task_id"])
        self.assertEqual(post.call_count, 4)
        self.assertFalse(self.pipeline.inputs[0][2])
        self.assertFalse(job["with_sam"])
        self.assertEqual(job["pictures"][0]["result"]["measurement_status"], "sam_disabled")
        self.assertNotIn("mask_url", job["pictures"][0]["result"]["detections"][0])
        self.assertEqual(job["pictures"][0]["callback"]["status"], "delivered")
        self.assertTrue(all(set(call.kwargs["json"]["result"]) == {"type", "typeName", "bbox", "confidence"}
                            for call in post.call_args_list))

    def test_nonvehicle_only_image_does_not_create_or_post_alarm(self):
        self.manager.callback_format = "report"
        self.manager.callback_url = "http://backend.example/api/alarm/record/report"
        original = self.pipeline.process

        def nonvehicles(*args, **kwargs):
            result = original(*args, **kwargs)
            template = result["detections"][0]
            result["detections"] = [dict(template, instance_id=index, class_id=class_id, class_name=name)
                                    for index, (class_id, name) in enumerate([
                                        (0, "pedestrian"), (1, "people"), (2, "bicycle"),
                                        (6, "tricycle"), (7, "awning-tricycle"), (9, "motor"),
                                    ], 1)]
            result["detections"].append(dict(template, instance_id=7, model_id="uav_pdd2023",
                                              class_id=3, class_name="crack"))
            return result

        with patch("ours.service.batch.requests.get", return_value=Response(self.image)), \
                patch("ours.service.batch.requests.post") as post, \
                patch.object(self.pipeline, "process", side_effect=nonvehicles):
            accepted = self.client.post("/detect", json={"types": [12001], "pictures": ["http://minio.example/image"]})
            job = self.wait_task(accepted.json()["task_id"])
        post.assert_not_called()
        item = job["pictures"][0]
        self.assertEqual((job["status"], item["report_status"], item["callback"]["status"]),
                         ("completed", "no_matching_targets", "skipped"))
        self.assertEqual(item["reports"], [])

    def test_report_failure_is_not_retried_and_continues_other_targets(self):
        self.manager.callback_format = "report"
        self.manager.callback_url = "http://backend.example/api/alarm/record/report"
        original = self.pipeline.process

        def dual(*args, **kwargs):
            result = original(*args, **kwargs)
            result["detections"].append(dict(result["detections"][0], instance_id=2))
            return result

        with patch("ours.service.batch.requests.get", return_value=Response(self.image)), \
                patch("ours.service.batch.requests.post", side_effect=[
                    Response(status=503), Response(b"json", body={"code": 200, "success": True, "data": True}),
                ]) as post, patch("ours.service.batch.time.sleep") as sleep, \
                patch.object(self.pipeline, "process", side_effect=dual):
            accepted = self.client.post("/detect", json={"types": [12001], "pictures": ["http://minio.example/image.jpeg"]})
            job = self.wait_task(accepted.json()["task_id"])
        item = job["pictures"][0]
        self.assertEqual((job["status"], item["status"], item["callback"]["status"]),
                         ("completed", "completed", "failed"))
        self.assertIsNotNone(item["result"])
        self.assertEqual(post.call_count, 2)
        self.assertTrue(all(call.args == (0.01,) for call in sleep.call_args_list))  # Task polling only.
        self.assertEqual(item["callback"]["attempts"], 2)
        self.assertEqual(item["reports"][0]["callback"],
                         {"status": "failed", "attempts": 1, "error": "Callback returned HTTP 503"})
        self.assertEqual(item["reports"][1]["callback"]["status"], "delivered")
        events = [call.kwargs["json"]["eventId"] for call in post.call_args_list]
        self.assertNotEqual(events[0], events[1])

    def test_report_timeout_or_application_failure_is_not_retried(self):
        self.manager.callback_format = "report"
        self.manager.callback_url = "http://backend.example/api/alarm/record/report"
        for failure in [requests.Timeout("Uncertain acceptance"),
                        Response(b"json", body={"code": 500, "success": False})]:
            with patch("ours.service.batch.requests.get", return_value=Response(self.image)), \
                    patch("ours.service.batch.requests.post", side_effect=[failure]) as post:
                accepted = self.client.post("/detect", json={"types": [12001], "pictures": ["http://minio.example/image"]})
                job = self.wait_task(accepted.json()["task_id"])
            post.assert_called_once()
            item = job["pictures"][0]
            self.assertEqual((job["status"], item["callback"]["status"]), ("completed", "failed"))
            self.assertEqual(item["reports"][0]["callback"]["attempts"], 1)
            self.assertEqual(item["reports"][0]["payload"], post.call_args.kwargs["json"])

    def test_report_skips_empty_and_failed_images_and_saves_when_unconfigured(self):
        self.manager.callback_format = "report"
        self.manager.callback_url = "http://backend.example/api/alarm/record/report"
        original = self.pipeline.process

        def empty(*args, **kwargs):
            result = original(*args, **kwargs)
            result["detections"] = []
            return result

        with patch("ours.service.batch.requests.get", side_effect=[Response(self.image), Response(status=404)]), \
                patch("ours.service.batch.requests.post") as post, \
                patch.object(self.pipeline, "process", side_effect=empty):
            accepted = self.client.post("/detect", json={"types": [12001], "pictures": ["http://minio.example/empty",
                                                                       "http://minio.example/missing"]})
            job = self.wait_task(accepted.json()["task_id"])
        post.assert_not_called()
        self.assertEqual(job["status"], "partial_failed")
        self.assertEqual([item["report_status"] for item in job["pictures"]],
                         ["no_matching_targets", "image_failed"])
        self.assertTrue(all(item["callback"]["status"] == "skipped" for item in job["pictures"]))

        self.manager.callback_url = None
        with patch("ours.service.batch.requests.get", return_value=Response(self.image)), \
                patch("ours.service.batch.requests.post") as post:
            accepted = self.client.post("/detect", json={"types": [12001], "pictures": ["http://minio.example/image"]})
            job = self.wait_task(accepted.json()["task_id"])
        post.assert_not_called()
        self.assertEqual(job["pictures"][0]["reports"][0]["callback"]["status"], "disabled")

    def test_request_callback_url_enables_report_without_forwarding_configured_token(self):
        self.manager.callback_format = "report"
        self.manager.callback_url = "http://backend.example/api/alarm/record/report"
        self.manager.callback_token = "test-secret"
        with patch("ours.service.batch.requests.get", return_value=Response(self.image)), \
                patch("ours.service.batch.requests.post", return_value=Response()) as post:
            accepted = self.client.post("/detect", json={"types": [12001], "pictures": ["http://minio.example/image"],
                "callbackUrl": "http://other.example/api/alarm/record/report"})
            job = self.wait_task(accepted.json()["task_id"])
        self.assertEqual(post.call_args.args[0], "http://other.example/api/alarm/record/report")
        self.assertEqual(post.call_args.kwargs["headers"], {})
        self.assertEqual(post.call_args.kwargs["json"]["source"], "offline")
        self.assertEqual(job["pictures"][0]["callback"]["status"], "delivered")

    def test_relative_documented_picture_path_and_unsupported_types(self):
        self.manager.callback_format = "report"
        object_path = "DJ-MEDIA-OSS/workspace/job/image.jpeg"
        body = {"pictures": [{"pictureId": "image-id", "picturePath": object_path}], "types": [12001, 12001]}
        self.assertEqual(self.client.post("/detect", json=body).status_code, 422)
        self.manager.image_base = "http://minio.example:9000/dbxq/"
        with patch("ours.service.batch.requests.get", return_value=Response(self.image)) as get:
            accepted = self.client.post("/alarm/record/detection2", json=body)
            job = self.wait_task(accepted.json()["task_id"])
        self.assertEqual(get.call_args.args[0], "http://minio.example:9000/dbxq/" + object_path)
        self.assertEqual(job["pictures"][0]["picturePath"], object_path)
        self.assertEqual(job["pictures"][0]["reports"][0]["payload"]["pictureId"], "image-id")
        body["types"] = [3]
        self.assertEqual(self.client.post("/detect", json=body).status_code, 422)
        for invalid in ["../image.jpeg", "%2e%2e/image.jpeg", "//other.example/image.jpeg"]:
            body["pictures"][0]["picturePath"] = invalid
            self.assertEqual(self.client.post("/detect", json=body).status_code, 422)


if __name__ == "__main__":
    unittest.main()
