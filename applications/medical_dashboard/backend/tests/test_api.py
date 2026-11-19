import sys
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

from fastapi.testclient import TestClient

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import main


class DashboardApiTests(unittest.TestCase):
    def setUp(self):
        self.record = {
            "image_id": "41_normal.png",
            "dataset": "BUSI",
            "image_path": str(Path(__file__).resolve().parents[5] / "data/raw/BUSI/Test_Folder/img/41_normal.png"),
            "seg_mask_path": "/missing/mask.png",
            "unc_png_path": "/missing/uncertainty.png",
            "u_fg": None,
            "u_mean": 0.0,
            "count_fg": 0,
            "dsc": 1.0,
            "nsd": 1.0,
            "status": "Accepted",
        }
        accepted = SimpleNamespace(
            status="Accepted", u_fg=None, u_mean=0.0, threshold=0.0001,
            reason="Empty mask policy", clinical_note="Accepted by policy.",
        )
        self.system = {
            "records": [self.record],
            "triage": SimpleNamespace(
                threshold=0.064058324,
                empty_fallback_threshold=0.0001,
                evaluate_case=lambda **_: accepted,
            ),
            "segmentation": SimpleNamespace(get_segmentation_for_query=lambda *_, **__: {
                "u_fg": None, "u_mean": 0.0, "count_fg": 0,
                "image_id": "41_normal.png", "dataset": "BUSI", "is_unseen": False,
            }),
            "embeddings": [[0.0]], "key_to_idx": {"41_normal.png@BUSI": 0},
            "retrieval": SimpleNamespace(search_top_k=lambda **_: ([], SimpleNamespace(calls=1))), "nprobe": 4,
        }
        self.patch = patch.object(main, "get_system", return_value=self.system)
        self.patch.start()
        self.client = TestClient(main.app)

    def tearDown(self):
        self.patch.stop()

    def test_health_and_case_list(self):
        self.assertEqual(self.client.get("/api/health").json()["status"], "ok")
        response = self.client.get("/api/cases?dataset=BUSI")
        self.assertEqual(response.status_code, 200)
        self.assertEqual(response.json()["items"][0]["image_id"], "41_normal.png")

    def test_case_analysis_uses_empty_mask_fallback_and_returns_no_stub_matches(self):
        response = self.client.get("/api/cases/BUSI/41_normal.png/analyze")
        self.assertEqual(response.status_code, 200)
        payload = response.json()
        self.assertEqual(payload["case"]["status"], "Accepted")
        self.assertEqual(payload["case"]["threshold"], 0.0001)
        self.assertEqual(payload["neighbors"], [])
        self.assertEqual(payload["case"]["source"], "Precomputed benchmark result")

    def test_manifest_backed_image_is_served_and_unknown_kind_is_rejected(self):
        response = self.client.get("/api/image", params={"dataset": "BUSI", "image_id": "41_normal.png", "kind": "original"})
        self.assertEqual(response.status_code, 200)
        self.assertTrue(response.headers["content-type"].startswith("image/"))
        self.assertEqual(self.client.get("/api/image?dataset=BUSI&image_id=41_normal.png&kind=weights").status_code, 422)

    def test_batch_upload_matches_canonical_pixels_and_evaluates_each_match(self):
        image_path = Path(self.record["image_path"])
        with image_path.open("rb") as image_file:
            payload = image_file.read()
        response = self.client.post(
            "/api/uploads/evaluate",
            data={"prompt": "Segment and find 2 similar cases"},
            files=[
                ("files", ("renamed-a.png", payload, "image/png")),
                ("files", ("renamed-b.png", payload, "image/png")),
            ],
        )
        self.assertEqual(response.status_code, 200)
        body = response.json()
        self.assertEqual(body["evaluated_count"], 2)
        self.assertEqual(body["not_evaluated_count"], 0)
        self.assertTrue(all(item["matched"] for item in body["items"]))
        self.assertTrue(all(item["analysis"]["case"]["status"] == "Accepted" for item in body["items"]))


if __name__ == "__main__":
    unittest.main()
