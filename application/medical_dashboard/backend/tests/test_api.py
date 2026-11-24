import csv
import sys
from io import BytesIO
from tempfile import TemporaryDirectory
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

import numpy as np
from fastapi.testclient import TestClient
from PIL import Image

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import main


class DashboardApiTests(unittest.TestCase):
    def setUp(self):
        folder = TemporaryDirectory()
        self.addCleanup(folder.cleanup)
        original = Path(folder.name) / "fixture-normal.png"
        Image.new("L", (8, 12)).save(original)
        self.record = {
            "image_id": "fixture-normal.png",
            "dataset": "BUSI",
            "image_path": str(original),
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
                "image_id": "fixture-normal.png", "dataset": "BUSI", "is_unseen": False,
            }),
            "embeddings": [[0.0]], "key_to_idx": {"fixture-normal.png@BUSI": 0},
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
        self.assertEqual(response.json()["items"][0]["image_id"], "fixture-normal.png")

    def test_case_analysis_uses_empty_mask_fallback_and_returns_no_stub_matches(self):
        response = self.client.get("/api/cases/BUSI/fixture-normal.png/analyze")
        self.assertEqual(response.status_code, 200)
        payload = response.json()
        self.assertEqual(payload["case"]["status"], "Accepted")
        self.assertEqual(payload["case"]["threshold"], 0.0001)
        self.assertEqual(payload["neighbors"], [])
        self.assertEqual(payload["case"]["source"], "Precomputed benchmark result")

    def test_manifest_backed_image_is_served_and_unknown_kind_is_rejected(self):
        response = self.client.get("/api/image", params={"dataset": "BUSI", "image_id": "fixture-normal.png", "kind": "original"})
        self.assertEqual(response.status_code, 200)
        self.assertTrue(response.headers["content-type"].startswith("image/"))
        self.assertEqual(self.client.get("/api/image?dataset=BUSI&image_id=fixture-normal.png&kind=weights").status_code, 422)

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


class PortableGalleryTests(unittest.TestCase):
    def setUp(self):
        folder = TemporaryDirectory()
        self.addCleanup(folder.cleanup)
        self.root = Path(folder.name)
        self.catalog = self.root / "cases.csv"
        self.vectors = self.root / "vecs.npz"
        env = patch.dict("os.environ", {
            "MEDCLIPSEG_GALLERY": str(self.catalog),
            "MEDCLIPSEG_EMBEDDINGS": str(self.vectors),
        })
        env.start()
        self.addCleanup(env.stop)
        main.get_system.cache_clear()
        self.addCleanup(main.get_system.cache_clear)
        self.client = TestClient(main.app)

    def create_gallery(self, ids, vector_ids=None):
        fields = ("dataset", "image_id", "image_path", "seg_mask_path", "unc_npy_path", "u_fg", "u_mean")
        with self.catalog.open("w", newline="") as f:
            writer = csv.writer(f)
            writer.writerow(fields)
            for i, case in enumerate(ids):
                source = f"original-{i}.png"
                mask = f"mask-{i}.png"
                entropy = f"entropy-{i}.npy"
                Image.new("L", (8, 12), color=10 + i).save(self.root / source)
                mask_array = np.zeros((4, 4), np.uint8)
                if i != 1:
                    mask_array[1:3, 1:3] = 1
                Image.fromarray(mask_array).save(self.root / mask)
                if i == 0:
                    np.save(self.root / entropy, np.full((4, 4), 0.03, dtype=np.float32))
                    writer.writerow(("site & one", case, source, mask, entropy, "", ""))
                elif i == 1:
                    writer.writerow(("site & one", case, source, mask, "", "", "0.00003"))
                else:
                    writer.writerow(("site & one", case, source, mask, "", "0.15", ""))
        vector_ids = vector_ids or list(reversed(ids))
        embeddings = np.zeros((len(vector_ids), 512), dtype=np.float32)
        for i in range(len(vector_ids)):
            embeddings[i, i] = 1
        np.savez(self.vectors, keys=[f"{case}@site & one" for case in vector_ids], embeddings=embeddings)

    def test_portable_gallery_upload_gates_retrieval_and_aligns_overlay(self):
        self.create_gallery(["case a", "case b", "case c"])
        summary = self.client.get("/api/summary")
        self.assertEqual(summary.status_code, 200, summary.text)
        self.assertEqual(summary.json()["accepted_count"], 2)
        self.assertEqual(summary.json()["review_count"], 1)
        self.assertEqual(main.get_system()["embeddings"][0].argmax(), 2)

        with (self.root / "original-0.png").open("rb") as source:
            response = self.client.post("/api/uploads/evaluate", data={"prompt": "Segment and find 2 similar cases"},
                                        files={"files": ("renamed.png", source, "image/png")})
        self.assertEqual(response.status_code, 200, response.text)
        analysis = response.json()["items"][0]["analysis"]
        self.assertEqual(analysis["case"]["status"], "Accepted")
        self.assertEqual([match["image_id"] for match in analysis["neighbors"]], ["case b"])
        self.assertNotIn("uncertainty", analysis["images"])
        mask = self.client.get(analysis["images"]["mask"])
        with Image.open(BytesIO(mask.content)) as image:
            self.assertEqual(image.size, (8, 12))
            self.assertEqual(image.getextrema(), (0, 255))
        self.assertEqual(self.client.get("/api/cases?limit=1&offset=1").json()["items"][0]["image_id"], "case b")

    def test_single_case_and_mismatched_embedding_ids(self):
        self.create_gallery(["only"], ["another"])
        mismatch = self.client.get("/api/summary")
        self.assertEqual(mismatch.status_code, 503)
        self.assertIn("must exactly match", mismatch.json()["detail"])
        self.create_gallery(["only"])
        self.assertEqual(self.client.get("/api/summary").json()["case_count"], 1)


if __name__ == "__main__":
    unittest.main()
