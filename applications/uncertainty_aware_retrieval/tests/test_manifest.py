"""Tests for ManifestBuilder and Data Adapters."""

import unittest
from pathlib import Path
from applications.uncertainty_aware_retrieval.uncertainty_aware_retrieval.manifest_builder import ManifestBuilder


class TestManifestBuilder(unittest.TestCase):
    def setUp(self):
        self.builder = ManifestBuilder()
        self.records = self.builder.build_or_load_manifest()

    def test_total_cohort_count(self):
        self.assertEqual(len(self.records), 542)

    def test_canonical_status_breakdown(self):
        n_accepted = sum(1 for r in self.records if r["status"] == "Accepted")
        n_rejected = sum(1 for r in self.records if r["status"] == "Rejected")
        # Canonical numbers: 298 Accepted, 244 Rejected
        self.assertEqual(n_accepted, 298)
        self.assertEqual(n_rejected, 244)

    def test_file_paths_exist(self):
        # Sample check across records
        for r in self.records[::20]:
            self.assertTrue(Path(r["image_path"]).exists(), f"Image missing: {r['image_path']}")
            self.assertTrue(Path(r["seg_mask_path"]).exists(), f"Mask missing: {r['seg_mask_path']}")
            self.assertTrue(Path(r["unc_png_path"]).exists(), f"Unc PNG missing: {r['unc_png_path']}")

    def test_partition_demo_queries_exclusion(self):
        queries, gallery = self.builder.partition_demo_queries(self.records, m=10)
        self.assertEqual(len(queries), 10)
        self.assertEqual(len(gallery), 532)

        query_keys = {q["image_id"] + "@" + q["dataset"] for q in queries}
        gallery_keys = {g["image_id"] + "@" + g["dataset"] for g in gallery}

        # Mathematical guarantee: query images are strictly disjoint from gallery
        self.assertEqual(len(query_keys.intersection(gallery_keys)), 0)


if __name__ == "__main__":
    unittest.main()

