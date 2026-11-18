"""Tests for Visualization and Overlay utilities."""

import unittest
import numpy as np
from PIL import Image
from applications.uncertainty_aware_retrieval.uncertainty_aware_retrieval.visualizer import (
    create_segmentation_overlay,
    load_uncertainty_visualization,
)
from applications.uncertainty_aware_retrieval.uncertainty_aware_retrieval.manifest_builder import (
    ManifestBuilder,
)


class TestVisualizer(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        builder = ManifestBuilder()
        cls.records = builder.build_or_load_manifest()

    def test_overlay_creation_synthetic(self):
        # Create synthetic test image and mask
        img_arr = np.ones((100, 100, 3), dtype=np.uint8) * 128
        mask_arr = np.zeros((100, 100), dtype=np.uint8)
        mask_arr[25:75, 25:75] = 255

        composite = create_segmentation_overlay(img_arr, mask_arr)
        self.assertIsInstance(composite, Image.Image)
        self.assertEqual(composite.size, (100, 100))
        self.assertEqual(composite.mode, "RGB")

    def test_overlay_creation_empty_mask(self):
        img_arr = np.ones((100, 100, 3), dtype=np.uint8) * 128
        mask_arr = np.zeros((100, 100), dtype=np.uint8)

        composite = create_segmentation_overlay(img_arr, mask_arr)
        self.assertIsInstance(composite, Image.Image)
        self.assertEqual(composite.size, (100, 100))

    def test_overlay_creation_real_case(self):
        # Use first non-empty real case
        rec = next(r for r in self.records if r["count_fg"] > 0)
        composite = create_segmentation_overlay(rec["image_path"], rec["seg_mask_path"])
        self.assertIsInstance(composite, Image.Image)
        orig_img = Image.open(rec["image_path"])
        self.assertEqual(composite.size, orig_img.size)

    def test_load_uncertainty_visualization_real_case(self):
        rec = next(r for r in self.records if r["count_fg"] > 0)
        unc_img = load_uncertainty_visualization(rec["unc_png_path"], rec["unc_npy_path"])
        self.assertIsNotNone(unc_img)
        self.assertIsInstance(unc_img, Image.Image)


if __name__ == "__main__":
    unittest.main()

