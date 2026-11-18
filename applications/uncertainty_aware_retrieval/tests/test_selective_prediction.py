"""Tests for Selective Prediction Triage (Stage 3)."""

import unittest
from applications.uncertainty_aware_retrieval.uncertainty_aware_retrieval.selective_prediction import (
    SelectivePredictionService,
)
from applications.uncertainty_aware_retrieval.uncertainty_aware_retrieval.manifest_builder import (
    ManifestBuilder,
)
from applications.uncertainty_aware_retrieval.uncertainty_aware_retrieval.config import (
    FROZEN_THRESHOLD,
    EMPTY_MASK_FALLBACK_THRESHOLD,
)


class TestSelectivePrediction(unittest.TestCase):
    def setUp(self):
        self.triage = SelectivePredictionService()

    def test_demo_cohort_exact_counts(self):
        """Verify that the 10 demonstration queries evaluate to EXACTLY 6 Accepted and 4 Rejected."""
        builder = ManifestBuilder()
        records = builder.build_or_load_manifest()
        queries, _ = builder.partition_demo_queries(records, m=10)

        acc = 0
        rej = 0
        not_eval = 0

        for q in queries:
            res = self.triage.evaluate_case(
                u_fg=q["u_fg"],
                u_mean=q["u_mean"],
                count_fg=q["count_fg"],
                image_id=q["image_id"],
                dataset=q["dataset"],
                is_unseen=q.get("is_unseen", False),
            )
            if res.status == "Accepted":
                acc += 1
            elif res.status == "Rejected":
                rej += 1
            else:
                not_eval += 1

        self.assertEqual(len(queries), 10)
        self.assertEqual(acc, 6, "Expected exactly 6 Accepted cases in demo cohort")
        self.assertEqual(rej, 4, "Expected exactly 4 Rejected cases in demo cohort")
        self.assertEqual(not_eval, 0)

    def test_empty_mask_policy_accepted(self):
        """Normal case with empty mask and U_mean <= 1e-4 should be Accepted."""
        res = self.triage.evaluate_case(
            u_fg=None,
            u_mean=0.000000,
            count_fg=0,
            image_id="normal_test.png",
        )
        self.assertEqual(res.status, "Accepted")
        self.assertIn("negative screening policy", res.clinical_note.lower())

    def test_empty_mask_policy_rejected(self):
        """Suspicious empty mask with U_mean > 1e-4 should be Rejected."""
        res = self.triage.evaluate_case(
            u_fg=None,
            u_mean=0.005000,
            count_fg=0,
            image_id="suspicious_empty.png",
        )
        self.assertEqual(res.status, "Rejected")
        self.assertIn("suspicious background entropy", res.reason.lower())

    def test_lesion_accepted(self):
        """Lesion scan with U_fg <= tau should be Accepted."""
        res = self.triage.evaluate_case(
            u_fg=FROZEN_THRESHOLD - 0.001,
            u_mean=0.01,
            count_fg=1500,
            image_id="benign.png",
        )
        self.assertEqual(res.status, "Accepted")

    def test_lesion_rejected(self):
        """Lesion scan with U_fg > tau should be Rejected."""
        res = self.triage.evaluate_case(
            u_fg=FROZEN_THRESHOLD + 0.001,
            u_mean=0.01,
            count_fg=1500,
            image_id="malignant.png",
        )
        self.assertEqual(res.status, "Rejected")

    def test_unseen_image_not_evaluated(self):
        """Unseen image should be marked 'Not Evaluated' rather than falsely Rejected."""
        res = self.triage.evaluate_case(
            u_fg=None,
            u_mean=None,
            count_fg=None,
            image_id="uploaded_user_scan.png",
            is_unseen=True,
        )
        self.assertEqual(res.status, "Not Evaluated")
        self.assertIn("uncalibrated", res.reason.lower())


if __name__ == "__main__":
    unittest.main()

