"""Tests for IntentRouter (Stage 1)."""

import unittest
from application.uncertainty_aware_retrieval.uncertainty_aware_retrieval.intent_router import IntentRouter


class TestIntentRouter(unittest.TestCase):
    def setUp(self):
        self.router = IntentRouter(default_k=5)

    def test_segment_intent(self):
        res = self.router.parse("Please segment these ultrasound scans and outline the lesions", images=[1, 2])
        self.assertEqual(res.intent, "segment")
        self.assertEqual(res.image_count, 2)
        self.assertFalse(res.is_ambiguous)

    def test_retrieve_intent(self):
        res = self.router.parse("Find 3 similar cases from the historical database", images=[1])
        self.assertEqual(res.intent, "retrieve")
        self.assertEqual(res.top_k, 3)
        self.assertFalse(res.is_ambiguous)

    def test_joint_segment_and_retrieve_intent(self):
        res = self.router.parse(
            "Segment these images and show me the most similar cases",
            images=[1, 2, 3],
        )
        self.assertEqual(res.intent, "segment_retrieve")
        self.assertEqual(res.image_count, 3)
        self.assertEqual(res.top_k, 5)

    def test_ambiguous_fallback(self):
        res = self.router.parse("Hello, can you help with these scans?", images=[1])
        self.assertEqual(res.intent, "segment_retrieve")
        self.assertTrue(res.is_ambiguous)
        self.assertIn("ambiguous", res.explanation.lower())

    def test_custom_k_extraction(self):
        cases = [
            ("retrieve top 7 similar cases", 7),
            ("show 10 similar ultrasound images", 10),
            ("find nearest matches k=4", 4),
            ("outline tumor without retrieval", 5),
        ]
        for instruction, expected_k in cases:
            res = self.router.parse(instruction)
            self.assertEqual(res.top_k, expected_k)


if __name__ == "__main__":
    unittest.main()

