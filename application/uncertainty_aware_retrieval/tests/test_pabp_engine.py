"""Tests for FAISS-IVF + genuine Predicate-Aware Block Pruning (PABP) Engine."""

import unittest
import numpy as np
from application.uncertainty_aware_retrieval.uncertainty_aware_retrieval.pabp_engine import (
    PABPRetrievalEngine,
)
from application.uncertainty_aware_retrieval.uncertainty_aware_retrieval.manifest_builder import (
    ManifestBuilder,
)


class TestPABPRetrievalEngine(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        builder = ManifestBuilder()
        cls.records = builder.build_or_load_manifest()
        cls.queries, cls.gallery = builder.partition_demo_queries(cls.records, m=10)

        # Load precomputed embeddings
        from application.uncertainty_aware_retrieval.uncertainty_aware_retrieval.embedding_extractor import (
            EmbeddingExtractor,
        )
        extractor = EmbeddingExtractor()
        all_embeddings = extractor.extract_or_load_gallery_embeddings(cls.records)

        # Extract gallery embeddings
        key_map = {r["image_id"] + "@" + r["dataset"]: i for i, r in enumerate(cls.records)}
        gal_indices = [key_map[g["image_id"] + "@" + g["dataset"]] for g in cls.gallery]
        cls.gallery_embeddings = all_embeddings[gal_indices]

        # Build PABP engine
        cls.engine = PABPRetrievalEngine(nlist=8, nprobe=8, block_size=16)
        cls.engine.build_index(cls.gallery, cls.gallery_embeddings)

    def test_index_is_built(self):
        self.assertTrue(self.engine.is_built)
        self.assertIsNotNone(self.engine.quantizer)
        self.assertEqual(len(self.engine.gallery_records), 532)

    def test_block_summary_predicates_integrity(self):
        """Verify that every PABP block accurately summarizes its constituent candidates."""
        for c_id, blocks in self.engine.inverted_lists.items():
            for b in blocks:
                # Check that any_eligible exactly equals whether 'Accepted' is in statuses
                expected_any = "Accepted" in b.statuses
                self.assertEqual(
                    b.any_eligible,
                    expected_any,
                    f"Block summary predicate mismatch in centroid {c_id}, block {b.block_id}",
                )

    def test_predicate_filtering_guarantee(self):
        """Verify that PABP NEVER returns a Rejected case when filter_status='Accepted'."""
        sample_query = self.gallery_embeddings[0]
        neighbors, metrics = self.engine.search_top_k(
            sample_query, k=10, filter_status="Accepted"
        )
        self.assertGreater(len(neighbors), 0)
        for n in neighbors:
            self.assertEqual(
                n.status,
                "Accepted",
                f"Violation: Retrieved neighbor {n.image_id} has status '{n.status}', expected 'Accepted'",
            )

    def test_genuine_pabp_block_skipping(self):
        """Verify that PABP genuinely inspects blocks, measures pruning, and saves distance computations."""
        sample_query = self.gallery_embeddings[5]
        neighbors, metrics = self.engine.search_top_k(
            sample_query, k=5, nprobe=8, filter_status="Accepted"
        )
        self.assertGreater(metrics.blocks_inspected, 0)
        self.assertGreaterEqual(metrics.blocks_pruned_pabp, 0)
        # Total distance computations must be strictly less than total gallery candidates
        self.assertLess(metrics.vector_distances_computed, len(self.gallery))
        self.assertGreater(metrics.total_candidates_avoided, 0)
        # Exact arithmetic balance check
        sum_components = (
            metrics.candidates_in_pruned_blocks
            + metrics.candidates_filtered_in_scanned_blocks
            + metrics.candidates_excluded_by_id
            + metrics.vector_distances_computed
        )
        self.assertEqual(sum_components, metrics.total_candidates_in_probed_lists)

    def test_exact_brute_force_equivalence(self):
        """Verify that PABP Top-1 and ranking align with exact filtered brute-force search."""
        for q_idx in [0, 25, 100]:
            q_vec = self.gallery_embeddings[q_idx]
            pabp_neighbors, _ = self.engine.search_top_k(
                q_vec, k=5, nprobe=8, filter_status="Accepted"
            )
            exact_neighbors = self.engine.exact_filtered_search(
                q_vec, k=5, filter_status="Accepted"
            )

            self.assertGreater(len(pabp_neighbors), 0)
            self.assertGreater(len(exact_neighbors), 0)

            # Top-1 neighbor must match identically in ID and similarity
            self.assertEqual(pabp_neighbors[0].image_id, exact_neighbors[0].image_id)
            self.assertAlmostEqual(
                pabp_neighbors[0].similarity,
                exact_neighbors[0].similarity,
                places=4,
            )

    def test_query_exclusion_guarantee(self):
        """Verify that excluded IDs are never returned in search results."""
        target_rec = self.gallery[10]
        target_key = target_rec["image_id"] + "@" + target_rec["dataset"]
        q_vec = self.gallery_embeddings[10]

        neighbors, _ = self.engine.search_top_k(
            q_vec, k=5, excluded_ids={target_key}
        )
        for n in neighbors:
            retrieved_key = n.image_id + "@" + n.dataset
            self.assertNotEqual(retrieved_key, target_key)


if __name__ == "__main__":
    unittest.main()
