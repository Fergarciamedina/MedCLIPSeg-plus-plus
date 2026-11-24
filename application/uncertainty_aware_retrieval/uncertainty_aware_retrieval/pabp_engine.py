"""Stage 4: FAISS-IVF with genuine Predicate-Aware Block Pruning (PABP).

Implements ultrasound similarity retrieval using an IVF-Flat structure with
block-level predicate summaries to prune ineligible candidate blocks prior to
vector distance computation.
"""

import heapq
import os
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Tuple
import faiss
import numpy as np

# Ensure single-threaded FAISS execution to prevent thread pool contention on macOS
faiss.omp_set_num_threads(1)

from .config import (
    DEFAULT_BLOCK_SIZE,
    DEFAULT_K,
    DEFAULT_NLIST,
    DEFAULT_NPROBE,
    EMBEDDING_DIM,
)


@dataclass
class PABPBlock:
    """A contiguous block of candidates within an IVF inverted list."""
    block_id: int
    centroid_id: int
    indices: np.ndarray  # Indices in the original gallery
    vectors: np.ndarray  # (B, D) float32 embeddings
    statuses: List[str]  # e.g. ['Accepted', 'Rejected', ...]
    eligible_mask: np.ndarray  # Boolean mask where status == 'Accepted'
    any_eligible: bool  # Block summary predicate: True if >= 1 eligible candidate


@dataclass
class RetrievalMetricRecord:
    """Detailed research profiling metrics clearly distinguishing block pruning from per-item filtering."""
    lists_probed: int
    blocks_inspected: int
    blocks_pruned_pabp: int
    blocks_scanned: int
    candidates_in_pruned_blocks: int
    candidates_filtered_in_scanned_blocks: int
    candidates_excluded_by_id: int
    vector_distances_computed: int
    total_candidates_in_probed_lists: int
    block_pruning_ratio: float
    total_candidates_avoided: int
    overall_candidate_saving_ratio: float


@dataclass
class RetrievedNeighbor:
    """A single retrieved historical case."""
    gallery_index: int
    image_id: str
    dataset: str
    similarity: float  # Cosine similarity in [-1, 1]
    status: str
    u_fg: Optional[float]
    dsc: Optional[float]
    image_path: str
    seg_mask_path: str
    unc_png_path: str


class PABPRetrievalEngine:
    """FAISS-IVF Indexing Engine with genuine Predicate-Aware Block Pruning."""

    def __init__(
        self,
        embedding_dim: int = EMBEDDING_DIM,
        nlist: int = DEFAULT_NLIST,
        nprobe: int = DEFAULT_NPROBE,
        block_size: int = DEFAULT_BLOCK_SIZE,
    ):
        self.embedding_dim = embedding_dim
        self.nlist = nlist
        self.nprobe = nprobe
        self.block_size = block_size

        self.quantizer: Optional[faiss.IndexFlatIP] = None
        self.gallery_records: List[Dict[str, Any]] = []
        self.gallery_embeddings: Optional[np.ndarray] = None
        self.inverted_lists: Dict[int, List[PABPBlock]] = {}
        self.is_built: bool = False

    def build_index(
        self,
        gallery_records: List[Dict[str, Any]],
        gallery_embeddings: np.ndarray,
    ) -> None:
        """Construct the FAISS coarse quantizer and organize inverted lists into PABP blocks."""
        n_samples = len(gallery_records)
        if n_samples == 0:
            raise ValueError("Gallery records cannot be empty.")

        self.gallery_records = gallery_records
        # Ensure float32 and strict L2 normalization
        embeddings = gallery_embeddings.astype(np.float32)
        norms = np.linalg.norm(embeddings, axis=-1, keepdims=True)
        norms[norms == 0] = 1.0
        self.gallery_embeddings = embeddings / norms

        # Adapt nlist for small datasets so points per centroid >= 39
        eff_nlist = min(self.nlist, n_samples, max(2, n_samples // 39))

        # Train FAISS coarse quantizer with Inner Product (Cosine similarity on unit vectors)
        self.quantizer = faiss.IndexFlatIP(self.embedding_dim)
        clustering = faiss.Clustering(self.embedding_dim, eff_nlist)
        clustering.niter = 20
        clustering.verbose = False
        clustering.train(self.gallery_embeddings, self.quantizer)

        # Assign all gallery vectors to their nearest centroid
        _, cluster_assignments = self.quantizer.search(self.gallery_embeddings, 1)
        cluster_assignments = cluster_assignments.flatten()

        # Build IVF inverted lists and divide each into PABP blocks
        self.inverted_lists = {}
        for c_id in range(eff_nlist):
            member_indices = np.where(cluster_assignments == c_id)[0]
            if len(member_indices) == 0:
                self.inverted_lists[c_id] = []
                continue

            # Partition member indices into fixed-size blocks
            blocks: List[PABPBlock] = []
            for b_idx, start_pos in enumerate(range(0, len(member_indices), self.block_size)):
                b_indices = member_indices[start_pos : start_pos + self.block_size]
                b_vectors = self.gallery_embeddings[b_indices]
                b_statuses = [self.gallery_records[idx]["status"] for idx in b_indices]
                eligible_mask = np.array([s == "Accepted" for s in b_statuses], dtype=bool)
                any_eligible = bool(eligible_mask.any())

                blocks.append(
                    PABPBlock(
                        block_id=b_idx,
                        centroid_id=c_id,
                        indices=b_indices,
                        vectors=b_vectors,
                        statuses=b_statuses,
                        eligible_mask=eligible_mask,
                        any_eligible=any_eligible,
                    )
                )
            self.inverted_lists[c_id] = blocks

        self.is_built = True

    def search_top_k(
        self,
        query_vector: np.ndarray,
        k: int = DEFAULT_K,
        nprobe: Optional[int] = None,
        filter_status: str = "Accepted",
        excluded_ids: Optional[set] = None,
    ) -> Tuple[List[RetrievedNeighbor], RetrievalMetricRecord]:
        """Execute genuine Predicate-Aware Block Pruning (PABP) search.

        CRITICAL ALGORITHM:
        1. Find top nprobe coarse centroids via FAISS quantizer.
        2. In each probed inverted list, inspect the block summary predicate: block.any_eligible.
        3. If False, SKIP THE BLOCK entirely! Zero candidate dot products computed.
        4. If True, SCAN only the eligible candidates within the retained block.
        5. Maintain top-K candidates in a min-heap.
        """
        if not self.is_built or self.quantizer is None:
            raise RuntimeError("PABP Retrieval Engine index is not built.")

        # Normalize query vector
        q_vec = query_vector.astype(np.float32).reshape(1, -1)
        q_norm = np.linalg.norm(q_vec)
        if q_norm > 0:
            q_vec = q_vec / q_norm

        probe_count = min(nprobe or self.nprobe, self.quantizer.ntotal)
        _, probed_centroids = self.quantizer.search(q_vec, probe_count)
        probed_centroids = probed_centroids.flatten()

        excluded = excluded_ids or set()

        # Profiling counters with strict separation
        lists_probed = len(probed_centroids)
        blocks_inspected = 0
        blocks_pruned_pabp = 0
        blocks_scanned = 0
        candidates_in_pruned_blocks = 0
        candidates_filtered_in_scanned_blocks = 0
        candidates_excluded_by_id = 0
        vector_distances_computed = 0
        total_candidates_in_probed_lists = 0

        # Min-heap storing tuples of (similarity, unique_counter, neighbor_obj)
        heap: List[Tuple[float, int, RetrievedNeighbor]] = []
        counter = 0

        for c_id in probed_centroids:
            blocks = self.inverted_lists.get(int(c_id), [])
            for block in blocks:
                blocks_inspected += 1
                b_size = len(block.indices)
                total_candidates_in_probed_lists += b_size

                # ==========================================================
                # GENUINE PABP BLOCK PRUNING CHECK
                # ==========================================================
                if not block.any_eligible:
                    # Ineligible block: zero Accepted candidates.
                    # SKIP ENTIRE BLOCK BEFORE COMPUTING ANY VECTOR DISTANCES!
                    blocks_pruned_pabp += 1
                    candidates_in_pruned_blocks += b_size
                    continue

                # Block contains at least one eligible candidate
                blocks_scanned += 1

                # Evaluate candidates within retained block
                for i, idx in enumerate(block.indices):
                    # Check candidate-level eligibility
                    if not block.eligible_mask[i]:
                        candidates_filtered_in_scanned_blocks += 1
                        continue

                    # Check query self-exclusion
                    rec = self.gallery_records[idx]
                    cand_key = rec["image_id"] + "@" + rec["dataset"]
                    if cand_key in excluded:
                        candidates_excluded_by_id += 1
                        continue

                    # Vector distance computation (Inner Product / Cosine Similarity)
                    sim = float(np.dot(q_vec[0], block.vectors[i]))
                    vector_distances_computed += 1

                    neighbor = RetrievedNeighbor(
                        gallery_index=idx,
                        image_id=rec["image_id"],
                        dataset=rec["dataset"],
                        similarity=sim,
                        status=rec["status"],
                        u_fg=rec.get("u_fg"),
                        dsc=rec.get("dsc"),
                        image_path=rec["image_path"],
                        seg_mask_path=rec["seg_mask_path"],
                        unc_png_path=rec["unc_png_path"],
                    )

                    if len(heap) < k:
                        heapq.heappush(heap, (sim, counter, neighbor))
                        counter += 1
                    elif sim > heap[0][0]:
                        heapq.heappushpop(heap, (sim, counter, neighbor))
                        counter += 1

        # Extract top-K ordered by similarity descending
        sorted_results = [
            item[2] for item in sorted(heap, key=lambda x: x[0], reverse=True)
        ]

        # Calculate metrics
        b_ratio = (
            blocks_pruned_pabp / blocks_inspected
            if blocks_inspected > 0
            else 0.0
        )
        total_avoided = candidates_in_pruned_blocks + candidates_filtered_in_scanned_blocks + candidates_excluded_by_id
        d_ratio = (
            total_avoided / total_candidates_in_probed_lists
            if total_candidates_in_probed_lists > 0
            else 0.0
        )

        metrics = RetrievalMetricRecord(
            lists_probed=lists_probed,
            blocks_inspected=blocks_inspected,
            blocks_pruned_pabp=blocks_pruned_pabp,
            blocks_scanned=blocks_scanned,
            candidates_in_pruned_blocks=candidates_in_pruned_blocks,
            candidates_filtered_in_scanned_blocks=candidates_filtered_in_scanned_blocks,
            candidates_excluded_by_id=candidates_excluded_by_id,
            vector_distances_computed=vector_distances_computed,
            total_candidates_in_probed_lists=total_candidates_in_probed_lists,
            block_pruning_ratio=b_ratio,
            total_candidates_avoided=total_avoided,
            overall_candidate_saving_ratio=d_ratio,
        )

        return sorted_results, metrics

    def exact_filtered_search(
        self,
        query_vector: np.ndarray,
        k: int = DEFAULT_K,
        filter_status: str = "Accepted",
        excluded_ids: Optional[set] = None,
    ) -> List[RetrievedNeighbor]:
        """Ground-truth exact brute-force search over all eligible candidates.

        Used strictly for regression testing and numerical equivalence verification.
        """
        q_vec = query_vector.astype(np.float32).flatten()
        q_norm = np.linalg.norm(q_vec)
        if q_norm > 0:
            q_vec = q_vec / q_norm

        excluded = excluded_ids or set()
        candidates: List[RetrievedNeighbor] = []

        for idx, rec in enumerate(self.gallery_records):
            if rec["status"] != filter_status:
                continue
            cand_key = rec["image_id"] + "@" + rec["dataset"]
            if cand_key in excluded:
                continue

            cand_vec = self.gallery_embeddings[idx]
            sim = float(np.dot(q_vec, cand_vec))

            candidates.append(
                RetrievedNeighbor(
                    gallery_index=idx,
                    image_id=rec["image_id"],
                    dataset=rec["dataset"],
                    similarity=sim,
                    status=rec["status"],
                    u_fg=rec.get("u_fg"),
                    dsc=rec.get("dsc"),
                    image_path=rec["image_path"],
                    seg_mask_path=rec["seg_mask_path"],
                    unc_png_path=rec["unc_png_path"],
                )
            )

        candidates.sort(key=lambda x: x.similarity, reverse=True)
        return candidates[:k]
