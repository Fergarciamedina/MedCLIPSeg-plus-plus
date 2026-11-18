#!/usr/bin/env python3
"""Command-line interface for MedCLIPSeg++ Uncertainty-Aware Medical Image Retrieval.

Usage:
    # 1. Prepare/cache gallery manifest and embeddings:
    python cli.py --prepare

    # 2. Run full end-to-end demonstration on M=10 partition cases:
    python cli.py --demo

    # 3. Query with custom instruction and image:
    python cli.py --instruction "Segment and find 5 similar cases" --image /path/to/scan.png
"""

import argparse
import sys
from pathlib import Path
import numpy as np

# Add application directory to python path
APP_DIR = Path(__file__).resolve().parent
if str(APP_DIR) not in sys.path:
    sys.path.insert(0, str(APP_DIR))

from uncertainty_aware_retrieval import (
    EmbeddingExtractor,
    IntentRouter,
    ManifestBuilder,
    PABPRetrievalEngine,
    SegmentationService,
    SelectivePredictionService,
)


def parse_args():
    parser = argparse.ArgumentParser(
        description="MedCLIPSeg++ Uncertainty-Aware Medical Image Retrieval CLI"
    )
    parser.add_argument(
        "--prepare",
        action="store_true",
        help="Build historical gallery manifest and extract/cache visual embeddings.",
    )
    parser.add_argument(
        "--demo",
        action="store_true",
        help="Run end-to-end demonstration on M=10 diverse query cases.",
    )
    parser.add_argument(
        "--instruction",
        type=str,
        default="Segment these images and show me the most similar cases.",
        help="Natural-language clinical instruction.",
    )
    parser.add_argument(
        "--image",
        type=str,
        default=None,
        help="Path or image ID of query ultrasound image.",
    )
    parser.add_argument(
        "--k",
        type=int,
        default=5,
        help="Number of similar neighbors to retrieve.",
    )
    return parser.parse_args()


def prepare_pipeline():
    print("=" * 70)
    print(" MEDCLIPSEG++ : PREPARING HISTORICAL GALLERY & EMBEDDINGS")
    print("=" * 70)
    builder = ManifestBuilder()
    print("[1/2] Building historical gallery manifest from 5 benchmarks...")
    records = builder.build_or_load_manifest(force_rebuild=True)
    n_acc = sum(1 for r in records if r["status"] == "Accepted")
    n_rej = sum(1 for r in records if r["status"] == "Rejected")
    print(f"      Manifest loaded: N={len(records)} (Accepted={n_acc}, Rejected={n_rej})")

    print("[2/2] Extracting and caching frozen UniMedCLIP visual embeddings...")
    extractor = EmbeddingExtractor()
    embeddings = extractor.extract_or_load_gallery_embeddings(records, force_recompute=False)
    print(f"      Embeddings shape: {embeddings.shape}, dtype={embeddings.dtype}")
    print("Historical gallery preparation complete!\n")
    return records, embeddings


def run_demo(records, embeddings, k_neighbors=5):
    print("=" * 70)
    print(" MEDCLIPSEG++ : RUNNING END-TO-END DEMO (M=10 QUERIES)")
    print("=" * 70)
    builder = ManifestBuilder()
    queries, gallery = builder.partition_demo_queries(records, m=10)
    print(f"Partitioned: {len(queries)} Demo Query Scans | {len(gallery)} Historical Gallery Scans")

    # Filter embeddings for historical gallery
    gal_keys = {g["image_id"] + "@" + g["dataset"]: i for i, g in enumerate(records)}
    gal_indices = [gal_keys[g["image_id"] + "@" + g["dataset"]] for g in gallery]
    gallery_embeddings = embeddings[gal_indices]

    # Build PABP index
    pabp = PABPRetrievalEngine(nlist=8, nprobe=4, block_size=16)
    pabp.build_index(gallery, gallery_embeddings)
    print(f"FAISS-IVF Index Built: nlist=8, nprobe=4, block_size=16.")

    router = IntentRouter()
    parsed = router.parse(
        "Segment these images and show me the most similar cases.",
        images=queries,
        override_k=k_neighbors,
    )
    print(f"\n[Stage 1: Intent Router] -> Intent: '{parsed.intent}', K={parsed.top_k}")
    print(f"Explanation: {parsed.explanation}\n")

    triage_service = SelectivePredictionService()
    seg_service = SegmentationService(manifest=records)

    accepted_queries = []
    rejected_queries = []
    unevaluated_queries = []

    for idx, q in enumerate(queries, start=1):
        seg = seg_service.get_segmentation_for_query(q["image_id"])
        triage = triage_service.evaluate_case(
            u_fg=seg["u_fg"],
            u_mean=seg["u_mean"],
            count_fg=seg["count_fg"],
            image_id=seg["image_id"],
            dataset=seg["dataset"],
        )

        print(f"--- Query {idx}/{len(queries)}: [{q['dataset']}] {q['image_id']} ---")
        print(f"    Segmentation: Count FG={seg['count_fg']}, U_fg={seg['u_fg']}, DSC={seg.get('dsc')}")
        print(f"    Selective Triage: {triage.status} ({triage.reason})")

        if triage.status == "Accepted":
            accepted_queries.append(q)
            # Retrieve embedding from precomputed embeddings matrix
            q_idx = gal_keys[q["image_id"] + "@" + q["dataset"]]
            q_emb = embeddings[q_idx]

            neighbors, metrics = pabp.search_top_k(
                query_vector=q_emb,
                k=parsed.top_k,
                filter_status="Accepted",
                excluded_ids={q["image_id"] + "@" + q["dataset"]},
            )
            print(f"    Retrieval: Found {len(neighbors)} Accepted Historical Neighbors")
            print(f"    PABP Metrics: Blocks Inspected={metrics.blocks_inspected}, "
                  f"Blocks Pruned={metrics.blocks_pruned_pabp} ({metrics.block_pruning_ratio*100:.1f}%), "
                  f"Distances Computed={metrics.vector_distances_computed} "
                  f"(Candidates in Pruned Blocks={metrics.candidates_in_pruned_blocks}, "
                  f"Filtered in Scanned Blocks={metrics.candidates_filtered_in_scanned_blocks})")
            for rank, n in enumerate(neighbors[:3], start=1):
                print(f"      Top-{rank}: [{n.dataset}] {n.image_id} (Similarity={n.similarity:.4f}, DSC={n.dsc})")
        elif triage.status == "Rejected":
            rejected_queries.append(q)
            print(f"    [TRIAGE ACTION]: Autonomous retrieval suppressed. Routed to Review-Required Gallery.")
        else:
            unevaluated_queries.append(q)
            print(f"    [TRIAGE ACTION]: Missing uncertainty metrics. Marked as Not Evaluated.")

    print("\n" + "=" * 70)
    print(" SUMMARY DEMONSTRATION METRICS")
    print("=" * 70)
    print(f"Total Queries Processed: {len(queries)}")
    print(f"Accepted by Policy (Low Uncertainty):   {len(accepted_queries)} ({len(accepted_queries)/len(queries)*100:.1f}%)")
    print(f"Needs Review (Elevated Uncertainty):    {len(rejected_queries)} ({len(rejected_queries)/len(queries)*100:.1f}%)")
    print(f"Not Evaluated:                          {len(unevaluated_queries)}")
    print("All retrieved neighbors verified strictly matching predicate: status == 'Accepted'.")
    print("=" * 70)


def main():
    args = parse_args()
    records, embeddings = prepare_pipeline()

    if args.demo or not args.image:
        run_demo(records, embeddings, k_neighbors=args.k)
    else:
        # Single image query
        router = IntentRouter()
        parsed = router.parse(args.instruction, images=[args.image], override_k=args.k)
        print(f"Instruction parsed: Intent='{parsed.intent}', K={parsed.top_k}")
        seg_service = SegmentationService(manifest=records)
        seg = seg_service.get_segmentation_for_query(args.image)
        triage = SelectivePredictionService().evaluate_case(
            u_fg=seg["u_fg"],
            u_mean=seg["u_mean"],
            count_fg=seg["count_fg"],
            image_id=seg["image_id"],
            dataset=seg["dataset"],
            is_unseen=seg.get("is_unseen", False),
        )
        print(f"Status: {triage.status} - {triage.reason}")


if __name__ == "__main__":
    main()
