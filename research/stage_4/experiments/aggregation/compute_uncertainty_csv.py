#!/usr/bin/env python3
"""
Research Pipeline - Stage 04: Per-Image Uncertainty CSV

Computes all seven image-level uncertainty aggregation metrics for every
sample in a specified dataset split and saves the results to a structured CSV.

Usage:
    python compute_uncertainty_csv.py \\
        --dataset BUSI --split test --seed 666 \\
        [--boundary-width 2] [--threshold-t 0.1] [--eps 1e-6] \\
        [--output-dir /path/to/outputs]

Output:
    <output-dir>/<dataset>/aggregation_results/seed<seed>/
        uncertainty_scores_<dataset>_<split>_seed<seed>.csv

IMPORTANT — Exploratory diagnostics only:
    The resulting CSV must NOT be used for method selection, threshold
    calibration, or any scientific claim on the test set.  All downstream
    analyses must carry this caveat explicitly.
"""

import os
import sys
import argparse
import datetime
import pandas as pd
import numpy as np

# ---------------------------------------------------------------------------
# Path setup
# ---------------------------------------------------------------------------
REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", "..", ".."))
sys.path.insert(0, os.path.join(REPO_ROOT, "research", "stage_4", "src"))
sys.path.insert(0, os.path.join(REPO_ROOT, "shared"))

from uncertainty.aggregation import extract_image_features  # noqa: E402


# ---------------------------------------------------------------------------
# CLI arguments
# ---------------------------------------------------------------------------

def get_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Compute seven per-image uncertainty scores and write them to CSV."
    )
    parser.add_argument("--dataset", type=str, default="BUSI",
                        help="Dataset name (must match output subdirectory).")
    parser.add_argument("--split", type=str, default="test",
                        help="Data split label written to the CSV (e.g. test, val).")
    parser.add_argument("--seed", type=int, default=666,
                        help="Inference seed used to identify result subdirectory.")
    parser.add_argument("--prompt-design", type=str, default="original",
                        help="Prompt variant tag (e.g. original).")
    parser.add_argument("--boundary-width", type=int, default=2,
                        help="Morphological boundary half-width in pixels (default 2).")
    parser.add_argument("--threshold-t", type=float, default=0.1,
                        help="Absolute uncertainty threshold t for U_threshold (default 0.1).")
    parser.add_argument("--eps", type=float, default=1e-6,
                        help="Denominator regularizer epsilon for U_ratio (default 1e-6).")
    parser.add_argument("--output-dir", type=str, default="",
                        help="Root output directory. Overrides MEDCLIPSEG_OUTPUT env var.")
    return parser.parse_args()


# ---------------------------------------------------------------------------
# Path resolution helpers
# ---------------------------------------------------------------------------

def resolve_output_root(args: argparse.Namespace) -> str:
    root = args.output_dir or os.environ.get(
        "MEDCLIPSEG_OUTPUT",
        os.path.join(REPO_ROOT, "..", "data", "outputs"),
    )
    return os.path.abspath(root)


def build_paths(output_root: str, dataset: str, seed: int, prompt: str, split: str = "test") -> dict:
    tag = f"MedCLIPSeg_unimedclip_ViT-B-16_Prompt-{prompt}"
    if split in ["val", "validation"]:
        val_root = os.path.join(output_root, dataset, "validation", f"seed{seed}")
        return {
            "csv": os.path.join(val_root, "seg_results", f"val_Prompt-{prompt}.csv"),
            "unc_dir": os.path.join(val_root, "unc_results", tag),
            "mask_dir": os.path.join(val_root, "seg_results", tag),
        }
    return {
        "csv": os.path.join(output_root, dataset, "seg_results",
                            f"seed{seed}", f"test_Prompt-{prompt}.csv"),
        "unc_dir": os.path.join(output_root, dataset, "unc_results",
                                f"seed{seed}", tag),
        "mask_dir": os.path.join(output_root, dataset, "seg_results",
                                 f"seed{seed}", tag),
    }


def build_output_path(output_root: str, dataset: str, split: str, seed: int) -> str:
    if split in ["val", "validation"]:
        out_dir = os.path.join(output_root, dataset, "validation", f"seed{seed}", "aggregation_results")
    else:
        out_dir = os.path.join(output_root, dataset, "aggregation_results", f"seed{seed}")
    os.makedirs(out_dir, exist_ok=True)
    return os.path.join(out_dir, f"uncertainty_scores_{dataset}_{split}_seed{seed}.csv")


# ---------------------------------------------------------------------------
# Processing
# ---------------------------------------------------------------------------

def process_sample(row: pd.Series, paths: dict, args: argparse.Namespace) -> dict:
    """Load one sample and compute all seven uncertainty metrics."""
    img_name = row["Name"]
    base_name = os.path.splitext(img_name)[0]

    npy_path = os.path.join(paths["unc_dir"], f"{base_name}.npy")
    mask_path = os.path.join(paths["mask_dir"], img_name)

    if not os.path.exists(npy_path):
        raise FileNotFoundError(f"Uncertainty map not found: {npy_path}")
    if not os.path.exists(mask_path):
        raise FileNotFoundError(f"Predicted mask not found: {mask_path}")

    feats = extract_image_features(
        unc_path=npy_path,
        mask_path=mask_path,
        boundary_width=args.boundary_width,
        threshold_t=args.threshold_t,
        eps=args.eps,
    )
    return feats


def build_record(meta: dict, feats: dict, row: pd.Series) -> dict:
    """Assemble a single CSV row from metadata, features, and reference metrics."""
    record = {
        "dataset": meta["dataset"],
        "split": meta["split"],
        "seed": meta["seed"],
        "image_id": row["Name"],
        # Seven uncertainty scores
        "U_mean": feats["U_mean"],
        "U_positive": feats["U_positive"],
        "U_fg": feats["U_fg"],
        "U_boundary": feats["U_boundary"],
        "U_threshold": feats["U_threshold"],
        "U_std_fg": feats["U_std_fg"],
        "U_ratio": feats["U_ratio"],
        # Region pixel counts
        "count_total": feats["count_total"],
        "count_pos": feats["count_pos"],
        "count_fg": feats["count_fg"],
        "count_boundary": feats["count_boundary"],
        "count_threshold": feats["count_threshold"],
        # Validity flags
        "valid_mean": feats["valid_mean"],
        "valid_pos": feats["valid_pos"],
        "valid_fg": feats["valid_fg"],
        "valid_boundary": feats["valid_boundary"],
        "valid_threshold": feats["valid_threshold"],
        "valid_std_fg": feats["valid_std_fg"],
        "valid_ratio": feats["valid_ratio"],
        # Hyperparameter configuration
        "boundary_width": feats["boundary_width"],
        "threshold_t": feats["threshold_t"],
        "eps": feats["eps"],
        # Reference segmentation metrics
        "DSC": float(row["DSC"]),
        "NSD": float(row["NSD"]),
    }
    return record


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    args = get_args()
    output_root = resolve_output_root(args)
    paths = build_paths(output_root, args.dataset, args.seed, args.prompt_design, split=args.split)

    print("=" * 70)
    print("  Uncertainty Aggregation CSV Pipeline (Stage 04)")
    print("  EXPLORATORY DIAGNOSTICS ONLY — Not for method selection")
    print("=" * 70)
    print(f"  Dataset      : {args.dataset}  |  Split: {args.split}  |  Seed: {args.seed}")
    print(f"  boundary_width = {args.boundary_width}  |  threshold_t = {args.threshold_t}  |  eps = {args.eps}")
    print(f"  Reference CSV: {paths['csv']}")
    print(f"  Unc maps dir : {paths['unc_dir']}")
    print(f"  Pred masks   : {paths['mask_dir']}")
    print("-" * 70)

    # Validate input paths
    for key, path in paths.items():
        if not os.path.exists(path):
            print(f"[ERROR] Required path not found: {path}", file=sys.stderr)
            sys.exit(1)

    # Load reference CSV (image list + DSC + NSD)
    ref_df = pd.read_csv(paths["csv"])
    total = len(ref_df)
    print(f"  Reference CSV loaded: {total} images")

    # Guard: do not overwrite existing output CSV
    out_csv = build_output_path(output_root, args.dataset, args.split, args.seed)
    if os.path.exists(out_csv):
        ts = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
        out_csv_conflict = out_csv.replace(".csv", f"_conflict_{ts}.csv")
        print(
            f"\n[WARNING] Output CSV already exists:\n  {out_csv}\n"
            f"  Writing to timestamped copy instead:\n  {out_csv_conflict}"
        )
        out_csv = out_csv_conflict

    meta = {"dataset": args.dataset, "split": args.split, "seed": args.seed}
    records = []
    n_error = 0

    for idx, row in ref_df.iterrows():
        img_name = row["Name"]
        try:
            feats = process_sample(row, paths, args)
            records.append(build_record(meta, feats, row))
        except Exception as exc:
            print(f"  [ERROR] {img_name}: {exc}")
            n_error += 1

    if not records:
        print("[ERROR] No records were computed. Aborting.", file=sys.stderr)
        sys.exit(1)

    result_df = pd.DataFrame(records)
    result_df.to_csv(out_csv, index=False, float_format="%.8f")

    # Summary report
    print(f"\n  Processed : {len(records)} / {total} images  ({n_error} errors)")
    print()

    METRICS = ["U_mean", "U_positive", "U_fg", "U_boundary",
               "U_threshold", "U_std_fg", "U_ratio"]
    VALID_FLAGS = ["valid_mean", "valid_pos", "valid_fg", "valid_boundary",
                   "valid_threshold", "valid_std_fg", "valid_ratio"]

    print(f"  {'Metric':<14} {'Valid':<8} {'NaN':<6} {'Mean':>10} {'Std':>10} {'Min':>10} {'Max':>10}")
    print("  " + "-" * 70)
    for metric, flag in zip(METRICS, VALID_FLAGS):
        col = result_df[metric]
        n_valid = int(result_df[flag].sum())
        n_nan = int(col.isna().sum())
        m = col.dropna()
        mean_s = f"{m.mean():.5f}" if len(m) else "—"
        std_s = f"{m.std():.5f}" if len(m) else "—"
        min_s = f"{m.min():.5f}" if len(m) else "—"
        max_s = f"{m.max():.5f}" if len(m) else "—"
        print(f"  {metric:<14} {n_valid:<8} {n_nan:<6} {mean_s:>10} {std_s:>10} {min_s:>10} {max_s:>10}")

    print()
    print(f"  Output CSV saved to:\n    {out_csv}")
    print("=" * 70)


if __name__ == "__main__":
    main()

