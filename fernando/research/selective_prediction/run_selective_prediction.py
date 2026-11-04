#!/usr/bin/env python3
"""
Research Pipeline - Stage 03: Selective Prediction & Control Evaluation
Evaluates rejection thresholds, risk-coverage curves, AURC, and control comparisons (Random, Oracle).
"""

import os
import sys
import argparse
import pandas as pd
import numpy as np

# Add paths for fernando modules and shared repo
REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", ".."))
sys.path.insert(0, REPO_ROOT)
sys.path.insert(0, os.path.join(REPO_ROOT, "fernando", "src"))
sys.path.insert(0, os.path.join(REPO_ROOT, "shared"))

from uncertainty.aggregation import extract_image_features
from uncertainty.selective_prediction import (
    calibrate_rejection_threshold,
    apply_selective_rejection,
    compute_risk_coverage_curve,
    compute_aurc,
    evaluate_rejection_controls,
)


def get_args():
    parser = argparse.ArgumentParser(description="Run selective prediction evaluation.")
    parser.add_argument("--dataset", type=str, default="BUSI", help="Dataset name.")
    parser.add_argument("--seed", type=int, default=666, help="Seed.")
    parser.add_argument("--feature", type=str, default="U_fg", help="Uncertainty feature for rejection.")
    parser.add_argument("--output-dir", type=str, default="", help="Outputs directory.")
    return parser.parse_args()


def main():
    args = get_args()
    medclip_out = args.output_dir or os.environ.get("MEDCLIPSEG_OUTPUT", os.path.join(REPO_ROOT, "..", "data", "outputs"))
    medclip_out = os.path.abspath(medclip_out)

    results_tag = f"MedCLIPSeg_unimedclip_ViT-B-16_Prompt-original"
    csv_path = os.path.join(medclip_out, args.dataset, "seg_results", f"seed{args.seed}", "test_Prompt-original.csv")
    unc_dir = os.path.join(medclip_out, args.dataset, "unc_results", f"seed{args.seed}", results_tag)
    mask_dir = os.path.join(medclip_out, args.dataset, "seg_results", f"seed{args.seed}", results_tag)

    if not os.path.exists(csv_path):
        raise FileNotFoundError(f"CSV not found: {csv_path}")

    df = pd.read_csv(csv_path)

    # Compute uncertainty features for each image
    feat_records = []
    for _, row in df.iterrows():
        img_name = row["Name"]
        base_name = os.path.splitext(img_name)[0]
        npy_path = os.path.join(unc_dir, f"{base_name}.npy")
        png_path = os.path.join(unc_dir, img_name)
        unc_file = npy_path if os.path.exists(npy_path) else png_path
        mask_file = os.path.join(mask_dir, img_name)
        feat_records.append(extract_image_features(unc_file, mask_file))

    for col in pd.DataFrame(feat_records).columns:
        df[col] = [r[col] for r in feat_records]

    print("=" * 70)
    print(f"  SELECTIVE PREDICTION & CONTROL BENCHMARK ({args.dataset}, Score: {args.feature})")
    print("=" * 70)

    # 1. Evaluate controls across coverages
    coverages = [1.0, 0.90, 0.80, 0.70]
    controls_df = evaluate_rejection_controls(
        df, feature_col=args.feature, target_coverages=coverages, metric_col="DSC"
    )

    print(f"{'Target Cov':<12} | {'Realized':<10} | {'Uncertainty DSC':<16} | {'Random DSC (CI)':<20} | {'Oracle DSC':<12}")
    print("-" * 75)
    for _, r in controls_df.iterrows():
        rand_str = f"{r['Random_DSC']:.4f} [{r['Random_CI_Lower']:.3f}-{r['Random_CI_Upper']:.3f}]"
        print(f"{r['Target_Coverage']:<12.2f} | {r['Realized_Coverage']:<10.2f} | {r['Uncertainty_DSC']:<16.4f} | {rand_str:<20} | {r['Oracle_DSC']:<12.4f}")
    print("-" * 75)

    # 2. Risk-Coverage Curve & AURC
    covs, risks = compute_risk_coverage_curve(df, score_col=args.feature, metric_col="DSC")
    aurc = compute_aurc(covs, risks)
    print(f"Area Under Risk-Coverage Curve (AURC): {aurc:.4f} (Lower is better)")
    print("=" * 70 + "\n")


if __name__ == "__main__":
    main()
