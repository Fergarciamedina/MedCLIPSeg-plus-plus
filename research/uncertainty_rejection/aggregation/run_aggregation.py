#!/usr/bin/env python3
"""
Research Pipeline - Stage 02: Aggregation Metrics & Failure Analysis
Computes Spearman correlations and failure AUROC (DSC < 0.5, NSD < 0.5) on extracted uncertainty maps.
"""

import os
import sys
import argparse
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

# Add repo root to path
REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", ".."))
sys.path.insert(0, REPO_ROOT)

from src.uncertainty.aggregation import extract_image_features
from src.uncertainty.metrics import compute_spearman_correlations, compute_failure_auroc


def get_args():
    parser = argparse.ArgumentParser(description="Run uncertainty aggregation and error correlation analysis.")
    parser.add_argument("--dataset", type=str, default="BUSI", help="Target dataset name.")
    parser.add_argument("--seed", type=int, default=666, help="Evaluation seed.")
    parser.add_argument("--prompt-design", type=str, default="original", help="Prompt variant.")
    parser.add_argument("--output-dir", type=str, default="", help="Canonical outputs directory.")
    parser.add_argument("--save-plot", type=str, default="", help="Path to save regression grid plot.")
    return parser.parse_args()


def main():
    args = get_args()
    sns.set_theme(style="whitegrid", font_scale=1.05)

    medclip_out = args.output_dir or os.environ.get("MEDCLIPSEG_OUTPUT", os.path.join(REPO_ROOT, "..", "data", "outputs"))
    medclip_out = os.path.abspath(medclip_out)

    results_tag = f"MedCLIPSeg_unimedclip_ViT-B-16_Prompt-{args.prompt_design}"
    csv_path = os.path.join(medclip_out, args.dataset, "seg_results", f"seed{args.seed}", f"test_Prompt-{args.prompt_design}.csv")
    unc_dir = os.path.join(medclip_out, args.dataset, "unc_results", f"seed{args.seed}", results_tag)
    mask_dir = os.path.join(medclip_out, args.dataset, "seg_results", f"seed{args.seed}", results_tag)

    print(f"Loading CSV:         {csv_path}")
    print(f"Uncertainty Maps:    {unc_dir}")
    print(f"Predicted Masks:     {mask_dir}")

    if not os.path.exists(csv_path):
        raise FileNotFoundError(f"CSV file not found: {csv_path}")
    if not os.path.exists(unc_dir):
        raise FileNotFoundError(f"Uncertainty dir not found: {unc_dir}")
    if not os.path.exists(mask_dir):
        raise FileNotFoundError(f"Mask dir not found: {mask_dir}")

    df = pd.read_csv(csv_path)

    # Collect features
    feat_records = []
    for _, row in df.iterrows():
        img_name = row["Name"]
        base_name = os.path.splitext(img_name)[0]

        # Prefer raw .npy float array, fallback to .png
        npy_path = os.path.join(unc_dir, f"{base_name}.npy")
        png_path = os.path.join(unc_dir, img_name)
        unc_file = npy_path if os.path.exists(npy_path) else png_path

        mask_file = os.path.join(mask_dir, img_name)
        feats = extract_image_features(unc_file, mask_file)
        feat_records.append(feats)

    feats_df = pd.DataFrame(feat_records)
    for col in feats_df.columns:
        df[col] = feats_df[col]

    feature_cols = ["U_mean", "U_fg", "U_ratio", "U_std_fg", "U_boundary"]
    available_feats = [f for f in feature_cols if f in df.columns]

    # Spearman Correlation
    corr_df, summary = compute_spearman_correlations(df, available_feats, ["DSC", "NSD"])
    print("\n" + "=" * 65)
    print("       SPEARMAN CORRELATION MATRIX (UNCERTAINTY vs METRICS)      ")
    print("=" * 65)
    print(f"{'Feature':<12} | {'DSC (rho)':<18} | {'NSD (rho)':<18}")
    print("-" * 65)
    for _, r in corr_df.iterrows():
        dsc_str = f"{r['DSC_rho']:+.4f} (p={r['DSC_pval']:.2e})" if "DSC_rho" in r else "N/A"
        nsd_str = f"{r['NSD_rho']:+.4f} (p={r['NSD_pval']:.2e})" if "NSD_rho" in r else "N/A"
        print(f"{r['Feature']:<12} | {dsc_str:<18} | {nsd_str:<18}")
    print("-" * 65)
    if summary["best_pair"]:
        bp = summary["best_pair"]
        print(f"* Strongest correlation: {bp[0]} vs {bp[1]} (rho = {bp[2]:+.4f})")
    print("=" * 65 + "\n")

    # Failure AUROC
    auroc_df = compute_failure_auroc(df, available_feats, failure_threshold=0.5)
    print("=" * 65)
    print("             AUROC PERFORMANCE FOR FAILURE DETECTION            ")
    print("=" * 65)
    print(f"{'Feature':<15} | {'AUROC (DSC < 0.5)':<22} | {'AUROC (NSD < 0.5)':<22}")
    print("-" * 65)
    for _, r in auroc_df.iterrows():
        auc_dsc = f"{r['AUROC_DSC']:.4f}" if "AUROC_DSC" in r and not np.isnan(r['AUROC_DSC']) else "N/A"
        auc_nsd = f"{r['AUROC_NSD']:.4f}" if "AUROC_NSD" in r and not np.isnan(r['AUROC_NSD']) else "N/A"
        print(f"{r['Feature']:<15} | {auc_dsc:<22} | {auc_nsd:<22}")
    print("=" * 65 + "\n")

    # Save visualization grid if requested
    plot_target = args.save_plot or os.path.join(REPO_ROOT, "docs", "assets", "step1_matrix_grid.png")
    os.makedirs(os.path.dirname(os.path.abspath(plot_target)), exist_ok=True)

    fig, axes = plt.subplots(nrows=3, ncols=2, figsize=(14, 15))
    fig.suptitle(f"Uncertainty Features vs. Segmentation Metrics ({args.dataset})", fontsize=15, fontweight="bold", y=0.995)

    plot_rows = [
        ("U_fg", "Foreground Uncertainty ($U_{fg}$)", "#2980b9"),
        ("U_ratio", "Uncertainty Ratio ($U_{ratio}$)", "#8e44ad"),
        ("U_std_fg", "Foreground Std Dev ($U_{std\_fg}$)", "#d35400"),
    ]
    plot_cols = [("DSC", "Dice Similarity Coefficient (DSC)"), ("NSD", "Normalized Surface Distance (NSD)")]

    for row_idx, (feat, feat_lbl, color) in enumerate(plot_rows):
        for col_idx, (metric, metric_lbl) in enumerate(plot_cols):
            ax = axes[row_idx, col_idx]
            sns.regplot(
                data=df, x=metric, y=feat, ax=ax, color=color,
                scatter_kws={"alpha": 0.7, "s": 50, "edgecolors": "k"},
                line_kws={"color": "#c0392b", "linewidth": 2, "label": "Trendline"}
            )
            ax.axvline(0.5, color="gray", linestyle="--", linewidth=1.2, alpha=0.8, label="Failure (0.5)")
            ax.set_title(f"{feat} vs. {metric}", fontsize=11, fontweight="bold")
            ax.set_xlabel(metric_lbl, fontsize=10)
            ax.set_ylabel(feat_lbl, fontsize=10)
            ax.set_xlim(-0.05, 1.05)
            ax.legend(loc="upper right", fontsize=8)

    plt.tight_layout()
    fig.savefig(plot_target, dpi=300, bbox_inches="tight")
    plt.close(fig)
    print(f"Regression plot saved to: {plot_target}")


if __name__ == "__main__":
    main()
