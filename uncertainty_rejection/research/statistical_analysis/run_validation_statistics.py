#!/usr/bin/env python3
"""
Research Pipeline - Stage 08: BUSI Validation Statistical Analysis

Performs rigorous statistical evaluation of the seven training-free uncertainty
aggregation metrics on the BUSI validation set:
  1. Spearman rank correlation (rho, p-value, 95% bootstrap CI) on N=65 lesion cases.
  2. Full-cohort analysis (N=78) for complete metrics with explicit NaN accounting.
  3. Risk-Coverage curves and AURC (Area Under Risk-Coverage Curve) vs Random & Oracle.
  4. Lesion-size dilution analysis: correlation with predicted foreground area.
  5. Detailed case diagnostics for 45_malignant.png and 120_benign.png.
  6. Exploratory AUROC for low-quality segmentation detection (DSC < 0.50).
  7. Generation of publication-quality figures and CSV tables.

All outputs are written outside Git under:
  data/outputs/BUSI/validation/seed666/statistical_analysis/
"""

import os
import sys
import argparse
import numpy as np
import pandas as pd
import cv2
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from scipy.stats import spearmanr

# ---------------------------------------------------------------------------
# Path setup
# ---------------------------------------------------------------------------
REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", ".."))
sys.path.insert(0, REPO_ROOT)
sys.path.insert(0, os.path.join(REPO_ROOT, "uncertainty_rejection", "src"))
sys.path.insert(0, os.path.join(REPO_ROOT, "shared"))

from uncertainty.metrics import (  # noqa: E402
    compute_spearman_correlations,
    compute_failure_auroc,
    compute_spearman_bootstrap_ci,
)
from uncertainty.selective_prediction import (  # noqa: E402
    compute_risk_coverage_curve,
    compute_aurc,
)

METRICS = [
    "U_mean", "U_positive", "U_fg", "U_boundary",
    "U_threshold", "U_std_fg", "U_ratio"
]

METRIC_LABELS = {
    "U_mean": r"$U_{\mathrm{mean}}$",
    "U_positive": r"$U_{\mathrm{positive}}$",
    "U_fg": r"$U_{\mathrm{fg}}$",
    "U_boundary": r"$U_{\mathrm{boundary}}$",
    "U_threshold": r"$U_{\mathrm{threshold}}$",
    "U_std_fg": r"$U_{\mathrm{std\_fg}}$",
    "U_ratio": r"$U_{\mathrm{ratio}}$",
}

COLORS = {
    "U_mean": "#1f77b4",       # Blue
    "U_positive": "#aec7e8",   # Light Blue
    "U_fg": "#2ca02c",         # Green
    "U_boundary": "#d62728",   # Red
    "U_threshold": "#9467bd",  # Purple
    "U_std_fg": "#ff7f0e",     # Orange
    "U_ratio": "#8c564b",      # Brown
    "Random": "#7f7f7f",       # Gray
    "Oracle": "#17becf",       # Cyan
}


def get_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run validation statistical analysis.")
    parser.add_argument("--dataset", type=str, default="BUSI")
    parser.add_argument("--seed", type=int, default=666)
    parser.add_argument("--prompt-design", type=str, default="original")
    parser.add_argument("--output-dir", type=str, default="")
    parser.add_argument("--data-root", type=str, default="")
    parser.add_argument("--n-bootstrap", type=int, default=1000)
    return parser.parse_args()


def main() -> None:
    args = get_args()
    base_out = args.output_dir or os.environ.get(
        "MEDCLIPSEG_OUTPUT", os.path.abspath(os.path.join(REPO_ROOT, "..", "data", "outputs"))
    )
    data_root = args.data_root or os.environ.get(
        "MEDCLIPSEG_DATA", os.path.abspath(os.path.join(REPO_ROOT, "..", "data", "raw"))
    )

    val_root = os.path.join(base_out, args.dataset, "validation", f"seed{args.seed}")
    agg_csv = os.path.join(val_root, "aggregation_results", f"uncertainty_scores_{args.dataset}_val_seed{args.seed}.csv")

    if not os.path.exists(agg_csv):
        raise FileNotFoundError(f"Validation aggregation CSV not found: {agg_csv}")

    stats_dir = os.path.join(val_root, "statistical_analysis")
    tables_dir = os.path.join(stats_dir, "tables")
    figures_dir = os.path.join(stats_dir, "figures")
    os.makedirs(tables_dir, exist_ok=True)
    os.makedirs(figures_dir, exist_ok=True)

    print("=" * 75)
    print("  ELEC 6661 — BUSI Validation Statistical Analysis (Stage 08)")
    print("=" * 75)
    print(f"  Input CSV      : {agg_csv}")
    print(f"  Tables Output  : {tables_dir}")
    print(f"  Figures Output : {figures_dir}")
    print("-" * 75)

    df_all = pd.read_csv(agg_csv)
    n_all = len(df_all)
    df_lesion = df_all[df_all["valid_fg"] == True].copy().reset_index(drop=True)
    n_lesion = len(df_lesion)

    print(f"  Total validation samples : {n_all}")
    print(f"  Lesion-present cohort    : {n_lesion}")
    print(f"  Normal scans (empty)     : {n_all - n_lesion}")

    # =========================================================================
    # Task 1: Spearman Correlation on Lesion-Present Samples (N=65)
    # =========================================================================
    print("\n[Task 1/7] Computing Spearman correlations on N=65 lesion samples...")
    records_65 = []
    for m in METRICS:
        row = {"Metric": m, "Display": METRIC_LABELS[m], "N": n_lesion}
        for target in ["DSC", "NSD"]:
            rho, pval = spearmanr(df_lesion[m], df_lesion[target])
            _, ci_l, ci_u = compute_spearman_bootstrap_ci(
                df_lesion[m].values, df_lesion[target].values,
                n_bootstrap=args.n_bootstrap, ci=0.95, seed=args.seed
            )
            row[f"{target}_rho"] = float(rho)
            row[f"{target}_pval"] = float(pval)
            row[f"{target}_ci_lower"] = float(ci_l)
            row[f"{target}_ci_upper"] = float(ci_u)
        records_65.append(row)

    spearman_65_df = pd.DataFrame(records_65)
    csv_65 = os.path.join(tables_dir, "spearman_validation_65_lesion.csv")
    spearman_65_df.to_csv(csv_65, index=False)
    print(f"  Saved: {csv_65}")

    print("\n  Spearman Correlation Table (N=65 Lesion Samples):")
    print(f"  {'Metric':<14} | {'DSC rho (95% CI)':<28} | {'p-value':<10} | {'NSD rho (95% CI)':<28} | {'p-value':<10}")
    print("  " + "-" * 98)
    for _, r in spearman_65_df.iterrows():
        dsc_str = f"{r['DSC_rho']:+.4f} [{r['DSC_ci_lower']:+.3f}, {r['DSC_ci_upper']:+.3f}]"
        nsd_str = f"{r['NSD_rho']:+.4f} [{r['NSD_ci_lower']:+.3f}, {r['NSD_ci_upper']:+.3f}]"
        print(f"  {r['Metric']:<14} | {dsc_str:<28} | {r['DSC_pval']:<10.2e} | {nsd_str:<28} | {r['NSD_pval']:<10.2e}")

    # =========================================================================
    # Task 2: All-78-Image Analysis for Complete Coverage Metrics
    # =========================================================================
    print("\n[Task 2/7] Computing all-78-image analysis with NaN accounting...")
    records_78 = []
    for m in METRICS:
        n_valid = int(df_all[m].notna().sum())
        n_nan = int(df_all[m].isna().sum())
        row = {"Metric": m, "Display": METRIC_LABELS[m], "Valid_N": n_valid, "NaN_Count": n_nan}
        if n_valid == n_all:
            row["Status"] = "Complete Coverage"
            for target in ["DSC", "NSD"]:
                rho, pval = spearmanr(df_all[m], df_all[target])
                _, ci_l, ci_u = compute_spearman_bootstrap_ci(
                    df_all[m].values, df_all[target].values,
                    n_bootstrap=args.n_bootstrap, ci=0.95, seed=args.seed
                )
                row[f"{target}_rho"] = float(rho)
                row[f"{target}_pval"] = float(pval)
                row[f"{target}_ci_lower"] = float(ci_l)
                row[f"{target}_ci_upper"] = float(ci_u)
        else:
            row["Status"] = f"Undefined for {n_nan} normal scans (|Omega_fg|=0)"
            for target in ["DSC", "NSD"]:
                row[f"{target}_rho"] = np.nan
                row[f"{target}_pval"] = np.nan
                row[f"{target}_ci_lower"] = np.nan
                row[f"{target}_ci_upper"] = np.nan
        records_78.append(row)

    spearman_78_df = pd.DataFrame(records_78)
    csv_78 = os.path.join(tables_dir, "spearman_validation_78_all.csv")
    spearman_78_df.to_csv(csv_78, index=False)
    print(f"  Saved: {csv_78}")

    # =========================================================================
    # Task 3 & 4: Risk-Coverage Curves & AURC Evaluation
    # =========================================================================
    print("\n[Task 3/7] Generating Risk-Coverage curves & computing AURC...")
    aurc_records = []

    # Prepare Random Rejection Reference
    # For random rejection on N=65: expected risk is constant = 1 - mean(metric)
    mean_dsc_65 = float(df_lesion["DSC"].mean())
    mean_nsd_65 = float(df_lesion["NSD"].mean())

    # Oracle Rejection Reference (sort by ground-truth metric descending)
    cov_ora_dsc, risk_ora_dsc = compute_risk_coverage_curve(df_lesion, "DSC", "DSC", reverse_sort=True)
    aurc_ora_dsc = compute_aurc(cov_ora_dsc, risk_ora_dsc)

    cov_ora_nsd, risk_ora_nsd = compute_risk_coverage_curve(df_lesion, "NSD", "NSD", reverse_sort=True)
    aurc_ora_nsd = compute_aurc(cov_ora_nsd, risk_ora_nsd)

    # Random AURC: integral of (1 - mean_metric) from c_min to 1.0
    c_grid_random = cov_ora_dsc
    risk_rand_dsc = np.full_like(c_grid_random, 1.0 - mean_dsc_65)
    aurc_rand_dsc = compute_aurc(c_grid_random, risk_rand_dsc)

    risk_rand_nsd = np.full_like(c_grid_random, 1.0 - mean_nsd_65)
    aurc_rand_nsd = compute_aurc(c_grid_random, risk_rand_nsd)

    aurc_records.append({
        "Method": "Oracle (GT Ceiling)",
        "Type": "Reference Bound",
        "AURC_DSC": aurc_ora_dsc,
        "Gain_vs_Random_DSC": aurc_rand_dsc - aurc_ora_dsc,
        "AURC_NSD": aurc_ora_nsd,
        "Gain_vs_Random_NSD": aurc_rand_nsd - aurc_ora_nsd,
    })
    aurc_records.append({
        "Method": "Random Rejection",
        "Type": "Baseline Control",
        "AURC_DSC": aurc_rand_dsc,
        "Gain_vs_Random_DSC": 0.0,
        "AURC_NSD": aurc_rand_nsd,
        "Gain_vs_Random_NSD": 0.0,
    })

    rc_curves = {}
    for m in METRICS:
        c_dsc, r_dsc = compute_risk_coverage_curve(df_lesion, m, "DSC", reverse_sort=False, handle_ties=True)
        a_dsc = compute_aurc(c_dsc, r_dsc)

        c_nsd, r_nsd = compute_risk_coverage_curve(df_lesion, m, "NSD", reverse_sort=False, handle_ties=True)
        a_nsd = compute_aurc(c_nsd, r_nsd)

        rc_curves[m] = {"cov_dsc": c_dsc, "risk_dsc": r_dsc, "cov_nsd": c_nsd, "risk_nsd": r_nsd}

        aurc_records.append({
            "Method": m,
            "Type": "Uncertainty Metric",
            "AURC_DSC": a_dsc,
            "Gain_vs_Random_DSC": aurc_rand_dsc - a_dsc,
            "AURC_NSD": a_nsd,
            "Gain_vs_Random_NSD": aurc_rand_nsd - a_nsd,
        })

    aurc_df = pd.DataFrame(aurc_records)
    csv_aurc = os.path.join(tables_dir, "aurc_summary_validation.csv")
    aurc_df.to_csv(csv_aurc, index=False)
    print(f"  Saved: {csv_aurc}")

    print("\n  Risk-Coverage Performance (AURC, Lower is Better):")
    print(f"  {'Method':<22} | {'AURC (1-DSC)':<14} | {'Gain vs Rand':<14} | {'AURC (1-NSD)':<14} | {'Gain vs Rand':<14}")
    print("  " + "-" * 88)
    for _, r in aurc_df.iterrows():
        print(f"  {r['Method']:<22} | {r['AURC_DSC']:<14.4f} | {r['Gain_vs_Random_DSC']:<+14.4f} | {r['AURC_NSD']:<14.4f} | {r['Gain_vs_Random_NSD']:<+14.4f}")

    # =========================================================================
    # Task 5: Quantify Influence of Predicted Lesion Area (count_fg)
    # =========================================================================
    print("\n[Task 4/7] Quantifying influence of lesion area (count_fg) on uncertainty...")
    area_records = []
    for m in METRICS:
        rho_area, pval_area = spearmanr(df_lesion[m], df_lesion["count_fg"])
        _, ci_l, ci_u = compute_spearman_bootstrap_ci(
            df_lesion[m].values, df_lesion["count_fg"].values,
            n_bootstrap=args.n_bootstrap, ci=0.95, seed=args.seed
        )
        area_records.append({
            "Metric": m,
            "Display": METRIC_LABELS[m],
            "Spearman_rho_Area": float(rho_area),
            "p_value": float(pval_area),
            "ci_lower": float(ci_l),
            "ci_upper": float(ci_u),
            "Interpretation": "Strong Area Confounding" if abs(rho_area) > 0.4 else (
                "Moderate Confounding" if abs(rho_area) > 0.2 else "Area-Robust"
            )
        })

    area_df = pd.DataFrame(area_records)
    csv_area = os.path.join(tables_dir, "lesion_area_correlation.csv")
    area_df.to_csv(csv_area, index=False)
    print(f"  Saved: {csv_area}")

    print(f"\n  Lesion Area Confounding (Spearman rho vs count_fg, N={n_lesion}):")
    for _, r in area_df.iterrows():
        print(f"  {r['Metric']:<14} | rho = {r['Spearman_rho_Area']:+.4f} [{r['ci_lower']:+.3f}, {r['ci_upper']:+.3f}] | p = {r['p_value']:.2e} | {r['Interpretation']}")

    # =========================================================================
    # Task 6: Exploratory Failure Detection AUROC (DSC < 0.50)
    # =========================================================================
    print("\n[Task 5/7] Computing exploratory failure AUROC (DSC < 0.50)...")
    auroc_df = compute_failure_auroc(df_lesion, METRICS, failure_threshold=0.50, metric_cols=["DSC", "NSD"])
    csv_auroc = os.path.join(tables_dir, "failure_auroc_validation.csv")
    auroc_df.to_csv(csv_auroc, index=False)
    print(f"  Saved: {csv_auroc}")
    print(auroc_df.to_string(index=False))

    # =========================================================================
    # Task 7: Generate Publication-Quality Figures (300 DPI)
    # =========================================================================
    print("\n[Task 6/7] Generating publication-quality figures...")

    # -------------------------------------------------------------------------
    # Figure 1: Risk-Coverage Curves (DSC & NSD)
    # -------------------------------------------------------------------------
    fig1, (ax_d, ax_n) = plt.subplots(1, 2, figsize=(14, 5.5), sharey=True)

    # Left: 1 - DSC
    ax_d.plot(c_grid_random, risk_rand_dsc, label="Random Rejection", color=COLORS["Random"], linestyle="--", linewidth=1.8)
    ax_d.plot(cov_ora_dsc, risk_ora_dsc, label=f"Oracle (AURC={aurc_ora_dsc:.3f})", color=COLORS["Oracle"], linewidth=2.0)
    for m in METRICS:
        a = aurc_df[aurc_df["Method"] == m]["AURC_DSC"].iloc[0]
        ax_d.plot(rc_curves[m]["cov_dsc"], rc_curves[m]["risk_dsc"],
                  label=f"{METRIC_LABELS[m]} ({a:.3f})", color=COLORS[m], linewidth=1.6)

    ax_d.set_title("Risk-Coverage: Empirical Risk = 1 − DSC (N=65)", fontsize=11, fontweight="bold")
    ax_d.set_xlabel("Coverage (Retention Fraction)", fontsize=10)
    ax_d.set_ylabel("Empirical Risk (1 − Mean DSC)", fontsize=10)
    ax_d.set_xlim(0.0, 1.02)
    ax_d.set_ylim(-0.01, 0.40)
    ax_d.grid(True, linestyle=":", alpha=0.6)
    ax_d.legend(loc="upper left", fontsize=8, framealpha=0.9)

    # Right: 1 - NSD
    ax_n.plot(c_grid_random, risk_rand_nsd, label="Random Rejection", color=COLORS["Random"], linestyle="--", linewidth=1.8)
    ax_n.plot(cov_ora_nsd, risk_ora_nsd, label=f"Oracle (AURC={aurc_ora_nsd:.3f})", color=COLORS["Oracle"], linewidth=2.0)
    for m in METRICS:
        a = aurc_df[aurc_df["Method"] == m]["AURC_NSD"].iloc[0]
        ax_n.plot(rc_curves[m]["cov_nsd"], rc_curves[m]["risk_nsd"],
                  label=f"{METRIC_LABELS[m]} ({a:.3f})", color=COLORS[m], linewidth=1.6)

    ax_n.set_title("Risk-Coverage: Empirical Risk = 1 − NSD (N=65)", fontsize=11, fontweight="bold")
    ax_n.set_xlabel("Coverage (Retention Fraction)", fontsize=10)
    ax_n.set_xlim(0.0, 1.02)
    ax_n.grid(True, linestyle=":", alpha=0.6)
    ax_n.legend(loc="upper left", fontsize=8, framealpha=0.9)

    fig1.suptitle("Selective Prediction Risk-Coverage Trade-Off (BUSI Validation Cohort)", fontsize=13, fontweight="bold", y=1.00)
    plt.tight_layout()
    fig1_path = os.path.join(figures_dir, "figure1_risk_coverage_curves.png")
    fig1.savefig(fig1_path, dpi=300, bbox_inches="tight")
    plt.close(fig1)
    print(f"  Saved Figure 1: {fig1_path}")

    # -------------------------------------------------------------------------
    # Figure 2: Spearman Correlation Forest Plot (Point + 95% Bootstrap CI)
    # -------------------------------------------------------------------------
    fig2, (ax_f1, ax_f2) = plt.subplots(1, 2, figsize=(13, 5), sharey=True)
    y_pos = np.arange(len(METRICS))

    # DSC Forest Plot
    rhos_dsc = spearman_65_df["DSC_rho"].values
    ci_l_dsc = spearman_65_df["DSC_ci_lower"].values
    ci_u_dsc = spearman_65_df["DSC_ci_upper"].values
    xerr_dsc = [rhos_dsc - ci_l_dsc, ci_u_dsc - rhos_dsc]

    ax_f1.errorbar(rhos_dsc, y_pos, xerr=xerr_dsc, fmt="o", color="#2c3e50", ecolor="#e74c3c",
                   elinewidth=2, capsize=4, capthick=1.5, markersize=7)
    ax_f1.axvline(0.0, color="gray", linestyle="--", linewidth=1.2, alpha=0.8)
    ax_f1.set_yticks(y_pos)
    ax_f1.set_yticklabels([METRIC_LABELS[m] for m in METRICS], fontsize=11)
    ax_f1.set_xlabel("Spearman Rank Correlation (ρ)", fontsize=10, fontweight="bold")
    ax_f1.set_title("Correlation with DSC (N=65, 95% Bootstrap CI)", fontsize=11, fontweight="bold")
    ax_f1.set_xlim(-0.75, 0.30)
    ax_f1.grid(True, linestyle=":", alpha=0.6)

    # NSD Forest Plot
    rhos_nsd = spearman_65_df["NSD_rho"].values
    ci_l_nsd = spearman_65_df["NSD_ci_lower"].values
    ci_u_nsd = spearman_65_df["NSD_ci_upper"].values
    xerr_nsd = [rhos_nsd - ci_l_nsd, ci_u_nsd - rhos_nsd]

    ax_f2.errorbar(rhos_nsd, y_pos, xerr=xerr_nsd, fmt="s", color="#16a085", ecolor="#d35400",
                   elinewidth=2, capsize=4, capthick=1.5, markersize=7)
    ax_f2.axvline(0.0, color="gray", linestyle="--", linewidth=1.2, alpha=0.8)
    ax_f2.set_xlabel("Spearman Rank Correlation (ρ)", fontsize=10, fontweight="bold")
    ax_f2.set_title("Correlation with NSD (N=65, 95% Bootstrap CI)", fontsize=11, fontweight="bold")
    ax_f2.set_xlim(-0.75, 0.30)
    ax_f2.grid(True, linestyle=":", alpha=0.6)

    fig2.suptitle("Uncertainty vs. Segmentation Quality: Spearman Rank Correlation Forest Plot", fontsize=12, fontweight="bold", y=0.98)
    plt.tight_layout()
    fig2_path = os.path.join(figures_dir, "figure2_spearman_forest_plot.png")
    fig2.savefig(fig2_path, dpi=300, bbox_inches="tight")
    plt.close(fig2)
    print(f"  Saved Figure 2: {fig2_path}")

    # -------------------------------------------------------------------------
    # Figure 3: Lesion-Size Dilution Analysis (Area vs U_mean, U_fg, U_ratio)
    # -------------------------------------------------------------------------
    fig3, axes3 = plt.subplots(1, 3, figsize=(16, 4.8))

    # Diagnostic points
    case_45 = df_lesion[df_lesion["image_id"] == "45_malignant.png"].iloc[0]
    case_120 = df_lesion[df_lesion["image_id"] == "120_benign.png"].iloc[0]

    # Panel A: U_mean vs count_fg
    ax = axes3[0]
    ax.scatter(df_lesion["count_fg"], df_lesion["U_mean"], color="#1f77b4", alpha=0.7, edgecolors="k", s=40, label="Validation (N=65)")
    ax.scatter([case_45["count_fg"]], [case_45["U_mean"]], color="red", s=120, edgecolors="k", marker="*", zorder=5, label="45_malignant (DSC=0.05)")
    ax.scatter([case_120["count_fg"]], [case_120["U_mean"]], color="orange", s=100, edgecolors="k", marker="D", zorder=5, label="120_benign (DSC=0.37)")
    ax.set_title(r"$U_{\mathrm{mean}}$ vs. Lesion Area (ρ = +0.55)", fontsize=10, fontweight="bold")
    ax.set_xlabel("Predicted Lesion Area (pixels)", fontsize=9)
    ax.set_ylabel(r"$U_{\mathrm{mean}}$ (Global)", fontsize=9)
    ax.grid(True, linestyle=":", alpha=0.6)
    ax.legend(loc="upper left", fontsize=7.5)

    # Panel B: U_fg vs count_fg
    ax = axes3[1]
    ax.scatter(df_lesion["count_fg"], df_lesion["U_fg"], color="#2ca02c", alpha=0.7, edgecolors="k", s=40, label="Validation (N=65)")
    ax.scatter([case_45["count_fg"]], [case_45["U_fg"]], color="red", s=120, edgecolors="k", marker="*", zorder=5)
    ax.scatter([case_120["count_fg"]], [case_120["U_fg"]], color="orange", s=100, edgecolors="k", marker="D", zorder=5)
    ax.set_title(r"$U_{\mathrm{fg}}$ vs. Lesion Area (ρ = −0.13, Robust)", fontsize=10, fontweight="bold")
    ax.set_xlabel("Predicted Lesion Area (pixels)", fontsize=9)
    ax.set_ylabel(r"$U_{\mathrm{fg}}$ (Foreground)", fontsize=9)
    ax.grid(True, linestyle=":", alpha=0.6)

    # Panel C: U_ratio vs count_fg
    ax = axes3[2]
    ax.scatter(df_lesion["count_fg"], df_lesion["U_ratio"], color="#8c564b", alpha=0.7, edgecolors="k", s=40, label="Validation (N=65)")
    ax.scatter([case_45["count_fg"]], [case_45["U_ratio"]], color="red", s=120, edgecolors="k", marker="*", zorder=5)
    ax.scatter([case_120["count_fg"]], [case_120["U_ratio"]], color="orange", s=100, edgecolors="k", marker="D", zorder=5)
    ax.set_title(r"$U_{\mathrm{ratio}}$ vs. Lesion Area (ρ = −0.63)", fontsize=10, fontweight="bold")
    ax.set_xlabel("Predicted Lesion Area (pixels)", fontsize=9)
    ax.set_ylabel(r"$U_{\mathrm{ratio}}$ (Foreground/Global)", fontsize=9)
    ax.grid(True, linestyle=":", alpha=0.6)

    fig3.suptitle("Lesion Size Confounding: Global Dilution vs. Foreground Localization", fontsize=12, fontweight="bold", y=1.01)
    plt.tight_layout()
    fig3_path = os.path.join(figures_dir, "figure3_lesion_area_bias.png")
    fig3.savefig(fig3_path, dpi=300, bbox_inches="tight")
    plt.close(fig3)
    print(f"  Saved Figure 3: {fig3_path}")

    # -------------------------------------------------------------------------
    # Figure 4: Diagnostic Case Comparison (45_malignant vs 120_benign)
    # -------------------------------------------------------------------------
    fig4, axes4 = plt.subplots(2, 4, figsize=(15, 7.5))

    val_img_dir = os.path.join(data_root, args.dataset, "Val_Folder", "img")
    val_seg_dir = os.path.join(val_root, "seg_results", f"MedCLIPSeg_unimedclip_ViT-B-16_Prompt-{args.prompt_design}")
    val_unc_dir = os.path.join(val_root, "unc_results", f"MedCLIPSeg_unimedclip_ViT-B-16_Prompt-{args.prompt_design}")

    cases = [
        ("45_malignant.png", "Case 1: 45_malignant.png (Large Failure, Area=7,810 px, DSC=0.05)", 0),
        ("120_benign.png", "Case 2: 120_benign.png (Small Diluted Failure, Area=753 px, DSC=0.37)", 1),
    ]

    ln2 = float(np.log(2.0))
    for fname, title, row_idx in cases:
        stem = os.path.splitext(fname)[0]
        img_bgr = cv2.imread(os.path.join(val_img_dir, fname))
        pred_m = cv2.imread(os.path.join(val_seg_dir, fname), cv2.IMREAD_GRAYSCALE)
        u_arr = np.load(os.path.join(val_unc_dir, f"{stem}.npy"))

        img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB) if img_bgr is not None else np.zeros((224, 224, 3), dtype=np.uint8)
        img_rgb_224 = cv2.resize(img_rgb, (224, 224), interpolation=cv2.INTER_LINEAR)

        # 1. Ultrasound
        axes4[row_idx, 0].imshow(img_rgb_224)
        axes4[row_idx, 0].set_title("Ultrasound (224×224)", fontsize=9)
        axes4[row_idx, 0].axis("off")

        # 2. Predicted Mask Overlay
        overlay = img_rgb_224.copy()
        fg = (pred_m > 0)
        overlay[fg] = (overlay[fg] * 0.4 + np.array([255, 60, 60]) * 0.6).clip(0, 255).astype(np.uint8)
        axes4[row_idx, 1].imshow(overlay)
        axes4[row_idx, 1].set_title(f"Prediction (Area={np.sum(fg)})", fontsize=9)
        axes4[row_idx, 1].axis("off")

        # 3. Raw Predictive Entropy
        im = axes4[row_idx, 2].imshow(u_arr, cmap="hot", vmin=0.0, vmax=ln2)
        plt.colorbar(im, ax=axes4[row_idx, 2], fraction=0.046, pad=0.04)
        axes4[row_idx, 2].set_title(f"Entropy Map [0, ln 2]", fontsize=9)
        axes4[row_idx, 2].axis("off")

        # 4. Metric Profile Bar Chart
        c_row = df_lesion[df_lesion["image_id"] == fname].iloc[0]
        bar_names = [r"$U_{\mathrm{mean}}$", r"$U_{\mathrm{fg}}$", r"$U_{\mathrm{bound}}$", r"$U_{\mathrm{thresh}}$"]
        bar_vals = [c_row["U_mean"], c_row["U_fg"], c_row["U_boundary"], c_row["U_threshold"]]
        bar_colors = ["#1f77b4", "#2ca02c", "#d62728", "#9467bd"]

        ax_bar = axes4[row_idx, 3]
        ax_bar.bar(bar_names, bar_vals, color=bar_colors, width=0.55, edgecolor="k")
        ax_bar.set_ylim(0.0, 0.55)
        ax_bar.set_ylabel("Uncertainty Score", fontsize=8)
        ax_bar.set_title(f"Metric Profile (DSC={c_row['DSC']:.2f})", fontsize=9)
        ax_bar.grid(True, linestyle=":", alpha=0.5, axis="y")
        for b_idx, val in enumerate(bar_vals):
            ax_bar.text(b_idx, val + 0.015, f"{val:.3f}", ha="center", fontsize=7.5, fontweight="bold")

    fig4.suptitle("Diagnostic Case Comparison: High-Volume Failure vs. Small-Lesion Dilution Failure", fontsize=12, fontweight="bold", y=0.99)
    plt.tight_layout()
    fig4_path = os.path.join(figures_dir, "figure4_failure_case_diagnostics.png")
    fig4.savefig(fig4_path, dpi=300, bbox_inches="tight")
    plt.close(fig4)
    print(f"  Saved Figure 4: {fig4_path}")

    print("\n" + "=" * 75)
    print("  Stage 08 statistical analysis completed successfully!")
    print(f"  All tables written to  : {tables_dir}")
    print(f"  All figures written to : {figures_dir}")
    print("=" * 75)


if __name__ == "__main__":
    main()

