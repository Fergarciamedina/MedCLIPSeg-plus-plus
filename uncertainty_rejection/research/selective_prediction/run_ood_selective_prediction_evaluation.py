#!/usr/bin/env python3
"""
Research Pipeline - Stage 11B.2: Out-Of-Distribution (OOD) Selective Prediction Evaluation

Executes locked, pre-registered selective prediction evaluation across four external
breast ultrasound benchmarks:
    - UDIAT (N=25, Siemens Elegra, Spain)
    - BUID (N=35, Clinical scanner, academic repository)
    - BUSUC (N=122, Multi-center registry)
    - BUSBRA (N=282, Multi-scanner, Brazil)
    Total OOD test cohort: N=464 images (100% lesion-present)

All uncertainty rejection thresholds, comparator methods, and policies derived strictly
on BUSI validation remain permanently frozen:
    - Primary Metric: U_fg
    - Prespecified Comparator: U_std_fg
    - Primary Operating Cutoff: tau = 0.064058324 (BUSI validation 80% coverage target)
    - Secondary Cutoffs: tau_95, tau_90, tau_85, tau_75, tau_70
    - Empty-Mask Gating Policy: two_stage (tau_neg = 1e-4)

Produces:
    - Comprehensive per-dataset and cross-domain statistical tables (CSV & Markdown).
    - Risk-coverage curves and empirical AURC (1-DSC and 1-NSD).
    - Paired bootstrap Delta_AURC tests against U_fg.
    - Lesion-size confounding analysis (Spearman rho with count_fg).
    - Publication-grade figures.
"""

import os
import sys
import argparse
import json
import yaml
import numpy as np
import pandas as pd
from scipy.stats import spearmanr
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

# ---------------------------------------------------------------------------
# Path setup
# ---------------------------------------------------------------------------
REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", ".."))
sys.path.insert(0, REPO_ROOT)
sys.path.insert(0, os.path.join(REPO_ROOT, "uncertainty_rejection", "src"))
sys.path.insert(0, os.path.join(REPO_ROOT, "shared"))

from uncertainty.selective_prediction import (  # noqa: E402
    compute_risk_coverage_curve,
    compute_aurc,
    compute_paired_bootstrap_aurc_ci,
    apply_empty_mask_policy,
    apply_selective_rejection,
)

METRICS_META = {
    "U_fg": {
        "display": r"$U_{\mathrm{fg}}$ (Primary)",
        "literature": "Roy et al. (Bayesian QuickNAT 2019)",
        "desc": "Foreground Mean Predictive Entropy",
        "color": "#1f77b4",
        "linestyle": "-",
    },
    "U_std_fg": {
        "display": r"$U_{\mathrm{std\_fg}}$ (Comparator)",
        "literature": "Jungo et al. (2020)",
        "desc": "Foreground Entropy Population Standard Deviation",
        "color": "#2ca02c",
        "linestyle": "--",
    },
    "U_mean": {
        "display": r"$U_{\mathrm{mean}}$",
        "literature": "Kendall & Gal (2017), Lakshminarayanan (2017)",
        "desc": "Global Full-Image Mean Entropy",
        "color": "#d62728",
        "linestyle": "-.",
    },
    "U_boundary": {
        "display": r"$U_{\mathrm{boundary}}$",
        "literature": "Wang et al. (TTA 2019), Mehrtash (2020)",
        "desc": "Morphological Boundary Band Mean Entropy",
        "color": "#ff7f0e",
        "linestyle": ":",
    },
    "U_threshold": {
        "display": r"$U_{\mathrm{threshold}}$",
        "literature": "Eaton-Rosen et al. (MICCAI 2018)",
        "desc": "High-Uncertainty Pixel Area Fraction",
        "color": "#9467bd",
        "linestyle": "-.",
    },
    "U_positive": {
        "display": r"$U_{\mathrm{positive}}$",
        "literature": "Ablation control (positive entropy)",
        "desc": "Non-zero Entropy Mean",
        "color": "#8c564b",
        "linestyle": ":",
    },
    "U_ratio": {
        "display": r"$U_{\mathrm{ratio}}$",
        "literature": "Ablation control (contrast ratio)",
        "desc": "Foreground-to-Global Entropy Ratio",
        "color": "#e377c2",
        "linestyle": "--",
    },
}

FROZEN_THRESHOLDS = {
    "tau_95": 0.101927518,
    "tau_90": 0.083463000,
    "tau_85": 0.072696542,
    "tau_80": 0.064058324,  # PRIMARY OPERATING POINT
    "tau_75": 0.059689360,
    "tau_70": 0.056208188,
}
PRIMARY_TAU = FROZEN_THRESHOLDS["tau_80"]
TAU_NEG = 1.0e-4


class NpEncoder(json.JSONEncoder):
    def default(self, obj):
        if isinstance(obj, (np.integer, np.int64, np.int32)):
            return int(obj)
        if isinstance(obj, (np.floating, np.float32, np.float64)):
            return float(obj)
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        if isinstance(obj, (np.bool_, bool)):
            return bool(obj)
        return super().default(obj)


def compute_bootstrap_metric_ci(
    values: np.ndarray,
    n_bootstrap: int = 1000,
    ci: float = 0.95,
    seed: int = 666,
) -> tuple:
    """Computes empirical percentile bootstrap CI for the mean of a 1D array."""
    if len(values) == 0:
        return np.nan, np.nan, np.nan
    rng = np.random.default_rng(seed)
    means = [float(np.mean(rng.choice(values, size=len(values), replace=True))) for _ in range(n_bootstrap)]
    alpha = 1.0 - ci
    return float(np.mean(values)), float(np.percentile(means, 100.0 * (alpha / 2.0))), float(np.percentile(means, 100.0 * (1.0 - alpha / 2.0)))


def compute_bootstrap_gain_ci(
    df: pd.DataFrame,
    feature_col: str,
    tau: float,
    metric_col: str = "DSC",
    n_bootstrap: int = 1000,
    ci: float = 0.95,
    seed: int = 666,
) -> tuple:
    """Computes bootstrap CI for the selective prediction gain: Mean(Accepted) - Mean(Full)."""
    n = len(df)
    if n == 0:
        return np.nan, np.nan, np.nan, np.nan, np.nan

    res_orig = apply_selective_rejection(df, feature_col, tau)
    acc_point = res_orig[f"accepted_mean_{metric_col.lower()}"]
    full_point = res_orig[f"full_mean_{metric_col.lower()}"]
    gain_point = acc_point - full_point

    rng = np.random.default_rng(seed)
    gains = []
    acc_means = []

    for _ in range(n_bootstrap):
        boot_idx = rng.integers(0, n, size=n)
        sample = df.iloc[boot_idx].reset_index(drop=True)
        res_b = apply_selective_rejection(sample, feature_col, tau)
        if res_b["n_accepted"] > 0:
            a_mean = res_b[f"accepted_mean_{metric_col.lower()}"]
            f_mean = res_b[f"full_mean_{metric_col.lower()}"]
            gains.append(a_mean - f_mean)
            acc_means.append(a_mean)

    alpha = 1.0 - ci
    gain_ci_lower = float(np.percentile(gains, 100.0 * (alpha / 2.0))) if gains else np.nan
    gain_ci_upper = float(np.percentile(gains, 100.0 * (1.0 - alpha / 2.0))) if gains else np.nan
    acc_ci_lower = float(np.percentile(acc_means, 100.0 * (alpha / 2.0))) if acc_means else np.nan
    acc_ci_upper = float(np.percentile(acc_means, 100.0 * (1.0 - alpha / 2.0))) if acc_means else np.nan

    return gain_point, gain_ci_lower, gain_ci_upper, acc_ci_lower, acc_ci_upper


def evaluate_dataset(
    dset_name: str,
    df_raw: pd.DataFrame,
    out_dir: str,
    n_bootstrap: int = 1000,
    seed: int = 666,
) -> dict:
    """Evaluates selective prediction on a single dataset."""
    os.makedirs(out_dir, exist_ok=True)
    n_total = len(df_raw)

    # 1. Baseline stats
    base_dsc, base_dsc_l, base_dsc_u = compute_bootstrap_metric_ci(
        df_raw["DSC"].values, n_bootstrap=n_bootstrap, seed=seed
    )
    base_nsd, base_nsd_l, base_nsd_u = compute_bootstrap_metric_ci(
        df_raw["NSD"].values, n_bootstrap=n_bootstrap, seed=seed
    )

    baseline_stats = {
        "dataset": dset_name,
        "n_total": n_total,
        "mean_dsc": base_dsc,
        "mean_dsc_95ci": [base_dsc_l, base_dsc_u],
        "std_dsc": float(df_raw["DSC"].std()),
        "min_dsc": float(df_raw["DSC"].min()),
        "max_dsc": float(df_raw["DSC"].max()),
        "mean_nsd": base_nsd,
        "mean_nsd_95ci": [base_nsd_l, base_nsd_u],
        "std_nsd": float(df_raw["NSD"].std()),
        "min_nsd": float(df_raw["NSD"].min()),
        "max_nsd": float(df_raw["NSD"].max()),
    }

    # 2. Check empty predictions
    empty_preds = df_raw[df_raw["valid_fg"] == False] if "valid_fg" in df_raw.columns else df_raw[df_raw["count_fg"] == 0]
    n_empty = len(empty_preds)

    # 3. Setting A: Lesion cohort (valid U_fg)
    df_valid = df_raw.dropna(subset=["U_fg"]).copy().reset_index(drop=True)
    n_valid = len(df_valid)

    # (a) Primary Operating Point Evaluation (tau = 0.064058)
    primary_eval = apply_selective_rejection(df_valid, "U_fg", PRIMARY_TAU)
    dsc_gain, dsc_gain_lo, dsc_gain_hi, dsc_acc_lo, dsc_acc_hi = compute_bootstrap_gain_ci(
        df_valid, "U_fg", PRIMARY_TAU, metric_col="DSC", n_bootstrap=n_bootstrap, seed=seed
    )
    nsd_gain, nsd_gain_lo, nsd_gain_hi, nsd_acc_lo, nsd_acc_hi = compute_bootstrap_gain_ci(
        df_valid, "U_fg", PRIMARY_TAU, metric_col="NSD", n_bootstrap=n_bootstrap, seed=seed
    )

    primary_summary = {
        "dataset": dset_name,
        "tau_primary": PRIMARY_TAU,
        "n_accepted": primary_eval["n_accepted"],
        "n_rejected": primary_eval["n_rejected"],
        "realized_coverage": primary_eval["realized_coverage"],
        "accepted_dsc": primary_eval["accepted_mean_dsc"],
        "accepted_dsc_ci": [dsc_acc_lo, dsc_acc_hi],
        "dsc_gain": dsc_gain,
        "dsc_gain_ci": [dsc_gain_lo, dsc_gain_hi],
        "rejected_dsc": primary_eval["rejected_mean_dsc"],
        "accepted_nsd": primary_eval["accepted_mean_nsd"],
        "accepted_nsd_ci": [nsd_acc_lo, nsd_acc_hi],
        "nsd_gain": nsd_gain,
        "nsd_gain_ci": [nsd_gain_lo, nsd_gain_hi],
        "rejected_nsd": primary_eval["rejected_mean_nsd"],
    }
    pd.DataFrame([primary_summary]).to_csv(os.path.join(out_dir, f"primary_operating_point_{dset_name}.csv"), index=False)

    # (b) Multi-Threshold Operating Points
    multi_point_rows = []
    for pt_name, tau_val in FROZEN_THRESHOLDS.items():
        res = apply_selective_rejection(df_valid, "U_fg", tau_val)
        g_dsc, g_dsc_l, g_dsc_h, a_dsc_l, a_dsc_h = compute_bootstrap_gain_ci(
            df_valid, "U_fg", tau_val, metric_col="DSC", n_bootstrap=n_bootstrap, seed=seed
        )
        multi_point_rows.append({
            "point": pt_name,
            "tau": tau_val,
            "n_accepted": res["n_accepted"],
            "realized_coverage": res["realized_coverage"],
            "accepted_dsc": res["accepted_mean_dsc"],
            "dsc_gain": g_dsc,
            "dsc_gain_95ci": f"[{g_dsc_l:+.4f}, {g_dsc_h:+.4f}]",
            "accepted_nsd": res["accepted_mean_nsd"],
            "rejected_dsc": res["rejected_mean_dsc"],
        })
    df_operating = pd.DataFrame(multi_point_rows)
    df_operating.to_csv(os.path.join(out_dir, f"operating_points_{dset_name}.csv"), index=False)

    # (c) AURC and Risk-Coverage Curves
    aurc_results = {"DSC": {}, "NSD": {}}
    rc_curves = {"DSC": {}, "NSD": {}}

    # Reference curves (Oracle & Random)
    cov_ora_d, risk_ora_d = compute_risk_coverage_curve(df_valid, "DSC", "DSC", reverse_sort=True, handle_ties=True)
    cov_ora_n, risk_ora_n = compute_risk_coverage_curve(df_valid, "NSD", "NSD", reverse_sort=True, handle_ties=True)
    aurc_ora_d = compute_aurc(cov_ora_d, risk_ora_d)
    aurc_ora_n = compute_aurc(cov_ora_n, risk_ora_n)

    cov_fg_d, risk_fg_d = compute_risk_coverage_curve(df_valid, "U_fg", "DSC", handle_ties=True)
    cov_fg_n, risk_fg_n = compute_risk_coverage_curve(df_valid, "U_fg", "NSD", handle_ties=True)

    r_rand_d = np.full_like(cov_fg_d, 1.0 - base_dsc)
    r_rand_n = np.full_like(cov_fg_n, 1.0 - base_nsd)
    aurc_rand_d = compute_aurc(cov_fg_d, r_rand_d)
    aurc_rand_n = compute_aurc(cov_fg_n, r_rand_n)

    aurc_results["DSC"]["Oracle"] = aurc_ora_d
    aurc_results["NSD"]["Oracle"] = aurc_ora_n
    aurc_results["DSC"]["Random"] = aurc_rand_d
    aurc_results["NSD"]["Random"] = aurc_rand_n

    rc_curves["DSC"]["Oracle"] = {"covs": cov_ora_d, "risks": risk_ora_d}
    rc_curves["NSD"]["Oracle"] = {"covs": cov_ora_n, "risks": risk_ora_n}
    rc_curves["DSC"]["Random"] = {"covs": cov_fg_d, "risks": r_rand_d}
    rc_curves["NSD"]["Random"] = {"covs": cov_fg_n, "risks": r_rand_n}

    method_comparison_rows = [
        {
            "Method": "Oracle (GT Ceiling)",
            "Type": "Reference Bound",
            "AURC_DSC": aurc_ora_d,
            "AURC_NSD": aurc_ora_n,
            "Gain_vs_Random_AURC": aurc_rand_d - aurc_ora_d,
        },
        {
            "Method": "Random Rejection",
            "Type": "Uninformed Control",
            "AURC_DSC": aurc_rand_d,
            "AURC_NSD": aurc_rand_n,
            "Gain_vs_Random_AURC": 0.0,
        },
    ]

    for m_name, meta in METRICS_META.items():
        if m_name not in df_valid.columns:
            continue
        c_d, r_d = compute_risk_coverage_curve(df_valid, m_name, "DSC", handle_ties=True)
        c_n, r_n = compute_risk_coverage_curve(df_valid, m_name, "NSD", handle_ties=True)
        a_d = compute_aurc(c_d, r_d)
        a_n = compute_aurc(c_n, r_n)

        aurc_results["DSC"][m_name] = a_d
        aurc_results["NSD"][m_name] = a_n
        rc_curves["DSC"][m_name] = {"covs": c_d, "risks": r_d}
        rc_curves["NSD"][m_name] = {"covs": c_n, "risks": r_n}

        method_comparison_rows.append({
            "Method": m_name,
            "Type": "Primary" if m_name == "U_fg" else ("Comparator" if m_name == "U_std_fg" else "Baseline"),
            "AURC_DSC": a_d,
            "AURC_NSD": a_n,
            "Gain_vs_Random_AURC": aurc_rand_d - a_d,
        })

    df_comp = pd.DataFrame(method_comparison_rows)
    df_comp.to_csv(os.path.join(out_dir, f"method_comparison_{dset_name}.csv"), index=False)

    # (d) Paired Bootstrap Delta_AURC against U_fg (on DSC)
    delta_aurc_rows = []
    comparisons = [
        ("U_fg", "U_std_fg", "Primary vs Prespecified Comparator"),
        ("U_fg", "U_mean", "Primary vs Global Full-Image Entropy"),
        ("U_fg", "U_boundary", "Primary vs Morphological Boundary Band"),
        ("U_fg", "U_threshold", "Primary vs High-Uncertainty Fraction"),
        ("U_fg", "U_positive", "Primary vs Positive-Entropy Filter"),
        ("U_fg", "U_ratio", "Primary vs Uncertainty Ratio"),
        ("U_std_fg", "U_mean", "Comparator vs Global Full-Image Entropy"),
    ]
    for col_a, col_b, label in comparisons:
        if col_a in df_valid.columns and col_b in df_valid.columns:
            d_dsc, d_l, d_u, p_dsc = compute_paired_bootstrap_aurc_ci(
                df_valid, col_a, col_b, metric_col="DSC", n_bootstrap=n_bootstrap, seed=seed
            )
            d_nsd, dn_l, dn_u, p_nsd = compute_paired_bootstrap_aurc_ci(
                df_valid, col_a, col_b, metric_col="NSD", n_bootstrap=n_bootstrap, seed=seed
            )
            delta_aurc_rows.append({
                "Comparison": f"{col_a} vs {col_b}",
                "Description": label,
                "Delta_AURC_DSC": d_dsc,
                "DSC_95CI": f"[{d_l:+.4f}, {d_u:+.4f}]",
                "p_value_DSC": p_dsc,
                "Significant_DSC": (p_dsc < 0.05),
                "Delta_AURC_NSD": d_nsd,
                "NSD_95CI": f"[{dn_l:+.4f}, {dn_u:+.4f}]",
                "p_value_NSD": p_nsd,
                "Significant_NSD": (p_nsd < 0.05),
            })
    df_delta_aurc = pd.DataFrame(delta_aurc_rows)
    df_delta_aurc.to_csv(os.path.join(out_dir, f"delta_aurc_{dset_name}.csv"), index=False)

    # (e) Lesion Size Confounding Analysis
    spearman_confounding = {}
    if "count_fg" in df_valid.columns:
        for m_name in ["U_fg", "U_std_fg", "U_mean", "U_boundary", "U_ratio"]:
            if m_name in df_valid.columns:
                sub = df_valid.dropna(subset=[m_name, "count_fg"])
                rho, p_val = spearmanr(sub[m_name], sub["count_fg"])
                spearman_confounding[m_name] = {"rho": float(rho), "p_value": float(p_val)}

    # (f) Setting B: Two-stage gating policy evaluation
    two_stage_res = apply_empty_mask_policy(
        df_raw,
        policy="two_stage",
        primary_score_col="U_fg",
        fallback_col="U_mean",
        two_stage_tau_neg=TAU_NEG,
    )
    drop_empty_res = apply_empty_mask_policy(
        df_raw,
        policy="drop_empty",
        primary_score_col="U_fg",
    )

    # Save summary json
    summary_data = {
        "baseline_stats": baseline_stats,
        "n_empty_masks": n_empty,
        "primary_evaluation": primary_summary,
        "aurc_dsc": {k: float(v) for k, v in aurc_results["DSC"].items()},
        "aurc_nsd": {k: float(v) for k, v in aurc_results["NSD"].items()},
        "confounding_spearman": spearman_confounding,
        "two_stage_actions": dict(two_stage_res["policy_action"].value_counts()),
        "drop_empty_actions": dict(drop_empty_res["policy_action"].value_counts()),
    }
    with open(os.path.join(out_dir, f"summary_evaluation_{dset_name}.json"), "w") as f:
        json.dump(summary_data, f, indent=2, cls=NpEncoder)

    return {
        "summary": summary_data,
        "rc_curves": rc_curves,
        "df_valid": df_valid,
        "df_operating": df_operating,
        "df_delta_aurc": df_delta_aurc,
    }


def plot_cross_domain_risk_coverage(all_results: dict, out_path: str):
    """Generates cross-domain risk-coverage comparison figure."""
    fig, axes = plt.subplots(1, 2, figsize=(14, 6), sharey=True)

    colors = {
        "BUSI": "#111111",
        "UDIAT": "#1f77b4",
        "BUID": "#ff7f0e",
        "BUSUC": "#2ca02c",
        "BUSBRA": "#d62728",
    }
    linestyles = {
        "BUSI": "--",
        "UDIAT": "-",
        "BUID": "-",
        "BUSUC": "-",
        "BUSBRA": "-",
    }

    # Plot (a) Risk = 1 - DSC
    ax1 = axes[0]
    for dset, res in all_results.items():
        if "DSC" in res["rc_curves"] and "U_fg" in res["rc_curves"]["DSC"]:
            curve = res["rc_curves"]["DSC"]["U_fg"]
            aurc_val = res["summary"]["aurc_dsc"]["U_fg"]
            lbl = f"{dset} (AURC = {aurc_val:.4f})"
            ax1.plot(
                curve["covs"],
                curve["risks"],
                label=lbl,
                color=colors.get(dset, "gray"),
                linestyle=linestyles.get(dset, "-"),
                linewidth=2.2 if dset != "BUSI" else 1.8,
            )

    ax1.set_xlabel("Coverage (Fraction Retained)", fontsize=12, fontweight="bold")
    ax1.set_ylabel("Risk: (1 - Mean DSC)", fontsize=12, fontweight="bold")
    ax1.set_title("Cross-Domain Risk-Coverage: U_fg (1 - DSC)", fontsize=13, fontweight="bold")
    ax1.grid(True, linestyle="--", alpha=0.6)
    ax1.legend(loc="upper left", fontsize=10, framealpha=0.9)
    ax1.set_xlim(0.0, 1.02)
    ax1.set_ylim(0.0, 0.45)

    # Plot (b) Risk = 1 - NSD
    ax2 = axes[1]
    for dset, res in all_results.items():
        if "NSD" in res["rc_curves"] and "U_fg" in res["rc_curves"]["NSD"]:
            curve = res["rc_curves"]["NSD"]["U_fg"]
            aurc_val = res["summary"]["aurc_nsd"]["U_fg"]
            lbl = f"{dset} (AURC = {aurc_val:.4f})"
            ax2.plot(
                curve["covs"],
                curve["risks"],
                label=lbl,
                color=colors.get(dset, "gray"),
                linestyle=linestyles.get(dset, "-"),
                linewidth=2.2 if dset != "BUSI" else 1.8,
            )

    ax2.set_xlabel("Coverage (Fraction Retained)", fontsize=12, fontweight="bold")
    ax2.set_ylabel("Risk: (1 - Mean NSD)", fontsize=12, fontweight="bold")
    ax2.set_title("Cross-Domain Risk-Coverage: U_fg (1 - NSD)", fontsize=13, fontweight="bold")
    ax2.grid(True, linestyle="--", alpha=0.6)
    ax2.legend(loc="upper left", fontsize=10, framealpha=0.9)
    ax2.set_xlim(0.0, 1.02)
    ax2.set_ylim(0.0, 0.45)

    plt.tight_layout()
    plt.savefig(out_path, dpi=300)
    plt.close()
    print(f"Saved cross-domain risk-coverage figure to: {out_path}")


def main():
    parser = argparse.ArgumentParser(description="Locked OOD Selective Prediction Evaluation.")
    parser.add_argument("--seed", type=int, default=666)
    parser.add_argument("--n-bootstrap", type=int, default=1000)
    parser.add_argument("--output-dir", type=str, default="")
    args = parser.parse_args()

    base_out = args.output_dir or os.environ.get(
        "MEDCLIPSEG_OUTPUT", os.path.abspath(os.path.join(REPO_ROOT, "..", "data", "outputs"))
    )

    ood_datasets = ["UDIAT", "BUID", "BUSUC", "BUSBRA"]
    all_datasets = ["BUSI"] + ood_datasets

    all_results = {}

    print("=" * 80)
    print("  ELEC 6661 — Out-Of-Distribution Selective Prediction Evaluation")
    print("  Frozen Model: MedCLIPSeg ViT-B/16 (BUSI seed 666)")
    print(f"  Primary Metric: U_fg  |  Primary Threshold tau: {PRIMARY_TAU:.6f}")
    print(f"  Target Cohorts: {', '.join(all_datasets)}")
    print("=" * 80)

    for dset in all_datasets:
        csv_path = os.path.join(
            base_out,
            dset,
            "aggregation_results",
            f"seed{args.seed}",
            f"uncertainty_scores_{dset}_test_seed{args.seed}.csv",
        )
        if not os.path.exists(csv_path):
            print(f"[WARNING] Skipping {dset}: Uncertainty CSV not found at {csv_path}")
            continue

        print(f"\n>>> Evaluating {dset} from {csv_path}...")
        df_raw = pd.read_csv(csv_path)

        # For BUSI, Setting A evaluates the lesion-present cohort (N=65) to match Step 10 exactly
        if dset == "BUSI":
            df_lesion = df_raw.dropna(subset=["U_fg"]).copy().reset_index(drop=True)
            print(f"  Total BUSI test images: {len(df_raw)} (Setting A Lesion cohort: {len(df_lesion)})")
            df_eval_target = df_lesion
        else:
            df_eval_target = df_raw

        dset_out = os.path.join(base_out, dset, "test_evaluation", f"seed{args.seed}")
        res = evaluate_dataset(
            dset_name=dset,
            df_raw=df_eval_target,
            out_dir=dset_out,
            n_bootstrap=args.n_bootstrap,
            seed=args.seed,
        )
        all_results[dset] = res

    # Global cross-domain reports and figures
    if len(all_results) > 1:
        global_eval_dir = os.path.join(base_out, "cross_domain_evaluation", f"seed{args.seed}")
        os.makedirs(global_eval_dir, exist_ok=True)

        # 1. Cross-domain Master Table
        master_rows = []
        for dset, res in all_results.items():
            base = res["summary"]["baseline_stats"]
            prim = res["summary"]["primary_evaluation"]
            aurc_d = res["summary"]["aurc_dsc"]
            cov_pct = prim["realized_coverage"] * 100.0
            total_eval = prim["n_accepted"] + prim["n_rejected"]
            master_rows.append({
                "Dataset": dset,
                "Cohort Size (N)": base["n_total"],
                "Prevalence": "100% Lesion" if dset != "BUSI" else "83.3% (65/78)",
                "Baseline DSC": f"{base['mean_dsc']:.4f} ± {base['std_dsc']:.4f}",
                "Baseline NSD": f"{base['mean_nsd']:.4f} ± {base['std_nsd']:.4f}",
                "Coverage at Frozen tau": f"{cov_pct:.1f}% ({prim['n_accepted']}/{total_eval})",
                "Accepted DSC": f"{prim['accepted_dsc']:.4f}",
                "DSC Absolute Gain": f"{prim['dsc_gain']:+.4f} (95% CI [{prim['dsc_gain_ci'][0]:+.4f}, {prim['dsc_gain_ci'][1]:+.4f}])",
                "Rejected DSC": f"{prim['rejected_dsc']:.4f}",
                "Accepted NSD": f"{prim['accepted_nsd']:.4f}",
                "NSD Absolute Gain": f"{prim['nsd_gain']:+.4f}",
                "U_fg AURC (1-DSC)": f"{aurc_d['U_fg']:.4f}",
                "U_mean AURC (1-DSC)": f"{aurc_d['U_mean']:.4f}",
                "Random AURC (1-DSC)": f"{aurc_d['Random']:.4f}",
                "Oracle AURC (1-DSC)": f"{aurc_d['Oracle']:.4f}",
            })
        # Compute Pooled OOD rows across the 4 OOD benchmarks
        ood_keys = [k for k in ["UDIAT", "BUID", "BUSUC", "BUSBRA"] if k in all_results]
        if len(ood_keys) == 4:
            # 1. Micro-average (Concatenating all 464 images)
            df_ood_pooled = pd.concat([all_results[k]["df_valid"] for k in ood_keys], ignore_index=True)
            n_ood_total = len(df_ood_pooled)

            prim_ood = apply_selective_rejection(df_ood_pooled, "U_fg", PRIMARY_TAU)
            g_d_pool, g_dl_pool, g_dh_pool, a_dl_pool, a_dh_pool = compute_bootstrap_gain_ci(
                df_ood_pooled, "U_fg", PRIMARY_TAU, metric_col="DSC", n_bootstrap=args.n_bootstrap, seed=args.seed
            )
            g_n_pool, g_nl_pool, g_nh_pool, a_nl_pool, a_nh_pool = compute_bootstrap_gain_ci(
                df_ood_pooled, "U_fg", PRIMARY_TAU, metric_col="NSD", n_bootstrap=args.n_bootstrap, seed=args.seed
            )

            c_p_fg, r_p_fg = compute_risk_coverage_curve(df_ood_pooled, "U_fg", "DSC", handle_ties=True)
            c_p_mean, r_p_mean = compute_risk_coverage_curve(df_ood_pooled, "U_mean", "DSC", handle_ties=True)
            c_p_ora, r_p_ora = compute_risk_coverage_curve(df_ood_pooled, "DSC", "DSC", reverse_sort=True, handle_ties=True)
            r_p_rand = np.full_like(c_p_fg, 1.0 - df_ood_pooled["DSC"].mean())

            p_aurc_fg = compute_aurc(c_p_fg, r_p_fg)
            p_aurc_mean = compute_aurc(c_p_mean, r_p_mean)
            p_aurc_rand = compute_aurc(c_p_fg, r_p_rand)
            p_aurc_ora = compute_aurc(c_p_ora, r_p_ora)

            master_rows.append({
                "Dataset": "POOLED OOD (Micro)",
                "Cohort Size (N)": n_ood_total,
                "Prevalence": "100% Lesion",
                "Baseline DSC": f"{df_ood_pooled['DSC'].mean():.4f} ± {df_ood_pooled['DSC'].std():.4f}",
                "Baseline NSD": f"{df_ood_pooled['NSD'].mean():.4f} ± {df_ood_pooled['NSD'].std():.4f}",
                "Coverage at Frozen tau": f"{prim_ood['realized_coverage']*100:.1f}% ({prim_ood['n_accepted']}/{n_ood_total})",
                "Accepted DSC": f"{prim_ood['accepted_mean_dsc']:.4f}",
                "DSC Absolute Gain": f"{g_d_pool:+.4f} (95% CI [{g_dl_pool:+.4f}, {g_dh_pool:+.4f}])",
                "Rejected DSC": f"{prim_ood['rejected_mean_dsc']:.4f}",
                "Accepted NSD": f"{prim_ood['accepted_mean_nsd']:.4f}",
                "NSD Absolute Gain": f"{g_n_pool:+.4f}",
                "U_fg AURC (1-DSC)": f"{p_aurc_fg:.4f}",
                "U_mean AURC (1-DSC)": f"{p_aurc_mean:.4f}",
                "Random AURC (1-DSC)": f"{p_aurc_rand:.4f}",
                "Oracle AURC (1-DSC)": f"{p_aurc_ora:.4f}",
            })

            # 2. Macro-average (Unweighted Mean of the 4 Datasets)
            macro_base_dsc = np.mean([all_results[k]["summary"]["baseline_stats"]["mean_dsc"] for k in ood_keys])
            macro_base_nsd = np.mean([all_results[k]["summary"]["baseline_stats"]["mean_nsd"] for k in ood_keys])
            macro_cov = np.mean([all_results[k]["summary"]["primary_evaluation"]["realized_coverage"] for k in ood_keys])
            macro_acc_dsc = np.mean([all_results[k]["summary"]["primary_evaluation"]["accepted_dsc"] for k in ood_keys])
            macro_rej_dsc = np.mean([all_results[k]["summary"]["primary_evaluation"]["rejected_dsc"] for k in ood_keys])
            macro_gain_dsc = macro_acc_dsc - macro_base_dsc
            macro_acc_nsd = np.mean([all_results[k]["summary"]["primary_evaluation"]["accepted_nsd"] for k in ood_keys])
            macro_gain_nsd = macro_acc_nsd - macro_base_nsd
            macro_aurc_fg = np.mean([all_results[k]["summary"]["aurc_dsc"]["U_fg"] for k in ood_keys])
            macro_aurc_mean = np.mean([all_results[k]["summary"]["aurc_dsc"]["U_mean"] for k in ood_keys])
            macro_aurc_rand = np.mean([all_results[k]["summary"]["aurc_dsc"]["Random"] for k in ood_keys])
            macro_aurc_ora = np.mean([all_results[k]["summary"]["aurc_dsc"]["Oracle"] for k in ood_keys])

            master_rows.append({
                "Dataset": "MACRO OOD (Unweighted)",
                "Cohort Size (N)": "4 Datasets",
                "Prevalence": "100% Lesion",
                "Baseline DSC": f"{macro_base_dsc:.4f}",
                "Baseline NSD": f"{macro_base_nsd:.4f}",
                "Coverage at Frozen tau": f"{macro_cov*100:.1f}%",
                "Accepted DSC": f"{macro_acc_dsc:.4f}",
                "DSC Absolute Gain": f"{macro_gain_dsc:+.4f}",
                "Rejected DSC": f"{macro_rej_dsc:.4f}",
                "Accepted NSD": f"{macro_acc_nsd:.4f}",
                "NSD Absolute Gain": f"{macro_gain_nsd:+.4f}",
                "U_fg AURC (1-DSC)": f"{macro_aurc_fg:.4f}",
                "U_mean AURC (1-DSC)": f"{macro_aurc_mean:.4f}",
                "Random AURC (1-DSC)": f"{macro_aurc_rand:.4f}",
                "Oracle AURC (1-DSC)": f"{macro_aurc_ora:.4f}",
            })

        df_master = pd.DataFrame(master_rows)
        df_master.to_csv(os.path.join(global_eval_dir, "cross_domain_summary.csv"), index=False)
        print("\n" + "=" * 80)
        print("  CROSS-DOMAIN MASTER SUMMARY TABLE")
        print("=" * 80)
        print(df_master.to_string(index=False))

        # 2. Cross-domain Plot
        plot_cross_domain_risk_coverage(
            all_results,
            os.path.join(global_eval_dir, "cross_domain_risk_coverage_overlay.png"),
        )

        # 3. Lesion size confounding cross-domain table
        conf_rows = []
        for dset, res in all_results.items():
            conf = res["summary"]["confounding_spearman"]
            conf_rows.append({
                "Dataset": dset,
                "U_fg vs Area (rho)": f"{conf.get('U_fg', {}).get('rho', np.nan):+.3f}",
                "U_fg p-value": f"{conf.get('U_fg', {}).get('p_value', np.nan):.3e}",
                "U_mean vs Area (rho)": f"{conf.get('U_mean', {}).get('rho', np.nan):+.3f}",
                "U_mean p-value": f"{conf.get('U_mean', {}).get('p_value', np.nan):.3e}",
            })
        df_conf = pd.DataFrame(conf_rows)
        df_conf.to_csv(os.path.join(global_eval_dir, "cross_domain_confounding.csv"), index=False)
        print("\n" + "=" * 80)
        print("  LESION-SIZE CONFOUNDING (SPEARMAN RHO WITH FOREGROUND AREA)")
        print("=" * 80)
        print(df_conf.to_string(index=False))

    print("\n[SUCCESS] OOD evaluation complete.")


if __name__ == "__main__":
    main()
