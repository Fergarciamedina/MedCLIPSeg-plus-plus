#!/usr/bin/env python3
"""Command-line interface for uncertainty-based selective prediction evaluation.

Evaluates medical image segmentation outputs under selective prediction:
1. Applies frozen rejection threshold tau (default: 0.064058324).
2. Computes realized coverage, retained accuracy (DSC/NSD), and selective gains.
3. Computes Area Under the Risk-Coverage Curve (AURC) vs. Random and Oracle baselines.
4. Generates standardized CSV reports and risk-coverage coordinate files.

Usage examples:
    # Evaluate a precomputed CSV
    python evaluate.py --input_csv /path/to/predictions.csv --output_dir ./results

    # Compare all 7 uncertainty aggregators
    python evaluate.py --input_csv /path/to/predictions.csv --output_dir ./results --compare_aggregators

    # Run on synthetic demonstration data
    python evaluate.py --demo --output_dir ./demo_results
"""

import argparse
import os
import sys
from typing import Dict, List, Optional, Union
import numpy as np
import pandas as pd

try:
    from .metrics import (
        compute_dice,
        compute_risk_coverage_curve,
        compute_aurc,
        compute_random_aurc,
        compute_oracle_aurc,
        compute_spearman_correlation,
        compute_paired_bootstrap_ci,
        apply_selective_rejection,
    )
    from .model_adapter import MockModelAdapter
except ImportError:
    # If run directly as a script
    from metrics import (
        compute_dice,
        compute_risk_coverage_curve,
        compute_aurc,
        compute_random_aurc,
        compute_oracle_aurc,
        compute_spearman_correlation,
        compute_paired_bootstrap_ci,
        apply_selective_rejection,
    )
    from model_adapter import MockModelAdapter


FROZEN_DEFAULT_THRESHOLD = 0.064058324


def parse_arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Selective Prediction Evaluation Toolkit for Medical Image Segmentation."
    )
    parser.add_argument(
        "--input_csv",
        type=str,
        default=None,
        help="Path to CSV containing per-image predictions and uncertainty scores.",
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default="./evaluation_output",
        help="Directory to save evaluation reports and curves.",
    )
    parser.add_argument(
        "--threshold",
        type=float,
        default=FROZEN_DEFAULT_THRESHOLD,
        help=f"Decision threshold tau (default: {FROZEN_DEFAULT_THRESHOLD}).",
    )
    parser.add_argument(
        "--aggregator",
        type=str,
        default="u_fg",
        help="Primary uncertainty column name (default: 'u_fg').",
    )
    parser.add_argument(
        "--metric_col",
        type=str,
        default="DSC",
        help="Segmentation metric column name (default: 'DSC').",
    )
    parser.add_argument(
        "--compare_aggregators",
        action="store_true",
        help="Evaluate and rank all available uncertainty aggregators in input CSV.",
    )
    parser.add_argument(
        "--two_stage",
        action="store_true",
        help="Enable two-stage empty-mask fallback (U_mean <= 1e-4) for screening cohorts with normal scans.",
    )
    parser.add_argument(
        "--dataset_name",
        type=str,
        default="Cohort",
        help="Human-readable label for dataset in summary tables.",
    )
    parser.add_argument(
        "--demo",
        action="store_true",
        help="Generate synthetic demonstration cohort and execute evaluation.",
    )
    return parser.parse_args()


def generate_demo_dataset(num_samples: int = 100, seed: int = 42) -> pd.DataFrame:
    """Generate realistic synthetic evaluation dataset with correlated error and uncertainty."""
    rng = np.random.RandomState(seed)
    
    # Ground truth loss (1 - DSC) drawn from beta-like distribution
    losses = rng.beta(2, 5, size=num_samples)
    dscs = np.clip(1.0 - losses, 0.0, 1.0)
    
    # U_fg strongly correlated with loss
    u_fg = 0.04 + 0.06 * losses + rng.normal(0, 0.015, size=num_samples)
    u_fg = np.clip(u_fg, 0.01, 0.25)
    
    # Area fraction
    areas = rng.uniform(0.01, 0.25, size=num_samples)
    
    # U_mean heavily diluted by area
    u_mean = areas * u_fg + rng.uniform(0.0001, 0.001, size=num_samples)
    
    # Additional aggregators
    u_std_fg = u_fg * rng.uniform(0.7, 1.2, size=num_samples)
    u_boundary = u_fg + rng.uniform(0.02, 0.08, size=num_samples)
    u_positive = u_mean * rng.uniform(1.0, 1.5, size=num_samples)
    u_threshold = u_fg * rng.uniform(1.1, 1.8, size=num_samples)
    u_ratio = u_fg / (u_mean + 1e-6)

    data = {
        "image_id": [f"case_{i:04d}" for i in range(num_samples)],
        "DSC": dscs,
        "u_fg": u_fg,
        "u_std_fg": u_std_fg,
        "u_boundary": u_boundary,
        "u_mean": u_mean,
        "u_positive": u_positive,
        "u_threshold": u_threshold,
        "u_ratio": u_ratio,
    }
    return pd.DataFrame(data)


def run_evaluation(
    df: pd.DataFrame,
    threshold: float,
    aggregator: str,
    metric_col: str,
    dataset_name: str,
    compare_aggregators: bool,
    output_dir: str,
    two_stage: bool = False,
) -> Dict[str, Union[float, str]]:
    """Execute complete selective prediction evaluation."""
    os.makedirs(output_dir, exist_ok=True)

    # Validate column presence
    # Check case-insensitive variations for common columns
    col_mapping = {}
    for c in df.columns:
        col_mapping[c.lower()] = c

    agg_key = aggregator.lower()
    if agg_key not in col_mapping:
        raise KeyError(f"Specified uncertainty aggregator '{aggregator}' not found in columns: {list(df.columns)}")
    actual_agg_col = col_mapping[agg_key]

    metric_key = metric_col.lower()
    if metric_key not in col_mapping:
        raise KeyError(f"Specified metric column '{metric_col}' not found in columns: {list(df.columns)}")
    actual_metric_col = col_mapping[metric_key]

    n_total = len(df)
    scores = df[actual_agg_col].to_numpy(dtype=float)
    metrics = df[actual_metric_col].to_numpy(dtype=float)
    risks = 1.0 - metrics

    # 1. Apply frozen selective rejection
    is_accepted = scores <= threshold
    if two_stage and "u_mean" in col_mapping:
        u_mean_scores = df[col_mapping["u_mean"]].to_numpy(dtype=float)
        # Empty prediction detection: NaN in primary score or zero foreground area
        empty_pred = np.isnan(scores)
        if "pred_area" in col_mapping:
            empty_pred = empty_pred | (df[col_mapping["pred_area"]].to_numpy(dtype=float) == 0)
        # Fallback gate: U_mean <= 1e-4
        is_accepted = is_accepted | (empty_pred & (u_mean_scores <= 1e-4))

    n_accepted = int(np.sum(is_accepted))
    n_rejected = n_total - n_accepted
    realized_coverage = float(n_accepted / n_total) if n_total > 0 else 0.0

    baseline_metric = float(np.mean(metrics))
    accepted_metric = float(np.mean(metrics[is_accepted])) if n_accepted > 0 else np.nan
    rejected_metric = float(np.mean(metrics[~is_accepted])) if n_rejected > 0 else np.nan
    selective_gain = accepted_metric - baseline_metric if not np.isnan(accepted_metric) else np.nan

    # Paired bootstrap CI for selective gain
    gain_ci_low, gain_ci_high = np.nan, np.nan
    if n_accepted > 0:
        _, gain_ci_low, gain_ci_high = compute_paired_bootstrap_ci(
            metrics[is_accepted],
            values_b=None,
            num_bootstrap=1000,
        )

    # 2. Compute Risk-Coverage Curve and AURC
    covs, sel_risks = compute_risk_coverage_curve(scores, risks)
    aurc_primary = compute_aurc(covs, sel_risks)
    aurc_random = compute_random_aurc(risks, coverages=covs)
    aurc_oracle = compute_oracle_aurc(risks)

    # Spearman rank correlation
    rho_val, p_val = compute_spearman_correlation(scores, metrics)

    # 3. Save Per-Image Evaluation CSV
    df_per_image = df.copy()
    df_per_image["selective_accepted"] = is_accepted
    df_per_image["selective_action"] = np.where(is_accepted, "Retain", "Refer")
    per_image_path = os.path.join(output_dir, "per_image_evaluation.csv")
    df_per_image.to_csv(per_image_path, index=False)

    # 4. Save Risk-Coverage Coordinates
    df_rc = pd.DataFrame({"coverage": covs, "selective_risk": sel_risks})
    rc_path = os.path.join(output_dir, "risk_coverage_curve.csv")
    df_rc.to_csv(rc_path, index=False)

    # 5. Aggregator Comparison Table (if requested)
    df_comp = None
    if compare_aggregators:
        comp_records = []
        possible_aggs = ["u_fg", "u_std_fg", "u_boundary", "u_ratio", "u_mean", "u_positive", "u_threshold"]
        for agg_name in possible_aggs:
            if agg_name in col_mapping:
                cand_col = col_mapping[agg_name]
                c_scores = df[cand_col].to_numpy(dtype=float)
                c_rho, c_p = compute_spearman_correlation(c_scores, metrics)
                c_covs, c_risks = compute_risk_coverage_curve(c_scores, risks)
                c_aurc = compute_aurc(c_covs, c_risks)
                comp_records.append({
                    "Aggregator": agg_name,
                    "Spearman_rho": c_rho,
                    "AURC": c_aurc,
                    "Delta_Random": c_aurc - aurc_random,
                })
        if comp_records:
            df_comp = pd.DataFrame(comp_records).sort_values("AURC")
            comp_path = os.path.join(output_dir, "aggregator_comparison.csv")
            df_comp.to_csv(comp_path, index=False)

    # 6. Save Master Summary Record
    summary = {
        "Dataset": dataset_name,
        "Total_Cases": n_total,
        "Accepted_Cases": n_accepted,
        "Rejected_Cases": n_rejected,
        "Realized_Coverage": realized_coverage,
        "Threshold": threshold,
        "Baseline_DSC": baseline_metric,
        "Accepted_DSC": accepted_metric,
        "Rejected_DSC": rejected_metric,
        "Selective_Gain": selective_gain,
        "Spearman_rho": rho_val,
        "AURC_Primary": aurc_primary,
        "AURC_Random": aurc_random,
        "AURC_Oracle": aurc_oracle,
    }
    df_summary = pd.DataFrame([summary])
    summary_path = os.path.join(output_dir, "selective_summary.csv")
    df_summary.to_csv(summary_path, index=False)

    # Print Clean Console Summary
    print("\n========================================================")
    print(" SELECTIVE PREDICTION EVALUATION SUMMARY")
    print("========================================================")
    print(f" Dataset Name:         {dataset_name}")
    print(f" Total Cohort Size:    N = {n_total}")
    print(f" Decision Threshold:   tau = {threshold:.6f} ({actual_agg_col})")
    print(f" Realized Coverage:    {realized_coverage * 100.0:.1f}% ({n_accepted} retained, {n_rejected} referred)")
    print("--------------------------------------------------------")
    print(f" Baseline Accuracy:    {baseline_metric:.4f} {metric_col}")
    print(f" Accepted Accuracy:    {accepted_metric:.4f} {metric_col}")
    print(f" Rejected Accuracy:    {rejected_metric:.4f} {metric_col}")
    print(f" Selective Gain:       {selective_gain:+.4f} {metric_col}")
    print(f" Spearman Correlation: rho = {rho_val:.4f} (p = {p_val:.4e})")
    print("--------------------------------------------------------")
    print(f" Primary AURC:         {aurc_primary:.4f}")
    print(f" Random Baseline AURC: {aurc_random:.4f}")
    print(f" Empirical Oracle AURC:{aurc_oracle:.4f}")
    print("========================================================")
    print(f" Results successfully exported to: {output_dir}\n")

    return summary


def main():
    args = parse_arguments()

    if args.demo:
        print("[INFO] Running evaluation in Demonstration Mode with synthetic data...")
        df = generate_demo_dataset()
        dataset_name = "Synthetic Demonstration Cohort"
    elif args.input_csv:
        if not os.path.exists(args.input_csv):
            print(f"[ERROR] Input CSV file not found: {args.input_csv}", file=sys.stderr)
            sys.exit(1)
        df = pd.read_csv(args.input_csv)
        dataset_name = args.dataset_name if args.dataset_name != "Cohort" else os.path.splitext(os.path.basename(args.input_csv))[0]
    else:
        print("[ERROR] Please provide either --input_csv <path> or --demo flag.", file=sys.stderr)
        sys.exit(1)

    run_evaluation(
        df=df,
        threshold=args.threshold,
        aggregator=args.aggregator,
        metric_col=args.metric_col,
        dataset_name=dataset_name,
        compare_aggregators=args.compare_aggregators,
        output_dir=args.output_dir,
        two_stage=args.two_stage,
    )


if __name__ == "__main__":
    main()
