"""Selective prediction, threshold calibration, and risk-coverage analysis."""

import numpy as np
import pandas as pd
from typing import Dict, List, Tuple, Any, Optional


def calibrate_rejection_threshold(
    val_df: pd.DataFrame,
    feature_col: str,
    target_coverage: float,
) -> float:
    """
    Calibrates the numerical rejection threshold tau strictly on validation data
    to achieve a target retention coverage c in (0, 1].

    Images with feature <= tau are accepted (retained); images with feature > tau are rejected.

    Args:
        val_df: Validation DataFrame (e.g., BUSI validation split).
        feature_col: Name of uncertainty feature column (e.g., 'U_fg').
        target_coverage: Desired fraction of accepted cases, e.g., 0.90 for 90% coverage.

    Returns:
        tau: Calibrated threshold value.
    """
    if not 0.0 < target_coverage <= 1.0:
        raise ValueError(f"Target coverage must be in (0, 1], got {target_coverage}")

    values = val_df[feature_col].dropna().values
    if len(values) == 0:
        raise ValueError(f"Feature column '{feature_col}' contains no valid data.")

    # Quantile at target coverage (e.g. 90th percentile if coverage=0.90)
    tau = float(np.percentile(values, target_coverage * 100.0))
    return tau


def apply_selective_rejection(
    df: pd.DataFrame,
    feature_col: str,
    tau: float,
) -> Dict[str, Any]:
    """
    Applies fixed numerical threshold tau to a dataset.

    Args:
        df: DataFrame containing evaluation images.
        feature_col: Uncertainty column name.
        tau: Numerical rejection threshold.

    Returns:
        Dictionary with accepted DataFrame, rejected DataFrame, realized coverage,
        and summary metrics.
    """
    accepted_mask = (df[feature_col] <= tau)
    accepted_df = df[accepted_mask].copy()
    rejected_df = df[~accepted_mask].copy()

    total_images = len(df)
    n_accepted = len(accepted_df)
    realized_coverage = float(n_accepted / total_images) if total_images > 0 else 0.0

    result = {
        "tau": tau,
        "total_images": total_images,
        "n_accepted": n_accepted,
        "n_rejected": len(rejected_df),
        "realized_coverage": realized_coverage,
        "accepted_df": accepted_df,
        "rejected_df": rejected_df,
    }

    if "DSC" in df.columns:
        result["accepted_mean_dsc"] = float(accepted_df["DSC"].mean()) if n_accepted > 0 else 0.0
        result["rejected_mean_dsc"] = float(rejected_df["DSC"].mean()) if len(rejected_df) > 0 else 0.0
        result["full_mean_dsc"] = float(df["DSC"].mean())

    if "NSD" in df.columns:
        result["accepted_mean_nsd"] = float(accepted_df["NSD"].mean()) if n_accepted > 0 else 0.0
        result["rejected_mean_nsd"] = float(rejected_df["NSD"].mean()) if len(rejected_df) > 0 else 0.0
        result["full_mean_nsd"] = float(df["NSD"].mean())

    return result


def compute_risk_coverage_curve(
    df: pd.DataFrame,
    score_col: str,
    metric_col: str = "DSC",
    reverse_sort: bool = False,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Computes empirical Risk-Coverage curve.
    Images are sorted by uncertainty score ascending (least uncertain first).
    Coverage increases as more uncertain images are accepted.
    Risk is defined as empirical error: 1 - mean(metric_col).

    Args:
        df: DataFrame.
        score_col: Column to rank by.
        metric_col: Metric to measure risk on (e.g. 'DSC').
        reverse_sort: If True, higher score is accepted first (e.g. for confidence).

    Returns:
        coverages: Array of coverage fractions from 1/N to 1.0.
        risks: Array of empirical risks (1 - mean DSC) at each coverage.
    """
    sorted_df = df.sort_values(by=score_col, ascending=(not reverse_sort)).reset_index(drop=True)
    n = len(sorted_df)
    metric_vals = sorted_df[metric_col].values

    cum_means = np.cumsum(metric_vals) / np.arange(1, n + 1)
    coverages = np.arange(1, n + 1) / float(n)
    risks = 1.0 - cum_means

    return coverages, risks


def compute_aurc(coverages: np.ndarray, risks: np.ndarray) -> float:
    """
    Computes Area Under the Risk-Coverage Curve (AURC) via trapezoidal integration.
    Lower AURC indicates superior selective prediction performance.
    """
    return float(np.trapz(risks, coverages))


def evaluate_rejection_controls(
    df: pd.DataFrame,
    feature_col: str,
    target_coverages: List[float] = [1.0, 0.90, 0.80, 0.70],
    metric_col: str = "DSC",
    num_bootstrap: int = 500,
    seed: int = 42,
) -> pd.DataFrame:
    """
    Compares uncertainty-based selective rejection against three control strategies:
    1. Uncertainty Rejection (descriptive ranking by feature_col)
    2. Random Rejection (bootstrap average over random subsets)
    3. Oracle Rejection (retaining highest ground-truth DSC first)

    Returns:
        DataFrame with rows per coverage level and columns per control strategy.
    """
    rng = np.random.default_rng(seed)
    records = []
    n_total = len(df)

    # Sorts
    df_unc_sorted = df.sort_values(by=feature_col, ascending=True).reset_index(drop=True)
    df_oracle_sorted = df.sort_values(by=metric_col, ascending=False).reset_index(drop=True)

    for cov in target_coverages:
        k = max(1, int(round(cov * n_total)))
        actual_cov = float(k / n_total)

        # 1. Uncertainty rejection
        unc_dsc = float(df_unc_sorted.iloc[:k][metric_col].mean())

        # 2. Oracle rejection (optimal ceiling)
        oracle_dsc = float(df_oracle_sorted.iloc[:k][metric_col].mean())

        # 3. Random rejection (Monte Carlo average)
        rand_draws = [float(df[metric_col].sample(n=k, replace=False, random_state=rng.integers(1e9)).mean())
                      for _ in range(num_bootstrap)]
        rand_dsc = float(np.mean(rand_draws))
        rand_ci_lower = float(np.percentile(rand_draws, 2.5))
        rand_ci_upper = float(np.percentile(rand_draws, 97.5))

        records.append({
            "Target_Coverage": cov,
            "Realized_Coverage": actual_cov,
            "Kept_Count": k,
            "Uncertainty_DSC": unc_dsc,
            "Random_DSC": rand_dsc,
            "Random_CI_Lower": rand_ci_lower,
            "Random_CI_Upper": rand_ci_upper,
            "Oracle_DSC": oracle_dsc,
            "Gain_Over_Random": unc_dsc - rand_dsc,
        })

    return pd.DataFrame(records)
