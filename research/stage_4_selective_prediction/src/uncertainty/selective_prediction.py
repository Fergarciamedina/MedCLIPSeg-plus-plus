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
    handle_ties: bool = True,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Computes empirical Risk-Coverage curve.
    Images are sorted by uncertainty score ascending (least uncertain first).
    Coverage increases as more uncertain images are accepted.
    Risk is defined as empirical error: 1 - mean(metric_col).

    NaN values in score_col or metric_col are automatically dropped.
    When handle_ties is True, unique threshold cutoffs are evaluated so that
    tied score permutations produce identical deterministic risk-coverage steps.

    Args:
        df: DataFrame.
        score_col: Column to rank by.
        metric_col: Metric to measure risk on (e.g. 'DSC').
        reverse_sort: If True, higher score is accepted first (e.g. for confidence).
        handle_ties: If True, evaluates at distinct threshold boundaries to resolve ties.

    Returns:
        coverages: Array of coverage fractions in (0, 1].
        risks: Array of empirical risks (1 - mean metric) at each coverage.
    """
    valid = (~df[score_col].isna()) & (~df[metric_col].isna())
    cols = list(dict.fromkeys([score_col, metric_col]))
    clean_df = df.loc[valid, cols].copy()
    n = len(clean_df)
    if n == 0:
        return np.array([]), np.array([])

    sorted_df = clean_df.sort_values(by=score_col, ascending=(not reverse_sort)).reset_index(drop=True)

    if not handle_ties:
        metric_vals = sorted_df[metric_col].values
        cum_means = np.cumsum(metric_vals) / np.arange(1, n + 1)
        coverages = np.arange(1, n + 1) / float(n)
        risks = 1.0 - cum_means
        return coverages, risks

    # Handle ties: evaluate risk at each unique score threshold
    scores = sorted_df[score_col].values
    metrics = sorted_df[metric_col].values

    # Find indices where the score changes, plus the final index
    diff_mask = np.diff(scores) != 0
    cutoff_indices = np.where(diff_mask)[0]
    cutoff_indices = np.append(cutoff_indices, n - 1)

    cum_sum = np.cumsum(metrics)
    k_counts = cutoff_indices + 1
    coverages = k_counts / float(n)
    risks = 1.0 - (cum_sum[cutoff_indices] / k_counts)

    return coverages, risks


def compute_aurc(coverages: np.ndarray, risks: np.ndarray) -> float:
    """
    Computes Area Under the Risk-Coverage Curve (AURC) via trapezoidal integration.
    Lower AURC indicates superior selective prediction performance.
    Handles empty arrays, single points, and degenerate curves gracefully.
    """
    if len(coverages) == 0 or len(risks) == 0:
        return np.nan
    if len(coverages) == 1:
        return float(risks[0])
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


def compute_paired_bootstrap_aurc_ci(
    df: pd.DataFrame,
    col_a: str,
    col_b: str,
    metric_col: str = "DSC",
    n_bootstrap: int = 1000,
    ci: float = 0.95,
    seed: int = 666,
) -> Tuple[float, float, float, float]:
    """
    Computes paired bootstrap confidence interval for the difference in AURC:
        Delta_AURC = AURC(col_a) - AURC(col_b)

    A negative Delta indicates that col_a achieves lower risk (better performance) than col_b.

    Args:
        df: DataFrame containing evaluation samples.
        col_a: First scoring column name.
        col_b: Second scoring column name.
        metric_col: Segmentation metric (e.g., 'DSC' or 'NSD').
        n_bootstrap: Number of paired bootstrap resamples.
        ci: Confidence level (default 0.95).
        seed: Random seed for reproducibility.

    Returns:
        (delta_point, ci_lower, ci_upper, p_value_two_tailed)
    """
    valid = (~df[col_a].isna()) & (~df[col_b].isna()) & (~df[metric_col].isna())
    clean_df = df.loc[valid, [col_a, col_b, metric_col]].copy().reset_index(drop=True)
    n = len(clean_df)

    if n < 5:
        return np.nan, np.nan, np.nan, np.nan

    # Point estimates
    cov_a, risk_a = compute_risk_coverage_curve(clean_df, col_a, metric_col, handle_ties=True)
    cov_b, risk_b = compute_risk_coverage_curve(clean_df, col_b, metric_col, handle_ties=True)
    aurc_a = compute_aurc(cov_a, risk_a)
    aurc_b = compute_aurc(cov_b, risk_b)
    delta_point = float(aurc_a - aurc_b)

    rng = np.random.default_rng(seed)
    deltas = []
    alpha = 1.0 - ci

    for _ in range(n_bootstrap):
        boot_idx = rng.integers(0, n, size=n)
        sample = clean_df.iloc[boot_idx].reset_index(drop=True)
        c_a, r_a = compute_risk_coverage_curve(sample, col_a, metric_col, handle_ties=True)
        c_b, r_b = compute_risk_coverage_curve(sample, col_b, metric_col, handle_ties=True)
        a_a = compute_aurc(c_a, r_a)
        a_b = compute_aurc(c_b, r_b)
        if not np.isnan(a_a) and not np.isnan(a_b):
            deltas.append(float(a_a - a_b))

    if len(deltas) < int(n_bootstrap * 0.5):
        return delta_point, np.nan, np.nan, np.nan

    lower_pct = 100.0 * (alpha / 2.0)
    upper_pct = 100.0 * (1.0 - alpha / 2.0)
    ci_lower = float(np.percentile(deltas, lower_pct))
    ci_upper = float(np.percentile(deltas, upper_pct))

    # Empirical two-tailed p-value testing null hypothesis Delta = 0
    deltas_arr = np.array(deltas)
    p_val = float(2.0 * min(np.mean(deltas_arr >= 0), np.mean(deltas_arr <= 0)))
    p_val = min(1.0, max(0.0, p_val))

    return delta_point, ci_lower, ci_upper, p_val


def apply_empty_mask_policy(
    df: pd.DataFrame,
    primary_score_col: str = "U_fg",
    fallback_col: str = "U_mean",
    policy: str = "global_fallback",
    two_stage_tau_neg: float = 0.001,
) -> pd.DataFrame:
    """
    Applies an inference-time deployment policy to resolve undefined uncertainty scores
    when the predicted binary mask is empty (|Omega_fg| == 0).

    Uses STRICTLY model outputs available at test time (mask emptiness and fallback uncertainty);
    NEVER inspects ground-truth labels or file category strings.

    Supported policies:
      - 'drop_empty': Retains only cases with |Omega_fg| > 0 (Setting A: lesion cohort).
      - 'global_fallback': Uses primary_score_col when |Omega_fg| > 0; falls back to
                           fallback_col when |Omega_fg| == 0.
      - 'conservative_abstention': When |Omega_fg| == 0, assigns +inf uncertainty so that
                                  empty predictions are always rejected before lesion cases.
      - 'two_stage': When |Omega_fg| == 0, if fallback_col <= two_stage_tau_neg, accepts scan
                     with score=fallback_col; if fallback_col > two_stage_tau_neg (suspicious
                     empty prediction), assigns +inf to trigger expert abstention.

    Returns:
        DataFrame with an added column 'deployed_score' and 'policy_action'.
    """
    res_df = df.copy()

    if "valid_fg" in res_df.columns:
        has_fg = res_df["valid_fg"].astype(bool)
    elif "count_fg" in res_df.columns:
        has_fg = (res_df["count_fg"] > 0)
    else:
        has_fg = (~res_df[primary_score_col].isna())

    deployed_scores = []
    actions = []

    for idx, row in res_df.iterrows():
        is_fg = has_fg.iloc[idx]
        if is_fg:
            deployed_scores.append(float(row[primary_score_col]))
            actions.append("primary_lesion_score")
        else:
            if policy == "drop_empty":
                deployed_scores.append(np.nan)
                actions.append("dropped")
            elif policy == "global_fallback":
                fallback_val = float(row[fallback_col]) if not np.isnan(row[fallback_col]) else 0.0
                deployed_scores.append(fallback_val)
                actions.append("fallback_global_uncertainty")
            elif policy == "conservative_abstention":
                # Assign highest possible uncertainty to abstain/reject immediately
                deployed_scores.append(1e9)
                actions.append("conservative_abstain_empty")
            elif policy == "two_stage":
                fb = float(row[fallback_col]) if not np.isnan(row[fallback_col]) else 0.0
                if fb <= two_stage_tau_neg:
                    deployed_scores.append(fb)
                    actions.append("two_stage_accept_negative")
                else:
                    deployed_scores.append(1e9)
                    actions.append("two_stage_abstain_suspicious_negative")
            else:
                raise ValueError(f"Unknown empty mask policy: {policy}")

    res_df["deployed_score"] = deployed_scores
    res_df["policy_action"] = actions

    if policy == "drop_empty":
        res_df = res_df[res_df["deployed_score"].notna()].reset_index(drop=True)

    return res_df
