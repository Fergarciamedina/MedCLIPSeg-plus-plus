"""Statistical metrics for evaluation of uncertainty correlation and failure discrimination."""

import numpy as np
import pandas as pd
from scipy.stats import spearmanr
from sklearn.metrics import roc_auc_score
from typing import Dict, List, Tuple, Any, Optional


def compute_spearman_bootstrap_ci(
    x: np.ndarray,
    y: np.ndarray,
    n_bootstrap: int = 1000,
    ci: float = 0.95,
    seed: int = 666,
) -> Tuple[float, float, float]:
    """
    Computes non-parametric percentile bootstrap confidence interval for Spearman rho.

    Args:
        x: Array of feature values.
        y: Array of target metric values.
        n_bootstrap: Number of bootstrap resamples (default 1000).
        ci: Confidence level (default 0.95).
        seed: Fixed random seed for reproducibility.

    Returns:
        (rho_point, ci_lower, ci_upper)
    """
    x_arr = np.asarray(x)
    y_arr = np.asarray(y)
    valid = (~np.isnan(x_arr)) & (~np.isnan(y_arr))
    xv = x_arr[valid]
    yv = y_arr[valid]
    n = len(xv)

    if n < 3:
        return np.nan, np.nan, np.nan

    rho_point, _ = spearmanr(xv, yv)
    if np.isnan(rho_point):
        return np.nan, np.nan, np.nan

    rng = np.random.default_rng(seed)
    boot_rhos = []
    alpha = 1.0 - ci

    for _ in range(n_bootstrap):
        idx = rng.integers(0, n, size=n)
        bx, by = xv[idx], yv[idx]
        # Guard against degenerate constant samples in bootstrap draw
        if np.all(bx == bx[0]) or np.all(by == by[0]):
            continue
        r, _ = spearmanr(bx, by)
        if not np.isnan(r):
            boot_rhos.append(float(r))

    if len(boot_rhos) < int(n_bootstrap * 0.5):
        return float(rho_point), np.nan, np.nan

    lower_pct = 100.0 * (alpha / 2.0)
    upper_pct = 100.0 * (1.0 - alpha / 2.0)
    ci_lower = float(np.percentile(boot_rhos, lower_pct))
    ci_upper = float(np.percentile(boot_rhos, upper_pct))
    return float(rho_point), ci_lower, ci_upper


def compute_spearman_correlations(
    df: pd.DataFrame,
    feature_cols: List[str],
    metric_cols: List[str] = ["DSC", "NSD"],
    compute_ci: bool = False,
    n_bootstrap: int = 1000,
    seed: int = 666,
) -> Tuple[pd.DataFrame, Dict[str, Any]]:
    """
    Computes Spearman rank correlation rho and p-values between uncertainty features and error metrics.
    Automatically excludes NaN entries for each feature/metric pair.
    Optionally computes 95% bootstrap confidence intervals with a fixed seed.

    Args:
        df: DataFrame containing feature and metric columns.
        feature_cols: List of uncertainty feature names.
        metric_cols: List of ground-truth segmentation metrics.
        compute_ci: Whether to calculate bootstrap confidence intervals.
        n_bootstrap: Number of bootstrap iterations if compute_ci is True.
        seed: Random seed for bootstrap iterations.

    Returns:
        corr_df: Formatted table with rho, p-values, sample counts, and optional CIs.
        summary_dict: Summary including the strongest feature/metric pair.
    """
    records = []
    max_abs_rho = 0.0
    best_pair = None

    for feat in feature_cols:
        row = {"Feature": feat}
        for metric in metric_cols:
            if metric not in df.columns:
                continue
            valid_mask = (~df[feat].isna()) & (~df[metric].isna())
            n_valid = int(valid_mask.sum())
            if n_valid >= 2:
                rho, pval = spearmanr(df.loc[valid_mask, feat], df.loc[valid_mask, metric])
                row[f"{metric}_rho"] = float(rho)
                row[f"{metric}_pval"] = float(pval)
                row[f"{metric}_n"] = n_valid

                if compute_ci:
                    _, ci_l, ci_u = compute_spearman_bootstrap_ci(
                        df.loc[valid_mask, feat].values,
                        df.loc[valid_mask, metric].values,
                        n_bootstrap=n_bootstrap,
                        ci=0.95,
                        seed=seed,
                    )
                    row[f"{metric}_ci_lower"] = ci_l
                    row[f"{metric}_ci_upper"] = ci_u

                if abs(rho) > max_abs_rho:
                    max_abs_rho = abs(rho)
                    best_pair = (feat, metric, float(rho), float(pval))
            else:
                row[f"{metric}_rho"] = np.nan
                row[f"{metric}_pval"] = np.nan
                row[f"{metric}_n"] = n_valid
                if compute_ci:
                    row[f"{metric}_ci_lower"] = np.nan
                    row[f"{metric}_ci_upper"] = np.nan
        records.append(row)

    corr_df = pd.DataFrame(records)
    summary = {
        "max_abs_rho": max_abs_rho,
        "best_pair": best_pair,
    }
    return corr_df, summary


def compute_failure_auroc(
    df: pd.DataFrame,
    feature_cols: List[str],
    failure_threshold: float = 0.5,
    metric_cols: List[str] = ["DSC", "NSD"],
) -> pd.DataFrame:
    """
    Evaluates AUROC of uncertainty features for detecting low-quality segmentations.
    Failure is defined a priori as metric < failure_threshold.
    Automatically excludes NaN entries per feature.

    Higher uncertainty is expected to indicate failure, so AUROC measures how well
    the uncertainty score ranks failures above non-failures.

    Args:
        df: DataFrame containing features and ground truth metrics.
        feature_cols: List of uncertainty feature names.
        failure_threshold: Cutoff below which a segmentation is classified as failure (default 0.5).
        metric_cols: Ground truth metrics to evaluate.

    Returns:
        auroc_df: Summary DataFrame with AUROC scores per feature and metric.
    """
    results = []
    for feat in feature_cols:
        row = {"Feature": feat}
        for m in metric_cols:
            if m not in df.columns:
                continue
            valid_mask = (~df[feat].isna()) & (~df[m].isna())
            n_eval = int(valid_mask.sum())
            if n_eval == 0:
                row[f"AUROC_{m}"] = np.nan
                row[f"n_fail_{m}"] = 0
                row[f"n_eval_{m}"] = 0
                continue

            y_fail = (df.loc[valid_mask, m] < failure_threshold).astype(int)
            n_fail = int(y_fail.sum())
            n_pass = int(len(y_fail) - n_fail)

            if n_fail == 0 or n_pass == 0:
                row[f"AUROC_{m}"] = np.nan
                row[f"n_fail_{m}"] = n_fail
                row[f"n_eval_{m}"] = n_eval
            else:
                score = roc_auc_score(y_fail, df.loc[valid_mask, feat])
                row[f"AUROC_{m}"] = float(score)
                row[f"n_fail_{m}"] = n_fail
                row[f"n_eval_{m}"] = n_eval
        results.append(row)

    return pd.DataFrame(results)
