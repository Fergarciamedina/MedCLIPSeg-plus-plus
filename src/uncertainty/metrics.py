"""Statistical metrics for evaluation of uncertainty correlation and failure discrimination."""

import numpy as np
import pandas as pd
from scipy.stats import spearmanr
from sklearn.metrics import roc_auc_score
from typing import Dict, List, Tuple, Any, Optional


def compute_spearman_correlations(
    df: pd.DataFrame,
    feature_cols: List[str],
    metric_cols: List[str] = ["DSC", "NSD"],
) -> Tuple[pd.DataFrame, Dict[str, Any]]:
    """
    Computes Spearman rank correlation rho and p-values between uncertainty features and error metrics.

    Args:
        df: DataFrame containing feature and metric columns.
        feature_cols: List of uncertainty feature names.
        metric_cols: List of ground-truth segmentation metrics.

    Returns:
        corr_df: Formatted table with rho and p-values.
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
            rho, pval = spearmanr(df[feat], df[metric])
            row[f"{metric}_rho"] = float(rho)
            row[f"{metric}_pval"] = float(pval)
            if abs(rho) > max_abs_rho:
                max_abs_rho = abs(rho)
                best_pair = (feat, metric, float(rho), float(pval))
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
            y_fail = (df[m] < failure_threshold).astype(int)
            n_fail = int(y_fail.sum())
            n_pass = int(len(y_fail) - n_fail)

            if n_fail == 0 or n_pass == 0:
                row[f"AUROC_{m}"] = np.nan
                row[f"n_fail_{m}"] = n_fail
            else:
                score = roc_auc_score(y_fail, df[feat])
                row[f"AUROC_{m}"] = float(score)
                row[f"n_fail_{m}"] = n_fail
        results.append(row)

    return pd.DataFrame(results)
