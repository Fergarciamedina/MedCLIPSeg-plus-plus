"""Metrics and uncertainty aggregation routines for selective prediction in medical image segmentation.

This module provides self-contained, model-independent implementations of:
1. Spatial uncertainty aggregators (U_fg, U_std_fg, U_boundary, U_ratio, U_mean, U_positive, U_threshold).
2. Segmentation metrics (Dice Similarity Coefficient and Normalized Surface Distance).
3. Selective prediction evaluation (Risk-Coverage curves, AURC, Random Baseline, Empirical Oracle).
4. Statistical evaluation routines (Spearman rank correlation, paired bootstrap confidence intervals).
"""

from typing import Dict, List, Optional, Tuple, Union
import numpy as np
import scipy.ndimage as ndi
from scipy.stats import spearmanr


# =========================================================================
# 1. Spatial Uncertainty Aggregation Operators
# =========================================================================

def compute_u_mean(entropy_map: np.ndarray) -> float:
    """Compute global mean predictive entropy across the entire image grid.

    Args:
        entropy_map: 2D array of dense predictive entropy values.

    Returns:
        Scalar global mean entropy.
    """
    if entropy_map.size == 0:
        return 0.0
    return float(np.mean(entropy_map))


def compute_u_positive(entropy_map: np.ndarray, eps: float = 1e-7) -> float:
    """Compute mean entropy conditioned on strictly positive entropy pixels.

    Args:
        entropy_map: 2D array of predictive entropy.
        eps: Threshold above which a pixel is considered positive.

    Returns:
        Mean entropy of positive pixels, or 0.0 if none exist.
    """
    pos_mask = entropy_map > eps
    if not np.any(pos_mask):
        return 0.0
    return float(np.mean(entropy_map[pos_mask]))


def compute_u_fg(
    entropy_map: np.ndarray,
    pred_mask: np.ndarray,
    fallback_mean: float = 0.0,
) -> float:
    """Compute foreground-averaged predictive entropy (U_fg).

    Restricts spatial averaging strictly to predicted foreground pixels (pred_mask > 0).
    Eliminates background area dilution confounding global spatial averages.

    Args:
        entropy_map: 2D array of predictive entropy.
        pred_mask: 2D binary segmentation prediction mask.
        fallback_mean: Score returned if predicted foreground is completely empty.

    Returns:
        Mean entropy over the predicted foreground region.
    """
    fg_mask = pred_mask > 0
    if not np.any(fg_mask):
        return float(fallback_mean)
    return float(np.mean(entropy_map[fg_mask]))


def compute_u_std_fg(
    entropy_map: np.ndarray,
    pred_mask: np.ndarray,
    fallback_std: float = 0.0,
) -> float:
    """Compute foreground standard deviation of predictive entropy (U_std_fg).

    Measures internal heterogeneity of predictive uncertainty within the predicted lesion.

    Args:
        entropy_map: 2D array of predictive entropy.
        pred_mask: 2D binary segmentation prediction mask.
        fallback_std: Score returned if predicted foreground has < 2 pixels.

    Returns:
        Standard deviation of entropy across the foreground region.
    """
    fg_mask = pred_mask > 0
    if np.sum(fg_mask) <= 1:
        return float(fallback_std)
    return float(np.std(entropy_map[fg_mask]))


def compute_boundary_band_mask(
    pred_mask: np.ndarray,
    dilation_radius: int = 5,
    erosion_radius: int = 5,
) -> np.ndarray:
    """Construct an annular boundary band mask around the predicted contour.

    Args:
        pred_mask: 2D binary segmentation mask.
        dilation_radius: Morphological dilation radius in pixels.
        erosion_radius: Morphological erosion radius in pixels.

    Returns:
        2D boolean array representing the boundary band.
    """
    fg = pred_mask > 0
    if not np.any(fg):
        return np.zeros_like(fg, dtype=bool)

    # Structuring elements
    struct_dil = ndi.generate_binary_structure(2, 1)
    struct_ero = ndi.generate_binary_structure(2, 1)

    dilated = ndi.binary_dilation(fg, structure=struct_dil, iterations=dilation_radius)
    eroded = ndi.binary_erosion(fg, structure=struct_ero, iterations=erosion_radius)

    return np.logical_xor(dilated, eroded)


def compute_u_boundary(
    entropy_map: np.ndarray,
    pred_mask: np.ndarray,
    dilation_radius: int = 5,
    erosion_radius: int = 5,
    fallback_mean: float = 0.0,
) -> float:
    """Compute predictive entropy averaged across a morphological boundary band.

    Args:
        entropy_map: 2D array of predictive entropy.
        pred_mask: 2D binary segmentation mask.
        dilation_radius: Pixel dilation radius.
        erosion_radius: Pixel erosion radius.
        fallback_mean: Score returned if boundary band is empty.

    Returns:
        Mean entropy in the boundary band.
    """
    band_mask = compute_boundary_band_mask(pred_mask, dilation_radius, erosion_radius)
    if not np.any(band_mask):
        return float(fallback_mean)
    return float(np.mean(entropy_map[band_mask]))


def compute_u_threshold(
    entropy_map: np.ndarray,
    percentile: float = 90.0,
    fallback_mean: float = 0.0,
) -> float:
    """Compute mean entropy of pixels exceeding a high-uncertainty percentile.

    Args:
        entropy_map: 2D array of predictive entropy.
        percentile: Threshold percentile (e.g., 90th percentile).
        fallback_mean: Score returned if map is empty or constant zero.

    Returns:
        Mean entropy of the top percentile uncertainty pixels.
    """
    if entropy_map.size == 0 or np.all(entropy_map == 0):
        return float(fallback_mean)
    threshold_val = float(np.percentile(entropy_map, percentile))
    tail_mask = entropy_map >= threshold_val
    if not np.any(tail_mask):
        return float(fallback_mean)
    return float(np.mean(entropy_map[tail_mask]))


def compute_u_ratio(
    entropy_map: np.ndarray,
    pred_mask: np.ndarray,
    eps: float = 1e-7,
) -> float:
    """Compute ratio of foreground mean entropy to global mean entropy.

    Args:
        entropy_map: 2D array of predictive entropy.
        pred_mask: 2D binary segmentation mask.
        eps: Small constant to avoid division by zero.

    Returns:
        Ratio U_fg / (U_mean + eps).
    """
    u_fg_val = compute_u_fg(entropy_map, pred_mask)
    u_mean_val = compute_u_mean(entropy_map)
    return float(u_fg_val / (u_mean_val + eps))


def aggregate_all_uncertainties(
    entropy_map: np.ndarray,
    pred_mask: np.ndarray,
) -> Dict[str, float]:
    """Compute all seven uncertainty aggregation operators for a single image.

    Args:
        entropy_map: 2D predictive entropy array.
        pred_mask: 2D predicted binary mask.

    Returns:
        Dictionary mapping aggregator names to scalar uncertainty values.
    """
    return {
        "u_fg": compute_u_fg(entropy_map, pred_mask),
        "u_std_fg": compute_u_std_fg(entropy_map, pred_mask),
        "u_boundary": compute_u_boundary(entropy_map, pred_mask),
        "u_ratio": compute_u_ratio(entropy_map, pred_mask),
        "u_mean": compute_u_mean(entropy_map),
        "u_positive": compute_u_positive(entropy_map),
        "u_threshold": compute_u_threshold(entropy_map),
    }


# =========================================================================
# 2. Segmentation Metrics
# =========================================================================

def compute_dice(
    pred_mask: np.ndarray,
    gt_mask: np.ndarray,
    empty_value: float = 1.0,
) -> float:
    """Compute Dice Similarity Coefficient (DSC) between binary masks.

    Handles empty masks according to clinical standard:
    - If both prediction and ground truth are empty: returns empty_value (default 1.0).
    - If one is empty and the other is non-empty: returns 0.0.

    Args:
        pred_mask: Binary predicted mask (0 or 1).
        gt_mask: Binary ground truth mask (0 or 1).
        empty_value: DSC value assigned when both masks are empty.

    Returns:
        Scalar DSC in [0, 1].
    """
    p_bool = pred_mask.astype(bool)
    g_bool = gt_mask.astype(bool)

    p_sum = int(np.sum(p_bool))
    g_sum = int(np.sum(g_bool))

    if p_sum == 0 and g_sum == 0:
        return float(empty_value)
    if p_sum == 0 or g_sum == 0:
        return 0.0

    intersection = int(np.sum(np.logical_and(p_bool, g_bool)))
    return float(2.0 * intersection / (p_sum + g_sum))


def compute_normalized_surface_distance(
    pred_mask: np.ndarray,
    gt_mask: np.ndarray,
    tolerance_pixels: float = 2.0,
    empty_value: float = 1.0,
) -> float:
    """Compute Normalized Surface Distance (NSD) with fixed pixel tolerance.

    Args:
        pred_mask: Binary predicted mask.
        gt_mask: Binary ground truth mask.
        tolerance_pixels: Distance tolerance in pixel units (default 2.0).
        empty_value: NSD assigned when both masks are empty.

    Returns:
        Scalar NSD in [0, 1].
    """
    p_bool = pred_mask.astype(bool)
    g_bool = gt_mask.astype(bool)

    p_sum = int(np.sum(p_bool))
    g_sum = int(np.sum(g_bool))

    if p_sum == 0 and g_sum == 0:
        return float(empty_value)
    if p_sum == 0 or g_sum == 0:
        return 0.0

    # Extract boundaries
    struct = ndi.generate_binary_structure(2, 1)
    p_border = np.logical_and(p_bool, np.logical_not(ndi.binary_erosion(p_bool, struct)))
    g_border = np.logical_and(g_bool, np.logical_not(ndi.binary_erosion(g_bool, struct)))

    n_p = int(np.sum(p_border))
    n_g = int(np.sum(g_border))

    if n_p == 0 and n_g == 0:
        return float(empty_value)
    if n_p == 0 or n_g == 0:
        return 0.0

    # Distance transforms from surfaces
    dt_g = ndi.distance_transform_edt(np.logical_not(g_border))
    dt_p = ndi.distance_transform_edt(np.logical_not(p_border))

    p_in_tau = np.sum(dt_g[p_border] <= tolerance_pixels)
    g_in_tau = np.sum(dt_p[g_border] <= tolerance_pixels)

    return float((p_in_tau + g_in_tau) / (n_p + n_g))


# =========================================================================
# 3. Selective Prediction and Risk-Coverage Formulation
# =========================================================================

def apply_selective_rejection(
    scores: np.ndarray,
    threshold: float,
    pred_is_empty: Optional[np.ndarray] = None,
    u_mean_scores: Optional[np.ndarray] = None,
    empty_fallback_threshold: float = 1e-4,
) -> np.ndarray:
    """Apply selective prediction decision rule: accept if score <= threshold.

    Supports optional two-stage policy for empty predicted masks.

    Args:
        scores: 1D array of primary uncertainty scores (e.g. U_fg).
        threshold: Frozen decision threshold tau.
        pred_is_empty: Boolean array indicating whether prediction is empty.
        u_mean_scores: 1D array of global mean entropy for two-stage fallback.
        empty_fallback_threshold: Threshold for empty mask acceptance.

    Returns:
        1D boolean array (True = accepted, False = rejected).
    """
    scores = np.asarray(scores)
    accepted = scores <= threshold

    if pred_is_empty is not None and u_mean_scores is not None:
        pred_is_empty = np.asarray(pred_is_empty, dtype=bool)
        u_mean_scores = np.asarray(u_mean_scores)
        # Apply two-stage rule for empty predictions
        accepted[pred_is_empty] = u_mean_scores[pred_is_empty] <= empty_fallback_threshold

    return accepted


def compute_risk_coverage_curve(
    uncertainties: np.ndarray,
    risks: np.ndarray,
    num_thresholds: int = 200,
) -> Tuple[np.ndarray, np.ndarray]:
    """Compute empirical Risk-Coverage (RC) curve by sorting uncertainty scores.

    Args:
        uncertainties: 1D array of uncertainty scores (higher = more uncertain).
        risks: 1D array of sample losses (e.g., 1.0 - DSC).
        num_thresholds: Number of evaluation points along the curve.

    Returns:
        Tuple of (coverages, selective_risks).
    """
    uncertainties = np.asarray(uncertainties)
    risks = np.asarray(risks)
    n = len(uncertainties)

    if n == 0:
        return np.array([0.0, 1.0]), np.array([0.0, 0.0])

    # Sort samples in ascending order of uncertainty (least uncertain retained first)
    order = np.argsort(uncertainties)
    sorted_risks = risks[order]

    coverages = []
    selective_risks = []

    # Evaluate at discrete retention fractions k / n
    step = max(1, n // num_thresholds)
    indices = list(range(1, n + 1, step))
    if indices[-1] != n:
        indices.append(n)

    for k in indices:
        cov = k / n
        sel_risk = float(np.mean(sorted_risks[:k]))
        coverages.append(cov)
        selective_risks.append(sel_risk)

    return np.asarray(coverages), np.asarray(selective_risks)


def compute_aurc(coverages: np.ndarray, selective_risks: np.ndarray) -> float:
    """Compute Area Under the Risk-Coverage Curve (AURC) via trapezoidal rule.

    Evaluates trapezoidal integration over empirical coverage points in (0, 1].
    Lower AURC indicates superior selective efficiency.

    Args:
        coverages: 1D array of coverage values in (0, 1].
        selective_risks: 1D array of corresponding selective risks.

    Returns:
        Scalar AURC value.
    """
    covs = np.asarray(coverages)
    risks = np.asarray(selective_risks)

    if len(covs) == 0 or len(risks) == 0:
        return np.nan
    if len(covs) == 1:
        return float(risks[0])

    return float(np.trapz(risks, covs))


def compute_random_aurc(risks: np.ndarray, coverages: Optional[np.ndarray] = None) -> float:
    """Compute AURC of an uninformative Random Rejection baseline.

    Under uniform random abstention, expected selective risk at all coverage levels
    equals the unselective full-cohort risk: R_random(c) = R(1.0) = 1 - mean(DSC).
    
    When coverages array is provided, evaluates discrete trapezoidal integration across
    the empirical support [1/N, 1.0], matching canonical benchmark outputs:
        AURC_empirical = (1 - mean(DSC)) * (N - 1) / N.
    When coverages is None, returns the analytical continuous full-domain expectation:
        AURC_analytical = 1 - mean(DSC).

    Args:
        risks: 1D array of sample losses (1 - DSC).
        coverages: Optional 1D array of empirical coverages.

    Returns:
        Scalar Random AURC value.
    """
    base_risk = float(np.mean(risks))
    if coverages is not None and len(coverages) > 1:
        covs = np.asarray(coverages)
        rand_risks = np.full_like(covs, base_risk)
        return float(np.trapz(rand_risks, covs))
    return base_risk


def compute_oracle_aurc(risks: np.ndarray) -> float:
    """Compute theoretical AURC of an Empirical Oracle baseline.

    The Oracle sorts samples in perfect ascending order of actual ground truth risk,
    achieving the theoretical minimum selective risk for any given coverage.

    Args:
        risks: 1D array of sample losses.

    Returns:
        Scalar Oracle AURC.
    """
    covs, sel_risks = compute_risk_coverage_curve(uncertainties=risks, risks=risks)
    return compute_aurc(covs, sel_risks)


# =========================================================================
# 4. Statistical Routines
# =========================================================================

def compute_spearman_correlation(
    scores: np.ndarray,
    metrics: np.ndarray,
) -> Tuple[float, float]:
    """Compute Spearman rank correlation between uncertainty scores and accuracy metric.

    More negative correlation indicates higher quality uncertainty ranking
    (elevated uncertainty predicts low segmentation accuracy).

    Args:
        scores: 1D array of uncertainty scores.
        metrics: 1D array of segmentation fidelity (e.g. DSC or NSD).

    Returns:
        Tuple of (correlation_rho, p_value).
    """
    rho, pval = spearmanr(scores, metrics)
    return float(rho), float(pval)


def compute_paired_bootstrap_ci(
    values_a: np.ndarray,
    values_b: Optional[np.ndarray] = None,
    num_bootstrap: int = 1000,
    alpha: float = 0.05,
    seed: int = 42,
) -> Tuple[float, float, float]:
    """Compute paired percentile bootstrap confidence interval and two-sided p-value.

    If values_b is provided, computes CI for paired difference (values_a - values_b).
    If values_b is None, computes CI for values_a mean.

    Args:
        values_a: 1D array of metric values.
        values_b: Optional 1D array of baseline/comparator values.
        num_bootstrap: Number of bootstrap resamples (default 1000).
        alpha: Significance level (default 0.05 for 95% CI).
        seed: Random seed for reproducibility.

    Returns:
        Tuple of (point_estimate, ci_lower, ci_upper).
    """
    rng = np.random.RandomState(seed)
    n = len(values_a)

    if values_b is not None:
        diffs = np.asarray(values_a) - np.asarray(values_b)
        point_est = float(np.mean(diffs))
        indices = rng.randint(0, n, size=(num_bootstrap, n))
        boot_means = np.mean(diffs[indices], axis=1)
    else:
        point_est = float(np.mean(values_a))
        indices = rng.randint(0, n, size=(num_bootstrap, n))
        boot_means = np.mean(np.asarray(values_a)[indices], axis=1)

    ci_lower = float(np.percentile(boot_means, 100.0 * (alpha / 2.0)))
    ci_upper = float(np.percentile(boot_means, 100.0 * (1.0 - alpha / 2.0)))

    return point_est, ci_lower, ci_upper

