"""Uncertainty quantification and selective prediction package."""

from .raw_entropy import compute_pixel_entropy, save_uncertainty_artifacts
from .aggregation import (
    compute_u_mean,
    compute_u_positive,
    compute_u_fg,
    compute_boundary_band_mask,
    compute_u_boundary,
    compute_u_threshold,
    compute_u_std_fg,
    compute_u_ratio,
    compute_uncertainty_features,
    extract_image_features,
)
from .metrics import (
    compute_spearman_correlations,
    compute_failure_auroc,
    compute_spearman_bootstrap_ci,
)
from .selective_prediction import (
    calibrate_rejection_threshold,
    apply_selective_rejection,
    compute_risk_coverage_curve,
    compute_aurc,
    evaluate_rejection_controls,
    compute_paired_bootstrap_aurc_ci,
    apply_empty_mask_policy,
)

__all__ = [
    "compute_pixel_entropy",
    "save_uncertainty_artifacts",
    "compute_u_mean",
    "compute_u_positive",
    "compute_u_fg",
    "compute_boundary_band_mask",
    "compute_u_boundary",
    "compute_u_threshold",
    "compute_u_std_fg",
    "compute_u_ratio",
    "compute_uncertainty_features",
    "extract_image_features",
    "compute_spearman_correlations",
    "compute_failure_auroc",
    "compute_spearman_bootstrap_ci",
    "calibrate_rejection_threshold",
    "apply_selective_rejection",
    "compute_risk_coverage_curve",
    "compute_aurc",
    "evaluate_rejection_controls",
    "compute_paired_bootstrap_aurc_ci",
    "apply_empty_mask_policy",
]
