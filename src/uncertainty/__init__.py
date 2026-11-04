"""Uncertainty quantification and selective prediction package."""

from .raw_entropy import compute_pixel_entropy, save_uncertainty_artifacts
from .aggregation import compute_uncertainty_features, extract_image_features
from .metrics import compute_spearman_correlations, compute_failure_auroc
from .selective_prediction import (
    calibrate_rejection_threshold,
    apply_selective_rejection,
    compute_risk_coverage_curve,
    compute_aurc,
    evaluate_rejection_controls,
)

__all__ = [
    "compute_pixel_entropy",
    "save_uncertainty_artifacts",
    "compute_uncertainty_features",
    "extract_image_features",
    "compute_spearman_correlations",
    "compute_failure_auroc",
    "calibrate_rejection_threshold",
    "apply_selective_rejection",
    "compute_risk_coverage_curve",
    "compute_aurc",
    "evaluate_rejection_controls",
]
