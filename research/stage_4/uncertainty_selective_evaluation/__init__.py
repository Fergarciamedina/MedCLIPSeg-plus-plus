"""Uncertainty-Based Selective Prediction Evaluation Toolkit for Medical Image Segmentation.

Provides modular, model-agnostic tools to evaluate post-hoc selective prediction:
- 7 spatial uncertainty aggregators (U_fg, U_std_fg, U_boundary, U_ratio, U_mean, U_positive, U_threshold).
- Dice Similarity Coefficient and Normalized Surface Distance metrics.
- Risk-Coverage curve generation and Area Under the Risk-Coverage Curve (AURC).
- Random baseline and Empirical Oracle reference implementations.
- Paired bootstrap confidence intervals and statistical testing.
- Model adapters for vision-language segmentation models.
"""

from .metrics import (
    compute_u_fg,
    compute_u_std_fg,
    compute_u_boundary,
    compute_u_ratio,
    compute_u_mean,
    compute_u_positive,
    compute_u_threshold,
    aggregate_all_uncertainties,
    compute_dice,
    compute_normalized_surface_distance,
    apply_selective_rejection,
    compute_risk_coverage_curve,
    compute_aurc,
    compute_random_aurc,
    compute_oracle_aurc,
    compute_spearman_correlation,
    compute_paired_bootstrap_ci,
)
from .model_adapter import (
    BaseSegmentationModelAdapter,
    MockModelAdapter,
    MedCLIPSegModelAdapter,
)

__version__ = "1.0.0"
__all__ = [
    "compute_u_fg",
    "compute_u_std_fg",
    "compute_u_boundary",
    "compute_u_ratio",
    "compute_u_mean",
    "compute_u_positive",
    "compute_u_threshold",
    "aggregate_all_uncertainties",
    "compute_dice",
    "compute_normalized_surface_distance",
    "apply_selective_rejection",
    "compute_risk_coverage_curve",
    "compute_aurc",
    "compute_random_aurc",
    "compute_oracle_aurc",
    "compute_spearman_correlation",
    "compute_paired_bootstrap_ci",
    "BaseSegmentationModelAdapter",
    "MockModelAdapter",
    "MedCLIPSegModelAdapter",
]

