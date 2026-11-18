"""Stage 4: Calibrated Selective Prediction (Triage) Package."""
from .decision_policy import (
    DecisionOutcome,
    evaluate_decision_policy,
    FROZEN_TAU_STAR,
    FROZEN_TAU_NEG,
)
from .uncertainty_aggregators import (
    compute_foreground_uncertainty,
    compute_mean_uncertainty,
    compute_boundary_uncertainty,
    compute_all_uncertainty_metrics,
    extract_all_aggregators,
)

__all__ = [
    "DecisionOutcome",
    "evaluate_decision_policy",
    "FROZEN_TAU_STAR",
    "FROZEN_TAU_NEG",
    "compute_foreground_uncertainty",
    "compute_mean_uncertainty",
    "compute_boundary_uncertainty",
    "compute_all_uncertainty_metrics",
    "extract_all_aggregators",
]
