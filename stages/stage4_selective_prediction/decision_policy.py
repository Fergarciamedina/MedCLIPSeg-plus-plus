"""Stage 4: Calibrated Selective Prediction (Triage) (MOD-SELECTIVE-DECISION / Stage 4).

Executes the calibrated triage policy:
1. "Foreground in M?"
   - Yes: compute mu operator U_fg (mean on M).
     - If U_fg <= tau_fg (tau* = 0.064058324): Accepted (auto) -> g = 1
     - Else: Review (radiologist) -> g = 0
   - No: compute mu operator U_mean (full image mean).
     - If U_mean <= tau_neg (1e-4): Accepted (auto) -> g = 1
     - Else: Review (radiologist) -> g = 0

Produces acceptance gate signal g in {0, 1} for Stage 5 retrieval gating (g AND r).
"""

from dataclasses import dataclass
from typing import Dict, Any, Tuple
import numpy as np

# Calibrated decision thresholds
FROZEN_TAU_STAR = 0.064058324  # tau_fg calibrated on BUSI validation at 80% coverage
FROZEN_TAU_NEG = 0.0001        # tau_neg calibrated on BUSI normal scans

@dataclass
class DecisionOutcome:
    """Standardized selective triage decision outcome."""
    status: str            # 'ACCEPTED', 'ACCEPTED_NEGATIVE', or 'REVIEW_REQUIRED'
    is_accepted: bool      # True if accepted (auto); False if flagged for review
    g: int                 # Acceptance gate signal: 1 if Accepted, 0 if Review
    primary_metric: float  # Value of U_fg (or U_mean for empty mask)
    threshold: float       # tau* or tau_neg applied
    margin: float          # threshold - score (positive = confident; negative = ambiguous)
    reason: str            # Human-readable clinical justification
    fg_pixels: int = 0
    u_fg: float = 0.0
    u_mean: float = 0.0

def evaluate_decision_policy(
    binary_mask: np.ndarray,
    entropy_map: np.ndarray,
    tau_fg: float = FROZEN_TAU_STAR,
    tau_neg: float = FROZEN_TAU_NEG,
) -> DecisionOutcome:
    """Evaluates Stage 4 triage policy directly from binary mask M and uncertainty map U.

    Follows the exact flowchart:
    - Foreground in M?
      - Yes: mu = U_fg. U_fg <= tau_fg? Yes -> Accepted (g=1), No -> Review (g=0)
      - No: mu = U_mean. U_mean <= tau_neg? Yes -> Accepted (g=1), No -> Review (g=0)
    """
    fg_pixels = int(np.sum(binary_mask > 0))
    u_mean = float(np.mean(entropy_map))

    if fg_pixels > 0:
        u_fg = float(np.mean(entropy_map[binary_mask > 0]))
    else:
        u_fg = float("nan")

    # Degeneracy Handling for Empty Masks
    if fg_pixels == 0 or np.isnan(u_fg):
        if u_mean <= tau_neg:
            return DecisionOutcome(
                status="ACCEPTED_NEGATIVE",
                is_accepted=True,
                g=1,
                primary_metric=u_mean,
                threshold=tau_neg,
                margin=float(tau_neg - u_mean),
                reason=f"Quiescent acoustic field with no lesion (U_mean={u_mean:.6f} <= {tau_neg})",
                fg_pixels=0,
                u_fg=0.0,
                u_mean=u_mean,
            )
        else:
            return DecisionOutcome(
                status="REVIEW_REQUIRED",
                is_accepted=False,
                g=0,
                primary_metric=u_mean,
                threshold=tau_neg,
                margin=float(tau_neg - u_mean),
                reason=f"Suspicious empty prediction with active background field (U_mean={u_mean:.6f} > {tau_neg})",
                fg_pixels=0,
                u_fg=0.0,
                u_mean=u_mean,
            )

    # Standard Foreground Stencil
    if u_fg <= tau_fg:
        margin = float(tau_fg - u_fg)
        return DecisionOutcome(
            status="ACCEPTED",
            is_accepted=True,
            g=1,
            primary_metric=u_fg,
            threshold=tau_fg,
            margin=margin,
            reason=f"Low predictive boundary ambiguity (U_fg={u_fg:.6f} <= {tau_fg:.6f}, margin=+{margin:.5f})",
            fg_pixels=fg_pixels,
            u_fg=u_fg,
            u_mean=u_mean,
        )
    else:
        excess = float(u_fg - tau_fg)
        return DecisionOutcome(
            status="REVIEW_REQUIRED",
            is_accepted=False,
            g=0,
            primary_metric=u_fg,
            threshold=tau_fg,
            margin=-excess,
            reason=f"High predictive boundary ambiguity (U_fg={u_fg:.6f} > {tau_fg:.6f}, excess=+{excess:.5f})",
            fg_pixels=fg_pixels,
            u_fg=u_fg,
            u_mean=u_mean,
        )

