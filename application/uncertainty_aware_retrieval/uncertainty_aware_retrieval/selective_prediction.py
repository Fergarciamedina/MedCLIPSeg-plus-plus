"""Stage 3: Selective Prediction Triage Service.

Implements the frozen Level 0 selective prediction policy:
- Primary threshold tau = 0.064058324 on foreground predictive entropy (U_fg).
- Conservative two-stage fallback tau_neg = 1e-4 on global mean entropy (U_mean) for empty predicted masks.

Strictly distinguishes between:
1. 'Accepted' (Accepted by uncertainty policy: low predictive uncertainty)
2. 'Rejected' (Rejected by uncertainty policy: elevated predictive uncertainty, manual review required)
3. 'Not Evaluated' (Unseen image or missing valid uncertainty calibration)
"""

from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Tuple, Union
import numpy as np

from .config import (
    EMPTY_MASK_FALLBACK_THRESHOLD,
    FROZEN_THRESHOLD,
)


@dataclass
class TriageResult:
    image_id: str
    dataset: str
    status: str  # 'Accepted', 'Rejected', or 'Not Evaluated'
    u_fg: Optional[float]
    u_mean: Optional[float]
    threshold: float
    reason: str
    clinical_note: str


class SelectivePredictionService:
    """Classifies query segmentations into Accepted, Rejected, or Not Evaluated."""

    def __init__(
        self,
        threshold: float = FROZEN_THRESHOLD,
        empty_fallback_threshold: float = EMPTY_MASK_FALLBACK_THRESHOLD,
    ):
        self.threshold = threshold
        self.empty_fallback_threshold = empty_fallback_threshold

    def evaluate_case(
        self,
        u_fg: Optional[float],
        u_mean: Optional[float] = 0.0,
        count_fg: Optional[int] = None,
        image_id: str = "case",
        dataset: str = "cohort",
        is_unseen: bool = False,
    ) -> TriageResult:
        """Evaluate a query case according to the frozen Level 0 policy."""
        # 1. Check for unseen or uncalibrated inputs
        if is_unseen:
            return TriageResult(
                image_id=image_id,
                dataset=dataset,
                status="Not Evaluated",
                u_fg=None,
                u_mean=u_mean,
                threshold=self.threshold,
                reason="Uncalibrated scan. Dense predictive entropy has not been evaluated.",
                clinical_note=(
                    "External scan without calibrated Level 0 predictive entropy maps. "
                    "Cannot apply selective triage policy without validated uncertainty quantification."
                ),
            )

        # 2. Empty predicted mask handling (Setting B negative screening policy)
        if count_fg is not None and count_fg == 0:
            if u_mean is not None and u_mean <= self.empty_fallback_threshold:
                return TriageResult(
                    image_id=image_id,
                    dataset=dataset,
                    status="Accepted",
                    u_fg=None,
                    u_mean=u_mean,
                    threshold=self.empty_fallback_threshold,
                    reason=f"Empty mask with low background entropy (U_mean = {u_mean:.6f} <= {self.empty_fallback_threshold:.1e}).",
                    clinical_note=(
                        "Accepted under negative screening policy. "
                        "Low background entropy indicates high model confidence in absence of focal lesion."
                    ),
                )
            else:
                u_m_val = u_mean if u_mean is not None else 0.0
                return TriageResult(
                    image_id=image_id,
                    dataset=dataset,
                    status="Rejected",
                    u_fg=None,
                    u_mean=u_m_val,
                    threshold=self.empty_fallback_threshold,
                    reason=f"Empty mask with suspicious background entropy (U_mean = {u_m_val:.6f} > {self.empty_fallback_threshold:.1e}).",
                    clinical_note=(
                        "Referred for sonographer review. "
                        "Elevated background uncertainty suggests possible missed lesion or domain artifact."
                    ),
                )

        # 3. Lesion cohort (count_fg > 0)
        if u_fg is None:
            return TriageResult(
                image_id=image_id,
                dataset=dataset,
                status="Not Evaluated",
                u_fg=None,
                u_mean=u_mean,
                threshold=self.threshold,
                reason="Missing foreground predictive entropy score (U_fg).",
                clinical_note="Cannot evaluate lesion uncertainty without U_fg score.",
            )

        if u_fg <= self.threshold:
            return TriageResult(
                image_id=image_id,
                dataset=dataset,
                status="Accepted",
                u_fg=u_fg,
                u_mean=u_mean,
                threshold=self.threshold,
                reason=f"Foreground predictive entropy U_fg = {u_fg:.6f} <= {self.threshold:.6f}.",
                clinical_note=(
                    "Low boundary uncertainty in predicted lesion area. "
                    "Case satisfies selective quality policy and proceeds to reference retrieval."
                ),
            )
        else:
            return TriageResult(
                image_id=image_id,
                dataset=dataset,
                status="Rejected",
                u_fg=u_fg,
                u_mean=u_mean,
                threshold=self.threshold,
                reason=f"Elevated foreground predictive entropy U_fg = {u_fg:.6f} > {self.threshold:.6f}.",
                clinical_note=(
                    "Elevated boundary uncertainty detected in lesion area. "
                    "Referred for clinical sonographer review. Automated retrieval suppressed to avoid clinical bias."
                ),
            )
