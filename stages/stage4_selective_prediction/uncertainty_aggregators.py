"""Stage 4: Spatial Uncertainty Aggregators (MOD-SELECTIVE-DECISION / Stage 4).

Implements all 7 spatial uncertainty operators evaluated in the research:
1. U_fg: Foreground mean entropy (Primary proposed operator; eliminates background dilution)
2. U_std_fg: Foreground standard deviation of predictive entropy
3. U_boundary: Morphological boundary band mean entropy
4. U_ratio: Ratio of foreground mean entropy to global mean entropy
5. U_mean: Global grid-averaged entropy (area-confounded; kept for negative screening fallback)
6. U_positive: Strictly positive entropy pixels mean
7. U_threshold: Mean entropy above the 90th percentile
"""

from typing import Dict
import numpy as np
from scipy.ndimage import binary_dilation, binary_erosion

def compute_foreground_uncertainty(entropy_map: np.ndarray, binary_mask: np.ndarray) -> float:
    """Compute U_fg: mean predictive entropy on foreground mask M."""
    fg_pixels = np.sum(binary_mask > 0)
    if fg_pixels == 0:
        return float("nan")
    return float(np.mean(entropy_map[binary_mask > 0]))

def compute_mean_uncertainty(entropy_map: np.ndarray) -> float:
    """Compute U_mean: global spatial grid-averaged entropy."""
    return float(np.mean(entropy_map))

def compute_boundary_uncertainty(
    entropy_map: np.ndarray, binary_mask: np.ndarray, width: int = 3
) -> float:
    """Compute U_boundary: mean predictive entropy along lesion contour."""
    if np.sum(binary_mask > 0) == 0:
        return float("nan")
    dilated = binary_dilation(binary_mask > 0, iterations=width)
    eroded = binary_erosion(binary_mask > 0, iterations=width)
    band = dilated ^ eroded
    if not np.any(band):
        return float("nan")
    return float(np.mean(entropy_map[band]))

def compute_all_uncertainty_metrics(
    entropy_map: np.ndarray,
    binary_mask: np.ndarray,
    boundary_width: int = 3,
) -> Dict[str, float]:
    """Compute the complete suite of 7 spatial uncertainty metrics for a single prediction."""
    total_pixels = entropy_map.size
    fg_pixels = int(np.sum(binary_mask > 0))

    # 1. Global Mean Entropy (U_mean)
    u_mean = float(np.mean(entropy_map))

    # 2. Strictly Positive Entropy (U_positive)
    pos_mask = entropy_map > 0
    u_positive = float(np.mean(entropy_map[pos_mask])) if np.any(pos_mask) else 0.0

    # 3. 90th Percentile Threshold Entropy (U_threshold)
    p90 = np.percentile(entropy_map, 90)
    top10_mask = entropy_map >= p90
    u_threshold = float(np.mean(entropy_map[top10_mask])) if np.any(top10_mask) else 0.0

    if fg_pixels == 0:
        return {
            "u_fg": float("nan"),
            "u_std_fg": float("nan"),
            "u_boundary": float("nan"),
            "u_ratio": 0.0,
            "u_mean": u_mean,
            "u_positive": u_positive,
            "u_threshold": u_threshold,
            "fg_pixels": 0,
            "fg_ratio": 0.0,
        }

    # 4. Foreground Mean Entropy (U_fg)
    fg_mask = binary_mask > 0
    fg_entropy = entropy_map[fg_mask]
    u_fg = float(np.mean(fg_entropy))

    # 5. Foreground Standard Deviation (U_std_fg)
    u_std_fg = float(np.std(fg_entropy))

    # 6. Morphological Boundary Band Entropy (U_boundary)
    dilated = binary_dilation(fg_mask, iterations=boundary_width)
    eroded = binary_erosion(fg_mask, iterations=boundary_width)
    boundary_mask = dilated ^ eroded
    if np.any(boundary_mask):
        u_boundary = float(np.mean(entropy_map[boundary_mask]))
    else:
        u_boundary = u_fg

    # 7. Foreground-to-Global Ratio (U_ratio)
    u_ratio = float(u_fg / (u_mean + 1e-8))

    return {
        "u_fg": u_fg,
        "u_std_fg": u_std_fg,
        "u_boundary": u_boundary,
        "u_ratio": u_ratio,
        "u_mean": u_mean,
        "u_positive": u_positive,
        "u_threshold": u_threshold,
        "fg_pixels": fg_pixels,
        "fg_ratio": float(fg_pixels / total_pixels),
    }

def extract_all_aggregators(entropy_map: np.ndarray, binary_mask: np.ndarray) -> Dict[str, float]:
    """Alias for compute_all_uncertainty_metrics."""
    return compute_all_uncertainty_metrics(entropy_map, binary_mask)
