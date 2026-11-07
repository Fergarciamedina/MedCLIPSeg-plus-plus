"""
Modular image-level uncertainty aggregation metrics.

This module implements seven training-free aggregation methods to condense
dense 2D predictive uncertainty maps into scalar image-level scores:
    1. U_mean: Global spatial average over the entire image grid.
    2. U_positive: Conditional average over strictly positive uncertainty pixels (U > 0).
    3. U_fg: Average uncertainty restricted to the predicted foreground mask (M > 0).
    4. U_boundary: Average uncertainty inside a symmetric morphological boundary band.
    5. U_threshold: Average uncertainty over pixels exceeding a fixed threshold t (U > t).
    6. U_std_fg: Population standard deviation of uncertainty within the predicted foreground.
    7. U_ratio: Foreground-to-global uncertainty ratio U_fg / (U_mean + epsilon).

Mathematical formulation and degeneracy rules:
- Empty regions or undefined operations evaluate to np.nan with valid=False.
- All functions consume raw float numpy arrays (e.g., float32 predictive entropy in nats [0, ln 2]).
"""

import os
from typing import Dict, Any, Tuple, Optional
import cv2
import numpy as np


def _validate_inputs(unc_map: np.ndarray, mask: Optional[np.ndarray] = None) -> None:
    """Validate dimensionality, shapes, and finiteness of inputs."""
    if unc_map.ndim != 2:
        raise ValueError(f"Expected 2D uncertainty map, got shape {unc_map.shape}")
    if not np.all(np.isfinite(unc_map)):
        raise ValueError("Uncertainty map contains non-finite values (NaN or Inf)")
    if mask is not None:
        if mask.ndim != 2:
            raise ValueError(f"Expected 2D mask, got shape {mask.shape}")
        if unc_map.shape != mask.shape:
            raise ValueError(
                f"Shape mismatch: unc_map shape {unc_map.shape} != mask shape {mask.shape}"
            )


def compute_u_mean(unc_map: np.ndarray) -> Tuple[float, bool, int]:
    """
    Method 1: Mean uncertainty over the entire image domain Omega.

    Formula:
        U_mean = (1 / |Omega|) * sum_{i in Omega} U(i)

    Args:
        unc_map: 2D array of pixel uncertainty values.

    Returns:
        (score, valid, count): Scalar score, validity flag, and number of contributing pixels.
    """
    _validate_inputs(unc_map)
    total_pixels = int(unc_map.size)
    if total_pixels == 0:
        return np.nan, False, 0
    return float(np.mean(unc_map)), True, total_pixels


def compute_u_positive(unc_map: np.ndarray) -> Tuple[float, bool, int]:
    """
    Method 2: Conditional mean uncertainty over strictly positive pixels (U > 0).

    Formula:
        Omega_pos = {i in Omega | U(i) > 0}
        U_positive = (1 / |Omega_pos|) * sum_{i in Omega_pos} U(i)

    Degeneracy handling:
        If no pixels have U > 0, returns (np.nan, False, 0).

    Args:
        unc_map: 2D array of pixel uncertainty values.

    Returns:
        (score, valid, count): Scalar score, validity flag, and number of positive pixels.
    """
    _validate_inputs(unc_map)
    pos_mask = (unc_map > 0.0)
    count = int(np.sum(pos_mask))
    if count == 0:
        return np.nan, False, 0
    return float(np.mean(unc_map[pos_mask])), True, count


def compute_u_fg(unc_map: np.ndarray, mask: np.ndarray) -> Tuple[float, bool, int]:
    """
    Method 3: Mean uncertainty within the predicted foreground mask (M > 0).

    Formula:
        Omega_fg = {i in Omega | M(i) > 0}
        U_fg = (1 / |Omega_fg|) * sum_{i in Omega_fg} U(i)

    Degeneracy handling:
        If the predicted mask is empty (|Omega_fg| == 0), returns (np.nan, False, 0).

    Args:
        unc_map: 2D array of pixel uncertainty values.
        mask: 2D predicted binary or integer segmentation mask.

    Returns:
        (score, valid, count): Scalar score, validity flag, and foreground pixel count.
    """
    _validate_inputs(unc_map, mask)
    fg_mask = (mask > 0)
    count = int(np.sum(fg_mask))
    if count == 0:
        return np.nan, False, 0
    return float(np.mean(unc_map[fg_mask])), True, count


def compute_boundary_band_mask(
    mask: np.ndarray,
    boundary_width: int = 2,
    kernel_shape: str = "rect",
) -> np.ndarray:
    """
    Constructs a symmetric morphological boundary band around the predicted foreground.

    Definition:
        Let M_bin = (M > 0).
        Structuring element K_w has dimensions (2*w + 1, 2*w + 1) with width/radius w.
        M_dilated = M_bin (+) K_w
        M_eroded  = M_bin (-) K_w
        Omega_boundary = {i in Omega | M_dilated(i) == 1 and M_eroded(i) == 0}

    Args:
        mask: 2D predicted segmentation mask.
        boundary_width: Half-width w in pixels. Must be >= 1.
        kernel_shape: Structuring element shape ("rect" or "ellipse").

    Returns:
        2D boolean array representing the boundary band.
    """
    if mask.ndim != 2:
        raise ValueError(f"Expected 2D mask, got shape {mask.shape}")
    if boundary_width < 1:
        raise ValueError(f"boundary_width must be >= 1, got {boundary_width}")

    fg_mask = (mask > 0).astype(np.uint8)
    if np.sum(fg_mask) == 0:
        return np.zeros_like(fg_mask, dtype=bool)

    k_size = 2 * boundary_width + 1
    morph_type = cv2.MORPH_ELLIPSE if kernel_shape == "ellipse" else cv2.MORPH_RECT
    kernel = cv2.getStructuringElement(morph_type, (k_size, k_size))

    dilated = cv2.dilate(fg_mask, kernel, iterations=1)
    eroded = cv2.erode(fg_mask, kernel, iterations=1)
    boundary_band = ((dilated > 0) & (eroded == 0))
    return boundary_band


def compute_u_boundary(
    unc_map: np.ndarray,
    mask: np.ndarray,
    boundary_width: int = 2,
    kernel_shape: str = "rect",
) -> Tuple[float, bool, int]:
    """
    Method 4: Mean uncertainty within a narrow morphological boundary band.

    Formula:
        U_boundary = (1 / |Omega_boundary|) * sum_{i in Omega_boundary} U(i)

    Degeneracy handling:
        If the foreground is empty or the boundary band contains zero pixels,
        returns (np.nan, False, 0).

    Args:
        unc_map: 2D array of pixel uncertainty values.
        mask: 2D predicted segmentation mask.
        boundary_width: Half-width w of morphological dilation/erosion (default 2).
        kernel_shape: Kernel geometry ("rect" or "ellipse").

    Returns:
        (score, valid, count): Scalar score, validity flag, and boundary pixel count.
    """
    _validate_inputs(unc_map, mask)
    b_mask = compute_boundary_band_mask(mask, boundary_width=boundary_width, kernel_shape=kernel_shape)
    count = int(np.sum(b_mask))
    if count == 0:
        return np.nan, False, 0
    return float(np.mean(unc_map[b_mask])), True, count


def compute_u_threshold(
    unc_map: np.ndarray,
    threshold_t: float = 0.1,
) -> Tuple[float, bool, int]:
    """
    Method 5: Mean uncertainty over high-uncertainty pixels exceeding threshold t.

    Formula:
        Omega_t = {i in Omega | U(i) > t}
        U_threshold(t) = (1 / |Omega_t|) * sum_{i in Omega_t} U(i)

    Degeneracy handling:
        If no pixels exceed threshold t, returns (np.nan, False, 0).

    Args:
        unc_map: 2D array of pixel uncertainty values.
        threshold_t: Absolute uncertainty threshold cut-off t (default 0.1).

    Returns:
        (score, valid, count): Scalar score, validity flag, and thresholded pixel count.
    """
    _validate_inputs(unc_map)
    thresh_mask = (unc_map > threshold_t)
    count = int(np.sum(thresh_mask))
    if count == 0:
        return np.nan, False, 0
    return float(np.mean(unc_map[thresh_mask])), True, count


def compute_u_std_fg(unc_map: np.ndarray, mask: np.ndarray) -> Tuple[float, bool, int]:
    """
    Method 6: Population standard deviation of uncertainty inside predicted foreground.

    Formula:
        sigma_fg = sqrt( (1 / |Omega_fg|) * sum_{i in Omega_fg} (U(i) - U_fg)^2 )
        computed with population degrees of freedom (ddof=0).

    Degeneracy handling:
        If |Omega_fg| == 0, returns (np.nan, False, 0).
        If |Omega_fg| == 1, population standard deviation is 0.0 with valid=True.

    Args:
        unc_map: 2D array of pixel uncertainty values.
        mask: 2D predicted segmentation mask.

    Returns:
        (score, valid, count): Scalar score, validity flag, and foreground pixel count.
    """
    _validate_inputs(unc_map, mask)
    fg_mask = (mask > 0)
    count = int(np.sum(fg_mask))
    if count == 0:
        return np.nan, False, 0
    if count == 1:
        return 0.0, True, 1
    return float(np.std(unc_map[fg_mask], ddof=0)), True, count


def compute_u_ratio(
    unc_map: np.ndarray,
    mask: np.ndarray,
    eps: float = 1e-6,
) -> Tuple[float, bool, int]:
    """
    Method 7: Ratio of foreground mean uncertainty to global mean uncertainty.

    Formula:
        U_ratio = U_fg / (U_mean + epsilon)

    Degeneracy handling:
        If foreground is empty (U_fg is NaN) or global mean is undefined,
        returns (np.nan, False, 0).
        If (U_mean + epsilon) <= 0, returns (np.nan, False, count_fg).

    Args:
        unc_map: 2D array of pixel uncertainty values.
        mask: 2D predicted segmentation mask.
        eps: Small non-negative denominator regularizer (default 1e-6).

    Returns:
        (score, valid, count): Scalar score, validity flag, and foreground pixel count.
    """
    if eps < 0.0:
        raise ValueError(f"epsilon must be non-negative, got {eps}")

    u_fg, fg_valid, count_fg = compute_u_fg(unc_map, mask)
    u_mean, mean_valid, _ = compute_u_mean(unc_map)

    if not fg_valid or not mean_valid:
        return np.nan, False, count_fg

    denom = u_mean + eps
    if denom <= 0.0:
        return np.nan, False, count_fg

    return float(u_fg / denom), True, count_fg


def compute_uncertainty_features(
    unc_map: np.ndarray,
    mask: np.ndarray,
    boundary_width: int = 2,
    threshold_t: float = 0.1,
    eps: float = 1e-6,
) -> Dict[str, Any]:
    """
    Computes all seven uncertainty aggregation features alongside diagnostic counts
    and validity flags from raw 2D uncertainty maps and predicted binary masks.

    Returns:
        Dictionary containing:
            - Seven scalar scores: U_mean, U_positive, U_fg, U_boundary, U_threshold,
                                   U_std_fg, U_ratio.
            - Five region counts: count_total, count_pos, count_fg, count_boundary,
                                  count_threshold.
            - Validity booleans: valid_mean, valid_pos, valid_fg, valid_boundary,
                                 valid_threshold, valid_std_fg, valid_ratio.
            - Metadata: boundary_width, threshold_t, eps.
    """
    u_mean, v_mean, c_total = compute_u_mean(unc_map)
    u_pos, v_pos, c_pos = compute_u_positive(unc_map)
    u_fg, v_fg, c_fg = compute_u_fg(unc_map, mask)
    u_bound, v_bound, c_bound = compute_u_boundary(unc_map, mask, boundary_width=boundary_width)
    u_thresh, v_thresh, c_thresh = compute_u_threshold(unc_map, threshold_t=threshold_t)
    u_std_fg, v_std_fg, _ = compute_u_std_fg(unc_map, mask)
    u_ratio, v_ratio, _ = compute_u_ratio(unc_map, mask, eps=eps)

    return {
        "U_mean": u_mean,
        "U_positive": u_pos,
        "U_fg": u_fg,
        "U_boundary": u_bound,
        "U_threshold": u_thresh,
        "U_std_fg": u_std_fg,
        "U_ratio": u_ratio,
        "count_total": c_total,
        "count_pos": c_pos,
        "count_fg": c_fg,
        "count_boundary": c_bound,
        "count_threshold": c_thresh,
        "valid_mean": v_mean,
        "valid_pos": v_pos,
        "valid_fg": v_fg,
        "valid_boundary": v_bound,
        "valid_threshold": v_thresh,
        "valid_std_fg": v_std_fg,
        "valid_ratio": v_ratio,
        "boundary_width": boundary_width,
        "threshold_t": threshold_t,
        "eps": eps,
    }


def extract_image_features(
    unc_path: str,
    mask_path: str,
    boundary_width: int = 2,
    threshold_t: float = 0.1,
    eps: float = 1e-6,
) -> Dict[str, Any]:
    """
    Loads raw float32 uncertainty map (.npy strictly required) and predicted mask (.png),
    then extracts all seven uncertainty metrics.

    Args:
        unc_path: Path to raw .npy array (or corresponding file stem).
        mask_path: Path to binary mask image.
        boundary_width: Half-width of boundary band in pixels.
        threshold_t: Threshold cut-off t for U_threshold.
        eps: Small regularizer for U_ratio.

    Returns:
        Dictionary of computed features and metadata.
    """
    if unc_path.endswith(".npy"):
        resolved_unc = unc_path
    elif os.path.exists(os.path.splitext(unc_path)[0] + ".npy"):
        resolved_unc = os.path.splitext(unc_path)[0] + ".npy"
    else:
        raise ValueError(
            f"Uncertainty map must be a raw float32 .npy file, got '{unc_path}'. "
            "Grayscale-converted colormapped PNGs are strictly rejected to avoid normalization artifacts."
        )

    unc_map = np.load(resolved_unc)

    mask_img = cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE)
    if mask_img is None:
        raise FileNotFoundError(f"Failed to load mask image: {mask_path}")

    return compute_uncertainty_features(
        unc_map=unc_map,
        mask=mask_img,
        boundary_width=boundary_width,
        threshold_t=threshold_t,
        eps=eps,
    )
