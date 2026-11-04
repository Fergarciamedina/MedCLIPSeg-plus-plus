"""Aggregation metrics for image-level uncertainty scores."""

import cv2
import numpy as np
from typing import Dict, Any, Optional


def compute_uncertainty_features(
    unc_map: np.ndarray,
    mask: np.ndarray,
    boundary_width: int = 2,
    eps: float = 1e-6,
) -> Dict[str, float]:
    """
    Computes image-level uncertainty aggregation features from a 2D uncertainty map
    and predicted binary mask.

    Features computed:
        - U_mean: Mean uncertainty over the entire image.
        - U_fg: Mean uncertainty within predicted foreground mask.
        - U_std_fg: Standard deviation of uncertainty within foreground mask.
        - U_ratio: U_fg / (U_mean + eps).
        - U_boundary: Mean uncertainty within a narrow morphological boundary band.

    Args:
        unc_map: 2D array of pixel uncertainty (raw float or normalized).
        mask: 2D binary mask (0 or >0).
        boundary_width: Pixel width for morphological boundary dilation/erosion.
        eps: Small epsilon to prevent division by zero.

    Returns:
        Dictionary mapping feature names to float values.
    """
    if unc_map.ndim != 2:
        raise ValueError(f"Expected 2D uncertainty map, got shape {unc_map.shape}")
    if mask.ndim != 2:
        raise ValueError(f"Expected 2D mask, got shape {mask.shape}")

    u_mean = float(np.mean(unc_map))
    fg_mask = (mask > 0)
    num_fg_pixels = int(np.sum(fg_mask))

    if num_fg_pixels > 0:
        fg_pixels = unc_map[fg_mask]
        u_fg = float(np.mean(fg_pixels))
        u_std_fg = float(np.std(fg_pixels))
    else:
        # Empty mask (e.g. true normal cases or false negative cases)
        u_fg = 0.0
        u_std_fg = 0.0

    u_ratio = float(u_fg / (u_mean + eps))

    # Boundary band uncertainty via morphological dilation and erosion
    if num_fg_pixels > 0:
        kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (2 * boundary_width + 1, 2 * boundary_width + 1))
        mask_uint8 = fg_mask.astype(np.uint8)
        dilated = cv2.dilate(mask_uint8, kernel, iterations=1)
        eroded = cv2.erode(mask_uint8, kernel, iterations=1)
        boundary_mask = ((dilated - eroded) > 0)
        if np.sum(boundary_mask) > 0:
            u_boundary = float(np.mean(unc_map[boundary_mask]))
        else:
            u_boundary = u_fg
    else:
        u_boundary = 0.0

    return {
        "U_mean": u_mean,
        "U_fg": u_fg,
        "U_std_fg": u_std_fg,
        "U_ratio": u_ratio,
        "U_boundary": u_boundary,
    }


def extract_image_features(
    unc_path: str,
    mask_path: str,
    boundary_width: int = 2,
) -> Dict[str, float]:
    """
    Loads uncertainty map (.npy preferred, falls back to .png) and mask (.png),
    then extracts uncertainty features.

    Args:
        unc_path: Path to .npy array or .png image.
        mask_path: Path to binary mask image.
        boundary_width: Morphological band width in pixels.

    Returns:
        Dictionary of computed features.
    """
    if unc_path.endswith(".npy"):
        unc_map = np.load(unc_path)
    else:
        img = cv2.imread(unc_path, cv2.IMREAD_GRAYSCALE)
        if img is None:
            raise FileNotFoundError(f"Failed to load uncertainty map: {unc_path}")
        unc_map = img.astype(np.float32) / 255.0

    mask_img = cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE)
    if mask_img is None:
        raise FileNotFoundError(f"Failed to load mask image: {mask_path}")

    return compute_uncertainty_features(unc_map, mask_img, boundary_width=boundary_width)
