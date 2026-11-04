"""Raw pixel-level entropy computation and serialization."""

import os
import cv2
import numpy as np
import matplotlib.pyplot as plt
from typing import Tuple


def compute_pixel_entropy(logits_samples: np.ndarray, eps: float = 1e-8) -> Tuple[np.ndarray, np.ndarray]:
    """
    Computes predictive mean probability and binary pixel entropy from Monte Carlo logits samples.

    Args:
        logits_samples: Array of shape (S, H, W) or (S, C, H, W) of sigmoid probabilities.
        eps: Small epsilon for numerical stability.

    Returns:
        mean_prob: Predictive mean probability array.
        entropy_map: Raw unnormalized binary entropy array with values in [0, ln(2)].
    """
    mean_prob = np.mean(logits_samples, axis=0)
    entropy_map = -(
        mean_prob * np.log(mean_prob + eps) + (1.0 - mean_prob) * np.log(1.0 - mean_prob + eps)
    )
    return mean_prob.astype(np.float32), entropy_map.astype(np.float32)


def save_uncertainty_artifacts(
    u_map_raw: np.ndarray,
    save_dir: str,
    base_name: str,
    save_npy: bool = True,
    save_png: bool = True,
    colormap_name: str = "nipy_spectral",
) -> Tuple[str, str]:
    """
    Saves raw float32 uncertainty array (.npy) strictly before normalization,
    and optionally saves a normalized colormapped PNG for visualization.

    Args:
        u_map_raw: Raw float32 2D array of entropy values.
        save_dir: Destination directory.
        base_name: Filename base without extension.
        save_npy: Whether to persist uncompressed .npy array.
        save_png: Whether to persist colormapped visualization PNG.
        colormap_name: Matplotlib colormap name.

    Returns:
        Tuple of (npy_path, png_path).
    """
    os.makedirs(save_dir, exist_ok=True)
    clean_base = os.path.splitext(base_name)[0]

    npy_path = ""
    png_path = ""

    if save_npy:
        npy_path = os.path.join(save_dir, f"{clean_base}.npy")
        np.save(npy_path, u_map_raw.astype(np.float32))

    if save_png:
        png_path = os.path.join(save_dir, f"{clean_base}.png")
        # Per-image min-max normalization strictly for visualization
        u_min, u_max = float(np.min(u_map_raw)), float(np.max(u_map_raw))
        if u_max - u_min > 1e-8:
            u_norm = (u_map_raw - u_min) / (u_max - u_min)
        else:
            u_norm = np.zeros_like(u_map_raw)

        cmap = plt.get_cmap(colormap_name)
        color_rgb = (cmap(u_norm)[:, :, :3] * 255).astype(np.uint8)
        color_bgr = cv2.cvtColor(color_rgb, cv2.COLOR_RGB2BGR)
        cv2.imwrite(png_path, color_bgr)

    return npy_path, png_path
