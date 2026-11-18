"""Stage 6: System Output Composer (MOD-SYSTEM-OUTPUT / Stage 6).

Assembles multi-modal clinical system outputs according to execution flags:
1. s=1, r=0 (Segment only):
   Outputs {M, U} -> Original image, Segmentation Mask M, Predictive Uncertainty U.
2. s=0, r=1 (Retrieve only):
   Outputs {X_i, T_i, d_i}_{i=1}^K -> Query image, Top-K retrieved gallery cases with similarities.
3. s=1, r=1 (Both):
   Outputs {M, U, {X_i, T_i, d_i}_{i=1}^K} -> Unified clinical dashboard with M, U, and Top-K cases.
"""

from typing import Tuple, List, Optional, Any, Union
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
from scipy.ndimage import binary_dilation

def compose_system_output(
    s: int,
    r: int,
    raw_image: Union[np.ndarray, Image.Image],
    binary_mask: Optional[np.ndarray] = None,
    entropy_map: Optional[np.ndarray] = None,
    decision_outcome: Optional[Any] = None,
    retrieved_neighbors: Optional[List[Any]] = None,
    save_path: Optional[str] = None,
) -> plt.Figure:
    """Assembles adaptive clinical figures matching Stage 6 system output specifications."""
    # Ensure raw_image is numpy RGB (256, 256, 3)
    if isinstance(raw_image, Image.Image):
        img_np = np.array(raw_image.resize((256, 256)))
    else:
        img_np = raw_image.copy()
        if img_np.shape[:2] != (256, 256):
            img_np = np.array(Image.fromarray(img_np).resize((256, 256)))

    if len(img_np.shape) == 2:
        img_np = np.stack([img_np] * 3, axis=-1)

    # Mode 1: Segment Only (s=1, r=0) -> {M, U}
    if s == 1 and r == 0:
        fig, axes = plt.subplots(1, 4, figsize=(16, 4), gridspec_kw={'width_ratios': [1, 1, 1, 1.2]})
        
        # Panel 1: Original
        axes[0].imshow(img_np, cmap='gray')
        axes[0].set_title("1. Original Ultrasound (X)", fontsize=11, fontweight='bold')
        axes[0].axis('off')

        # Panel 2: Segmentation Mask M
        overlay = img_np.copy()
        if binary_mask is not None:
            boundary = binary_dilation(binary_mask > 0) ^ (binary_mask > 0)
            overlay[binary_mask > 0] = overlay[binary_mask > 0] * 0.6 + np.array([0, 180, 0]) * 0.4
            overlay[boundary > 0] = [0, 255, 0]
            fg_area = int(np.sum(binary_mask > 0))
            axes[1].imshow(overlay)
            axes[1].set_title(f"2. Mask M ({fg_area} px)", fontsize=11, fontweight='bold')
        else:
            axes[1].imshow(overlay)
            axes[1].set_title("2. Mask M (N/A)", fontsize=11, fontweight='bold')
        axes[1].axis('off')

        # Panel 3: Uncertainty U
        if entropy_map is not None:
            im = axes[2].imshow(entropy_map, cmap='inferno')
            axes[2].set_title("3. Uncertainty U", fontsize=11, fontweight='bold')
            plt.colorbar(im, ax=axes[2], fraction=0.046, pad=0.04)
        axes[2].axis('off')

        # Panel 4: Telemetry Card
        axes[3].axis('off')
        _render_triage_card(axes[3], decision_outcome)

    # Mode 2: Retrieve Only (s=0, r=1) -> {X_i, T_i, d_i}
    elif s == 0 and r == 1:
        n_neighbors = len(retrieved_neighbors) if retrieved_neighbors else 0
        cols = 1 + max(1, n_neighbors)
        fig, axes = plt.subplots(1, cols, figsize=(3.5 * cols, 3.8))
        if cols == 1:
            axes = [axes]

        axes[0].imshow(img_np, cmap='gray')
        axes[0].set_title("Query Image (X)", fontsize=11, fontweight='bold')
        axes[0].axis('off')

        if retrieved_neighbors:
            for idx, nb in enumerate(retrieved_neighbors):
                ax = axes[1 + idx]
                dist = getattr(nb, 'distance', 0.0)
                status = getattr(nb, 'status', 'Accepted')
                concept = getattr(nb, 'concept', 'lesion')
                # Render neighbor placeholder
                ax.imshow(img_np, cmap='gray')
                ax.set_title(f"Neighbor #{idx+1}\nDist: {dist:.3f} | {status}", fontsize=9, fontweight='bold')
                ax.axis('off')
        else:
            axes[1].text(0.5, 0.5, "No eligible neighbors retrieved", ha='center', va='center')
            axes[1].axis('off')

    # Mode 3: Both (s=1, r=1) -> {M, U, {X_i, T_i, d_i}}
    else:
        n_neighbors = len(retrieved_neighbors) if retrieved_neighbors else 0
        fig, axes = plt.subplots(1, 4, figsize=(18, 4.2), gridspec_kw={'width_ratios': [1, 1, 1, 1.4]})

        # Panel 1: Original
        axes[0].imshow(img_np, cmap='gray')
        axes[0].set_title("1. Original Ultrasound (X)", fontsize=11, fontweight='bold')
        axes[0].axis('off')

        # Panel 2: Mask M
        overlay = img_np.copy()
        if binary_mask is not None:
            boundary = binary_dilation(binary_mask > 0) ^ (binary_mask > 0)
            overlay[binary_mask > 0] = overlay[binary_mask > 0] * 0.6 + np.array([0, 180, 0]) * 0.4
            overlay[boundary > 0] = [0, 255, 0]
            fg_area = int(np.sum(binary_mask > 0))
            axes[1].imshow(overlay)
            axes[1].set_title(f"2. Mask M ({fg_area} px)", fontsize=11, fontweight='bold')
        axes[1].axis('off')

        # Panel 3: Uncertainty U
        if entropy_map is not None:
            im = axes[2].imshow(entropy_map, cmap='inferno')
            axes[2].set_title("3. Uncertainty U", fontsize=11, fontweight='bold')
            plt.colorbar(im, ax=axes[2], fraction=0.046, pad=0.04)
        axes[2].axis('off')

        # Panel 4: Combined Triage & Retrieval Summary
        axes[3].axis('off')
        _render_triage_card(axes[3], decision_outcome, retrieved_neighbors)

    plt.tight_layout()
    if save_path:
        fig.savefig(save_path, bbox_inches='tight', dpi=150)
    return fig

def _render_triage_card(ax, decision_outcome, retrieved_neighbors=None):
    if decision_outcome is None:
        ax.text(0.1, 0.5, "No triage evaluation requested", fontsize=11)
        return

    is_acc = decision_outcome.is_accepted
    bg_color = '#dcfce7' if is_acc else '#fee2e2'
    border_color = '#16a34a' if is_acc else '#dc2626'
    badge_text = "AUTO ACCEPTED (g=1)" if is_acc else "FLAGGED FOR REVIEW (g=0)"
    
    bbox = dict(boxstyle='round,pad=0.7', facecolor=bg_color, edgecolor=border_color, linewidth=2)
    
    ret_str = ""
    if retrieved_neighbors is not None:
        k_found = len(retrieved_neighbors)
        ret_str = f"\n• Retrieved Cases: {k_found} accepted cases"

    status_card = (
        f"STAGE 4/6 DECISION\n"
        f"-------------------------------\n"
        f"Status: {badge_text}\n"
        f"Gate Signal g: {decision_outcome.g}\n"
        f"Decision Metric: {decision_outcome.primary_metric:.5f}\n"
        f"Calibrated Threshold: {decision_outcome.threshold:.5f}\n"
        f"Confidence Margin: {decision_outcome.margin:+.5f}\n"
        f"Foreground Area: {decision_outcome.fg_pixels} px"
        f"{ret_str}\n\n"
        f"Rationale:\n{decision_outcome.reason}"
    )
    ax.text(0.05, 0.5, status_card, fontsize=9.5, family='monospace', va='center', bbox=bbox)

