"""Visualization and Clinical Overlay Utilities.

Renders high-contrast segmentation overlays, uncertainty maps, and clinical triage cards
for ultrasound images without modifying original arrays.
"""

from pathlib import Path
from typing import Optional, Tuple, Union
import numpy as np
from PIL import Image, ImageDraw, ImageOps
import matplotlib.cm as cm


def create_segmentation_overlay(
    image_input: Union[str, Path, Image.Image, np.ndarray],
    mask_input: Optional[Union[str, Path, Image.Image, np.ndarray]],
    color: Tuple[int, int, int] = (0, 255, 127),  # Spring green
    alpha: float = 0.40,
    contour_width: int = 2,
) -> Image.Image:
    """Creates a clinical-style semi-transparent segmentation overlay on an ultrasound image."""
    # Load base image
    if isinstance(image_input, (str, Path)):
        base_img = Image.open(str(image_input)).convert("RGBA")
    elif isinstance(image_input, Image.Image):
        base_img = image_input.convert("RGBA")
    elif isinstance(image_input, np.ndarray):
        base_img = Image.fromarray(image_input).convert("RGBA")
    else:
        raise TypeError(f"Unsupported image input type: {type(image_input)}")

    if mask_input is None:
        return base_img.convert("RGB")

    # Load mask
    if isinstance(mask_input, (str, Path)):
        mask_path = Path(mask_input)
        if not mask_path.exists():
            return base_img.convert("RGB")
        mask_img = Image.open(str(mask_path)).convert("L")
    elif isinstance(mask_input, Image.Image):
        mask_img = mask_input.convert("L")
    elif isinstance(mask_input, np.ndarray):
        mask_img = Image.fromarray((mask_input > 0).astype(np.uint8) * 255).convert("L")
    else:
        return base_img.convert("RGB")

    # Resize mask to base image size if needed
    if mask_img.size != base_img.size:
        mask_img = mask_img.resize(base_img.size, Image.NEAREST)

    mask_values = np.asarray(mask_img)
    mask_arr = (mask_values == 1) | (mask_values > 127)
    if not mask_arr.any():
        # Empty predicted mask (Setting B negative scan)
        return base_img.convert("RGB")

    # Create colored overlay layer
    overlay_arr = np.zeros((base_img.size[1], base_img.size[0], 4), dtype=np.uint8)
    overlay_arr[mask_arr] = [color[0], color[1], color[2], int(255 * alpha)]
    overlay_img = Image.fromarray(overlay_arr)

    # Composite overlay on base image
    composite = Image.alpha_composite(base_img, overlay_img)

    # Add crisp boundary contour
    from scipy.ndimage import binary_dilation
    dilated = binary_dilation(mask_arr, iterations=contour_width)
    contour = dilated ^ mask_arr
    contour_arr = np.array(composite)
    contour_arr[contour] = [255, 255, 255, 255]  # Crisp white outline

    return Image.fromarray(contour_arr).convert("RGB")


def load_uncertainty_visualization(
    unc_png_path: Optional[Union[str, Path]],
    unc_npy_path: Optional[Union[str, Path]] = None,
    target_size: Optional[Tuple[int, int]] = None,
) -> Optional[Image.Image]:
    """Loads precomputed uncertainty PNG map or renders from raw entropy NPY array."""
    if unc_png_path:
        p = Path(unc_png_path)
        if p.exists():
            img = Image.open(str(p)).convert("RGB")
            if target_size and img.size != target_size:
                img = img.resize(target_size, Image.BILINEAR)
            return img

    if unc_npy_path:
        p = Path(unc_npy_path)
        if p.exists():
            arr = np.load(str(p))
            # Normalize to [0, 1] for colormap rendering
            max_val = arr.max()
            norm_arr = arr / max_val if max_val > 0 else arr
            colored = (cm.viridis(norm_arr)[:, :, :3] * 255).astype(np.uint8)
            img = Image.fromarray(colored)
            if target_size and img.size != target_size:
                img = img.resize(target_size, Image.BILINEAR)
            return img

    return None
