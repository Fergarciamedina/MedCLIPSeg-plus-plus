"""Stage 1: Inputs (MOD-INPUTS / Stage 1).

Handles ultrasound image X and clinical natural language request R:
- Image loading, integrity verification, spatial resizing (default 256x256), normalization.
- Clinical request prompt standardization (default: 'Segment and find similar cases').
"""

from dataclasses import dataclass
from typing import Optional, Union, Tuple
from pathlib import Path
import numpy as np
from PIL import Image

@dataclass
class ClinicalInputs:
    """Standardized Stage 1 Input Container."""
    image_path: Optional[str]
    image_rgb: np.ndarray  # HxWxC uint8
    prompt: str
    original_size: Tuple[int, int]
    target_size: Tuple[int, int] = (256, 256)

def load_clinical_inputs(
    image_source: Union[str, Path, np.ndarray, Image.Image],
    prompt: Optional[str] = None,
    target_size: Tuple[int, int] = (256, 256),
) -> ClinicalInputs:
    """Ingests and validates ultrasound scan X and clinical request R.

    Args:
        image_source: File path to ultrasound scan, PIL Image, or numpy array.
        prompt: Clinical request string R (e.g. 'Segment and find similar cases').
        target_size: Standardized spatial resolution (height, width).

    Returns:
        ClinicalInputs containing standardized RGB array and sanitized prompt.
    """
    image_path = None
    if isinstance(image_source, (str, Path)):
        image_path = str(image_source)
        if not Path(image_path).exists():
            raise FileNotFoundError(f"Ultrasound image not found: {image_path}")
        pil_img = Image.open(image_path).convert("RGB")
    elif isinstance(image_source, Image.Image):
        pil_img = image_source.convert("RGB")
    elif isinstance(image_source, np.ndarray):
        if image_source.ndim == 2:
            pil_img = Image.fromarray(image_source).convert("RGB")
        elif image_source.ndim == 3:
            pil_img = Image.fromarray(image_source)
        else:
            raise ValueError(f"Invalid numpy array shape for image: {image_source.shape}")
    else:
        raise TypeError(f"Unsupported image_source type: {type(image_source)}")

    orig_size = pil_img.size  # (width, height)
    resized_pil = pil_img.resize(target_size, Image.Resampling.BILINEAR)
    image_rgb = np.array(resized_pil)

    # Standardize prompt R
    clean_prompt = (prompt or "").strip()
    if not clean_prompt:
        clean_prompt = "Segment and find similar cases"

    return ClinicalInputs(
        image_path=image_path,
        image_rgb=image_rgb,
        prompt=clean_prompt,
        original_size=orig_size,
        target_size=target_size,
    )

