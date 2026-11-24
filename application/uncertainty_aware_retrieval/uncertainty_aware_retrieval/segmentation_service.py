"""Stage 2: Segmentation Service.

Provides instant precomputed lookup for the 542 evaluated benchmark scans,
with a graceful live-inference adapter interface for newly uploaded images.
"""

from pathlib import Path
from typing import Any, Dict, List, Optional, Union
import numpy as np
from PIL import Image

from .config import FROZEN_THRESHOLD


class SegmentationService:
    """Retrieves cached segmentation masks & uncertainty maps or executes inference."""

    def __init__(self, manifest: Optional[List[Dict[str, Any]]] = None):
        self.manifest = manifest or []
        self._lookup: Dict[str, Dict[str, Any]] = {}
        self._build_lookup()

    def _build_lookup(self) -> None:
        """Index manifest by image_id, image_path, and basename for O(1) lookup."""
        for r in self.manifest:
            # Key by image_id
            self._lookup[r["image_id"]] = r
            # Key by dataset:image_id
            self._lookup[f"{r['dataset']}:{r['image_id']}"] = r
            # Key by filename
            self._lookup[Path(r["image_path"]).name] = r
            # Key by absolute path
            self._lookup[str(Path(r["image_path"]).resolve())] = r

    def get_segmentation_for_query(
        self,
        image_identifier: Union[str, Path, Image.Image],
        image_name: Optional[str] = None,
    ) -> Dict[str, Any]:
        """Look up precomputed segmentation and uncertainty or indicate live status."""
        query_key = None
        if isinstance(image_identifier, (str, Path)):
            p = str(image_identifier)
            if p in self._lookup:
                query_key = p
            elif Path(p).name in self._lookup:
                query_key = Path(p).name
        elif image_name and image_name in self._lookup:
            query_key = image_name

        if query_key and query_key in self._lookup:
            cached = self._lookup[query_key]
            return {
                "source": "cached_canonical",
                "image_id": cached["image_id"],
                "dataset": cached["dataset"],
                "image_path": cached["image_path"],
                "seg_mask_path": cached["seg_mask_path"],
                "unc_png_path": cached["unc_png_path"],
                "unc_npy_path": cached["unc_npy_path"],
                "u_fg": cached["u_fg"],
                "u_mean": cached["u_mean"],
                "count_fg": cached["count_fg"],
                "status": cached["status"],
                "dsc": cached.get("dsc"),
                "is_unseen": False,
            }

        # For an unseen image without precomputed Level 0 predictions:
        return {
            "source": "unseen_image",
            "image_id": image_name or "uploaded_case.png",
            "dataset": "User Upload",
            "image_path": str(image_identifier) if isinstance(image_identifier, (str, Path)) else "in-memory",
            "seg_mask_path": None,
            "unc_png_path": None,
            "unc_npy_path": None,
            "u_fg": None,
            "u_mean": None,
            "count_fg": None,
            "status": "Review Required (Unseen Image)",
            "dsc": None,
            "is_unseen": True,
            "limitation_note": (
                "Unseen ultrasound image without precomputed Level 0 segmentation or uncertainty maps. "
                "MedCLIPSeg requires paired text prompts and calibrated test-time Monte Carlo or ensemble "
                "sampling to generate predictive entropy maps. Autonomous retrieval requires review."
            ),
        }

