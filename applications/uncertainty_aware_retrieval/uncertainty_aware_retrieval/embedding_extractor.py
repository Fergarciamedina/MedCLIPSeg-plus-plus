"""Frozen UniMedCLIP Visual Embedding Extractor.

Extracts L2-normalized 512-dimensional visual feature vectors from ultrasound images
using the pretrained MedCLIPSeg / UniMedCLIP image backbone (ViT-B/16-quickgelu).
"""

import os
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional, Union
import numpy as np
import torch
from PIL import Image

from .config import (
    CHECKPOINT_PATH,
    EMBEDDINGS_CACHE_PATH,
    EMBEDDING_DIM,
    REPO_ROOT,
)

# Ensure shared medclipseg library is importable
SHARED_MEDCLIPSEG_DIR = REPO_ROOT / "shared" / "medclipseg"
if str(SHARED_MEDCLIPSEG_DIR) not in sys.path:
    sys.path.insert(0, str(SHARED_MEDCLIPSEG_DIR))
SHARED_DIR = REPO_ROOT / "shared"
if str(SHARED_DIR) not in sys.path:
    sys.path.insert(0, str(SHARED_DIR))

# Offline HuggingFace execution to avoid internet calls in sandbox
os.environ["HF_HUB_OFFLINE"] = "1"
os.environ["TRANSFORMERS_OFFLINE"] = "1"


class EmbeddingExtractor:
    """Extracts and caches frozen visual embeddings for ultrasound cases."""

    def __init__(
        self,
        checkpoint_path: Optional[Path] = None,
        cache_path: Optional[Path] = None,
        device: str = "cpu",
    ):
        self.checkpoint_path = Path(checkpoint_path or CHECKPOINT_PATH)
        self.cache_path = Path(cache_path or EMBEDDINGS_CACHE_PATH)
        self.device = device
        self._model = None
        self._preprocess = None

    def _load_model(self):
        """Lazy load the UniMedCLIP vision encoder."""
        if self._model is not None:
            return

        if not self.checkpoint_path.exists():
            raise FileNotFoundError(
                f"MedCLIPSeg/UniMedCLIP checkpoint not found at: {self.checkpoint_path}"
            )

        from open_clip_lib import create_model_and_transforms, get_mean_std

        mean, std = get_mean_std()
        model, _, preprocess = create_model_and_transforms(
            "ViT-B-16-quickgelu",
            str(self.checkpoint_path),
            precision="fp32",
            device=self.device,
            force_quick_gelu=True,
            mean=mean,
            std=std,
            inmem=True,
            text_encoder_name="microsoft/BiomedNLP-BiomedBERT-base-uncased-abstract",
        )
        model.eval()
        self._model = model
        self._preprocess = preprocess

    def extract_image_embedding(self, image_input: Union[str, Path, Image.Image]) -> np.ndarray:
        """Extract a single L2-normalized 512-dim embedding vector."""
        self._load_model()

        if isinstance(image_input, (str, Path)):
            img = Image.open(str(image_input)).convert("RGB")
        elif isinstance(image_input, Image.Image):
            img = image_input.convert("RGB")
        else:
            raise TypeError(f"Unsupported image input type: {type(image_input)}")

        tensor = self._preprocess(img).unsqueeze(0).to(self.device)
        with torch.no_grad():
            feat = self._model.encode_image(tensor)
            feat = feat / feat.norm(dim=-1, keepdim=True)

        vec = feat.cpu().numpy().squeeze(0).astype(np.float32)
        norm = np.linalg.norm(vec)
        if norm > 0:
            vec = vec / norm
        return vec

    def extract_or_load_gallery_embeddings(
        self,
        manifest: List[Dict[str, Any]],
        force_recompute: bool = False,
        batch_size: int = 32,
    ) -> np.ndarray:
        """Load embeddings from disk cache if present and matching count; otherwise compute and cache."""
        expected_n = len(manifest)

        if not force_recompute and self.cache_path.exists():
            try:
                embeddings = np.load(str(self.cache_path))
                if embeddings.shape == (expected_n, EMBEDDING_DIM):
                    return embeddings.astype(np.float32)
            except Exception:
                pass

        # Compute embeddings across manifest
        self._load_model()
        all_embeddings: List[np.ndarray] = []

        for i in range(0, expected_n, batch_size):
            batch_records = manifest[i : i + batch_size]
            tensors = []
            for r in batch_records:
                img_path = r["image_path"]
                img = Image.open(img_path).convert("RGB")
                tensors.append(self._preprocess(img))

            batch_tensor = torch.stack(tensors).to(self.device)
            with torch.no_grad():
                feats = self._model.encode_image(batch_tensor)
                feats = feats / feats.norm(dim=-1, keepdim=True)

            arr = feats.cpu().numpy().astype(np.float32)
            # Ensure strict L2 normalization
            norms = np.linalg.norm(arr, axis=-1, keepdims=True)
            norms[norms == 0] = 1.0
            arr = arr / norms
            all_embeddings.append(arr)

        full_embeddings = np.concatenate(all_embeddings, axis=0).astype(np.float32)

        # Cache to disk
        self.cache_path.parent.mkdir(parents=True, exist_ok=True)
        np.save(str(self.cache_path), full_embeddings)

        return full_embeddings

