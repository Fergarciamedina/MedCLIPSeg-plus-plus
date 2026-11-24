"""Frozen UniMedCLIP Visual Embedding Extractor.

Extracts L2-normalized 512-dimensional visual feature vectors from ultrasound images
using the pretrained MedCLIPSeg / UniMedCLIP image backbone (ViT-B/16-quickgelu).
"""

import hashlib
import json
import os
import tempfile
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
    MANIFEST_CACHE_PATH,
    REPO_ROOT,
)

# Ensure shared medclipseg library is importable
SHARED_MEDCLIPSEG_DIR = REPO_ROOT / "shared" / "medclipseg"
if str(SHARED_MEDCLIPSEG_DIR) not in sys.path:
    sys.path.insert(0, str(SHARED_MEDCLIPSEG_DIR))
SHARED_DIR = REPO_ROOT / "shared"
if str(SHARED_DIR) not in sys.path:
    sys.path.insert(0, str(SHARED_DIR))

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
        supplied = os.getenv("MEDCLIPSEG_EMBEDDINGS")
        self.supplied_embeddings = Path(supplied).expanduser() if supplied else None
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
        """Load keyed vectors or compute them from the configured visual encoder."""
        keys = [f"{r['image_id']}@{r['dataset']}" for r in manifest]
        if not keys or len(set(keys)) != len(keys):
            raise ValueError("Gallery embeddings require a nonempty list of unique cases")
        if self.supplied_embeddings:
            return self._load_vectors(self.supplied_embeddings, keys)

        signature = self._signature(manifest)
        cache = self.cache_path.with_name(f"{self.cache_path.stem}-{signature}.npz")
        if not force_recompute and cache.is_file():
            return self._load_vectors(cache, keys)
        if not force_recompute and self.cache_path == EMBEDDINGS_CACHE_PATH:
            vectors = self._legacy_vectors(manifest)
            if vectors is not None:
                self._write_vectors(cache, keys, vectors)
                return vectors

        self._load_model()
        all_embeddings: List[np.ndarray] = []

        for i in range(0, len(manifest), batch_size):
            batch_records = manifest[i : i + batch_size]
            tensors = []
            for r in batch_records:
                img_path = r["image_path"]
                with Image.open(img_path) as img:
                    tensors.append(self._preprocess(img.convert("RGB")))

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

        full_embeddings = self._validate_vectors(full_embeddings, len(keys))
        self._write_vectors(cache, keys, full_embeddings)
        return full_embeddings

    @staticmethod
    def _validate_vectors(vectors, count):
        if vectors.shape != (count, EMBEDDING_DIM) or not np.issubdtype(vectors.dtype, np.floating):
            raise ValueError(f"Expected {count} floating-point vectors of dimension {EMBEDDING_DIM}")
        vectors = vectors.astype(np.float32)
        if not np.isfinite(vectors).all():
            raise ValueError("Gallery embeddings must contain finite values")
        norms = np.linalg.norm(vectors.astype(np.float64), axis=1, keepdims=True)
        if np.any(norms == 0):
            raise ValueError("Gallery embeddings must be nonzero")
        return (vectors / norms).astype(np.float32)

    def _load_vectors(self, path, keys):
        with np.load(path, allow_pickle=False) as archive:
            stored = archive["keys"]
            if stored.ndim != 1 or stored.dtype.kind != "U":
                raise ValueError("Embedding keys must be a one-dimensional Unicode array")
            positions = {key: i for i, key in enumerate(stored.tolist())}
            if len(positions) != len(stored) or positions.keys() != set(keys):
                raise ValueError(f"Embedding keys in {path} must exactly match the gallery cases")
            vectors = self._validate_vectors(archive["embeddings"], len(keys))
        return vectors[[positions[key] for key in keys]]

    def _signature(self, manifest):
        paths = [Path(r["image_path"]).resolve() for r in manifest]
        checkpoint = self.checkpoint_path.resolve()
        signature = hashlib.sha256()
        for path in [checkpoint, *paths]:
            stat = path.stat() if path.is_file() else None
            signature.update(json.dumps([str(path), stat.st_size if stat else None,
                                         stat.st_mtime_ns if stat else None]).encode("utf-8"))
        for row in manifest:
            signature.update(json.dumps([row["dataset"], row["image_id"]]).encode("utf-8"))
        return signature.hexdigest()[:24]

    def _legacy_vectors(self, manifest):
        """Migrate the original benchmark cache only with its matching ordered manifest."""
        if any(os.getenv(name) for name in (
            "MEDCLIPSEG_GALLERY", "MEDCLIPSEG_DATA", "MEDCLIPSEG_OUTPUT", "MEDCLIPSEG_CHECKPOINT",
        )):
            return None
        legacy = self.cache_path.with_suffix(".npy")
        if not legacy.is_file() or not MANIFEST_CACHE_PATH.is_file():
            return None
        fields = ("dataset", "image_id", "image_path")
        def identity(rows):
            return [tuple(row[field] for field in fields) for row in rows]
        try:
            with MANIFEST_CACHE_PATH.open(encoding="utf-8") as source:
                cached = json.load(source)
            if identity(cached) != identity(manifest):
                return None
            cutoff = legacy.stat().st_mtime_ns
            if any(Path(r["image_path"]).stat().st_mtime_ns > cutoff for r in manifest):
                return None
            return self._validate_vectors(np.load(legacy, allow_pickle=False), len(manifest))
        except (OSError, ValueError, KeyError, TypeError):
            return None

    @staticmethod
    def _write_vectors(path, keys, vectors):
        path.parent.mkdir(parents=True, exist_ok=True)
        with tempfile.NamedTemporaryFile(dir=path.parent, suffix=".npz", delete=False) as handle:
            temp = Path(handle.name)
        try:
            np.savez(temp, keys=np.asarray(keys, dtype=str), embeddings=vectors)
            temp.replace(path)
        finally:
            temp.unlink(missing_ok=True)
