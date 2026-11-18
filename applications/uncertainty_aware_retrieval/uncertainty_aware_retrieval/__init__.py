"""MedCLIPSeg++ Uncertainty-Aware Medical Image Retrieval Package."""

from .config import (
    FROZEN_THRESHOLD,
    EMPTY_MASK_FALLBACK_THRESHOLD,
    DEFAULT_K,
    DEFAULT_NLIST,
    DEFAULT_NPROBE,
    DEFAULT_BLOCK_SIZE,
)
from .intent_router import IntentRouter, ParsedInstruction
from .manifest_builder import ManifestBuilder
from .embedding_extractor import EmbeddingExtractor
from .segmentation_service import SegmentationService
from .selective_prediction import SelectivePredictionService, TriageResult
from .pabp_engine import PABPRetrievalEngine, RetrievedNeighbor, RetrievalMetricRecord
from .visualizer import create_segmentation_overlay, load_uncertainty_visualization

__all__ = [
    "FROZEN_THRESHOLD",
    "EMPTY_MASK_FALLBACK_THRESHOLD",
    "DEFAULT_K",
    "DEFAULT_NLIST",
    "DEFAULT_NPROBE",
    "DEFAULT_BLOCK_SIZE",
    "IntentRouter",
    "ParsedInstruction",
    "ManifestBuilder",
    "EmbeddingExtractor",
    "SegmentationService",
    "SelectivePredictionService",
    "TriageResult",
    "PABPRetrievalEngine",
    "RetrievedNeighbor",
    "RetrievalMetricRecord",
    "create_segmentation_overlay",
    "load_uncertainty_visualization",
]
