"""Stage 3: MedCLIPSeg Segmentation Backbone Package."""
from .model_adapter import (
    BaseSegmentationModelAdapter,
    MedCLIPSegAdapter,
    MockModelAdapter,
)

__all__ = [
    "BaseSegmentationModelAdapter",
    "MedCLIPSegAdapter",
    "MockModelAdapter",
]

