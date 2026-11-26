# Segmentation model builders

Location `shared/medclipseg/trainers/` · [Up to medclipseg/](../BACKBONE_SOURCE.md)

The builders assemble the upstream vision–language encoders and probabilistic segmentation adapters expected by the released checkpoints. [Stage 3](../../../research/stage_3_medclipseg/FROZEN_SEGMENTATION.md) documents the variant used for the experiment.

| File | Purpose |
| --- | --- |
| [__init__.py](__init__.py) | Exposes loaders for the four encoder variants. |
| [medclipseg_unimedclip.py](medclipseg_unimedclip.py) | Builds the UniMedCLIP-based network used in the breast ultrasound study. |
| [medclipseg_biomedclip.py](medclipseg_biomedclip.py) | Builds the upstream BiomedCLIP variant. |
| [medclipseg_clip.py](medclipseg_clip.py) | Builds the upstream OpenAI CLIP variant. |
| [medclipseg_pubmedclip.py](medclipseg_pubmedclip.py) | Loads PubMedCLIP weights and configures the corresponding segmentation variant. |
| [layers.py](layers.py) | Implements the probabilistic cross-attention and vision–language adapters. |
| [scale_block.py](scale_block.py) | Defines the feature-rescaling blocks reused by the segmentation variants. |

[Backbone source map](../BACKBONE_SOURCE.md) · [Original MedCLIPSeg credit](../../../ATTRIBUTION.md)
