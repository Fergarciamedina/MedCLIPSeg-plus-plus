# MedCLIPSeg backbone source

Location `shared/medclipseg/` · [Up to shared/](../SHARED_COMPONENTS.md)

This directory contains the adapted upstream segmentation code. Use the [offline launchers](../scripts/SEGMENTATION_COMMANDS.md) and [YAML profiles](../configs/DATASET_CONFIGURATIONS.md) to run it. Original model and license information is in [Attribution](../../ATTRIBUTION.md).

| File or folder | Role |
| --- | --- |
| [test.py](test.py) | Loads a configured model and checkpoint, runs stochastic test-set inference, and exports the predicted masks and uncertainty. |
| [__init__.py](__init__.py) | Marks the MedCLIPSeg namespace. |
| [trainers/](trainers/SEGMENTATION_MODEL_BUILDERS.md) | Backbone-specific model builders, probabilistic adapters and feature scaling. |
| [datasets/](datasets/SEGMENTATION_DATASET_LOADER.md) | Image and mask pairing and input transforms. |
| [utils/](utils/BACKBONE_UTILITIES.md) | Configuration and checkpoint helpers and reference-mask evaluation. |
| [clip/](clip/CLIP_SOURCE.md) | Bundled OpenAI CLIP model and tokenization helpers. |
| [open_clip_lib/](open_clip_lib/OPEN_CLIP_COMPONENTS.md) | Bundled model factory, encoder and tokenizer infrastructure used by the UniMedCLIP path. |
| [biomedclip/](biomedclip/BIOMEDCLIP_COMPONENTS.md) | Model support files for the upstream BiomedCLIP alternative. |

The reported experiments use the UniMedCLIP ViT-B/16 variant and its saved outputs. [Frozen segmentation settings](../../research/stage_3/FROZEN_SEGMENTATION.md).
