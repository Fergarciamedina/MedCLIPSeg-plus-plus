# Segmentation dataset loader

Location `shared/medclipseg/datasets/` · [Up to medclipseg/](../BACKBONE_SOURCE.md)

[dataloader.py](dataloader.py) defines the dataset that loads prepared image and label pairs. Its validation transform puts images and masks on the model grid. It also retains training-time augmentation helpers from the source workflow. [__init__.py](__init__.py) marks the dataset subpackage.

The study's prepared folder and workbook contract is documented in [Stage 1](../../../research/stage_1_study_inputs/STUDY_INPUTS.md). The [YAML profiles](../../configs/DATASET_CONFIGURATIONS.md) specify the paths and grid size.
