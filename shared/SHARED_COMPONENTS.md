# Shared segmentation components

Location `shared/` · [Up to repository root](../README.md)

The research workflow and offline application preparation use the same frozen segmentation backbone, dataset profiles and launchers. Use this directory to prepare masks and entropy before running the study or registering cases in the application.

| File or folder | Role |
| --- | --- |
| [medclipseg/](medclipseg/BACKBONE_SOURCE.md) | Adapted upstream MedCLIPSeg inference and its model dependencies. |
| [configs/](configs/DATASET_CONFIGURATIONS.md) | Dataset, model and inference profiles, including the five study cohorts. |
| [scripts/](scripts/SEGMENTATION_COMMANDS.md) | Launchers for checkpoints, predictions and reference-mask evaluation. |
| [LICENSE-MedCLIPSeg](LICENSE-MedCLIPSeg) | Upstream Apache-2.0 license copy included for the MedCLIPSeg source. |

[Project attribution](../ATTRIBUTION.md) · [Frozen study configuration](../research/stage_3/FROZEN_SEGMENTATION.md)
