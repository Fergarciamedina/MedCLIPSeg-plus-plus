# Offline segmentation commands

Location `shared/scripts/` · [Up to shared/](../SHARED_COMPONENTS.md)

Run these shell scripts from the repository root. They connect a [dataset profile](../configs/DATASET_CONFIGURATIONS.md) to the [MedCLIPSeg inference source](../medclipseg/BACKBONE_SOURCE.md).

| Launcher | Purpose |
| --- | --- |
| [download_checkpoint.sh](download_checkpoint.sh) | Retrieves a released checkpoint and places it in the configured output tree. |
| [run_inference.sh](run_inference.sh) | Runs frozen test inference for the selected profile and seed, writing predicted masks and uncertainty artifacts. |
| [run_evaluation.sh](run_evaluation.sh) | Compares stored test masks to references and exports per-case Dice and NSD. |

The Stage 4 [validation launcher](../../research/stage_4_selective_prediction/experiments/validation_selection/VALIDATION_SELECTION.md) has its own split-specific pipeline. For the shared paths and command order, see [application artifact preparation](../../application/APPLICATION_GUIDE.md#gallery-catalog) and the [locked BUSI test guide](../../research/stage_4_selective_prediction/experiments/locked_busi_test/LOCKED_BUSI_TEST.md).
