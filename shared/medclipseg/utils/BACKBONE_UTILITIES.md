# MedCLIPSeg utilities

Location `shared/medclipseg/utils/` · [Up to medclipseg/](../BACKBONE_SOURCE.md)

These utilities support preparing upstream artifacts and evaluating the predictions against reference masks.

| File | Purpose |
| --- | --- |
| [__init__.py](__init__.py) | Marks the utilities subpackage. |
| [main_utils.py](main_utils.py) | Loads YAML settings and text workbooks and provides image normalization helpers. |
| [download_ckpt.py](download_ckpt.py) | Downloads a named checkpoint from the model release. |
| [eval.py](eval.py) | Pairs saved masks with ground truth and writes Dice and NSD results. |
| [SurfaceDice.py](SurfaceDice.py) | Computes overlap and surface distances and derives surface Dice at a chosen tolerance. |
| [preprocess_EUS.py](preprocess_EUS.py) | Prepares the upstream endoscopic ultrasound dataset layout, including empty masks where needed. The breast ultrasound study uses its own prepared cohorts. |

[Offline commands](../../scripts/SEGMENTATION_COMMANDS.md) · [Metric conventions](../../../research/stage_4_selective_prediction/study_reference/THEORY_AND_METRICS.md)
