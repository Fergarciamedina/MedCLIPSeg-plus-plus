# Stage 3 · Frozen segmentation artifacts

Location `research/stage_3_medclipseg/` · [Up to research/](../RESEARCH_GUIDE.md)

The backbone is [MedCLIPSeg by Koleilat et al.](https://arxiv.org/abs/2602.20423) with the UniMedCLIP ViT-B/16 variant. From a prepared image and lesion description, it provides two aligned outputs that are required by Stage 4. A binary mask identifies predicted foreground and a raw entropy map measures pixelwise predictive uncertainty. Keeping these outputs fixed allows the study to compare scan-level scores on the same predictions.

| Setting in the study | Configuration |
| --- | --- |
| Weights | Frozen BUSI MedCLIPSeg checkpoint, with the `latest` variant and seed label 666 in the study workflow. |
| Model grid | 224 × 224, with the upstream image normalization and label-preserving mask resampling. |
| Predictive sampling | 30 stochastic predictions per image. Predictive entropy is computed from the mean foreground probability. |
| Output contract | Binary PNG mask from mean foreground probability above 0.5 and an aligned raw float32 entropy `.npy` array in nats. |

The [study input audit](../stage_1_study_inputs/STUDY_INPUTS.md#what-was-validated-on-the-inputs) checked 620 mask–entropy pairs. A [baseline comparison](../stage_4_selective_prediction/study_reference/PUBLISHED_BASELINE_COMPARISON.md) compares aggregate Dice with the upstream paper before selective prediction. Stage 4 evaluates the uncertainty rule on these outputs. The new study does not train the backbone or compare backbone architectures.

[Upstream code](https://github.com/HealthX-Lab/MedCLIPSeg) · [Released checkpoints](https://huggingface.co/TahaKoleilat/MedCLIPSeg) · [Local YAML guide](../../shared/configs/DATASET_CONFIGURATIONS.md) · [Inference choices and provenance](../stage_4_selective_prediction/study_reference/EXPERIMENTAL_RATIONALE.md#backbone-and-digital-image-processing) · [Application artifact lookup](../../application/stages/stage3_segmentation_backbone/SEGMENTATION_ARTIFACTS.md)
