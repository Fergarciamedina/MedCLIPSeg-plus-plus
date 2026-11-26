# Prepare per-image uncertainty scores

Location `research/stage_4_selective_prediction/experiments/prepare_uncertainty_scores/` · [Up to experiments/](../RESULTS_GUIDE.md)

Use this branch after frozen MedCLIPSeg has produced a binary mask and a raw predictive entropy map for each case. The [input guide](../../../stage_1_study_inputs/STUDY_INPUTS.md) explains the dataset files and case pairing. The [frozen segmentation guide](../../../stage_3_medclipseg/FROZEN_SEGMENTATION.md) gives the model source and inference settings.

## Inputs and outputs

The [aggregation script](compute_uncertainty_csv.py) joins masks, entropy arrays, and reference Dice and NSD by image ID. It writes one row per case with seven uncertainty scores and validity fields. The same script handles validation and test so the score definitions are consistent across splits. Its study settings are boundary radius 2, high-entropy cutoff 0.1 nats and ratio denominator offset `1e-6`. See the [equations and edge cases](../../study_reference/THEORY_AND_METRICS.md#4-seven-aggregation-functions-operational-definitions-and-anticipated-failure-modes).

Outputs go under `MEDCLIPSEG_OUTPUT`. For BUSI validation, the score table is at `BUSI/validation/seed666/aggregation_results/uncertainty_scores_BUSI_val_seed666.csv`. For a test cohort it is at `<cohort>/aggregation_results/seed666/uncertainty_scores_<cohort>_test_seed666.csv`. These per-image inputs are kept in the local output tree. The study branches contain aggregate tables and figures.

## Scripts in this folder

| Python script | Shell launcher | When to use it |
| --- | --- | --- |
| [compute_uncertainty_csv.py](compute_uncertainty_csv.py) | [run_compute_uncertainty_csv.sh](run_compute_uncertainty_csv.sh) | Create the per-image input to validation selection and locked tests. |
| [run_aggregation.py](run_aggregation.py) | [run_aggregation.sh](run_aggregation.sh) | Explore score–quality correlations and failure AUROC. |
| [visualize_aggregation.py](visualize_aggregation.py) | [run_visualize_aggregation.sh](run_visualize_aggregation.sh) | Plot image, mask and entropy checks and other aggregation diagnostics. |
| [run_selective_prediction.py](run_selective_prediction.py) | [run_selective_prediction.sh](run_selective_prediction.sh) | Explore risk–coverage with random and oracle controls on saved scores. The locked analyses use their phase-specific evaluators. |

## Run

From the repository root with the [study environment](../../SELECTIVE_PREDICTION_STUDY.md#reproduction) configured, run the command for the split whose inference artifacts are available.

```bash
bash research/stage_4_selective_prediction/experiments/prepare_uncertainty_scores/run_compute_uncertainty_csv.sh BUSI val 666
bash research/stage_4_selective_prediction/experiments/prepare_uncertainty_scores/run_compute_uncertainty_csv.sh BUSI test 666
```

[Explore aggregate correlations](run_aggregation.sh), [visualize spatial diagnostics](run_visualize_aggregation.sh), and [explore selective prediction on a saved cohort](run_selective_prediction.sh) are optional analysis tools. The validation-only selection and locked test use their own dedicated runners.

[Next · Validation and selection](../validation_selection/VALIDATION_SELECTION.md) · [Stage 4 study](../../SELECTIVE_PREDICTION_STUDY.md)
