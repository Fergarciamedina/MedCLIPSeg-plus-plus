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
| [visualize_aggregation.py](visualize_aggregation.py) | [run_visualize_aggregation.sh](run_visualize_aggregation.sh) | Plot image, mask and entropy checks and other aggregation diagnostics. |

## Run

From the repository root with the [study environment](../../SELECTIVE_PREDICTION_STUDY.md#reproduction) configured, run the command for the split whose inference artifacts are available.

```bash
bash research/stage_4_selective_prediction/experiments/prepare_uncertainty_scores/run_compute_uncertainty_csv.sh BUSI val 666
bash research/stage_4_selective_prediction/experiments/prepare_uncertainty_scores/run_compute_uncertainty_csv.sh BUSI test 666
```

Use the [visual diagnostics](run_visualize_aggregation.sh) to inspect how each summary uses the spatial uncertainty map. The phase-specific runners reproduce rank correlations and risk–coverage results from the per-image scores.

[Next · Validation and selection](../validation_selection/VALIDATION_SELECTION.md) · [Stage 4 study](../../SELECTIVE_PREDICTION_STUDY.md)
