# BUSI validation and score selection

Location `research/stage_4_selective_prediction/experiments/validation_selection/` · [Up to experiments/](../RESULTS_GUIDE.md)

This branch contains the scripts, frozen policy and aggregate results for the only split used to select a score and set the cutoff. The study used 78 BUSI validation images, with 65 nonempty predicted masks in the main comparison and 13 empty predictions examined separately. See the [input checks](../../../stage_1_study_inputs/STUDY_INPUTS.md), [model configuration](../../../stage_3_medclipseg/FROZEN_SEGMENTATION.md) and [experimental protocol](../../study_reference/EXPERIMENTAL_PROTOCOL.md).

## What is in this folder

| File or folder | What to expect |
| --- | --- |
| [run_validation_inference.py](run_validation_inference.py) · [launcher](run_validation_inference.sh) | Generate masks and raw entropy with the frozen BUSI model. The launcher uses seed 666, batch size 8 and 30 stochastic predictions. The [BUSI model YAML](../../../../shared/configs/BUSI.yaml) is shared with the other inference workflows. |
| [run_validation_statistics.py](run_validation_statistics.py) · [launcher](run_validation_statistics.sh) | Analyze the validation scores, including score–quality associations and foreground area. |
| [run_validation_selection.py](run_validation_selection.py) · [launcher](run_validation_selection.sh) | Compare the seven scores and freeze the numeric operating point. The nominal 80% target retained 52/65 cases and gave the cutoff `0.064058324`. |
| [Frozen policy YAML](frozen_selective_prediction_config.yaml) and [JSON](frozen_selective_prediction_config.json) | Archived numeric policy from the completed study. This is the rule carried into both test branches. |
| [results/](results/VALIDATION_RESULTS.md) | Validation method comparisons, paired bootstrap, six operating points, and diagnostic figures. |

## Run on local artifacts

From the repository root with the [study environment](../../SELECTIVE_PREDICTION_STUDY.md#reproduction) configured, generate predictions if needed, build the per-image table, and run the two validation analyses.

```bash
bash research/stage_4_selective_prediction/experiments/validation_selection/run_validation_inference.sh 666 cpu
bash research/stage_4_selective_prediction/experiments/prepare_uncertainty_scores/run_compute_uncertainty_csv.sh BUSI val 666
bash research/stage_4_selective_prediction/experiments/validation_selection/run_validation_statistics.sh 666
bash research/stage_4_selective_prediction/experiments/validation_selection/run_validation_selection.sh 666
```

The runners write to `MEDCLIPSEG_OUTPUT/BUSI/validation/seed666/`. Both test branches read `selective_prediction/frozen_selective_prediction_config.yaml` from that local tree. To evaluate the archived policy on already prepared scores, place the bundled YAML at that path in your output tree. Use the [paired-bootstrap table](results/paired_bootstrap_aurc_ci.csv) for the score comparison and the [numeric fields in the YAML](../RESULTS_GUIDE.md#how-to-read-the-frozen-yaml) for the decision rule.

[Next · Locked BUSI test](../locked_busi_test/LOCKED_BUSI_TEST.md) · [External tests](../external_test/EXTERNAL_TEST.md) · [Stage 4 study](../../SELECTIVE_PREDICTION_STUDY.md)
