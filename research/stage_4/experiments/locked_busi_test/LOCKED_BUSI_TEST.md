# Locked BUSI test

Location `research/stage_4/experiments/locked_busi_test/` · [Up to experiments/](../RESULTS_GUIDE.md)

This branch applies the [validation-derived policy](../validation_selection/VALIDATION_SELECTION.md) to the held-out BUSI test split. It uses 78 images, of which 65 have nonempty predicted masks. The locked foreground rule retained 47 of those 65 predictions. The full-split analysis also compares the handling of the 13 empty predictions.

## Contents

[run_locked_test_evaluation.py](run_locked_test_evaluation.py) applies the frozen threshold and writes comparisons for the nonempty and full-cohort analyses. [run_locked_test_evaluation.sh](run_locked_test_evaluation.sh) sets the local experiment paths and invokes the evaluator. The [results/](results/LOCKED_BUSI_RESULTS.md) guide explains every bundled table and plot.

## Inputs and run

The evaluator uses the frozen validation YAML and the BUSI test uncertainty CSV under the same local `MEDCLIPSEG_OUTPUT` tree. The [input preparation branch](../prepare_uncertainty_scores/PREPARE_UNCERTAINTY_SCORES.md) documents how to build the CSV. Use the same BUSI checkpoint and original prompt workflow as validation. The shared [inference configuration](../../../../shared/configs/BUSI.yaml) defines the model.

From the repository root, prepare the test predictions and score table if needed, then run the locked evaluation.

```bash
bash shared/scripts/run_inference.sh BUSI 666
bash shared/scripts/run_evaluation.sh BUSI 666
bash research/stage_4/experiments/prepare_uncertainty_scores/run_compute_uncertainty_csv.sh BUSI test 666
bash research/stage_4/experiments/locked_busi_test/run_locked_test_evaluation.sh BUSI 666
```

The runner writes under `MEDCLIPSEG_OUTPUT/BUSI/test_evaluation/seed666/`. Start with the bundled [primary operating point](results/primary_operating_point_test.csv) for the held-out behavior at the selected cutoff.

[Next · External tests](../external_test/EXTERNAL_TEST.md) · [Stage 4 study](../../SELECTIVE_PREDICTION_STUDY.md)
