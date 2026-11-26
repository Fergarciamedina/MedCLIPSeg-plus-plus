# External cohort test

Location `research/stage_4/experiments/external_test/` · [Up to experiments/](../RESULTS_GUIDE.md)

This branch measures how the frozen BUSI rule transfers to UDIAT, BUID, BUSUC and BUSBRA. The 464 external images use predictions from the same BUSI source checkpoint and the [policy fixed at validation](../validation_selection/VALIDATION_SELECTION.md). The four cohorts vary in size and data source. The [experimental rationale](../../study_reference/EXPERIMENTAL_RATIONALE.md#cohort-scope-and-sample-sizes) explains the split selection and hardware scope.

## Contents

[run_external_inference_and_aggregation.sh](run_external_inference_and_aggregation.sh) builds external score tables using the BUSI checkpoint. [run_external_test_evaluation.py](run_external_test_evaluation.py) applies the frozen policy to saved scores and exports the cohort and cross-domain comparisons. The [results/](results/CROSS_DOMAIN_RESULTS.md) guide links each cohort and explains its evidence files.

## Inputs and run

The evaluator reads the local frozen YAML and the per-image score CSV for each cohort. The external inference launcher can build the four external CSVs from their images, prompts and reference masks using the [shared model YAMLs](../../../../shared/configs/DATASET_CONFIGURATIONS.md). Prepare the BUSI test CSV as well to include the in-domain comparison in the same output.

With the [study environment](../../SELECTIVE_PREDICTION_STUDY.md#reproduction) configured, run from the repository root.

```bash
# Run this step if external predictions and score CSVs are unavailable.
bash research/stage_4/experiments/external_test/run_external_inference_and_aggregation.sh UDIAT BUID BUSUC BUSBRA

.venv/bin/python research/stage_4/experiments/external_test/run_external_test_evaluation.py --seed 666
```

The runner writes per-cohort reports and a cross-domain summary under `MEDCLIPSEG_OUTPUT`. The [cross-domain summary](results/cross_domain_summary.csv) reports separate pooled and cohort-average results. The fixed cutoff retained 238/464 external cases.

[Cross-phase results guide](../RESULTS_GUIDE.md) · [Stage 4 study](../../SELECTIVE_PREDICTION_STUDY.md)
