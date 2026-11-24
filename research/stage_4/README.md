# Stage 4 · Selective prediction

Stage 4 evaluates whether predictive entropy from frozen MedCLIPSeg can rank segmentation quality and support a policy that sends uncertain cases to review. The choice of score and cutoff was made on BUSI validation. The same cutoff was then applied to the locked BUSI test set and four external cohorts.

| Evidence | What it covers |
| --- | --- |
| [Theory and metrics](THEORY_AND_METRICS.md) | Predictive entropy, seven image-level aggregators, DSC and NSD, risk–coverage and AURC |
| [Experimental protocol](EXPERIMENTAL_PROTOCOL.md) | Cohorts, hypotheses, validation-only method selection, frozen cutoff, and test design |
| [Results](results/README.md) | Frozen policy, validation selection, locked test, external results, tables, and figures |
| [Scientific claims](SCIENTIFIC_EVIDENCE_AND_CLAIMS.md) | Quantitative evidence, paired comparisons, and scope of claims |
| [Decisions and limitations](RESEARCH_DECISIONS_AND_LIMITATIONS.md) | Audit trail, confounders, and retained low-quality predictions |
| [Published baseline comparison](PUBLISHED_BASELINE_COMPARISON.md) | Attribution and scope relative to the underlying MedCLIPSeg paper |
| [Benchmarking notebook](../notebooks/02_scientific_benchmarking.ipynb) | Interactive walkthrough of the saved experiment |

Foreground mean entropy, `U_fg`, had the lowest validation AURC among the seven candidates at 0.0839. Foreground entropy standard deviation had AURC 0.0850, with the paired confidence interval for the difference including zero. The validation-derived cutoff `0.064058324` retained 47/65 lesion cases on locked BUSI test and 238/464 external cases pooled. Mean DSC among retained cases was 0.9031 and 0.8718, respectively. The test also documented a low-DSC lesion that passed the filter. See the [result tables](results/README.md) for baselines, confidence intervals and coverage.

## Reproduction

Use the [notebook](../README.md#notebooks) to rerender the published aggregate evidence from the checkout. To recompute the study, install the repository [Python requirements](../../requirements.txt), obtain the [prepared cohorts](../stage_1/README.md) and the [checkpoint and prompts documented in the protocol](EXPERIMENTAL_PROTOCOL.md), then configure your private data and output paths. Launch the scripts below from the repository root.

```bash
export MEDCLIPSEG_DATA=/path/to/data/raw
export MEDCLIPSEG_OUTPUT=/path/to/experiment/outputs
# Permit first-time model asset downloads when the cache is empty.
export HF_HUB_OFFLINE=0 TRANSFORMERS_OFFLINE=0

bash research/stage_4/scripts/run_validation_inference.sh 666 cpu
bash research/stage_4/scripts/run_compute_uncertainty_csv.sh BUSI val 666
bash research/stage_4/scripts/run_validation_statistics.sh 666
bash research/stage_4/scripts/run_validation_selection.sh 666
```

For test evaluation, review the five cohort YAMLs under [shared/configs](../../shared/configs) and set `MODEL.DEVICE` to match the inference environment. Use the same BUSI checkpoint and original prompts throughout. Use these commands to generate the test artifacts.

```bash
bash shared/scripts/run_inference.sh BUSI 666
bash shared/scripts/run_evaluation.sh BUSI 666
bash research/stage_4/scripts/run_compute_uncertainty_csv.sh BUSI test 666
bash research/stage_4/scripts/run_ood_inference_and_aggregation.sh UDIAT BUID BUSUC BUSBRA
```

The next commands apply the frozen validation policy to the saved artifacts and run the regression suite.

```bash
bash research/stage_4/scripts/run_locked_test_evaluation.sh BUSI 666
.venv/bin/python research/stage_4/experiments/selective_prediction/run_ood_selective_prediction_evaluation.py
bash research/stage_4/scripts/run_tests.sh
```

The complete regression suite includes tests that read the local experiment artifacts. The [evaluation toolkit](uncertainty_selective_evaluation/README.md) can also run a standalone synthetic demonstration or read an existing per-image results CSV.
