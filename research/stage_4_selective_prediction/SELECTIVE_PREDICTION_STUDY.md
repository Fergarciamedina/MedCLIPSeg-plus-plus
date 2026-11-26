# Stage 4 · Selective prediction

Location `research/stage_4_selective_prediction/` · [Up to research/](../RESEARCH_GUIDE.md)

The [selective segmentation review chapter](../../docs/learning/LEARNING_GUIDE.md) provides an interactive introduction to the probability-to-Dice argument, score comparison and operating point.

Stage 4 asks whether a single uncertainty score per scan can identify which frozen MedCLIPSeg segmentations should be reviewed. Every candidate receives the same saved predicted mask and raw entropy map for each scan. Reference annotations supply the quality measures for evaluation. See [Stage 1](../stage_1_study_inputs/STUDY_INPUTS.md) for the data and input checks and [application Stage 4](../../application/stages/stage4_selective_prediction/RUNTIME_TRIAGE.md) for how the selected rule is applied in the dashboard.

## Follow the experiment

| Branch | What you will find |
| --- | --- |
| [Prepare uncertainty scores](experiments/prepare_uncertainty_scores/PREPARE_UNCERTAINTY_SCORES.md) | Shared scripts that turn masks and entropy into per-image scores for a given split. |
| [BUSI validation and selection](experiments/validation_selection/VALIDATION_SELECTION.md) | Validation inference, statistics, score selection, the frozen YAML and JSON, and aggregate validation results. |
| [Locked BUSI test](experiments/locked_busi_test/LOCKED_BUSI_TEST.md) | Held-out evaluation scripts and BUSI results using the fixed policy. |
| [External cohort tests](experiments/external_test/EXTERNAL_TEST.md) | External inference and evaluation scripts, per-cohort tables and cross-domain comparisons. |

Each branch explains its inputs, command, outputs, and the next dependency. The scripts share the scientific calculations in [src/](src/RESEARCH_SOURCE.md). The [tests/](tests/STUDY_TESTS.md) exercise numerical and protocol invariants. The [uncertainty_selective_evaluation/](uncertainty_selective_evaluation/EVALUATION_TOOLKIT.md) folder provides a standalone toolkit for evaluating other saved score tables. [Cross-phase results guide](experiments/RESULTS_GUIDE.md) · [Study-wide technical references](study_reference/STUDY_REFERENCE.md).

## From entropy map to review decision

The study compares global mean, positive-pixel mean, foreground mean, contour-band mean, high-entropy mean, foreground standard deviation, and the foreground-to-global ratio. Foreground mean, `U_fg`, excludes background pixels that can dilute uncertainty for small predictions. A predicted mask with no foreground has no foreground mean. The separate empty-mask branch uses whole-image mean entropy, `U_mean`.

The selected foreground cutoff is **0.064058324 nats**. The empty-mask branch uses **0.0001 nats** on the whole-image score. See the [frozen configuration guide](experiments/RESULTS_GUIDE.md#how-to-read-the-frozen-yaml) and [mathematical definitions](study_reference/THEORY_AND_METRICS.md#4-seven-aggregation-functions-operational-definitions-and-anticipated-failure-modes).

## Methodological sources

The study's model outputs come from MedCLIPSeg. Per-structure aggregation motivates testing foreground mean, spatial aggregation studies motivate comparing other spatial summaries, and selective prediction motivates the risk–coverage evaluation. Our contribution is the empirical comparison on fixed breast ultrasound outputs, selection on BUSI validation, and test of a frozen operating point across cohorts. See the [source-to-implementation evidence](study_reference/SCIENTIFIC_EVIDENCE_AND_CLAIMS.md) for how individual methods relate to prior work. [Koleilat et al.](https://arxiv.org/abs/2602.20423) · [Roy et al.](https://doi.org/10.1016/j.neuroimage.2019.03.042) · [Guarino et al.](https://openaccess.thecvf.com/content/CVPR2026/html/Guarino_Better_than_Average_Spatially-Aware_Aggregation_of_Segmentation_Uncertainty_Improves_Downstream_CVPR_2026_paper.html) · [Geifman and El-Yaniv](https://arxiv.org/abs/1705.08500).

## Validation and test design

| Step | Cases and comparisons | Purpose |
| --- | --- | --- |
| Input checks | All 620 prepared images across one validation and five test splits. | Check file and table identity, model-grid alignment, reference fields and split membership. [Input audit](../stage_1_study_inputs/STUDY_INPUTS.md#what-was-validated-on-the-inputs). |
| Validation selection | One BUSI split of 78 images. Compare seven scores on its 65 nonempty predictions and inspect the 13 empty predictions separately. | Select the primary method using Dice-loss AURC and paired comparisons. Compare nominal coverage targets of 95%, 90%, 85%, 80%, 75% and 70%. The 80% target retained 52/65 cases. [Validation tables](experiments/RESULTS_GUIDE.md#what-to-look-at). |
| Locked BUSI test | 78 held-out images, including 65 nonempty predictions in the main analysis. | Measure in-domain coverage and retained quality with the cutoff fixed. Use the full split for empty-mask policy comparisons. [Primary BUSI result](experiments/locked_busi_test/results/primary_operating_point_test.csv). |
| External test | UDIAT, BUID, BUSUC and BUSBRA, totaling 464 images. | Measure transfer with the same checkpoint, score and cutoff. Report per-cohort, pooled and unweighted summaries. [Cross-domain table](experiments/external_test/results/cross_domain_summary.csv). |
| Software verification | 31 automated regression checks. | Exercise imports, config loading, entropy bounds, aggregation edge cases, threshold reproducibility and external-cohort results. [Test runner](tests/run_tests.sh). |

Foreground mean had the lowest Dice-loss AURC on validation at **0.0839**. The value for foreground standard deviation was **0.0850**, with the paired interval for the difference crossing zero. The frozen cutoff retained **47/65** BUSI test cases at mean Dice **0.9031** and **238/464** external cases at mean Dice **0.8718**. A poor BUSI prediction still passed the rule. Retained Dice describes the selected subset and should be read alongside its coverage. [Results and interpretation](experiments/RESULTS_GUIDE.md).

## Go deeper

| If you want to understand… | Read next |
| --- | --- |
| Why these settings were chosen | [Experimental rationale](study_reference/EXPERIMENTAL_RATIONALE.md) for cohort scope, compute, image processing and policy choices. |
| How the scores and metrics are calculated | [Theory and metrics](study_reference/THEORY_AND_METRICS.md) for entropy, seven aggregators, Dice, surface Dice, risk–coverage and bootstrap. |
| How the selection and tests were run | [Experimental protocol](study_reference/EXPERIMENTAL_PROTOCOL.md) for frozen inputs, validation-only selection, locked tests and reproducibility. |
| Which observations support each claim | [Scientific evidence and limits](study_reference/SCIENTIFIC_EVIDENCE_AND_CLAIMS.md) for comparisons, counterexamples and source attribution. |
| How a decision evolved | [Research decisions and limitations](study_reference/RESEARCH_DECISIONS_AND_LIMITATIONS.md) for the historical decisions and unresolved limits. |
| Where the tables and visual evidence are | [Results and frozen policy](experiments/RESULTS_GUIDE.md) for CSVs and figures, [upstream baseline comparison](study_reference/PUBLISHED_BASELINE_COMPARISON.md) for the frozen model, and the [benchmarking notebook](../notebooks/02_scientific_benchmarking.ipynb) for the walkthrough. |

## Reproduction

### Inspect the archived study

The [benchmarking notebook](../notebooks/02_scientific_benchmarking.ipynb) redraws the bundled aggregate results. See the [notebook dependencies](../notebooks/NOTEBOOK_GUIDE.md#scientific-benchmarking-notebook). This path requires no inference or dataset download.

### Regenerate the study from prepared images

Use Python 3.9 or newer, Bash, the [research requirements](../../requirements.txt), the prepared BUSI, UDIAT, BUID, BUSUC and BUSBRA partitions, and the released BUSI MedCLIPSeg checkpoint. The dataset tree contains PNG images, reference masks and lesion-description workbooks. Obtain the data under the terms of each provider. [Cohort sources](../stage_1_study_inputs/STUDY_INPUTS.md#dataset-sources) · [Dataset layout and model settings](../../shared/configs/DATASET_CONFIGURATIONS.md) · [Checkpoint and prompt protocol](study_reference/EXPERIMENTAL_PROTOCOL.md).

From the repository root, configure a local output folder and a supported device. The study used Apple Silicon for inference. CPU is a slower fallback. Inference values can vary across devices and dependency versions. The saved-score path isolates the statistical analysis from those differences.

```bash
python3 -m venv .venv
.venv/bin/python -m pip install -r requirements.txt
export MEDCLIPSEG_DATA="/absolute/path/to/prepared/cohorts"
export MEDCLIPSEG_OUTPUT="/absolute/path/to/experiment/outputs"
export MEDCLIPSEG_DEVICE=cpu  # or mps or cuda
export HF_HUB_OFFLINE=0 TRANSFORMERS_OFFLINE=0  # first initialization if needed
bash shared/scripts/download_checkpoint.sh BUSI 666 100
bash research/stage_4_selective_prediction/reproduce_study.sh
```

If you already have the checkpoint, place it at `$MEDCLIPSEG_OUTPUT/BUSI/trained_models/seed666/MedCLIPSeg_unimedclip_ViT-B-16_latest.pth`. The [study launcher](reproduce_study.sh) runs the original phase commands in validation, held-out BUSI and external test order. It uses the selected validation policy for both test branches and writes regenerated results into the local output tree.

### Recalculate from per-image score tables

This path uses the same Python environment. It skips segmentation inference and the illustrative figure that needs raw validation images and maps. Set `MEDCLIPSEG_OUTPUT` to a directory containing these six local input files:

- `BUSI/validation/seed666/aggregation_results/uncertainty_scores_BUSI_val_seed666.csv`
- `<cohort>/aggregation_results/seed666/uncertainty_scores_<cohort>_test_seed666.csv` for BUSI, UDIAT, BUID, BUSUC and BUSBRA

```bash
bash research/stage_4_selective_prediction/reproduce_study.sh --from-scores
```

The [benchmarking notebook](../notebooks/02_scientific_benchmarking.ipynb) can also start either run. For the individual commands and expected outputs, follow [validation and selection](experiments/validation_selection/VALIDATION_SELECTION.md), [locked BUSI test](experiments/locked_busi_test/LOCKED_BUSI_TEST.md) and [external cohort test](experiments/external_test/EXTERNAL_TEST.md).

Run `bash research/stage_4_selective_prediction/tests/run_tests.sh` for the regression suite. The [standalone evaluation toolkit](uncertainty_selective_evaluation/EVALUATION_TOOLKIT.md) can also evaluate an existing score table from another collection.
