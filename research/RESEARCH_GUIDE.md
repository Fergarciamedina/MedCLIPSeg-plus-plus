# Research guide

Location `research/` · [Up to repository root](../README.md)

**Can predictive uncertainty tell us which breast ultrasound segmentations should be reviewed?**

[Study inputs](stage_1_study_inputs/STUDY_INPUTS.md) · [Stage 4 method](stage_4_selective_prediction/SELECTIVE_PREDICTION_STUDY.md) · [Results and frozen policy](stage_4_selective_prediction/experiments/RESULTS_GUIDE.md) · [Notebooks](#notebooks) · [Reproduce the analysis](stage_4_selective_prediction/SELECTIVE_PREDICTION_STUDY.md#reproduction)

Follow [From Ultrasound to Mask Review](../docs/learning/LEARNING_GUIDE.md) for a visual introduction to the model, the seven uncertainty scores, and the evaluation. The same synthetic image carries through to the dashboard examples.

## Overview

A binary segmentation mask identifies the pixels a model predicts as lesion. MedCLIPSeg also produces a predictive entropy map that records how uncertain the model is at each pixel. MedCLIPSeg++ studies whether that map can support a practical scan-level decision to retain the predicted mask or flag it for review. Stage 4 evaluates this decision using saved outputs from one frozen segmentation backbone.

The study starts with [prepared breast ultrasound data](stage_1_study_inputs/STUDY_INPUTS.md). It uses per-image lesion descriptions from the upstream dataset workflow to generate the frozen predictions. Stage 4 consumes the saved masks and entropy maps. Reference masks supply Dice and normalized surface Dice for evaluation. The text that a web user enters is a separate [application input](../application/stages/stage2_intent_router/INTENT_ROUTER.md) that chooses an operation such as retrieving similar cases.

## Method and validation

1. **Prepare six data splits.** Check BUSI validation, BUSI held-out test, and the four external test partitions. The study inputs contain 78 validation images and 542 test images. See [Stage 1](stage_1_study_inputs/STUDY_INPUTS.md) for exactly which assets, checks and denominators were used.
2. **Generate one set of frozen predictions.** Reuse MedCLIPSeg with the UniMedCLIP ViT-B/16 configuration. The same binary mask and float32 entropy map for each image are passed to every score. See [Stage 3](stage_3_medclipseg/FROZEN_SEGMENTATION.md) for settings and upstream credit.
3. **Compare seven scores on one validation split.** The primary comparison uses the 65 BUSI validation scans with nonempty predicted masks. Compare the ranking by area under the risk–coverage curve (AURC) for Dice loss. Evaluate six nominal retention targets and freeze the 80% target. It retains **52/65** validation cases at `U_fg ≤ 0.064058324` nats. BUSI's 13 empty predictions also support an evaluation of separate empty-mask policies. See [Stage 4](stage_4_selective_prediction/SELECTIVE_PREDICTION_STUDY.md) for the complete comparison and selection process.
4. **Test the frozen rule.** Apply the same score and cutoff in one held-out BUSI test and four external test splits. The study tests whether the rule transfers with useful coverage and retained quality. It also reports the cases for which it fails. See the [test result tables](stage_4_selective_prediction/experiments/RESULTS_GUIDE.md).

## Results

On BUSI validation, the mean entropy inside the predicted mask had Dice-loss AURC **0.0839**. The whole-image mean had **0.1438**, and the foreground standard deviation had **0.0850**. Lower values indicate lower loss across the coverage range studied. The paired interval for foreground mean versus foreground standard deviation crossed zero. [Validation evidence](stage_4_selective_prediction/experiments/RESULTS_GUIDE.md#what-to-look-at).

| Test cohort | Cases retained | Mean Dice before selection | Mean Dice of retained cases |
| --- | ---: | ---: | ---: |
| BUSI, primary nonempty-mask analysis | 47/65 (72.3%) | 0.8286 | 0.9031 |
| UDIAT | 14/25 (56.0%) | 0.7467 | 0.8747 |
| BUID | 20/35 (57.1%) | 0.7900 | 0.8818 |
| BUSUC | 78/122 (63.9%) | 0.8430 | 0.8982 |
| BUSBRA | 126/282 (44.7%) | 0.7511 | 0.8536 |
| **Four external cohorts pooled** | **238/464 (51.3%)** | **0.7780** | **0.8718** |

The increase in mean Dice describes the retained predictions, which are compared against the full cohort before selection. The lower external coverage shows how review workload changed when the BUSI cutoff was transferred. A low-quality BUSI mask passed the cutoff. The offline model descriptions include diagnosis-related terms, so performance under diagnosis-blind prompting remains open. [Cross-domain results](stage_4_selective_prediction/experiments/external_test/results/cross_domain_summary.csv) · [Evidence and limits](stage_4_selective_prediction/study_reference/SCIENTIFIC_EVIDENCE_AND_CLAIMS.md).

## Explore the six stages

The research folders use `stage_<number>_<purpose>` so each name identifies its place in the architecture and what its guide covers.

| Stage | What to read here | Where to see it run |
| --- | --- | --- |
| [1 · Study inputs](stage_1_study_inputs/STUDY_INPUTS.md) | Datasets, split counts, reference annotations, text provenance and input integrity checks. | [Application input contract](../application/stages/stage1_inputs/APPLICATION_INPUTS.md) |
| [2 · Intent routing](stage_2_intent_routing/INTENT_ROUTING_IN_CONTEXT.md) | Where the reused router fits and how application instructions differ from study inputs. | [Routing configuration and checks](../application/stages/stage2_intent_router/INTENT_ROUTER.md) |
| [3 · Segmentation artifacts](stage_3_medclipseg/FROZEN_SEGMENTATION.md) | Upstream backbone, frozen model settings and the two outputs Stage 4 receives. | [Runtime lookup and offline preparation](../application/stages/stage3_segmentation_backbone/SEGMENTATION_ARTIFACTS.md) |
| [4 · Selective prediction](stage_4_selective_prediction/SELECTIVE_PREDICTION_STUDY.md) | Scan-level score selection, thresholds, locked tests, external results and a guide to the detailed protocol and evidence. | [Runtime decision rules](../application/stages/stage4_selective_prediction/RUNTIME_TRIAGE.md) |
| [5 · Similarity retrieval](stage_5_similarity_retrieval/RETRIEVAL_IN_CONTEXT.md) | PABP source, how eligibility is passed to retrieval and the scope of its verification. | [Search settings and checks](../application/stages/stage5_similarity_retrieval/SIMILARITY_SEARCH.md) |
| [6 · Outputs](stage_6_study_outputs/STUDY_OUTPUTS.md) | Research outputs and how application outputs use the same study decision. | [Dashboard output contract](../application/stages/stage6_system_output/DASHBOARD_OUTPUTS.md) |

The experimental rationale explains the choices behind the cohort sizes, hardware settings, image processing and statistics. The project attribution guide traces the original works and publications for all six stages. [Experimental rationale](stage_4_selective_prediction/study_reference/EXPERIMENTAL_RATIONALE.md) · [Sources by stage](../ATTRIBUTION.md#sources-by-stage) · [System architecture](../docs/architecture/SYSTEM_ARCHITECTURE.md).

## Follow the Stage 4 experiment

[Prepare per-image scores](stage_4_selective_prediction/experiments/prepare_uncertainty_scores/PREPARE_UNCERTAINTY_SCORES.md) → [Select and freeze on BUSI validation](stage_4_selective_prediction/experiments/validation_selection/VALIDATION_SELECTION.md) → [Test on held-out BUSI](stage_4_selective_prediction/experiments/locked_busi_test/LOCKED_BUSI_TEST.md) → [Evaluate four external cohorts](stage_4_selective_prediction/experiments/external_test/EXTERNAL_TEST.md).

The phase guides contain the scripts and link the evidence made in that phase. The [Stage 4 overview](stage_4_selective_prediction/SELECTIVE_PREDICTION_STUDY.md) also links the study-wide protocol, theory, decisions and evidence.

## Notebooks

The [notebook folder guide](notebooks/NOTEBOOK_GUIDE.md) lists each notebook's inputs, dependencies and expected outputs.

- [Scientific benchmarking](notebooks/02_scientific_benchmarking.ipynb) reads the bundled aggregate results. Optional local per-image artifacts enable qualitative examples and additional plots.
- [Six-stage software walkthrough](notebooks/01_complete_system_pipeline.ipynb) uses synthetic segmentation outputs so you can inspect the interfaces. It accepts a local folder of PNG or JPEG inputs.

See the [notebook instructions](notebooks/NOTEBOOK_GUIDE.md) for the separate environment, launch command and optional data inputs for each notebook.
