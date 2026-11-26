# Selective prediction for segmentation

Stage 4 uses uncertainty to decide which breast ultrasound segmentations to retain for downstream use and which to flag for review. It operates on the outputs of a frozen MedCLIPSeg model. This guide follows the decision for one scan. The [Stage 4 study](../../research/stage_4/SELECTIVE_PREDICTION_STUDY.md) covers how the rule was selected and evaluated.

## Inputs and decision

MedCLIPSeg produces a binary lesion mask and a dense map of predictive entropy in nats. The aggregation runs on the aligned model grid. Display resizing and color mapping are used for viewing.

| Available output | Decision rule |
| --- | --- |
| Nonempty predicted mask | Compute `U_fg`, the mean entropy inside the predicted mask. Accept at or below **0.064058324 nats**. Flag larger scores for review. |
| Empty predicted mask | Compute `U_mean`, the mean entropy over the entire image. The separate empty-mask branch accepts at or below **0.0001 nats**. |
| No matching precomputed result in the dashboard | Report **Not Evaluated**. |

The mask is predicted by the backbone. Reference annotations are used offline to measure Dice and boundary agreement. For the exact definitions of all seven scores, see the [theory guide](../../research/stage_4/study_reference/THEORY_AND_METRICS.md). For syntax and units of the two thresholds, see the [frozen policy guide](../../research/stage_4/experiments/RESULTS_GUIDE.md#how-to-read-the-frozen-yaml).

## How the study evaluates the decision

The experiment uses BUSI validation to compare seven scores and calibrate the cutoff. The locked BUSI test and four external cohorts measure how much of each cohort passes the same cutoff and how well the retained masks agree with the reference annotations. A risk–coverage curve plots mean segmentation loss as the retained fraction changes. A higher mean Dice for the retained cases describes a selected subset of the original predictions.

Low uncertainty can accompany an incorrect prediction. One poor BUSI segmentation passed the cutoff. The external cohorts did not include predicted-empty cases, which limits what they can tell us about missed lesions under the empty-mask branch. [Cross-domain results](../../research/stage_4/experiments/RESULTS_GUIDE.md) · [Experimental rationale](../../research/stage_4/study_reference/EXPERIMENTAL_RATIONALE.md).

## Figures

| Figure | What it shows |
| --- | --- |
| [Segmentation architecture](figures/block_diagram/selective_segmentation_architecture.pdf) | Frozen backbone, mask and entropy outputs, and image-level aggregation. |
| [Selective decision flow](figures/flowcharts/selective_decision_flowchart.pdf) | The foreground and empty-mask branches and their review decisions. |
| [Experimental validation framework](figures/framework/experimental_validation_framework.pdf) | Validation selection followed by locked BUSI and external testing. |

The [editable source files](figures) sit alongside the exported figures. The [experimental protocol](../../research/stage_4/study_reference/EXPERIMENTAL_PROTOCOL.md) and [current application guide](../../application/APPLICATION_GUIDE.md) describe the settings of the evaluation and running software.
