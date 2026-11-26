# Stage 4 · Apply the uncertainty policy

Location `application/stages/stage4_selective_prediction/` · [Up to stages/](../APPLICATION_STAGES.md)

The [application triage service](../../uncertainty_aware_retrieval/uncertainty_aware_retrieval/selective_prediction.py) takes the saved scores and the predicted foreground count. If a custom case supplies raw entropy, the [gallery loader](../../uncertainty_aware_retrieval/uncertainty_aware_retrieval/manifest_builder.py) can compute missing scores using the aligned binary mask.

| Case | Score and setting | Runtime outcome |
| --- | --- | --- |
| Predicted mask has foreground | Mean entropy over the predicted foreground, `U_fg ≤ 0.064058324` nats. | At or below the cutoff the case is accepted by the policy. A larger value flags it for review. |
| Predicted mask is empty | Mean entropy over the entire model grid, `U_mean ≤ 0.0001` nats. | Apply the same pass or review comparison with the separate empty-mask cutoff. |
| No matching stored result | The score is unavailable. | Return **Not Evaluated**. |

The rule returns the decision, the score and cutoff used, and a reason. An accepted case becomes eligible for [Stage 5](../stage5_similarity_retrieval/SIMILARITY_SEARCH.md) when the request includes retrieval. The decision is a quality-screening result. Low entropy can occur in a poor prediction or a missed lesion.

Foreground averaging has a precedent in structure-wise segmentation quality estimation. Selective prediction provides the risk–coverage framework for deciding which predictions to retain. The two numeric cutoffs come from this project's frozen study policy. The foreground threshold has validation-derived provenance. The empty-mask threshold is a separate engineering rule evaluated on the available empty predictions. [Roy et al., *Bayesian QuickNAT*](https://doi.org/10.1016/j.neuroimage.2019.03.042) · [Geifman and El-Yaniv, *Selective Classification for Deep Neural Networks*](https://arxiv.org/abs/1705.08500) · [Local policy values](../../uncertainty_aware_retrieval/uncertainty_aware_retrieval/config.py).

The [six service tests](../../uncertainty_aware_retrieval/tests/test_selective_prediction.py) cover the foreground and empty-mask branches, unknown scans, and the demo-case counts. The [research Stage 4 guide](../../../research/stage_4_selective_prediction/SELECTIVE_PREDICTION_STUDY.md#validation-and-test-design) explains how the cutoff was selected and tested. The [archived policy](../../../research/stage_4_selective_prediction/experiments/RESULTS_GUIDE.md#how-to-read-the-frozen-yaml) records the frozen research settings.

## Notebook files in this folder

| File | Role |
| --- | --- |
| [decision_policy.py](decision_policy.py) | Maps scan-level scores and predicted foreground presence to a review decision for the walkthrough. |
| [uncertainty_aggregators.py](uncertainty_aggregators.py) | Derives scan-level mean, foreground, boundary and related scores from notebook masks and entropy. |
| [__init__.py](__init__.py) | Exports the decision and scoring interfaces. |

[Application stage map](../APPLICATION_STAGES.md) · [Next · Similarity retrieval](../stage5_similarity_retrieval/SIMILARITY_SEARCH.md)
