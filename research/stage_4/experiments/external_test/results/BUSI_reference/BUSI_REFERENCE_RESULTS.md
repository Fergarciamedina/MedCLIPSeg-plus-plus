# BUSI in-domain reference results

Location `research/stage_4/experiments/external_test/results/BUSI_reference/` · [Up to results/](../CROSS_DOMAIN_RESULTS.md)

These records cover the 65 nonempty BUSI test predictions processed by the cross-domain evaluator. See the [locked BUSI results](../../../locked_busi_test/results/LOCKED_BUSI_RESULTS.md) for the detailed held-out analysis and its full-cohort empty-mask comparison. The frozen foreground cutoff retained 47/65 predictions.

| File | Use it to inspect |
| --- | --- |
| [primary_operating_point_BUSI.csv](primary_operating_point_BUSI.csv) | Coverage and segmentation quality at the primary frozen cutoff. |
| [operating_points_BUSI.csv](operating_points_BUSI.csv) | Coverage and retained quality at the other validation-derived thresholds. |
| [method_comparison_BUSI.csv](method_comparison_BUSI.csv) | How the scores and controls compare on ranking and retained quality. |
| [delta_aurc_BUSI.csv](delta_aurc_BUSI.csv) | Paired AURC differences and their bootstrap intervals. |
| [summary_evaluation_BUSI.json](summary_evaluation_BUSI.json) | A structured record of baseline statistics and evaluation results. |

[Cross-domain comparison](../CROSS_DOMAIN_RESULTS.md) · [Dataset sources](../../../../../stage_1/STUDY_INPUTS.md#dataset-sources)
