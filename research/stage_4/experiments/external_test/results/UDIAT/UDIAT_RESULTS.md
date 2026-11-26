# UDIAT results

Location `research/stage_4/experiments/external_test/results/UDIAT/` · [Up to results/](../CROSS_DOMAIN_RESULTS.md)

These records cover the 25 images in the prepared UDIAT test split. The frozen foreground cutoff retained 14/25 predictions.

| File | Use it to inspect |
| --- | --- |
| [primary_operating_point_UDIAT.csv](primary_operating_point_UDIAT.csv) | Coverage and segmentation quality at the primary frozen cutoff. |
| [operating_points_UDIAT.csv](operating_points_UDIAT.csv) | Coverage and retained quality at the other validation-derived thresholds. |
| [method_comparison_UDIAT.csv](method_comparison_UDIAT.csv) | How the scores and controls compare on ranking and retained quality. |
| [delta_aurc_UDIAT.csv](delta_aurc_UDIAT.csv) | Paired AURC differences and their bootstrap intervals. |
| [summary_evaluation_UDIAT.json](summary_evaluation_UDIAT.json) | A structured record of baseline statistics and evaluation results. |

[Cross-domain comparison](../CROSS_DOMAIN_RESULTS.md) · [Dataset sources](../../../../../stage_1/STUDY_INPUTS.md#dataset-sources)
