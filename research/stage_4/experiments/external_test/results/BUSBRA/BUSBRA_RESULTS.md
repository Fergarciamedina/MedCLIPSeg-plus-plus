# BUSBRA results

Location `research/stage_4/experiments/external_test/results/BUSBRA/` · [Up to results/](../CROSS_DOMAIN_RESULTS.md)

These records cover the 282 images in the prepared BUSBRA test split. The frozen foreground cutoff retained 126/282 predictions.

| File | Use it to inspect |
| --- | --- |
| [primary_operating_point_BUSBRA.csv](primary_operating_point_BUSBRA.csv) | Coverage and segmentation quality at the primary frozen cutoff. |
| [operating_points_BUSBRA.csv](operating_points_BUSBRA.csv) | Coverage and retained quality at the other validation-derived thresholds. |
| [method_comparison_BUSBRA.csv](method_comparison_BUSBRA.csv) | How the scores and controls compare on ranking and retained quality. |
| [delta_aurc_BUSBRA.csv](delta_aurc_BUSBRA.csv) | Paired AURC differences and their bootstrap intervals. |
| [summary_evaluation_BUSBRA.json](summary_evaluation_BUSBRA.json) | A structured record of baseline statistics and evaluation results. |

[Cross-domain comparison](../CROSS_DOMAIN_RESULTS.md) · [Dataset sources](../../../../../stage_1/STUDY_INPUTS.md#dataset-sources)
