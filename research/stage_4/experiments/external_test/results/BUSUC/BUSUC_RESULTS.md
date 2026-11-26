# BUSUC results

Location `research/stage_4/experiments/external_test/results/BUSUC/` · [Up to results/](../CROSS_DOMAIN_RESULTS.md)

These records cover the 122 images in the prepared BUSUC test split. The frozen foreground cutoff retained 78/122 predictions.

| File | Use it to inspect |
| --- | --- |
| [primary_operating_point_BUSUC.csv](primary_operating_point_BUSUC.csv) | Coverage and segmentation quality at the primary frozen cutoff. |
| [operating_points_BUSUC.csv](operating_points_BUSUC.csv) | Coverage and retained quality at the other validation-derived thresholds. |
| [method_comparison_BUSUC.csv](method_comparison_BUSUC.csv) | How the scores and controls compare on ranking and retained quality. |
| [delta_aurc_BUSUC.csv](delta_aurc_BUSUC.csv) | Paired AURC differences and their bootstrap intervals. |
| [summary_evaluation_BUSUC.json](summary_evaluation_BUSUC.json) | A structured record of baseline statistics and evaluation results. |

[Cross-domain comparison](../CROSS_DOMAIN_RESULTS.md) · [Dataset sources](../../../../../stage_1/STUDY_INPUTS.md#dataset-sources)
