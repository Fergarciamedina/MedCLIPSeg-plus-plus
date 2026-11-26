# BUSI validation results

Location `research/stage_4/experiments/validation_selection/results/` · [Up to validation_selection/](../VALIDATION_SELECTION.md)

These artifacts come from one BUSI validation split. Use the 65 nonempty predictions to compare scores and inspect all 78 predictions for empty-mask handling. The [validation phase](../VALIDATION_SELECTION.md) explains the commands and the [frozen YAML](../frozen_selective_prediction_config.yaml) records the selected policy.

| Artifact | What it answers |
| --- | --- |
| [validation_method_comparison_N65.csv](validation_method_comparison_N65.csv) · [step9_method_and_policy_comparison.png](step9_method_and_policy_comparison.png) | How the seven scores compare at fixed coverage levels and across the risk–coverage curve. |
| [calibrated_abstention_thresholds.csv](calibrated_abstention_thresholds.csv) | Which cutoff each validation coverage target yields. |
| [paired_bootstrap_aurc_ci.csv](paired_bootstrap_aurc_ci.csv) | How large and uncertain the paired AURC differences are. |
| [empty_mask_policy_evaluation.csv](empty_mask_policy_evaluation.csv) | How the proposed deployment policies handle empty predictions. |
| [aurc_summary_validation.csv](aurc_summary_validation.csv) · [figure1_risk_coverage_curves.png](figure1_risk_coverage_curves.png) | How retained loss changes with coverage for each score and for the controls. |
| [spearman_validation_65_lesion.csv](spearman_validation_65_lesion.csv) · [spearman_validation_78_all.csv](spearman_validation_78_all.csv) · [figure2_spearman_forest_plot.png](figure2_spearman_forest_plot.png) | Score–quality rank associations, their uncertainty and the number of valid scores for each analysis. |
| [failure_auroc_validation.csv](failure_auroc_validation.csv) | Exploratory AUROC for identifying low-Dice or low-NSD predictions. |
| [lesion_area_correlation.csv](lesion_area_correlation.csv) · [figure3_lesion_area_bias.png](figure3_lesion_area_bias.png) | Association between the score and predicted foreground area. |

[Compare phases and interpret the metrics](../../RESULTS_GUIDE.md)
