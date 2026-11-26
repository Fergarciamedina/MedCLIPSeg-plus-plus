# Held-out BUSI results

Location `research/stage_4_selective_prediction/experiments/locked_busi_test/results/` · [Up to locked_busi_test/](../LOCKED_BUSI_TEST.md)

These outputs apply the validation-derived policy to the held-out BUSI cohort. The [test phase](../LOCKED_BUSI_TEST.md) documents its inputs and commands.

| Artifact | What it answers |
| --- | --- |
| [primary_operating_point_test.csv](primary_operating_point_test.csv) | Retained case count, realized coverage and Dice and NSD at the primary frozen cutoff. |
| [operating_points_profile_test.csv](operating_points_profile_test.csv) · [busi_test_operating_profile.png](busi_test_operating_profile.png) | What coverage and quality the other validation-derived cutoffs achieve on test. |
| [method_comparison_test_n65.csv](method_comparison_test_n65.csv) · [busi_test_risk_coverage_curves.png](busi_test_risk_coverage_curves.png) | How the uncertainty scores and controls rank the nonempty predictions. |
| [paired_bootstrap_delta_aurc_test.csv](paired_bootstrap_delta_aurc_test.csv) | Paired AURC differences and confidence intervals. |
| [setting_b_deployment_policies_test.csv](setting_b_deployment_policies_test.csv) | Full-cohort comparison of policies for handling empty predictions. |
| [subgroup_pathology_breakdown_test.csv](subgroup_pathology_breakdown_test.csv) | Coverage and quality by the available pathology labels. |
| [lesion_area_confounding_test.csv](lesion_area_confounding_test.csv) | Association between score rankings and predicted lesion size. |

[Compare phases and interpret the metrics](../../RESULTS_GUIDE.md)
