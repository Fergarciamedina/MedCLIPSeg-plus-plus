# Stage 4 results

These aggregate artifacts come from the saved seed-666 evaluation runs. The subdirectories mirror their paths within the experiment output tree. The tables report the study's splits and thresholds and provide the data behind the [benchmarking notebook](../../notebooks/02_scientific_benchmarking.ipynb).

| Run | Saved evidence |
| --- | --- |
| BUSI validation | [Seven-method selection](BUSI/validation/seed666/selective_prediction/tables/validation_method_comparison_N65.csv), [paired bootstrap](BUSI/validation/seed666/selective_prediction/tables/paired_bootstrap_aurc_ci.csv), [method and policy comparison figure](BUSI/validation/seed666/selective_prediction/figures/step9_method_and_policy_comparison.png), [risk–coverage curves](BUSI/validation/seed666/statistical_analysis/figures/figure1_risk_coverage_curves.png), [frozen policy YAML](BUSI/validation/seed666/selective_prediction/frozen_selective_prediction_config.yaml) |
| Locked BUSI test | [Primary operating point](BUSI/test_evaluation/seed666/tables/primary_operating_point_test.csv), [method comparison](BUSI/test_evaluation/seed666/tables/method_comparison_test_n65.csv), [pathology breakdown](BUSI/test_evaluation/seed666/tables/subgroup_pathology_breakdown_test.csv), [empty-mask policies](BUSI/test_evaluation/seed666/tables/setting_b_deployment_policies_test.csv) |
| External test | [UDIAT](UDIAT/test_evaluation/seed666), [BUID](BUID/test_evaluation/seed666), [BUSUC](BUSUC/test_evaluation/seed666), [BUSBRA](BUSBRA/test_evaluation/seed666) |
| Cross-domain comparison | [Cohorts and pooled results](cross_domain_evaluation/seed666/cross_domain_summary.csv), [risk–coverage curves](cross_domain_evaluation/seed666/cross_domain_risk_coverage_overlay.png) |

The [protocol](../EXPERIMENTAL_PROTOCOL.md) documents the prediction inputs and the conditions under which individual scan analysis can be reproduced from locally prepared data.
