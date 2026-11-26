# Comparison with the upstream MedCLIPSeg baseline

Koleilat and colleagues report whole-cohort segmentation Dice for five breast ultrasound test sets in Table 2 of their MedCLIPSeg paper. The comparison below checks whether the frozen-model outputs used by MedCLIPSeg++ have similar whole-cohort Dice before Stage 4 selects a subset. [MedCLIPSeg, CVPR 2026](https://openaccess.thecvf.com/content/CVPR2026/html/Koleilat_MedCLIPSeg_Probabilistic_Vision-Language_Adaptation_for_Data-Efficient_and_Generalizable_Medical_Image_CVPR_2026_paper.html) · [Upstream dataset splits](https://github.com/HealthX-Lab/MedCLIPSeg/blob/main/assets/DATASETS.md).

| Test split | Images | Dice reported by MedCLIPSeg | Dice in this repository before selection |
| --- | ---: | ---: | ---: |
| BUSI, including normal scans | 78 | 85.72% | 85.72% |
| BUSBRA | 282 | 75.06% | 75.11% |
| BUSUC | 122 | 84.37% | 84.30% |
| BUID | 35 | 78.99% | 79.00% |
| UDIAT | 25 | 74.64% | 74.67% |

The differences range from 0 to 0.07 percentage points in absolute value at the displayed precision. This supports close aggregate agreement. Exact reproduction would require matching the checkpoint and prompt artifacts and comparing predictions under a controlled software and hardware environment. The recorded runs do not isolate the effect of seed or compute backend on the differences. [Checkpoint, prompts and hardware](EXPERIMENTAL_RATIONALE.md).

## Compare the same denominator

The full BUSI test split includes 13 normal scans. The primary Stage 4 comparison uses the 65 nonempty predictions and therefore starts from a different mean Dice of **0.8286**. On that subset the frozen rule retained 47 cases with mean Dice **0.9031**. The [empty-mask policy comparison](../experiments/locked_busi_test/results/setting_b_deployment_policies_test.csv) gives the results for all 78 BUSI scans. The [cross-domain results](../experiments/external_test/results/cross_domain_summary.csv) give preselection and retained Dice for every external cohort.
