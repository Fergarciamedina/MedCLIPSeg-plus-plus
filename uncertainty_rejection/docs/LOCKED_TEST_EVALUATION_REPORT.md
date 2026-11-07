# ELEC 6661 — Stage 10: Locked BUSI Test Set Evaluation Report

**Date:** 2026-11-07  
**Evaluation Status:** LOCKED TEST EVALUATION (Pre-Registered Execution)  
**Primary Method:** `U_fg` (Foreground Mean Predictive Entropy)  
**Prespecified Comparator:** `U_std_fg` (Foreground Entropy Population Standard Deviation)  
**Primary Threshold:** `tau = 0.064058` (Validation 80% coverage calibration)  

---

## 1. Executive Summary & Locked Test Verdict

The frozen selective prediction strategy developed in Step 9 and independently audited in Step 9.5 was evaluated on the independent BUSI test cohort without modifying any threshold, hyperparameter, model weight, or post-processing rule.

### Key Quantitative Findings:
1. **Full Cohort Test Size:** Total $N=78$ images ($N=65$ lesion-present scans: 44 benign, 21 malignant; $N=13$ normal scans).
2. **Primary Operating Point ($	au = 0.064058$):**
   - **Realized Test Coverage:** `72.3%` (47/65 lesion scans accepted, 18 rejected).
   - **Accepted Mean DSC:** `0.9031` vs. Baseline `0.8286` (**Absolute Gain = +0.0745**, 95% CI: `[0.0349, 0.1235]`).
   - **Accepted Mean NSD:** `0.9349` vs. Baseline `0.8602` (**Absolute Gain = +0.0747**, 95% CI: `[0.0341, 0.1232]`).
   - **Rejected Cohort Mean DSC:** `0.6341` (efficient triage of low-quality segmentations).
3. **AURC Minimization:** `U_fg` achieved the lowest empirical AURC (1-DSC) among all candidate methods on test (`0.0790` vs Random `0.1687`, Gain = `+0.0898`).
4. **Failure Triage:** 4 out of the 5 worst segmentation failures in the test set (DSC $\le 0.45$) were correctly rejected by the primary threshold.

---

## 2. Primary Operating Point Results (Setting A: Lesion Cohort, N=65)

| Metric | Baseline (Full Cohort) | Accepted Cohort | Absolute Gain | 95% Bootstrap CI | Rejected Cohort |
| :--- | :---: | :---: | :---: | :---: | :---: |
| **DSC** | `0.8286` | `0.9031` | **+0.0745** | `[0.0349, 0.1235]` | `0.6341` |
| **NSD** | `0.8602` | `0.9349` | **+0.0747** | `[0.0341, 0.1232]` | `0.6651` |
| **Coverage** | 100.0% (65/65) | 72.3% (47/65) | - | - | 18/65 rejected |

---

## 3. Secondary Frozen Operating Points Profile (Setting A: N=65)

All thresholds were fixed strictly on validation data. No recalibration or tuning was performed on test data.

| Coverage Level | Target Cov | Frozen $\tau$ | Realized Cov | Accepted / Rejected | Accepted DSC | DSC Gain | 95% CI Gain | Rejected DSC | Accepted NSD | NSD Gain |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| `coverage_95` | 95% | 0.10193 | 90.8% | 59 / 6 | 0.8733 | +0.0447 | [0.0098, 0.0893] | 0.3891 | 0.9063 | +0.0461 |
| `coverage_90` | 90% | 0.08346 | 84.6% | 55 / 10 | 0.8883 | +0.0597 | [0.0201, 0.1077] | 0.5003 | 0.9217 | +0.0615 |
| `coverage_85` | 85% | 0.07270 | 72.3% | 47 / 18 | 0.9031 | +0.0745 | [0.0349, 0.1235] | 0.6341 | 0.9349 | +0.0747 |
| `coverage_80` **(Primary)** | 80% | 0.06406 | 72.3% | 47 / 18 | 0.9031 | +0.0745 | [0.0349, 0.1235] | 0.6341 | 0.9349 | +0.0747 |
| `coverage_75` | 75% | 0.05969 | 64.6% | 42 / 23 | 0.9197 | +0.0910 | [0.0451, 0.1421] | 0.6624 | 0.9522 | +0.0920 |
| `coverage_70` | 70% | 0.05621 | 61.5% | 40 / 25 | 0.9217 | +0.0930 | [0.0474, 0.1454] | 0.6798 | 0.9542 | +0.0940 |

---

## 4. Comprehensive Method Comparison on Test Cohort (N=65)

| Method | Type | Literature Citation | AURC (1-DSC) | Gain vs Random | DSC @ 90% | DSC @ 80% | DSC @ 70% |
| :--- | :--- | :--- | :---: | :---: | :---: | :---: | :---: |
| **Oracle (GT Ceiling)** | Reference Bound | Optimal Theoretical Bound | 0.0692 | +0.0995 | 0.8958 | 0.9103 | 0.9206 |
| **Random Rejection** | Uninformed Control | Monte Carlo Mean Baseline | 0.1687 | +0.0000 | 0.8286 | 0.8286 | 0.8286 |
| **U_fg** | Primary | Roy et al. (Bayesian QuickNAT 2019) | 0.0790 | +0.0898 | 0.8748 | 0.8939 | 0.9034 |
| **U_std_fg** | Comparator | Jungo et al. (2020) | 0.0793 | +0.0895 | 0.8734 | 0.8889 | 0.9034 |
| **U_mean** | Baseline | Kendall & Gal (2017), Lakshminarayanan (2017) | 0.1262 | +0.0426 | 0.8387 | 0.8324 | 0.8561 |
| **U_boundary** | Baseline | Wang et al. (TTA 2019), Mehrtash (2020) | 0.1113 | +0.0574 | 0.8512 | 0.8630 | 0.8837 |
| **U_threshold** | Baseline | Eaton-Rosen et al. (MICCAI 2018) | 0.2045 | -0.0358 | 0.8600 | 0.8579 | 0.8516 |
| **U_positive** | Baseline | Ablation control (positive entropy) | 0.1282 | +0.0405 | 0.8365 | 0.8324 | 0.8561 |
| **U_ratio** | Baseline | Ablation control (contrast ratio) | 0.1499 | +0.0188 | 0.8250 | 0.8183 | 0.8262 |

---

## 5. Paired Bootstrap Hypothesis Testing on $\Delta\text{AURC}$ (Test N=65)

Testing difference in AURC: $\Delta = \text{AURC}(A) - \text{AURC}(B)$. Negative $\Delta$ indicates Method A achieves lower risk.

| Comparison | Description | $\Delta\text{AURC}$ (DSC) | 95% Bootstrap CI | p-value | Statistically Significant? |
| :--- | :--- | :---: | :---: | :---: | :---: |
| `U_fg vs U_std_fg` | Primary vs Prespecified Comparator | -0.0003 | [-0.0024, +0.0014] | 0.786 | No (overlaps 0) |
| `U_fg vs U_mean` | Primary vs Global Full-Image Entropy | -0.0472 | [-0.0690, -0.0277] | 0.000 | Yes (p < 0.05) |
| `U_fg vs U_boundary` | Primary vs Morphological Boundary Band | -0.0324 | [-0.0440, -0.0198] | 0.000 | Yes (p < 0.05) |
| `U_fg vs U_threshold` | Primary vs High-Uncertainty Fraction | -0.1256 | [-0.1974, -0.0428] | 0.000 | Yes (p < 0.05) |
| `U_fg vs U_positive` | Primary vs Positive-Entropy Filter | -0.0493 | [-0.0721, -0.0293] | 0.000 | Yes (p < 0.05) |
| `U_fg vs U_ratio` | Primary vs Uncertainty Ratio | -0.0710 | [-0.1075, -0.0386] | 0.000 | Yes (p < 0.05) |
| `U_std_fg vs U_mean` | Comparator vs Global Full-Image Entropy | -0.0469 | [-0.0686, -0.0277] | 0.000 | Yes (p < 0.05) |
| `U_std_fg vs U_boundary` | Comparator vs Morphological Boundary Band | -0.0320 | [-0.0431, -0.0196] | 0.000 | Yes (p < 0.05) |

---

## 6. Setting B Evaluation: Full Clinical Screening Cohort (N=78)

| Deployment Policy | Total Scans | Realized Coverage | Accepted / Rejected | Accepted DSC | Baseline DSC | DSC Gain | Accepted NSD |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| `two_stage` | 78 | 76.9% | 60 / 18 | 0.9241 | 0.8572 | +0.0669 | 0.9490 |
| `drop_empty` | 65 | 72.3% | 47 / 18 | 0.9031 | 0.8286 | +0.0745 | 0.9349 |
| `global_fallback` | 78 | 76.9% | 60 / 18 | 0.9241 | 0.8572 | +0.0669 | 0.9490 |
| `conservative_abstention` | 78 | 60.3% | 47 / 31 | 0.9031 | 0.8572 | +0.0459 | 0.9349 |

---

## 7. Subgroup Breakdown & Pathological Analysis

| Pathology Subgroup | Test Samples | Baseline DSC | Realized Coverage | Accepted DSC | Absolute DSC Gain | Rejected DSC |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: |
| **Benign** | 44 | 0.8655 | 84.1% | 0.9247 | +0.0592 | 0.5526 |
| **Malignant** | 21 | 0.7514 | 47.6% | 0.8234 | +0.0720 | 0.6859 |

---

## 8. Severe Failure Case Triage Analysis (Bottom 5 Test DSC Cases)

| Image ID | Pathology | Lesion Area (px) | Test DSC | Test NSD | $U_{\mathrm{fg}}$ | Primary $\tau$ | Triage Decision |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| `116_benign.png` | benign | 1270 | 0.0000 | 0.0000 | 0.12920 | 0.06406 | **Rejected (Triaged to Expert)** |
| `169_malignant.png` | malignant | 4203 | 0.0877 | 0.1010 | 0.15480 | 0.06406 | **Rejected (Triaged to Expert)** |
| `116_malignant.png` | malignant | 3615 | 0.2193 | 0.2393 | 0.06279 | 0.06406 | Accepted (Failure Leak) |
| `299_benign.png` | benign | 2347 | 0.2297 | 0.2574 | 0.17911 | 0.06406 | **Rejected (Triaged to Expert)** |
| `198_malignant.png` | malignant | 6294 | 0.3539 | 0.3782 | 0.14096 | 0.06406 | **Rejected (Triaged to Expert)** |

---

## 9. Methodological Verification & Comparison with Validation

1. **Generalization of Primary Method:** `U_fg` generalized from validation to test without degradation. On validation, `U_fg` achieved AURC = `0.0839`; on test, it achieved AURC = `0.0790`.
2. **Area-Robustness Confirmed:** `U_fg` exhibited near-zero correlation with lesion area on test, confirming that it evaluates genuine model hesitation rather than tumor scale.
3. **Dispersion Comparator (`U_std_fg`):** Consistent with validation hypothesis testing, `U_std_fg` performed closely to `U_fg` and also significantly outperformed global `U_mean`.

---

## 10. Recommendations for Stage 11 (Out-of-Distribution & Multi-Center Benchmarks)

1. **Proceed with Frozen Specifications:** The primary method `U_fg` and operating point $\tau = 0.06406$ are locked and validated on the in-domain test set. They should be evaluated on external OOD datasets (e.g. DDTI, UDIAT) without adaptation.
2. **Preserve Setting A vs Setting B Distinction:** On OOD datasets with varying normal-to-abnormal prevalence, Setting A provides pure segmentation QC, while Setting B tests complete screening triage.
3. **Zero Further In-Domain Tuning:** In-domain BUSI weights, thresholds, and hyperparameters must remain permanently frozen.
