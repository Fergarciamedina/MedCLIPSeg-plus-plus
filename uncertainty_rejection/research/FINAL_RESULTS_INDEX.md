# Master Results Index and Authoritative Scientific Truth Table

**Document:** `FINAL_RESULTS_INDEX.md`  
**Location:** `uncertainty_rejection/research/`  
**Date:** 2026-11-07  
**Status:** Canonical Authoritative Anchor (Step 12A.3)  
**Scope:** Single, definitive numerical repository anchoring all experimental claims, tables, confidence intervals, risk-coverage curves, and statistical comparisons across BUSI and the four external benchmarks.

---

## 1. Master Cross-Domain Benchmark Summary Table

All selective evaluation metrics correspond strictly to the frozen in-distribution BUSI validation primary threshold ($\tau = 0.064058324$) and frozen empty-mask fallback ($\tau_{\text{neg}} = 10^{-4}$). Zero model retraining or test recalibration was performed.

| Cohort / Dataset | Protocol & Setting | Total $N$ | Baseline DSC (Mean $\pm$ SD) | Baseline NSD (Mean $\pm$ SD) | Accepted $N$ (%) | Rejected $N$ (%) | Accepted DSC | Rejected DSC | Selective DSC Gain | 95% Bootstrap CI [Gain] | Accepted NSD | Selective NSD Gain | Primary $U_{\text{fg}}$ AURC | Baseline $U_{\text{mean}}$ AURC | Random AURC Reference | Empirical Oracle AURC |
| :--- | :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **BUSI Val** | ID Calibration (Setting A) | 65 | $0.8520 \pm 0.1843$ | $0.8872 \pm 0.1812$ | 52 (80.0%) | 13 (20.0%) | 0.9082 | 0.6272 | **+0.0562** | [+0.0210, +0.0991] | 0.9391 | +0.0519 | **0.0839** | 0.1438 | 0.1434 | 0.0612 |
| **BUSI Test (A)** | ID Lesion Cohort (Setting A) | 65 | $0.8286 \pm 0.2110$ | $0.8602 \pm 0.2036$ | 47 (72.3%) | 18 (27.7%) | 0.9031 | 0.6341 | **+0.0745** | [+0.0349, +0.1235] | 0.9349 | +0.0747 | **0.0790** | 0.1262 | 0.1687 | 0.0692 |
| **BUSI Test (B)** | ID Full Screening (Setting B) | 78 | $0.8572 \pm 0.2078$ | $0.8841 \pm 0.1983$ | 60 (76.9%) | 18 (23.1%) | 0.9241 | 0.6341 | **+0.0669** | [+0.0335, +0.1118] | 0.9490 | +0.0649 | — | — | — | — |
| **UDIAT** | External OOD (Spain) | 25 | $0.7467 \pm 0.2992$ | $0.7849 \pm 0.3015$ | 14 (56.0%) | 11 (44.0%) | 0.8747 | 0.5838 | **+0.1281** | [+0.0325, +0.2569] | 0.8988 | +0.1139 | **0.1133** | 0.1979 | 0.2432 | 0.0898 |
| **BUID** | External OOD (Academic) | 35 | $0.7900 \pm 0.2024$ | $0.8138 \pm 0.2087$ | 20 (57.1%) | 15 (42.9%) | 0.8818 | 0.6675 | **+0.0919** | [+0.0339, +0.1644] | 0.8966 | +0.0828 | **0.1094** | 0.1708 | 0.2040 | 0.0886 |
| **BUSUC** | External OOD (Multi-center) | 122 | $0.8430 \pm 0.1152$ | $0.8899 \pm 0.1078$ | 78 (63.9%) | 44 (36.1%) | 0.8982 | 0.7452 | **+0.0551** | [+0.0363, +0.0758] | 0.9385 | +0.0486 | **0.0916** | 0.1368 | 0.1557 | 0.0833 |
| **BUSBRA** | External OOD (Brazil) | 282 | $0.7511 \pm 0.1962$ | $0.7942 \pm 0.2081$ | 126 (44.7%) | 156 (55.3%) | 0.8536 | 0.6683 | **+0.1025** | [+0.0789, +0.1278] | 0.8936 | +0.0994 | **0.1529** | 0.2078 | 0.2480 | 0.1169 |
| **Pooled OOD** | Micro-Evaluation (All 4 Sites) | **464** | **$0.7780 \pm 0.1900$** | **$0.8203 \pm 0.1996$** | **238 (51.3%)** | **226 (48.7%)** | **0.8718** | **0.6791** | **+0.0939** | **[+0.0775, +0.1113]** | **0.9088** | **+0.0885** | **0.1267** | **0.1864** | **0.2216** | **0.1046** |
| *Macro OOD* | Unweighted Dataset Average | 4 Datasets | 0.7827 | 0.8207 | 55.4% | 44.6% | 0.8771 | 0.6662 | **+0.0944** | — | 0.9069 | +0.0862 | 0.1168 | 0.1825 | 0.2127 | 0.0978 |
| *Weighted OOD* | Sample-Weighted Dataset Avg | 464 Cases | 0.7780 | 0.8203 | 51.3% | 48.7% | 0.8718 | 0.6791 | **+0.0939** | — | 0.9088 | +0.0885 | 0.1314 | 0.1849 | 0.2202 | 0.1070 |

> [!IMPORTANT]
> **Reporting Distinctions:**
> 1. **Micro Pooled Evaluation ($N=464$):** Evaluates all 464 external scans as a unified clinical cohort. The case-level global AURC ($0.1267$) reflects global ranking across all external sites.
> 2. **Sample-Weighted Average AURC ($0.1314$):** Represents $\sum_{k} \frac{N_k}{N_{\text{total}}} \text{AURC}_k$, averaging dataset-internal rankings without cross-site score calibration.
> 3. **Unweighted Macro Average ($0.1168$):** Arithmetic mean of the four dataset-specific AURCs, giving equal weight to UDIAT ($N=25$) and BUSBRA ($N=282$).

---

## 2. In-Domain Uncertainty Aggregator Comparison (BUSI Validation vs. Locked Test)

Evaluation across all seven investigated uncertainty aggregation strategies on the lesion-only cohort ($N=65$):

| Aggregator Metric | Mathematical Formulation | BUSI Val Spearman $\rho$ (DSC) | BUSI Val Spearman $\rho$ (NSD) | BUSI Val AURC (1-DSC) | BUSI Test Spearman $\rho$ (DSC) | BUSI Test AURC (1-DSC) | Paired $\Delta\text{AURC}$ vs. $U_{\text{fg}}$ (Test) | $p$-value (Bootstrap) |
| :--- | :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **$U_{\text{fg}}$ (Primary)** | $\frac{1}{\|\Omega_{\text{fg}}\|} \sum_{i \in \Omega_{\text{fg}}} \mathcal{H}_i$ | **-0.7202** | **-0.7410** | **0.0839** | **-0.8741** | **0.0790** | **Reference** | — |
| **$U_{\text{std\_fg}}$ (Comparator)** | $\sqrt{\frac{1}{\|\Omega_{\text{fg}}\|} \sum_{i \in \Omega_{\text{fg}}} (\mathcal{H}_i - U_{\text{fg}})^2}$ | -0.7068 | -0.7285 | 0.0850 | -0.8694 | 0.0793 | -0.0003 | 0.786 (NS) |
| **$U_{\text{boundary}}$** | $\frac{1}{\|\mathcal{B}\|} \sum_{i \in \mathcal{B}} \mathcal{H}_i$ ($5 \times 5$ band) | -0.2273 | -0.4174 | 0.1302 | -0.4812 | 0.1113 | -0.0324 | $< 0.001$ |
| **$U_{\text{mean}}$ (Global)** | $\frac{1}{\|\Omega\|} \sum_{i \in \Omega} \mathcal{H}_i$ | -0.0451 | -0.0712 | 0.1438 | -0.2114 | 0.1262 | -0.0472 | $< 0.001$ |
| **$U_{\text{positive}}$** | $\frac{1}{\|\{i: \mathcal{H}_i > 0\}\|} \sum_{\mathcal{H}_i > 0} \mathcal{H}_i$ | -0.0398 | -0.0655 | 0.1455 | -0.1988 | 0.1282 | -0.0492 | $< 0.001$ |
| **$U_{\text{ratio}}$** | $U_{\text{fg}} / (U_{\text{mean}} + \epsilon)$ | +0.0812 | +0.0945 | 0.1368 | -0.1124 | 0.1499 | -0.0709 | $< 0.001$ |
| **$U_{\text{threshold}}$** | $\frac{1}{\|\{i: \mathcal{H}_i > 0.1\}\|} \sum_{\mathcal{H}_i > 0.1} \mathcal{H}_i$ | -0.1245 | -0.1582 | 0.1750 | -0.1845 | 0.2045 | -0.1255 | $< 0.001$ |
| *Random Selection* | Uninformative uniform ranking | $0.0000$ | $0.0000$ | 0.1434 | $0.0000$ | 0.1687 | -0.0897 | $< 0.001$ |
| *Empirical Oracle* | Perfect sorting by true error ($1 - \text{DSC}$) | -1.0000 | -1.0000 | 0.0612 | -1.0000 | 0.0692 | +0.0098 | $< 0.001$ |

---

## 3. Coverage-Target Operating Points (BUSI In-Domain Calibration)

Calibrated operating thresholds computed on BUSI validation lesion cohort ($N=65$) and their realized test performance:

| Target Coverage | Calibrated Threshold $\tau$ | Val Realized Coverage | Val Retained DSC | Val Retained NSD | Test Realized Coverage | Test Retained DSC | Test Retained NSD | Test Selective Gain |
| :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **95%** | 0.10648211 | 95.4% (62/65) | 0.8674 | 0.9015 | 93.8% (61/65) | 0.8521 | 0.8834 | +0.0235 |
| **90%** | 0.08831456 | 90.8% (59/65) | 0.8812 | 0.9142 | 87.7% (57/65) | 0.8698 | 0.9012 | +0.0412 |
| **85%** | 0.07542190 | 86.2% (56/65) | 0.8945 | 0.9268 | 80.0% (52/65) | 0.8874 | 0.9195 | +0.0588 |
| **80% (Primary)**| **0.06405832**| **80.0% (52/65)**| **0.9082** | **0.9391** | **72.3% (47/65)**| **0.9031** | **0.9349** | **+0.0745** |
| **75%** | 0.05718944 | 75.4% (49/65) | 0.9178 | 0.9472 | 67.7% (44/65) | 0.9142 | 0.9458 | +0.0856 |
| **70%** | 0.05102488 | 70.8% (46/65) | 0.9264 | 0.9548 | 61.5% (40/65) | 0.9258 | 0.9562 | +0.0972 |

---

## 4. Failure Mode Analysis and Falsification Cases

1. **Accepted Severe Failure (Under-Rejection Mode):**
   - **Case:** `116_malignant.png` (BUSI Test Split).
   - **Segmentation Performance:** $\text{DSC} = 0.2193$, $\text{NSD} = 0.2458$.
   - **Uncertainty Score:** $U_{\text{fg}} = 0.062791$ (falls strictly below rejection threshold $\tau = 0.064058324$).
   - **Mechanism:** The model generated a highly confident but false-positive extension into posterior acoustic shadowing; because the internal entropy across the incorrect segmentation was uniformly low, the image bypassed filtering.
   - **Scientific Impact:** Falsifies the hypothesis that $U_{\text{fg}}$ guarantees detection of every severe error.

2. **Rejected High-Quality Case (Over-Rejection Mode):**
   - **Case:** `042_benign.png` (BUSI Test Split).
   - **Segmentation Performance:** $\text{DSC} = 0.9312$, $\text{NSD} = 0.9614$.
   - **Uncertainty Score:** $U_{\text{fg}} = 0.078415$ (exceeds threshold, correctly rejected according to algorithm, but clinically wasteful).
   - **Mechanism:** Highly ambiguous contour boundary amidst dense fibroglandular parenchyma inflated edge entropy despite accurate core placement.

3. **Coverage Drift Under Domain Shift:**
   - While the threshold was calibrated to retain 80.0% of in-distribution validation cases, realized coverage dropped to **72.3%** on BUSI test, **56.0%** on UDIAT, **57.1%** on BUID, **63.9%** on BUSUC, and **44.7%** on BUSBRA (pooled external coverage: **51.3%**).
   - **Scientific Impact:** Demonstrates that deployment workloads shift under scanner distribution changes, requiring adaptive or re-centered thresholds if fixed referral capacities must be maintained.

---

## 5. Artifact Provenance, Data Paths, and Execution Traceability

All reported results are anchored in immutable CSV outputs generated by the frozen pipeline:

- **BUSI Validation Predictions & Metrics:** `uncertainty_rejection/results/validation_evaluation_results.csv`
- **BUSI Locked Test Predictions & Metrics:** `uncertainty_rejection/results/test_evaluation_results.csv`
- **UDIAT Evaluation Predictions & Metrics:** `uncertainty_rejection/results/ood/UDIAT_evaluation_results.csv`
- **BUID Evaluation Predictions & Metrics:** `uncertainty_rejection/results/ood/BUID_evaluation_results.csv`
- **BUSUC Evaluation Predictions & Metrics:** `uncertainty_rejection/results/ood/BUSUC_evaluation_results.csv`
- **BUSBRA Evaluation Predictions & Metrics:** `uncertainty_rejection/results/ood/BUSBRA_evaluation_results.csv`
- **Aggregated OOD Master Table:** `uncertainty_rejection/results/ood/pooled_ood_summary.csv`
- **Bootstrap Statistical Tests (1,000 resamples):** `uncertainty_rejection/results/statistical_tests/paired_bootstrap_results.json`
- **Primary Frozen Configuration:** `uncertainty_rejection/configs/frozen_eval_config.yaml`
- **Frozen Calibration Artifact:** `uncertainty_rejection/configs/calibrated_thresholds.json`

