# ELEC 6661 — Scientific Evidence, Claims & Cross-Domain Validation Matrix

**Document:** `SCIENTIFIC_EVIDENCE_AND_CLAIMS.md`  
**Location:** `uncertainty_rejection/docs/`  
**Date:** 2026-11-07  
**Scope:** Formal inventory of verified scientific claims, supporting empirical evidence, statistical hypothesis test results, and cross-domain benchmark tables.

---

## 1. Summary of Core Scientific Claims

The project establishes four validated scientific claims supported by empirical evidence across 5 clinical cohorts ($N = 542$ total test ultrasound images):

| # | Scientific Claim | Key Empirical Evidence | Verification Status |
| :-: | :--- | :--- | :---: |
| **Claim 1** | **Universal Zero-Shot Selective Gain:** Training-free selective prediction using $U_{\mathrm{fg}}$ at frozen threshold $\tau = 0.064058$ yields statistically significant improvements in accepted segmentation accuracy across diverse clinical centers, scanners, and demographics without retraining. | 95% bootstrap confidence intervals for absolute DSC gain are strictly positive on all 5 datasets: BUSI ($+0.0745$ $[+0.035, +0.124]$), UDIAT ($+0.1281$ $[+0.033, +0.257]$), BUID ($+0.0919$ $[+0.034, +0.164]$), BUSUC ($+0.0551$ $[+0.036, +0.076]$), BUSBRA ($+0.1025$ $[+0.079, +0.128]$). | **VALIDATED** |
| **Claim 2** | **Decisive Superiority Over Global Baselines:** Foreground-normalized predictive entropy ($U_{\mathrm{fg}}$) significantly outperforms conventional global full-image entropy ($U_{\mathrm{mean}}$) across all domains. | Paired bootstrap hypothesis tests on $\Delta\text{AURC}$ demonstrate $p < 0.005$ on all 5 datasets. In BUID, $U_{\mathrm{mean}}$ performs worse than uninformed random guessing ($\text{AURC} = 0.2411$ vs. Random $0.2040$), whereas $U_{\mathrm{fg}}$ achieves $0.1094$. | **VALIDATED** |
| **Claim 3** | **Mechanistic Root Cause of Baseline Failure:** Global entropy ($U_{\mathrm{mean}}$) fails because it is severely confounded by lesion area, artificially penalizing large tumors. $U_{\mathrm{fg}}$ eliminates this positive confounding. | $U_{\mathrm{mean}}$ displays massive positive rank correlation with lesion area across all scanners ($\rho = +0.570$ to $+0.896$, all $p < 10^{-3}$). $U_{\mathrm{fg}}$ eliminates this positive artifact ($\rho = -0.033$ in BUSI, $+0.074$ in UDIAT). | **VALIDATED** |
| **Claim 4** | **Effective Clinical Triage via Two-Stage Gating:** When evaluating full screening populations containing normal scans, the two-stage gating policy cleanly resolves undefined foreground scores. | On BUSI test, two-stage gating with $\tau_{\mathrm{neg}} = 10^{-4}$ achieves $100\%$ accuracy in categorizing true normal scans ($13/13$ accepted as confident negatives with background entropy $U_{\mathrm{mean}} \le 10^{-4}$). | **VALIDATED** |

---

## 2. Reconciled Cross-Domain Master Evidence Table

All numbers are reproduced directly from the canonical per-image CSV files (`uncertainty_scores_<DATASET>_test_seed666.csv`):

| Dataset | Cohort Origin / Hardware | Sample Count ($N$) | Lesion Preval. | Baseline Mean DSC | Realized Cov. at $\tau=0.064058$ | Accepted Mean DSC | Absolute DSC Gain [95% Bootstrap CI] | Rejected Mean DSC | $U_{\mathrm{fg}}$ AURC (1-DSC) | $U_{\mathrm{mean}}$ AURC | Random AURC | Oracle Ceiling |
| :--- | :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **BUSI (ID)** | Baheya Hospital, Egypt (GE) | 65 | 83.3% | $0.8286 \pm 0.2110$ | 72.3% (47/65) | **0.9031** | **+0.0745** [$+0.0349, +0.1235$] | 0.6341 | **0.0790** | 0.1262 | 0.1687 | 0.0692 |
| **UDIAT** | UDIAT Diagnostic Centre, Spain (Siemens) | 25 | 100% | $0.7467 \pm 0.2992$ | 56.0% (14/25) | **0.8747** | **+0.1281** [$+0.0325, +0.2569$] | 0.5837 | **0.1133** | 0.1503 | 0.2432 | 0.0944 |
| **BUID** | Public Clinical Repository | 35 | 100% | $0.7900 \pm 0.2024$ | 57.1% (20/35) | **0.8818** | **+0.0919** [$+0.0339, +0.1644$] | 0.6675 | **0.1094** | 0.2411 | 0.2040 | 0.0935 |
| **BUSUC** | Multi-Center Clinical Registry | 122 | 100% | $0.8430 \pm 0.1152$ | 63.9% (78/122) | **0.8982** | **+0.0551** [$+0.0363, +0.0758$] | 0.7453 | **0.0916** | 0.1365 | 0.1557 | 0.0834 |
| **BUSBRA** | Multi-Hospital, Brazil (GE/Phil/Tosh) | 282 | 100% | $0.7511 \pm 0.1962$ | 44.7% (126/282) | **0.8536** | **+0.1025** [$+0.0789, +0.1278$] | 0.6683 | **0.1529** | 0.2020 | 0.2480 | 0.1200 |
| **POOLED OOD (Micro)** | All External Test Scans | **464** | **100%** | **$0.7780 \pm 0.1900$** | **51.3% (238/464)** | **0.8718** | **+0.0939** [$+0.0775, +0.1113$] | **0.6791** | **0.1267** | **0.1864** | **0.2216** | **0.1046** |
| **MACRO OOD (Unweighted)** | 4 External Datasets | 4 Sets | 100% | $0.7827$ | 55.4% | $0.8771$ | **+0.0944** | $0.6662$ | $0.1168$ | $0.1825$ | $0.2127$ | $0.0978$ |

---

## 3. Paired Bootstrap Statistical Significance ($\Delta\text{AURC}$)

Paired empirical bootstrap hypothesis tests ($B = 1000$ resamples) comparing the Primary method ($U_{\mathrm{fg}}$) against all comparator operators on the 1-DSC Risk metric:

| Dataset | $U_{\mathrm{fg}}$ vs. $U_{\mathrm{std\_fg}}$ ($\Delta\text{AURC}$, $p$) | $U_{\mathrm{fg}}$ vs. $U_{\mathrm{mean}}$ ($\Delta\text{AURC}$, $p$) | $U_{\mathrm{fg}}$ vs. $U_{\mathrm{boundary}}$ ($\Delta\text{AURC}$, $p$) | $U_{\mathrm{fg}}$ vs. $U_{\mathrm{threshold}}$ ($\Delta\text{AURC}$, $p$) | $U_{\mathrm{fg}}$ vs. $U_{\mathrm{positive}}$ ($\Delta\text{AURC}$, $p$) |
| :--- | :---: | :---: | :---: | :---: | :---: |
| **BUSI (ID)** | $-0.0003$ ($p = 0.786$, ns) | **$-0.0472$ ($p < 0.001$, \*\*\*)** | **$-0.0324$ ($p < 0.001$, \*\*\*)** | **$-0.1256$ ($p < 0.001$, \*\*\*)** | **$-0.0493$ ($p < 0.001$, \*\*\*)** |
| **UDIAT** | $-0.0019$ ($p = 0.360$, ns) | **$-0.0369$ ($p = 0.004$, \*\*)** | **$-0.0315$ ($p = 0.010$, \*)** | **$-0.0436$ ($p = 0.030$, \*)** | **$-0.0359$ ($p = 0.004$, \*\*)** |
| **BUID** | $-0.0134$ ($p = 0.646$, ns) | **$-0.1317$ ($p < 0.001$, \*\*\*)** | **$-0.0181$ ($p = 0.002$, \*\*)** | **$-0.0369$ ($p < 0.001$, \*\*\*)** | **$-0.1348$ ($p < 0.001$, \*\*\*)** |
| **BUSUC** | $+0.0003$ ($p = 0.440$, ns) | **$-0.0448$ ($p < 0.001$, \*\*\*)** | **$-0.0220$ ($p < 0.001$, \*\*\*)** | **$-0.0513$ ($p < 0.001$, \*\*\*)** | **$-0.0492$ ($p < 0.001$, \*\*\*)** |
| **BUSBRA** | $-0.0004$ ($p = 0.732$, ns) | **$-0.0492$ ($p < 0.001$, \*\*\*)** | **$-0.0152$ ($p = 0.008$, \*\*)** | **$-0.0846$ ($p < 0.001$, \*\*\*)** | **$-0.0523$ ($p < 0.001$, \*\*\*)** |

---

## 4. Lesion Area Confounding Analysis

Spearman rank correlation $\rho$ between uncertainty metrics and predicted lesion pixel area $|\Omega_{\mathrm{fg}}|$:

| Dataset | $U_{\mathrm{fg}}$ vs. Area ($\rho$, $p$-value) | $U_{\mathrm{mean}}$ vs. Area ($\rho$, $p$-value) | Confounding Severity |
| :--- | :---: | :---: | :--- |
| **BUSI (ID)** | $\rho = -0.033$ ($p = 0.795$) | $\mathbf{\rho = +0.896}$ ($p = 6.8 \times 10^{-24}$) | Massive positive confounding in $U_{\mathrm{mean}}$ |
| **UDIAT** | $\rho = +0.074$ ($p = 0.726$) | $\mathbf{\rho = +0.774}$ ($p = 5.7 \times 10^{-6}$) | Severe positive confounding in $U_{\mathrm{mean}}$ |
| **BUID** | $\rho = -0.360$ ($p = 0.034$) | $\mathbf{\rho = +0.570}$ ($p = 3.5 \times 10^{-4}$) | Moderate positive confounding in $U_{\mathrm{mean}}$ |
| **BUSUC** | $\rho = -0.312$ ($p = 4.7 \times 10^{-4}$) | $\mathbf{\rho = +0.596}$ ($p = 4.2 \times 10^{-13}$) | Moderate positive confounding in $U_{\mathrm{mean}}$ |
| **BUSBRA** | $\rho = -0.316$ ($p = 5.6 \times 10^{-8}$) | $\mathbf{\rho = +0.616}$ ($p = 7.5 \times 10^{-31}$) | Heavy positive confounding in $U_{\mathrm{mean}}$ |

### Key Nuance & Limitation Statement:
$U_{\mathrm{fg}}$ eliminates the severe positive confounding of $U_{\mathrm{mean}}$ that artificially penalizes large tumors. However, $U_{\mathrm{fg}}$ displays a mild-to-moderate residual **negative** correlation ($\rho \approx -0.31$ to $-0.36$) in external multicenter cohorts because tiny lesions have fewer interior pixels, making average entropy more sensitive to boundary speckle.

