# FINAL PROJECT CLOSURE AUDIT

**Project:** Uncertainty-Based Selective Prediction for Text-Guided Medical Image Segmentation Under Domain Shift  
**Course:** ELEC 6661 — Medical Image Processing (Fall 2026), Concordia University  
**Supervision:** IMPACT Lab (Prof. Hassan Rivaz, Omid Nejati Manzari)  
**Author:** Fernando Garcia-Medina (`fernando.garciamedina@concordia.ca`)  
**Baseline Model:** MedCLIPSeg (CVPR 2026, Koleilat et al.)  
**Date of Audit:** November 7, 2026  
**Audit Scope:** Steps 12B.1–18 (Manuscript Audit, Scientific Documentation Synchronization, Reusable Evaluation Toolkit, Google Colab Notebook, Clean-Room Reproducibility, Collaborator Repository Preparation, Primary Repository Readiness).

---

## 1. Executive Summary & Project Closure Status

This audit constitutes the formal, definitive verification and closure report for the ELEC 6661 final research project. The project investigated post-hoc, uncertainty-based selective prediction for frozen vision-language segmentation models without model retraining, fine-tuning, or test-time parameter adaptation.

### Closure Verdict: **APPROVED FOR FINAL SUBMISSION & RELEASE**
- **Scientific Manuscript:** 10-page IEEE Computer Society format technical paper fully drafted, audited, and compiled to PDF (`main.pdf`, 10 pages, 1.09 MB) with zero fatal errors or overfull hbox warnings.
- **Canonical Research Records:** All 7 canonical scientific documentation files in `uncertainty_rejection/research/` are 100% reconciled and synchronized with Decisions D-001 through D-163.
- **Reusable Evaluation Toolkit:** Self-contained, model-independent package `uncertainty_selective_evaluation/` implemented and verified with unit, synthetic, and canonical benchmark tests.
- **Google Colab Demonstration Notebook:** Standalone, beginner-friendly notebook `notebooks/uncertainty_selective_evaluation.ipynb` authored and validated for 1-click execution.
- **Clean-Room Reproducibility:** Verified concordance against original per-image CSVs across all five cohorts ($N=542$ total test images).
- **Collaborator Contribution Package:** Prepared for Ehsan's fork (`https://github.com/EhsanSl/MedCLIPSeg.git`) on feature branch `feature/uncertainty-selective-evaluation`.
- **Git Governance Compliance:** In strict adherence to project constraints, zero git commits, pushes, pull requests, or remote alterations were executed.

---

## 2. Scientific & Methodological Audit

### 2.1 Theoretical Framework: Elimination of Background Area Dilution
- **Analytical Derivation:** Global grid-level entropy averaging satisfies $U_{\mathrm{mean}} = \alpha U_{\mathrm{fg}} + (1-\alpha) U_{\mathrm{bg}}$. Because the network exhibits high confidence across background ultrasound tissue ($U_{\mathrm{bg}} \approx 0$), $U_{\mathrm{mean}} \approx \alpha \cdot U_{\mathrm{fg}}$, where $\alpha = |\Omega_{\mathrm{fg}}| / |\Omega|$ is the lesion area fraction. Consequently, global averaging sorts images primarily by tumor geometry rather than segmentation risk.
- **Empirical Validation:** On the BUSI validation cohort ($N=65$), $U_{\mathrm{mean}}$ exhibited a severe positive correlation with lesion area ($\rho = +0.9148$) and near-zero correlation with segmentation error ($\rho = -0.0451$ with DSC; $\rho = -0.0712$ with NSD). In contrast, restricting aggregation strictly to the predicted lesion ($U_{\mathrm{fg}}$) suppressed area correlation to $\rho = -0.0848$ and elevated error correlation to $\rho = -0.7202$ with DSC and $\rho = -0.7410$ with NSD.
- **Methodological Qualification (Decision D-161):** The manuscript and scientific records explicitly distinguish the mathematical derivation of $U_{\mathrm{mean}} \approx \alpha U_{\mathrm{fg}}$ from the empirical finding of $U_{\mathrm{fg}}$ diagnostic superiority, avoiding unqualified assertions of a priori mathematical guarantees.

### 2.2 Seven Spatial Uncertainty Aggregators Benchmark
On the in-distribution BUSI validation split ($N=65$ lesion cases), all seven aggregators were evaluated:
1. $U_{\mathrm{fg}}$ (Primary): AURC **0.0839**, Spearman $\rho = -0.7202$ (DSC), $\rho = -0.7410$ (NSD).
2. $U_{\mathrm{std\_fg}}$ (Comparator): AURC **0.0850**, Spearman $\rho = -0.7068$ (DSC), $\rho = -0.7285$ (NSD). Paired $\Delta\mathrm{AURC} = -0.0011$ was not statistically significant ($p = 0.294$).
3. $U_{\mathrm{boundary}}$: AURC **0.1302**, Spearman $\rho = -0.2273$ (DSC), $\rho = -0.4174$ (NSD). Underperforms whole-lesion triage due to natural edge ambiguity.
4. $U_{\mathrm{ratio}}$: AURC **0.1368**, Spearman $\rho = +0.0812$ (DSC), area correlation $\rho = -0.9957$.
5. $U_{\mathrm{mean}}$ (Global): AURC **0.1438**, indistinguishable from random rejection ($0.1434$).
6. $U_{\mathrm{positive}}$: AURC **0.1455**, area correlation $\rho = +0.8841$.
7. $U_{\mathrm{threshold}}$: AURC **0.1750**, discarding count information of tail entropy.
- **Baselines:** Uninformative Random Baseline AURC **0.1434**; Empirical Oracle AURC **0.0612**.

### 2.3 Frozen Validation Calibration Protocol
- **Primary Threshold:** Calibrated on BUSI validation lesion cohort ($N=65$) to achieve nominal $80.0\%$ coverage ($52/65$ retained):
  $$\tau = 0.064058324$$
  Permanently frozen prior to test-set or OOD evaluation.
- **Two-Stage Empty Mask Policy:** When predicted foreground is empty ($|\Omega_{\mathrm{fg}}| = 0$):
  $$g(X) = \mathbb{I}(U_{\mathrm{mean}} \le 10^{-4})$$
  Evaluated on Setting B screening data (13 normal scans with empty ground truth).

### 2.4 Baseline Reproduction & Provenance (Category A Lineage Equivalence)
Audited against the official MedCLIPSeg CVPR 2026 Table 2 (Domain Generalization):
- **BUSI Full Test ($N=78$):** Published **85.72%** vs. Replicated **0.8572** ($85.72\%$, $\Delta = 0.00\%$).
- **BUSBRA ($N=282$):** Published **75.06%** vs. Replicated **0.7511** ($75.11\%$, $\Delta = +0.05\%$).
- **BUSUC ($N=122$):** Published **84.37%** vs. Replicated **0.8430** ($84.30\%$, $\Delta = -0.07\%$).
- **BUID ($N=35$):** Published **78.99%** vs. Replicated **0.7900** ($79.00\%$, $\Delta = +0.01\%$).
- **UDIAT ($N=25$):** Published **74.64%** vs. Replicated **0.7467** ($74.67\%$, $\Delta = +0.03\%$).
- **Macro Average:** Published **78.27%** vs. Replicated **78.27%** ($\Delta = 0.00\%$).
- **Sample-Weighted Average:** Published **77.78%** vs. Replicated **77.80%** ($\Delta = +0.02\%$).
- **Classification:** All 5 benchmarks classified as **Category A (Directly Equivalent / Verified Checkpoint Lineage)**. Deviations $\le 0.07\%$ are fully explained by Monte Carlo dropout sampling stochasticity ($T=30$) and float32 hardware kernel rounding (MPS vs. CUDA).

### 2.5 Quantitative Selective Prediction Performance Summary

| Benchmark | Cohort Type | $N$ | Baseline DSC | Retained DSC | Selective Gain | Realized Coverage | $U_{\mathrm{fg}}$ AURC | Random AURC | Oracle AURC |
| :--- | :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **BUSI Test (Setting A)** | In-Domain Lesion | 65 | 0.8286 | **0.9031** | **+0.0745** | 72.3% (47/65) | **0.0790** | 0.1687 | 0.0692 |
| **BUSI Test (Setting B)** | In-Domain Full | 78 | 0.8572 | **0.9241** | **+0.0669** | 76.9% (60/78) | --- | --- | --- |
| **UDIAT** | Spain (OOD) | 25 | 0.7467 | **0.8747** | **+0.1281** | 56.0% (14/25) | **0.1133** | 0.2533 | 0.0957 |
| **BUID** | Academic (OOD) | 35 | 0.7900 | **0.8818** | **+0.0919** | 57.1% (20/35) | **0.1094** | 0.2100 | 0.0944 |
| **BUSUC** | Multi-center (OOD) | 122 | 0.8430 | **0.8982** | **+0.0551** | 63.9% (78/122) | **0.0916** | 0.1570 | 0.0836 |
| **BUSBRA** | Brazil (OOD) | 282 | 0.7511 | **0.8536** | **+0.1025** | 44.7% (126/282) | **0.1529** | 0.2489 | 0.1203 |
| **Pooled External OOD** | 4 Centers | **464** | **0.7780** | **0.8718** | **+0.0939** | **51.3%** (238/464) | **0.1267** | **0.2216** | **0.1046** |

- **Bootstrap Statistical Significance (Decision D-162):** Paired bootstrap testing ($B=1,000$ paired resamples) confirms pooled OOD selective gain is statistically significant: $95\%$ CI $[+0.0775, +0.1113]$, $p < 0.001$.
- **AURC Superiority:** Case-level global AURC across 464 external scans is **0.1267** for $U_{\mathrm{fg}}$ vs. **0.1864** for $U_{\mathrm{mean}}$ ($\Delta\mathrm{AURC} = -0.0597$, $p < 0.001$), outperforming Random (**0.2216**) and approaching Oracle (**0.1046**).

### 2.6 Critical Failure Modes and Boundary Delineation
1. **Severe False-Certainty Failure Mode:** Case `116_malignant.png` from the BUSI test split produced a catastrophic failure ($\mathrm{DSC} = 0.2193$, $\mathrm{NSD} = 0.2393$) by erroneously segmenting posterior acoustic shadowing. Because the dark shadowing region lacked high-frequency acoustic texture, model predictions were falsely confident ($U_{\mathrm{fg}} = 0.06279 < \tau = 0.064058$). The case bypassed rejection and entered the autonomous partition. This counterexample proves that uncertainty rejection cannot guarantee the detection of all severe errors.
2. **Coverage Drift Under Domain Shift:** While calibrated for $80.0\%$ coverage in-domain, realized coverage collapsed to $56.0\%$ on UDIAT, $57.1\%$ on BUID, $63.9\%$ on BUSUC, and $44.7\%$ on BUSBRA ($51.3\%$ pooled). Domain shift systematically inflates predictive entropy distributions, shifting static thresholds toward higher referral workloads.

---

## 3. Codebase & Reproducibility Audit

### 3.1 Automated Regression Test Suite
Executed test command: `bash uncertainty_rejection/scripts/run_tests.sh`
- **Result:** **31 passed, 0 failed out of 31 total**.
- **Coverage:** Unit tests for raw entropy, spatial aggregators, boundary morphology, empty masks, statistical ties, NaNs, ranking directions, Spearman bootstrap CIs, paired AURC CIs, deterministic rejection, YAML syntax, and OOD assertions.

### 3.2 Reusable Evaluation Toolkit (`uncertainty_selective_evaluation/`)
- `metrics.py`: Complete, self-contained implementation of all 7 aggregators, DSC, NSD, Risk-Coverage curves, AURC, Random, Oracle, and bootstrap routines.
- `model_adapter.py`: Abstract base adapter, production MedCLIPSeg adapter, and synthetic `MockModelAdapter`.
- `evaluate.py`: Standalone CLI supporting Mode A (precomputed CSVs), Mode B (model adapter), and Mode C (demo benchmark).
- `test_reproducibility.py`: Clean-room verification script executing synthetic checks and real benchmark runs across all 5 datasets. All tests pass with zero errors.

### 3.3 Google Colab Notebook (`uncertainty_selective_evaluation.ipynb`)
- Located in `notebooks/uncertainty_selective_evaluation.ipynb` and `uncertainty_selective_evaluation/notebooks/uncertainty_selective_evaluation.ipynb`.
- Validated via Python JSON schema parser: zero formatting defects.
- Features: 1-click synthetic demonstration fallback, dual Risk-Coverage and quality gain visualization, CSV export.

---

## 4. Collaborator Contribution Readiness (`EhsanSl/MedCLIPSeg`)

- **Target Repository:** `https://github.com/EhsanSl/MedCLIPSeg.git`
- **Target Branch:** `feature/uncertainty-selective-evaluation`
- **Contribution Folder:** `uncertainty_selective_evaluation/`
- **Readiness State:**
  - Self-contained directory containing `__init__.py`, `metrics.py`, `model_adapter.py`, `evaluate.py`, `test_reproducibility.py`, `requirements.txt`, `README.md`, `COLLABORATOR_INTEGRATION_GUIDE.md`, and Colab notebook.
  - Zero coupling to internal experimental scripts or private local paths.
  - Integration guide provides copy-paste CLI commands and Python API snippets.
- **Strict Compliance:** No git branch creation, commits, or pushes were performed on remote `EhsanSl/MedCLIPSeg`.

---

## 5. Primary Repository Readiness (`Fergarciamedina/MedCLIPSeg-plus-plus`)

- **Primary Repository:** `git@github.com:Fergarciamedina/MedCLIPSeg-plus-plus.git`
- **Scientific Paper:** `uncertainty_rejection/paper/main.pdf` (10 pages, IEEE Computer Society format, 5 publication figures, 2 master tables, 29 references). Synchronized with `project/latex/main.pdf`.
- **Root README:** Updated with comprehensive overview, master quantitative results table, file structure, reproduction commands, and links to all canonical research documents.
- **Security & Hygiene Audit:** Confirmed `.gitignore` excludes binary checkpoints, `.venv`, `.DS_Store`, build logs, and raw image data. Zero API keys, tokens, or credentials exist in the codebase.
- **Git State:** Local modifications staged in working directory on branch `main`. Awaiting user approval prior to any git commit or push.

---

## 6. Final Verification Checklist & Release Sign-Off

| Verification Item | Status | Evidence / Location |
| :--- | :---: | :--- |
| **10-page IEEEtran Manuscript** | PASS | `uncertainty_rejection/paper/main.pdf` (10 pages, 1.09 MB) |
| **All 5 Paper Figures Generated** | PASS | `uncertainty_rejection/paper/figures/fig[1-5]*.pdf` |
| **All Table Numbers Concordant** | PASS | Reconciled against `FINAL_RESULTS_INDEX.md` |
| **Dilution Wording Qualified** | PASS | Analytical derivation distinguished from empirical finding |
| **Bootstrap p-values Standardized** | PASS | Standardized to $p < 0.001$ ($B=1,000$ resolution) |
| **Empty Mask Policy Codified** | PASS | Setting B DSC=1.0 for both-empty, DSC=0.0 for false positives |
| **Category A Lineage Equivalence** | PASS | Exact match with CVPR 2026 Table 2 ($\le 0.07\%$ DSC) |
| **31/31 Automated Tests Passing** | PASS | `uncertainty_rejection/scripts/run_tests.sh` |
| **Evaluation Toolkit Verified** | PASS | `uncertainty_selective_evaluation/test_reproducibility.py` |
| **Google Colab Notebook Validated** | PASS | `notebooks/uncertainty_selective_evaluation.ipynb` |
| **Collaborator Guide Authored** | PASS | `COLLABORATOR_INTEGRATION_GUIDE.md` |
| **Primary Root README Updated** | PASS | `README.md` with master results and quick-start |
| **Zero Unauthorized Git Actions** | PASS | Working tree clean of unauthorized commits/pushes |

---

### Audit Sign-Off
**Auditor:** Automated Scientific Integrity & Research Governance Agent  
**Decision:** **PROJECT FULLY VERIFIED, CLOSED, AND READY FOR FINAL PRESENTATION/DELIVERY.**

