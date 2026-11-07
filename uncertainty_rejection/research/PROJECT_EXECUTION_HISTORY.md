# ELEC 6661 — Comprehensive Project Execution History

**Project Title:** Training-Free Predictive Uncertainty Quantification and Selective Rejection for Text-Guided Medical Image Segmentation Under Domain Shift  
**Author:** Fernando (Concordia University, Course ELEC 6661)  
**Auditor / Recorder:** Automated Scientific Assistant  
**Repository:** `project/MedCLIPSeg-plus-plus`  
**Execution Period:** October 2026 – November 2026  
**Status:** All Experimental Stages Completed (Steps 1–11C); Pre-Manuscript Phase (Step 12)  
**Primary Target Model:** MedCLIPSeg (ViT-B/16, UniMedCLIP text encoder, 10 residual adapter layers)  
**Total Clinical Datasets Evaluated:** 5 Datasets ($N_{\mathrm{total}} = 542$ test ultrasound images)  
- **In-Distribution:** BUSI ($N=78$ total, $N=65$ lesion-present, $N=13$ normal)
- **External Out-of-Distribution:** UDIAT ($N=25$), BUID ($N=35$), BUSUC ($N=122$), BUSBRA ($N=282$)

---

## Stage 01: Mathematical Formulation of Uncertainty Aggregation Metrics

- **Objective & Research Question:** How can high-dimensional pixel-level predictive entropy maps generated from stochastic Monte Carlo forward passes be rigorously aggregated into a single scalar image-level score that accurately reflects segmentation quality without requiring retraining?
- **Inputs & Dependencies:** Predictive entropy formula in natural units:
  $$H(p_i) = - \left[ p_i \ln(p_i + \epsilon) + (1 - p_i) \ln(1 - p_i + \epsilon) \right], \quad H(p_i) \in [0, \ln 2]$$
- **Scripts & Mathematical Formulations:**
  Formulated seven distinct spatial aggregation operators:
  1. Global Mean Entropy ($U_{\mathrm{mean}}$): Average over all image pixels $\Omega$.
  2. Positive-Entropy Filter ($U_{\mathrm{positive}}$): Average over pixels with $H(p_i) > 0$.
  3. Foreground Mean Predictive Entropy ($U_{\mathrm{fg}}$): Average strictly over predicted lesion pixels $\Omega_{\mathrm{fg}} = \{i \mid \hat{y}_i = 1\}$.
  4. Boundary Band Entropy ($U_{\mathrm{boundary}}$): Average over morphological dilation/erosion contour band $\Omega_{\mathrm{boundary}} = \Omega_{\mathrm{dilated}} \setminus \Omega_{\mathrm{eroded}}$.
  5. High-Uncertainty Pixel Area Fraction ($U_{\mathrm{threshold}}$): Fraction of image pixels with $H(p_i) > t$.
  6. Foreground Entropy Standard Deviation ($U_{\mathrm{std\_fg}}$): Population standard deviation of entropy within $\Omega_{\mathrm{fg}}$.
  7. Foreground-to-Global Entropy Ratio ($U_{\mathrm{ratio}}$): Ratio $\frac{U_{\mathrm{fg}}}{U_{\mathrm{mean}} + \epsilon}$.
- **Scientific Decisions:** Adopt natural logarithm (nats) $[0, \ln 2 \approx 0.6931]$ to match PyTorch conventions. Enforce strict mathematical separation between candidate scores.
- **Cross-References:** `uncertainty_rejection/src/uncertainty/aggregation.py`, `THEORY_AND_METRICS.md`.

---

## Stage 02 & 03: Modular Source Implementation and Synthetic Test Suite

- **Objective & Research Question:** Implement the seven aggregation operators in a reusable, modular Python package and verify numerical correctness on edge cases with analytically exact ground truth.
- **Inputs & Dependencies:** `numpy`, `scipy.ndimage`, `cv2`.
- **Scripts Used:** `uncertainty_rejection/src/uncertainty/aggregation.py`, `uncertainty_rejection/src/uncertainty/metrics.py`, `uncertainty_rejection/tests/test_uncertainty.py`.
- **Actions Executed:** Developed `extract_image_features` and individual metric functions. Built an initial suite of 15 synthetic unit tests covering empty masks, all-zero entropy, uniform uncertainty, border-touching lesions, L-shaped contours, morphological kernel geometry, and shape mismatch exceptions.
- **Quantitative Results:** 15/15 unit tests passing. Verified mathematical identities: for an all-zero map, all metrics return 0.0; for an empty mask ($|\Omega_{\mathrm{fg}}| = 0$), $U_{\mathrm{fg}}$, $U_{\mathrm{std\_fg}}$, $U_{\mathrm{boundary}}$, and $U_{\mathrm{ratio}}$ return `NaN` safely without raising unhandled exceptions.
- **Problems & Corrections:** Fixed morphological boundary extraction to handle single-pixel lesions where erosion reduces the mask to empty (applied fallback to original mask).
- **Cross-References:** `uncertainty_rejection/tests/test_uncertainty.py`.

---

## Stage 04 & 05: BUSI Test Inference and Diagnostic Pipeline

- **Objective & Research Question:** Execute baseline probabilistic inference on the BUSI test set ($N=78$), extract raw float32 predictive entropy maps, compute per-image aggregation scores, and verify data bounds.
- **Inputs & Dependencies:** Pretrained MedCLIPSeg checkpoint (`seed666`), BUSI test images and prompts (`Test_text_original.xlsx`).
- **Scripts Used:** `shared/medclipseg/test.py`, `uncertainty_rejection/research/aggregation/compute_uncertainty_csv.py`, `uncertainty_rejection/research/aggregation/visualize_aggregation.py`.
- **Actions Executed:** Ran inference with $T=30$ Monte Carlo samples on Apple Silicon MPS backend. Saved binary predicted masks (PNG) and raw float32 predictive entropy maps (`.npy`). Built CSV pipeline matching segmentation DSC/NSD to uncertainty metrics.
- **Quantitative Results:** All 78 test images processed. Mean DSC = $0.8572$, Mean NSD = $0.8835$. 65 lesion cases had valid $U_{\mathrm{fg}}$; 13 normal cases had undefined $U_{\mathrm{fg}}$ due to $|\Omega_{\mathrm{fg}}| = 0$.
- **Scientific Decisions:** To prevent exploratory data snooping, the resulting test CSV was strictly quarantined and banned from being used for threshold calibration or method selection.
- **Cross-References:** `uncertainty_rejection/docs/PRE_TEST_AUDIT.md`.

---

## Stage 06 & 07: BUSI Validation Inference and Integrity Audit

- **Objective & Research Question:** Generate predictions and uncertainty maps on the BUSI validation set ($N=78$) to serve as the sole, locked tuning cohort for method selection and threshold calibration.
- **Inputs & Dependencies:** `project/data/raw/BUSI/Val_Folder/`, `Val_text.xlsx`, frozen checkpoint.
- **Scripts Used:** `uncertainty_rejection/scripts/run_validation_inference.sh`, `run_validation_inference.py`.
- **Actions Executed:** Executed frozen inference on $N=78$ validation images outside Git under `project/data/outputs/BUSI/validation/seed666/`. Extracted raw `.npy` entropy maps and computed all 7 aggregation scores.
- **Quantitative Results:** Validation Mean DSC = $0.8767$, Mean NSD = $0.9040$. Exactly 65 lesion-present cases and 13 normal cases. Zero missing or corrupt files.
- **Problems & Corrections:** MPS Metal compiler cache permission failure resolved by configuring `MPLCONFIGDIR=/tmp/matplotlib_cache` and `HF_HUB_OFFLINE=1`.
- **Cross-References:** `uncertainty_rejection/docs/PRE_TEST_AUDIT.md`.

---

## Stage 08: BUSI Validation Statistical Analysis

- **Objective & Research Question:** Evaluate the association between the seven uncertainty metrics and segmentation quality (DSC, NSD) on the validation cohort, and calculate empirical Risk-Coverage curves.
- **Inputs & Dependencies:** Validation uncertainty CSV ($N=78$, 65 lesion-present).
- **Scripts Used:** `uncertainty_rejection/research/statistical_analysis/`.
- **Actions Executed:** Computed Spearman rank correlation $\rho$ with 95% bootstrap confidence intervals ($B=1000$). Computed empirical risk-coverage curves with risk defined as $1 - \text{DSC}$ and $1 - \text{NSD}$.
- **Quantitative Results:**
  - $U_{\mathrm{fg}}$ achieved Spearman $\rho = -0.7202$ against DSC ($p = 2.4 \times 10^{-11}$) and $\text{AURC} = 0.0839$.
  - $U_{\mathrm{std\_fg}}$ achieved Spearman $\rho = -0.7068$ ($p = 8.1 \times 10^{-11}$) and $\text{AURC} = 0.0850$.
  - Global $U_{\mathrm{mean}}$ suffered severe lesion-area confounding ($\rho = +0.9148$ with predicted area $|\Omega_{\mathrm{fg}}|$).
- **Scientific Decisions:** Confirmed that $U_{\mathrm{mean}}$ is mathematically invalid for selective prediction because it penalizes large tumors regardless of segmentation accuracy.
- **Cross-References:** `uncertainty_rejection/docs/PRE_TEST_AUDIT.md`.

---

## Stage 09: Validation-Only Method Selection & Threshold Calibration

- **Objective & Research Question:** Select the primary uncertainty metric, prespecify a secondary comparator, establish deployment policies for empty masks, and calibrate numerical rejection thresholds strictly on validation data.
- **Inputs & Dependencies:** BUSI validation results ($N=65$ lesion cohort).
- **Scripts Used:** `uncertainty_rejection/research/selective_prediction/run_validation_selection.py`.
- **Actions Executed:**
  - Primary metric selected: $U_{\mathrm{fg}}$ (minimal validation AURC $= 0.0839$).
  - Prespecified comparator: $U_{\mathrm{std\_fg}}$ ($\text{AURC} = 0.0850$, paired bootstrap $\Delta\text{AURC} = -0.0011$, $p = 0.884$, statistically indistinguishable).
  - Primary operating threshold calibrated for 80% validation retention: $\tau = 0.064058324$.
  - Operating ladder calibrated: $\tau_{95} = 0.10193$, $\tau_{90} = 0.08346$, $\tau_{85} = 0.07270$, $\tau_{75} = 0.05969$, $\tau_{70} = 0.05621$.
  - Empty-mask triage policy selected: `two_stage` with $\tau_{\mathrm{neg}} = 10^{-4}$ ($1.0 \times 10^{-4}$).
- **Outputs & Artifacts:** Frozen configuration locked in YAML (`frozen_selective_prediction_config.yaml`) and JSON (`frozen_selective_prediction_config.json`).
- **Regression Tests:** Expanded test suite from 20 to 23 tests (all passing).
- **Cross-References:** `project/data/outputs/BUSI/validation/seed666/selective_prediction/frozen_selective_prediction_config.yaml`.

---

## Stage 09.5: Pre-Test Scientific Audit

- **Objective & Research Question:** Independently audit the statistical implementation, mathematical formulas, and configuration files before unblinding the BUSI test set.
- **Inputs & Dependencies:** Frozen YAML/JSON configs, synthetic arrays, source code.
- **Actions Executed:**
  - Verified AURC trapezoidal integration by hand on synthetic micro-arrays.
  - Investigated why Random AURC was $0.1434$ on validation when baseline DSC was $0.8520$: discovered and documented the empirical boundary convention where integration begins at $c_1 = 1/N$, yielding $\text{AURC}_{\mathrm{rand}} = R_0 \times (1 - 1/N)$.
  - Conducted strict no-leakage audit: confirmed test data was never loaded during calibration.
- **Audit Verdict:** Passed with full clearance to execute locked test evaluation.
- **Cross-References:** `uncertainty_rejection/docs/PRE_TEST_AUDIT.md`.

---

## Stage 10: Locked BUSI Test Set Evaluation

- **Objective & Research Question:** Evaluate the frozen selective prediction strategy on the locked BUSI test set without adjusting any weights, thresholds, or code.
- **Inputs & Dependencies:** BUSI test aggregation CSV ($N=78$, 65 lesion-present), frozen YAML configuration.
- **Scripts Used:** `uncertainty_rejection/research/selective_prediction/run_locked_test_evaluation.py`.
- **Actions Executed:**
  - Evaluated Setting A (lesion cohort $N=65$) at primary operating cutoff $\tau = 0.064058324$.
  - Evaluated full 6-point operating ladder.
  - Computed empirical Risk-Coverage curves and AURC against Random, Oracle, and 5 baselines.
  - Executed paired bootstrap $\Delta\text{AURC}$ hypothesis tests ($B=1000$).
- **Quantitative Results:**
  - Realized coverage at primary threshold: **72.3%** ($47/65$ retained).
  - Accepted Mean DSC: **$0.9031$** (Baseline $= 0.8286$, Absolute Gain **$+0.0745$**, 95% CI $[+0.0349, +0.1235]$).
  - Rejected Mean DSC: **$0.6341$** ($18$ poor-quality segmentations filtered).
  - $U_{\mathrm{fg}}$ AURC (1-DSC): **$0.0790$** vs. Random $0.1687$ (Gain $+0.0898$) and Oracle $0.0692$.
  - Paired test vs. $U_{\mathrm{std\_fg}}$: $\Delta\text{AURC} = -0.0003$, $p = 0.786$ (no significant difference).
  - Paired test vs. $U_{\mathrm{mean}}$: $\Delta\text{AURC} = -0.0472$, $p < 0.001$ ($U_{\mathrm{fg}}$ decisively superior).
- **Regression Tests:** Expanded test suite from 23 to 27 tests (all passing).
- **Cross-References:** `uncertainty_rejection/docs/LOCKED_TEST_EVALUATION_REPORT.md`.

---

## Stage 11A: OOD Readiness Audit & Evaluation Protocol Formulation

- **Objective & Research Question:** Define a rigorous, leak-free evaluation protocol to test whether the frozen BUSI-calibrated selective prediction strategy transfers zero-shot to external clinical breast ultrasound benchmarks.
- **Actions Executed:** Evaluated potential candidate datasets. Determined that cross-organ transfer (e.g. DDTI thyroid) introduced severe anatomical confounding. Restored focus to breast-only domain generalization across four external clinical benchmarks: UDIAT, BUID, BUSUC, BUSBRA. Pre-registered prompt protocol: author baseline prompts ($P_{\mathrm{orig}}$) and label-free screening prompts ($P_{\mathrm{neutral}}$).
- **Cross-References:** `uncertainty_rejection/docs/STEP11A_OOD_READINESS.md`.

---

## Stage 11B.1: Staging of Four Breast Ultrasound OOD Datasets

- **Objective & Research Question:** Download, unpack, verify integrity, and configure the four official MedCLIPSeg external breast ultrasound datasets outside Git.
- **Inputs & Dependencies:** Official Hugging Face repository (`TahaKoleilat/MedCLIPSeg`), CC-BY-NC-4.0 licenses.
- **Actions Executed:** Retained original zip archives under `project/data/raw/downloads/`. Unpacked into canonical raw data paths. Verified 1-to-1 pairing between ultrasound images and ground-truth masks:
  - UDIAT ($N=25$ test images, Siemens Elegra, Spain)
  - BUID ($N=35$ test images, public clinical repository)
  - BUSUC ($N=122$ test images, multi-center registry)
  - BUSBRA ($N=282$ test images, multi-scanner, Brazil)
  Generated neutral prompt files (`Test_text_neutral.xlsx`) with diagnostic adjectives stripped. Configured dynamic YAML paths in `shared/configs/`.
- **Quantitative Results:** 464 total external test scans staged and verified ($100\%$ lesion prevalence).
- **Cross-References:** `uncertainty_rejection/docs/STEP11B1_STAGING_REPORT.md`.

---

## Stage 11B.2: Frozen OOD Inference & Selective Prediction Execution

- **Objective & Research Question:** Run frozen MedCLIPSeg probabilistic inference and evaluate the frozen selective rejection rules across all 464 external OOD images without model retraining or threshold recalibration.
- **Inputs & Dependencies:** Staged raw OOD data, frozen BUSI checkpoint (`seed666`), frozen YAML thresholds.
- **Scripts Used:** `shared/medclipseg/test.py`, `shared/medclipseg/utils/eval.py`, `uncertainty_rejection/research/aggregation/compute_uncertainty_csv.py`, `run_ood_selective_prediction_evaluation.py`.
- **Actions Executed:**
  - Ran 30-sample MC inference across UDIAT, BUID, BUSUC, BUSBRA on MPS.
  - Computed official DSC and NSD metrics.
  - Extracted all 7 uncertainty aggregation metrics into per-image CSVs.
  - Evaluated selective prediction using frozen primary threshold $\tau = 0.064058324$.
- **Quantitative Results:**
  - **UDIAT:** Baseline DSC $= 0.7467 \to$ Accepted DSC **$0.8747$** (Gain **$+0.1281$**, CI $[+0.0325, +0.2569]$).
  - **BUID:** Baseline DSC $= 0.7900 \to$ Accepted DSC **$0.8818$** (Gain **$+0.0919$**, CI $[+0.0339, +0.1644]$).
  - **BUSUC:** Baseline DSC $= 0.8430 \to$ Accepted DSC **$0.8982$** (Gain **$+0.0551$**, CI $[+0.0363, +0.0758]$).
  - **BUSBRA:** Baseline DSC $= 0.7511 \to$ Accepted DSC **$0.8536$** (Gain **$+0.1025$**, CI $[+0.0789, +0.1278]$).
  - $U_{\mathrm{fg}}$ decisively outperformed $U_{\mathrm{mean}}$ across all four datasets ($p < 0.005$ in all paired bootstrap tests).
- **Regression Tests:** Expanded test suite from 27 to 31 tests (all passing).
- **Cross-References:** `uncertainty_rejection/docs/STEP11B2_OOD_EVALUATION_REPORT.md`.

---

## Stage 11C: Final Cross-Domain Scientific Consistency Audit

- **Objective & Research Question:** Audit all numerical claims, formulas, and integration conventions across the full 5-cohort cross-domain benchmark prior to paper drafting.
- **Actions Executed:**
  - Reconciled draft aggregate OOD statistics against raw per-image CSVs: established exact sample-weighted micro-averages (Baseline DSC $= 0.7780$, Accepted DSC $= 0.8718$, Rejected DSC $= 0.6791$, Absolute Gain $= +0.0939$, realized coverage $= 51.3\%$).
  - Resolved BUSI Random AURC discrepancy: confirmed Step 10 value of $0.1687$ reflects the $N=65$ lesion cohort baseline, correcting an accidental inclusion of normal cases in Step 11B.2.
  - Disambiguated pooled case-level ranking AURC ($0.1267$) from sample-weighted average AURC ($0.1314$) and macro-average AURC ($0.1168$).
  - Re-ran the automated regression suite: 31/31 tests passed.
- **Cross-References:** `uncertainty_rejection/docs/STEP11C_CROSS_DOMAIN_AUDIT.md`.

---

## Stage 12: Final Reproducibility Audit, Published Baseline Comparison & Research Readiness

- **Substage 12A.1 & 12A.2 (Audit & Execution History):**
  - Verified published numerical claims against original per-image CSVs across all cohorts (BUSI Val, BUSI Test, UDIAT, BUID, BUSUC, BUSBRA).
  - Traced UniMedCLIP ViT-B/16 checkpoint provenance, prompt files, and official test split assignments.
  - Inspected DSC/NSD mathematical implementation and physical unit assumptions.
  - Audited patient clustering and documented potential within-patient dependencies.
  - Aligned all historical documentation files under `uncertainty_rejection/docs/`.
  - Cross-References: `uncertainty_rejection/docs/FINAL_REPRODUCIBILITY_AUDIT.md`.

- **Substage 12A.3 (Baseline Comparison, Documentation Integration & Final Readiness):**
  - Conducted systematic comparison against published MedCLIPSeg CVPR 2026 results (Koleilat et al., CVPR 2026).
  - Categorized all baseline comparisons under strict scientific criteria (Category A, B, C, D) in `PUBLISHED_BASELINE_COMPARISON.md`.
  - Formulated formal mathematical and empirical interpretations for RQ1–RQ5 in `SCIENTIFIC_EVIDENCE_AND_CLAIMS.md`.
  - Merged and consolidated the seven canonical scientific research documents under `uncertainty_rejection/research/`.
  - Established `FINAL_RESULTS_INDEX.md` as the single authoritative numerical anchor.
  - Preserved historical documentation under `uncertainty_rejection/docs/`.
  - Executed final quality control audit and verified 31/31 regression tests pass.

- **Substage 12A.3.1 (Critical Published-Baseline Correction & Final Resolution):**
  - Verified exact published Table 2 (Domain Generalization) MedCLIPSeg CVPR 2026 values: BUSI 85.72%, BUSBRA 75.06%, BUSUC 84.37%, BUID 78.99%, UDIAT 74.64%.
  - Confirmed replicated execution equivalence across all five datasets (differences $\le 0.07\%$, Category A across all benchmarks) attributable to Monte Carlo dropout sampling and float32 hardware precision.
  - Corrected `PUBLISHED_BASELINE_COMPARISON.md` to eliminate conflation between baseline segmentation and selective accepted performance.
  - Resolved full BUSI test (Setting B) accepted DSC discrepancy from canonical per-image outputs: confirmed true accepted mean DSC $= 0.924127$ ($0.9241$, gain $+0.0669$, accepted NSD $0.9490$), superseding draft $0.9193$ transcription error.
  - Reconciled patient provenance hierarchy: distinguished verified clinical IDs (absent in public benchmarks), filename-derived grouping proxies (BUSBRA `bus_XXXX` prefix grouping 262 clusters across 282 cases), and unknown patient identity (BUSI, UDIAT, BUID, BUSUC).
  - Adopted Decisions D-157, D-158, D-159 in `RESEARCH_DECISIONS_AND_LIMITATIONS.md`.
  - Updated `FINAL_RESULTS_INDEX.md` and `FINAL_DOCUMENTATION_INTEGRATION_AUDIT.md`.
  - Final Verdict: **READY FOR STEP 12B (LATEX MANUSCRIPT AUTHORING)**.

---

## Stage 12B: Scientific Manuscript Development and LaTeX Compilation

- **Objective:** Draft and compile the complete, formal 10-page academic manuscript formatted according to IEEE Computer Society standards.
- **Actions Executed:**
  - Designed and authored `uncertainty_rejection/paper/generate_paper_figures.py` to produce five publication-quality vector PDF and high-resolution PNG figures:
    - Figure 1: Schematic workflow of uncertainty extraction and selective prediction.
    - Figure 2: Qualitative segmentation and uncertainty triage cases (true positives, borderline accepted, rejected outliers, and false-certainty failure mode).
    - Figure 3: BUSI validation comparison of seven spatial uncertainty aggregators (Spearman $\rho$ and AURC).
    - Figure 4: Risk-Coverage curves and selective efficiency benchmarks on BUSI test.
    - Figure 5: Cross-domain selective prediction performance and coverage drift across four external benchmarks ($N=464$).
  - Drafted comprehensive 10-page IEEEtran manuscript (`uncertainty_rejection/paper/main.tex`, `references.bib`):
    - Rigorous theoretical derivation of background area dilution ($U_{\mathrm{mean}} \approx \alpha U_{\mathrm{fg}}$).
    - Formal selective prediction and risk-coverage formulation.
    - Multi-center empirical evaluation across 5 independent cohorts ($N=542$).
    - Critical failure analysis, coverage drift characterization, and false-certainty counterexample auditing.
  - Verified and compiled `main.pdf` using pdfTeX (TeX Live 2025): 10 pages, 1.09 MB, zero compilation errors. Synchronized identical source and output to `project/latex/`.
  - Authored `uncertainty_rejection/paper/MANUSCRIPT_VERIFICATION.md` verifying line-by-line numerical concordance with canonical CSVs.

---

## Stage 12B.1 & 13: Manuscript Audit, Scientific Documentation Synchronization & Closure

- **Objective:** Perform critical audit on manuscript claims, statistical reporting conventions, and baseline reproduction nuances; synchronize all 7 canonical scientific documentation files.
- **Actions Executed:**
  - Audited background dilution explanation: clarified that Eq. (10) explains why $U_{\mathrm{mean}}$ is confounded by area, while $U_{\mathrm{fg}}$ diagnostic superiority is an empirical finding across multi-center cohorts rather than an a priori mathematical guarantee for arbitrary metrics.
  - Standardized bootstrap p-value reporting to $p < 0.001$, aligning with $B = 1,000$ paired resamples minimum empirical resolution ($1/B = 0.001$), eliminating uncalibrated asymptotic $p < 0.0001$ claims.
  - Formalized Category A baseline reproduction wording: clarified that official checkpoint lineage + identical test splits match published CVPR 2026 Table 2 within $\le 0.07\%$ DSC, without asserting independent retraining from scratch.
  - Codified empty ground-truth evaluation policy in Setting B ($\mathrm{DSC}=1.0$ for both-empty, $\mathrm{DSC}=0.0$ for false-positive predictions on empty masks).
  - Adopted Decisions D-160 through D-163 in `RESEARCH_DECISIONS_AND_LIMITATIONS.md`.
  - Recompiled `main.pdf` in both `uncertainty_rejection/paper/` and `project/latex/` with zero errors.
  - Synchronized documentation across all 7 canonical research records.

