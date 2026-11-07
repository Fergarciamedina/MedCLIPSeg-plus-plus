# Manuscript Verification, Numerical Audit, and Reproducibility Report

**Document:** `MANUSCRIPT_VERIFICATION.md`  
**Location:** `uncertainty_rejection/paper/`  
**Date:** 2026-11-07  
**Substage:** Step 12B Delivery  
**Target Paper:** *Uncertainty-Based Selective Prediction for Text-Guided Medical Image Segmentation Under Domain Shift*  
**Compiled Artifact:** `main.pdf` (10 pages, IEEE Computer Society 10pt two-column format, 1,088,969 bytes).  
**Compiler:** pdfTeX Version 3.141592653-2.6-1.40.27 (TeX Live 2025) on macOS Darwin.

---

## 1. Executive Summary & Quality Assurance Sign-Off

This document certifies that the complete scientific manuscript has been drafted, verified, and compiled with zero fatal LaTeX errors. Every numerical claim, table cell, confidence interval, and citation in `main.tex` has been audited against the primary canonical experimental outputs, `FINAL_RESULTS_INDEX.md`, and the authoritative CVPR 2026 MedCLIPSeg publication (Koleilat et al., CVPR 2026, Table 2).

All mandatory editorial requirements established in Step 12A.3.1 have been strictly maintained:
1. CVPR 2026 Table 2 (Domain Generalization) is the sole baseline reference for breast ultrasound.
2. The published MedCLIPSeg Table 2 values (BUSI 85.72%, BUSBRA 75.06%, BUSUC 84.37%, BUID 78.99%, UDIAT 74.64%) are accurately cited.
3. Our baseline reproduction is classified as Category A (execution equivalence within $\le 0.07\%$ difference).
4. Full BUSI test accepted DSC is reported as **0.9241** ($76.9\%$ coverage, $+0.0669$ gain, $0.9490$ NSD).
5. Patient identifier provenance is rigorously distinguished (verified clinical IDs absent, BUSBRA filename proxy `bus_XXXX`, others unknown).

---

## 2. Numerical Verification Checklist Against Canonical CSVs

| Location in Manuscript | Reported Metric / Value | Canonical Source Artifact | Verified Value in Source | Audit Finding |
| :--- | :--- | :--- | :---: | :---: |
| **Abstract / Intro** | BUSI Full Test Baseline: $0.8572$ | `test_evaluation_results.csv` | $0.857192$ ($85.72\%$) | **PASS** |
| **Abstract / Sec 6.1** | Published CVPR 2026 Table 2: $85.72\%$ | Koleilat et al. CVPR 2026 Tab 2 | $85.72\%$ | **PASS** |
| **Abstract / Sec 6.3** | BUSI Lesion Baseline: $0.8286$ | `primary_operating_point_test.csv` | $0.828631$ | **PASS** |
| **Abstract / Sec 6.3** | BUSI Lesion Accepted DSC: $0.9031$ | `primary_operating_point_test.csv` | $0.903140$ | **PASS** |
| **Abstract / Sec 6.3** | BUSI Lesion Realized Coverage: $72.3\%$ | `primary_operating_point_test.csv` | $47/65 = 72.3077\%$ | **PASS** |
| **Abstract / Sec 6.3** | BUSI Full Test Accepted DSC: $0.9241$ | `setting_b_deployment_policies_test.csv` | $0.924127$ | **PASS** |
| **Abstract / Sec 6.3** | BUSI Full Test Coverage: $76.9\%$ | `setting_b_deployment_policies_test.csv` | $60/78 = 76.9231\%$ | **PASS** |
| **Abstract / Sec 6.3** | BUSI Full Test Accepted NSD: $0.9490$ | `setting_b_deployment_policies_test.csv` | $0.949005$ | **PASS** |
| **Abstract / Sec 6.4** | Pooled External OOD Baseline: $0.7780$ | `pooled_ood_summary.csv` | $0.7780 \pm 0.1900$ | **PASS** |
| **Abstract / Sec 6.4** | Pooled External OOD Accepted: $0.8718$ | `pooled_ood_summary.csv` | $0.8718 \pm 0.0934$ | **PASS** |
| **Abstract / Sec 6.4** | Pooled OOD Gain: $+0.0939$ | `pooled_ood_summary.csv` | $+0.093854$ | **PASS** |
| **Abstract / Sec 6.4** | Pooled OOD 95% Bootstrap CI | `pooled_ood_summary.csv` | $[+0.0775, +0.1113]$ | **PASS** |
| **Abstract / Sec 6.4** | Pooled OOD Realized Coverage: $51.3\%$ | `pooled_ood_summary.csv` | $238/464 = 51.2931\%$ | **PASS** |
| **Abstract / Sec 6.4** | Pooled Case-Level AURC $U_{\mathrm{fg}}$: $0.1267$ | `pooled_ood_summary.csv` | $0.126685$ | **PASS** |
| **Abstract / Sec 6.4** | Pooled Case-Level AURC $U_{\mathrm{mean}}$: $0.1864$ | `pooled_ood_summary.csv` | $0.186392$ | **PASS** |
| **Abstract / Sec 6.4** | Pooled Random AURC Reference: $0.2216$ | `pooled_ood_summary.csv` | $0.221600$ | **PASS** |
| **Abstract / Sec 6.4** | Pooled Empirical Oracle AURC: $0.1046$ | `pooled_ood_summary.csv` | $0.104612$ | **PASS** |
| **Table 1 / Sec 6.4** | UDIAT Baseline: $0.7467 \to 0.8747$ ($56.0\%$) | `primary_operating_point_UDIAT.csv` | $14/25$ accepted, DSC $0.8747$ | **PASS** |
| **Table 1 / Sec 6.4** | BUID Baseline: $0.7900 \to 0.8818$ ($57.1\%$) | `primary_operating_point_BUID.csv` | $20/35$ accepted, DSC $0.8818$ | **PASS** |
| **Table 1 / Sec 6.4** | BUSUC Baseline: $0.8430 \to 0.8982$ ($63.9\%$) | `primary_operating_point_BUSUC.csv` | $78/122$ accepted, DSC $0.8982$ | **PASS** |
| **Table 1 / Sec 6.4** | BUSBRA Baseline: $0.7511 \to 0.8536$ ($44.7\%$) | `primary_operating_point_BUSBRA.csv` | $126/282$ accepted, DSC $0.8536$ | **PASS** |
| **Table 2 / Sec 6.2** | BUSI Val Spearman $\rho(U_{\mathrm{fg}}, \mathrm{DSC})$: $-0.7202$ | `spearman_validation_65_lesion.csv` | $-0.720215$ | **PASS** |
| **Table 2 / Sec 6.2** | BUSI Val Spearman $\rho(U_{\mathrm{mean}}, \mathrm{DSC})$: $-0.0451$ | `spearman_validation_65_lesion.csv` | $-0.045120$ | **PASS** |
| **Table 2 / Sec 6.2** | BUSI Val Lesion Area Confounding: $+0.9148$ | `lesion_area_correlation.csv` | $+0.914822$ ($p < 10^{-20}$) | **PASS** |
| **Table 2 / Sec 6.2** | BUSI Val AURC $U_{\mathrm{fg}}$: $0.0839$ | `validation_method_comparison_N65.csv` | $0.083912$ | **PASS** |
| **Table 2 / Sec 6.2** | BUSI Val AURC $U_{\mathrm{std\_fg}}$: $0.0850$ | `validation_method_comparison_N65.csv` | $0.085024$ ($p=0.294$) | **PASS** |
| **Sec 6.5 / Fig 2** | Counterexample `116_malignant`: DSC $0.2193$ | `test_evaluation_results.csv` | DSC $0.2193$, $U_{\mathrm{fg}} = 0.06279$ | **PASS** |

---

## 3. Bibliographic Verification & Attribution Integrity

All 16 bibliography entries in `references.bib` have been verified for publication integrity:
1. `koleilat2026medclipseg`: IEEE/CVF CVPR 2026. Explicitly cited for the frozen foundation model, PVL adapters, and Table 2 Domain Generalization baselines.
2. `al2020dataset`: Data in Brief 2020. Official source for the BUSI benchmark.
3. `yap2017automated`: IEEE JBHI 2017. Official source for the UDIAT benchmark.
4. `gomezflores2024busbra`: Medical Physics 2024. Official source for the BUSBRA multi-scanner registry.
5. `guarino2026better`: IEEE/CVF CVPR 2026. Foundational citation for spatial aggregation sensitivity (AVG, ATA).
6. `roy2019bayesian`: NeuroImage 2019. Structure-wise Bayesian uncertainty for quality control.
7. `jungo2020analyzing`: Frontiers in Neuroscience 2020. Challenges of aggregated uncertainty in segmentation.
8. `mehrtash2020confidence`: IEEE TMI 2020. Uncalibrated deep segmentation networks.
9. `geifman2017selective`: NeurIPS 2017. Foundational selective classification formalism and risk-coverage definitions.
10. `gal2016dropout`: ICML 2016. Dropout as a Bayesian approximation.
11. `kendall2017uncertainties`: NeurIPS 2017. Aleatoric and epistemic uncertainty taxonomy.
12. `devries2018learning`: arXiv 2018. Learning confidence estimation for out-of-distribution detection.
13. `nikolov2018deep`: arXiv 2018. Normalized Surface Distance (NSD) definition.
14. `efron1993introduction`: Chapman & Hall/CRC 1993. Paired bootstrap resampling methodology.
15. `radford2021learning`: ICML 2021. Foundation CLIP vision-language model.
16. `wang2022medclip`: CHIL 2022. MedCLIP contrastive medical pre-training.

---

## 4. Source-to-Claim Traceability Mapping

| Manuscript Section | Research Question | Canonical Decision Anchor | Falsifiable Claim Anchor |
| :--- | :---: | :---: | :---: |
| **Section 1 (Introduction)** | Problem Formulation | D-001, D-002, D-048 | C01, C02, C04 |
| **Section 3 (MedCLIPSeg Baseline)** | Baseline Architecture | D-003, D-010, D-157 | C01, C14, C15 |
| **Section 4.2 (Spatial Operators)** | RQ3 (Aggregator Superiority) | D-015, D-018, D-085 | C05, C06, C07 |
| **Section 4.3 (Dilution Derivation)** | Mathematical Mechanism | D-020, D-152 | C05, C16 |
| **Section 4.4 (Selective Prediction)** | Formal Risk-Coverage | D-088, D-090, D-110 | C10, C11, C12 |
| **Section 4.5 (Validation Calibration)** | Protocol Lock & Two-Stage | D-092, D-105, D-133 | C13, C17, C18 |
| **Section 5.2 (Clinical Prompting)** | Prompt Neutrality ($P_{\mathrm{neutral}}$)| D-030, D-142 | C19, C20 |
| **Section 6.1 (Baseline Reproduction)** | RQ1 (Baseline Lineage) | D-157 | C01, C21 |
| **Section 6.2 (Aggregator Validation)** | RQ3 (Aggregator Superiority) | D-095, D-098, D-102 | C06, C07, C08 |
| **Section 6.3 (Locked BUSI Test)** | RQ2 (Selective Utility) | D-135, D-136, D-158 | C22, C23, C24 |
| **Section 6.4 (External OOD Transfer)** | RQ2 & RQ4 (Domain Shift) | D-140, D-145, D-151 | C25, C26, C27 |
| **Section 6.5 (Failure Analysis)** | RQ5 (Safety Boundaries) | D-131, D-136, D-150 | C28, C29, C30 |
| **Section 7 (Discussion)** | Causal Mechanics & Tradeoffs | D-152, D-153 | C31, C32, C33 |
| **Section 8 (Limitations)** | Boundary Limitations | D-132, D-153, D-159 | C34, C35, C36 |

---

## 5. Course Compliance & Template Specifications

- **Course:** ELEC 6661 -- Medical Image Processing (Fall 2026), Concordia University.
- **Supervision:** IMPACT Lab (Prof. Hassan Rivaz and Omid Nejati Manzari).
- **Template Choice:** Standard IEEE Computer Society (`IEEEtran` class, 10pt font, two-column layout).
- **Page Length:** Exactly 10 pages including title, abstract, 5 figures, 2 tables, 10 mathematical equations, and 16 references.
- **Academic Tone:** Impersonal, rigorous scientific English without marketing language or unverified claims of clinical safety.

---

## 6. Reproducibility Instructions

### To Regenerate All Figures:
From the repository root (`project/MedCLIPSeg-plus-plus/`):
```bash
MPLCONFIGDIR=/tmp/mpl .venv/bin/python3 uncertainty_rejection/paper/generate_paper_figures.py
```
Outputs: `fig1_methodology.{pdf,png}`, `fig2_qualitative_cases.{pdf,png}`, `fig3_val_aggregators.{pdf,png}`, `fig4_busi_risk_coverage.{pdf,png}`, `fig5_cross_domain_performance.{pdf,png}` located in `uncertainty_rejection/paper/figures/`.

### To Compile the Manuscript PDF:
From the paper directory (`project/MedCLIPSeg-plus-plus/uncertainty_rejection/paper/`):
```bash
pdflatex -interaction=nonstopmode main.tex
bibtex main
pdflatex -interaction=nonstopmode main.tex
pdflatex -interaction=nonstopmode main.tex
```
Output: `main.pdf` (10 pages, zero errors).

