# Authoritative Published Baseline Comparison and Experimental Equivalence

**Document:** `PUBLISHED_BASELINE_COMPARISON.md`  
**Location:** `research/stage_4/`
**Date:** 2026-11-07  
**Substage:** Step 12A.3.1 (Corrected Canonical Reference)  
**Target Publication:** Koleilat et al., *MedCLIPSeg: Probabilistic Vision-Language Adaptation for Data-Efficient and Generalizable Medical Image Segmentation*, IEEE/CVF CVPR 2026 (arXiv:2602.20423).  
**Table Source:** Original CVPR 2026 Paper **Table 2 (Domain Generalization)**, Breast Ultrasound Sub-section (Source: BUSI).  
**Official Artifacts:** Hugging Face `TahaKoleilat/MedCLIPSeg` & GitHub `HealthX-Lab/MedCLIPSeg`.

---

## 1. Executive Summary & Verification of Published Domain Generalization Table

In the official CVPR 2026 publication, **Table 2 (Domain Generalization)** reports the zero-shot cross-dataset performance of MedCLIPSeg trained on the source dataset **BUSI** and evaluated directly across the four unseen breast ultrasound target domains: **BUSBRA, BUSUC, BUID, and UDIAT** without test-time adaptation.

The authoritative published MedCLIPSeg Dice Similarity Coefficient ($\text{DSC }\%$) values, reported in exact paper dataset ordering (BUSI, BUSBRA, BUSUC, BUID, UDIAT), are:
- **BUSI (Source In-Distribution):** **85.72%**
- **BUSBRA (Target Domain):** **75.06%**
- **BUSUC (Target Domain):** **84.37%**
- **BUID (Target Domain):** **78.99%**
- **UDIAT (Target Domain):** **74.64%**

These values are distinct from data-efficiency tables (Table 1), ablation tables, or competing architecture rows (e.g., CAT-Seg, SAN, LViT).

---

## 2. Exact Numerical Comparison: Published vs. Our Baseline

The table below provides the direct side-by-side comparison between the published MedCLIPSeg CVPR 2026 Table 2 baseline and our evaluated baseline across all cohorts:

| Dataset | Modality & Cohort | Split / Sample $N$ | Published MedCLIPSeg Table 2 DSC (%) | Our MedCLIPSeg Baseline DSC | Absolute Difference ($\Delta$) | Our Baseline NSD (Mean $\pm$ SD) | Equivalence Classification | Attribution & Lineage Explanation |
| :--- | :--- | :---: | :---: | :---: | :---: | :---: | :---: | :--- |
| **BUSI** | Breast US (Screening Full) | Test ($N=78$) | **85.72%** | **0.8572** ($85.72\%$) | **0.00%** ($0.0000$) | $0.8841 \pm 0.1983$ | **Category A** | Exact match to 4 decimal places; identical split and weights. |
| **BUSBRA** | Breast US (Brazil Registry) | Test ($N=282$) | **75.06%** | **0.7511** ($75.11\%$) | **+0.05%** ($+0.0005$) | $0.7942 \pm 0.2081$ | **Category A** | Replicated lineage; MC sampling variance ($T=30$) and float32 backend. |
| **BUSUC** | Breast US (Multi-center) | Test ($N=122$) | **84.37%** | **0.8430** ($84.30\%$) | **-0.07%** ($-0.0007$) | $0.8899 \pm 0.1078$ | **Category A** | Replicated lineage; MC sampling variance ($T=30$) and float32 backend. |
| **BUID** | Breast US (Academic) | Test ($N=35$) | **78.99%** | **0.7900** ($79.00\%$) | **+0.01%** ($+0.0001$) | $0.8138 \pm 0.2087$ | **Category A** | Replicated lineage; MC sampling variance ($T=30$) and float32 backend. |
| **UDIAT** | Breast US (Spain) | Test ($N=25$) | **74.64%** | **0.7467** ($74.67\%$) | **+0.03%** ($+0.0003$) | $0.7849 \pm 0.3015$ | **Category A** | Replicated lineage; MC sampling variance ($T=30$) and float32 backend. |
| *Pooled OOD (Weighted)* | Multi-Center External | Test ($N=464$) | **77.78%** (Sample-Weighted) | **0.7780** ($77.80\%$) | **+0.02%** ($+0.0002$) | $0.8203 \pm 0.1996$ | **Category A** | Micro-pooled individual scans; exact weighted average alignment. |
| *Macro OOD (Unweighted)* | Arithmetic Mean (4 Sites) | 4 Datasets | **78.27%** (Unweighted Macro) | **0.7827** ($78.27\%$) | **0.00%** ($0.0000$) | 0.8207 | **Category A** | Exact match of unweighted macro-mean across the four external sites. |

---

## 3. Methodological Classification & Equivalence Analysis

### 3.1 Classification Framework
- **Category A (Directly Equivalent / Replicated Lineage):** The evaluation uses identical model weights, matching split partitions, identical image preprocessing, matching text prompts, and equivalent metric implementations. Minor numerical discrepancies ($\le 0.07\%$) stem purely from stochastic Monte Carlo dropout sampling and hardware float32 precision.
- **Category B (Partially Comparable):** Shared architecture and general cohort, but meaningful protocol variations (e.g., prompt modifications, different spatial aspect ratios).
- **Category C (Not Directly Comparable):** Different training datasets, different cohorts, or incompatible evaluation rules.
- **Category D (Unavailable):** No published value exists in the paper.

### 3.2 Evidence of Replicated Lineage (Why Category A Applies Across All Benchmarks)
Every evaluated benchmark achieves near-exact parity with published figures:
1. **Model Weights:** Our experiments utilize the official checkpoint released by the authors on Hugging Face (`TahaKoleilat/MedCLIPSeg`), embodying the frozen `UniMedCLIP ViT-B/16` with Probabilistic Vision-Language (PVL) adapters trained on BUSI.
2. **Dataset Splits:** The exact test partitions defined in `assets/DATASETS.md` were evaluated without alteration ($N=78$ BUSI, $N=282$ BUSBRA, $N=122$ BUSUC, $N=35$ BUID, $N=25$ UDIAT).
3. **Inference Procedure:** Exactly $T=30$ Monte Carlo stochastic attention passes at temperature $0.20$ and spatial resolution $224 \times 224$.
4. **Attribution Boundary:** We do **not** claim independent de novo reproduction (which would require re-training from random initialization on raw training splits). Rather, we establish **verified execution equivalence** of the official released model.
5. **Source of Residual Discrepancies ($\le 0.07\%$):**
   - *Stochastic Sampling:* 30 MC forward passes involve test-time variational attention sampling; different pseudorandom number sequences introduce small variance in predicted probability maps.
   - *Hardware Architecture:* Apple Silicon MPS (Metal Performance Shaders) float32 matrix multiplications exhibit minor rounding differences compared to NVIDIA CUDA float32 implementations.

---

## 4. Distinct Role of Selective Prediction (No Replacement of Baseline)

> [!IMPORTANT]
> **Selective Prediction is NOT a Replacement Baseline:**
> The published CVPR 2026 paper evaluates only full-cohort automated segmentation without rejection.
> Our core scientific contribution is a post-hoc **Uncertainty-Based Selective Prediction** framework that operates on top of this verified baseline without retraining:
> - **BUSI Test Full ($N=78$):** Baseline DSC **$0.8572$** $\to$ Retained DSC **$0.9241$** ($+0.0669$ gain, $76.9\%$ coverage).
> - **BUSI Test Lesion ($N=65$):** Baseline DSC **$0.8286$** $\to$ Retained DSC **$0.9031$** ($+0.0745$ gain, $72.3\%$ coverage).
> - **BUSBRA ($N=282$):** Baseline DSC **$0.7511$** $\to$ Retained DSC **$0.8536$** ($+0.1025$ gain, $44.7\%$ coverage).
> - **BUSUC ($N=122$):** Baseline DSC **$0.8430$** $\to$ Retained DSC **$0.8982$** ($+0.0551$ gain, $63.9\%$ coverage).
> - **BUID ($N=35$):** Baseline DSC **$0.7900$** $\to$ Retained DSC **$0.8818$** ($+0.0919$ gain, $57.1\%$ coverage).
> - **UDIAT ($N=25$):** Baseline DSC **$0.7467$** $\to$ Retained DSC **$0.8747$** ($+0.1281$ gain, $56.0\%$ coverage).
> - **Pooled External OOD ($N=464$):** Baseline DSC **$0.7780$** $\to$ Retained DSC **$0.8718$** ($+0.0939$ gain, $51.3\%$ coverage).

Selective prediction leaves every individual model prediction unchanged, but purges low-quality segmentations to yield high-confidence retained subsets for autonomous review, routing uncertain cases to expert sonographers.
