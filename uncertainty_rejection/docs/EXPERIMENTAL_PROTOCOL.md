# ELEC 6661 — Experimental Protocol & Evaluation Guidelines

**Document:** `EXPERIMENTAL_PROTOCOL.md`  
**Location:** `uncertainty_rejection/docs/`  
**Date:** 2026-11-07  
**Scope:** Formal specification of the multi-phase experimental design, data isolation protocols, prompt configurations, and frozen evaluation parameters.

---

## 1. Multi-Phase Experimental Architecture

The experimental protocol is divided into three strictly isolated phases to prevent data leakage and post-hoc threshold selection:

```
┌─────────────────────────────────────────────────────────────┐
│          PHASE 1: In-Domain Validation Calibration          │
│          Cohort: BUSI Validation Split (N=78, 65 Lesion)     │
│  - Method Selection: Compare 7 uncertainty operators        │
│  - Establish Primary: U_fg (AURC=0.0839) & Comparator: std  │
│  - Calibrate Thresholds: tau=0.064058 (80% target cov)      │
│  - Select Empty-Mask Policy: two_stage (tau_neg=1e-4)       │
└──────────────────────────────┬──────────────────────────────┘
                               │ PERMANENT PARAMETER LOCK
                               ▼
┌─────────────────────────────────────────────────────────────┐
│             PHASE 2: Locked In-Domain Evaluation            │
│             Cohort: BUSI Test Split (N=78, 65 Lesion)       │
│  - Setting A: Lesion cohort evaluation with tau=0.064058    │
│  - Setting B: Full screening triage with two_stage policy   │
│  - Paired bootstrap hypothesis tests vs. baselines          │
└──────────────────────────────┬──────────────────────────────┘
                               │ ZERO-SHOT DOMAIN TRANSFER
                               ▼
┌─────────────────────────────────────────────────────────────┐
│            PHASE 3: Frozen External OOD Evaluation          │
│       Cohorts: UDIAT (25), BUID (35), BUSUC (122),          │
│                BUSBRA (282) — Total N=464 Images            │
│  - Frozen weights, frozen tau=0.064058, zero retraining     │
│  - Protocol P_orig (Baseline) & Protocol P_neutral (Screen) │
│  - Evaluate calibration transfer, gains, & area confounding │
└─────────────────────────────────────────────────────────────┘
```

---

## 2. Dataset Cohorts & Partitioning

| Phase / Dataset | Clinical Site / Scanner | Split | Images ($N$) | Lesion Prevalence | Benign / Malignant / Normal |
| :--- | :--- | :---: | :---: | :---: | :---: |
| **BUSI (Val)** | Baheya Hospital, Egypt (GE Logiq E9/7) | Validation | 78 | 83.3% | 44 / 21 / 13 |
| **BUSI (Test)** | Baheya Hospital, Egypt (GE Logiq E9/7) | Test | 78 | 83.3% | 44 / 21 / 13 |
| **UDIAT** | UDIAT Diagnostic Centre, Spain (Siemens) | Test | 25 | 100% | 17 / 8 / 0 |
| **BUID** | Public Clinical Repository | Test | 35 | 100% | 16 / 19 / 0 |
| **BUSUC** | Multi-Center Clinical Registry | Test | 122 | 100% | 54 / 68 / 0 |
| **BUSBRA** | Multi-Hospital Registry, Brazil (GE/Philips/Toshiba) | Test | 282 | 100% | 191 / 91 / 0 |

---

## 3. Pre-Registered Prompt Protocols

To isolate model capabilities from language prompt design, two prompt protocols are formalized:

### 3.1. Protocol $P_{\mathrm{orig}}$: Official Baseline (Oracle-Pathology Prompting)
- **Source:** Official `Test_text_original.xlsx` provided by MedCLIPSeg authors.
- **Content:** Contains explicit diagnostic pathology terms (`benign tumor`, `malignant tumor`, `mass`).
- **Purpose:** Direct numerical benchmarking against published MedCLIPSeg CVPR 2026 baselines.
- **Reporting Requirement:** Must be explicitly documented as Oracle-Pathology Prompting because ground-truth pathology is provided in the prompt text.

### 3.2. Protocol $P_{\mathrm{neutral}}$: Fair Screening Protocol (Label-Free Prompting)
- **Source:** Generated `Test_text_neutral.xlsx` staged in each dataset directory.
- **Content:** All diagnostic adjectives (`benign`, `malignant`) are strictly removed while preserving sentence syntax and spatial indicators (`"A tumor is present in the center region of the breast."`).
- **Purpose:** Represents realistic clinical deployment where malignancy is unknown prior to biopsy.

---

## 4. Frozen Selective Prediction Parameters (Permanent Lock)

All parameters derived in Stage 09 remain permanently locked:

| Parameter | Identifier | Numerical Value | Description & Purpose |
| :--- | :---: | :---: | :--- |
| **Primary Metric** | $U_{\mathrm{fg}}$ | Formula 2.3 | Foreground Mean Predictive Entropy |
| **Comparator** | $U_{\mathrm{std\_fg}}$ | Formula 2.6 | Foreground Entropy Population Std. Dev. |
| **Primary Cutoff** | $\tau_{80}$ | **$0.064058324$** | Calibrated for 80% coverage on BUSI validation |
| **Negative Cutoff** | $\tau_{\mathrm{neg}}$ | **$1.0 \times 10^{-4}$** | Background threshold for two-stage empty mask triage |
| **Ladder Point 95** | $\tau_{95}$ | $0.101927518$ | High-coverage operating point (target 95%) |
| **Ladder Point 90** | $\tau_{90}$ | $0.083463000$ | Moderate-coverage operating point (target 90%) |
| **Ladder Point 85** | $\tau_{85}$ | $0.072696542$ | Intermediate operating point (target 85%) |
| **Ladder Point 75** | $\tau_{75}$ | $0.059689360$ | High-precision operating point (target 75%) |
| **Ladder Point 70** | $\tau_{70}$ | $0.056208188$ | Conservative operating point (target 70%) |
| **Boundary Width** | $w$ | 2 pixels | Half-width for morphological band dilation/erosion |
| **Threshold $t$** | $t$ | $0.1$ nats | Absolute entropy threshold for $U_{\mathrm{threshold}}$ |
| **MC Samples** | $T$ | 30 | Stochastic Monte Carlo forward passes |

---

## 5. Software & Execution Environment

- **Operating System:** macOS (Darwin 24.x, Apple Silicon M-series)
- **Primary PyTorch Backend:** Metal Performance Shaders (`DEVICE = "mps"`)
- **Python Version:** Python 3.9 (Virtual environment at `project/MedCLIPSeg-plus-plus/.venv`)
- **Key Dependencies:**
  - `torch == 2.2.0`
  - `torchvision == 0.17.0`
  - `timm == 0.9.16`
  - `scipy == 1.12.0`
  - `opencv-python == 4.9.0.80`
  - `pandas == 2.2.1`
  - `numpy == 1.26.4`
- **Random Seed:** $666$ fixed for PyTorch, NumPy, Python standard library, and bootstrap resampling.

