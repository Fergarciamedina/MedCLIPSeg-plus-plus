# Research Contribution: Uncertainty Quantification & Selective Prediction

**Research Area:** Post-Hoc Uncertainty Estimation, Error Correlation & Rejection Control\
**Course:** ELEC 6661 — Medical Image Processing (Fall 2026), Concordia University\
**Lab:** IMPACT Lab (Prof. Hassan Rivaz, Omid Nejati Manzari)\
**Development Branch:** `feature/uncertainty-rejection`

---

## 1. Research Scope & Contributions

This module implements the core scientific contribution for uncertainty-guided segmentation and rejection control, structured around three levels:

1. **Level 0: Scientific Research & Evaluation (Mandatory ELEC 6661 Scope)**
   - Post-hoc uncertainty quantification using MedCLIPSeg's Monte Carlo Value attention.
   - Persistence of uncompressed, raw `float32` pixel-level entropy arrays (`.npy`) prior to visualization normalization.
   - Extraction of 5 image-level aggregation features: $U_{\text{fg}}$ (foreground mean), $U_{\text{std\_fg}}$ (foreground std dev), $U_{\text{boundary}}$ (morphological boundary band), $U_{\text{mean}}$ (global mean), and $U_{\text{ratio}}$.
   - Statistical validation: Spearman rank correlation ($\rho$) against DSC and NSD, and failure discrimination AUROC ($\text{DSC} < 0.5$).
   - Selective prediction: calibrating rejection threshold $\tau_c$ on the BUSI validation set, evaluating Risk-Coverage curves and Area Under Risk-Coverage (AURC), and benchmarking against Random, Naive Confidence, and Oracle controls.
   - Out-of-distribution (OOD) evaluation across unseen breast ultrasound targets (`BUSBRA`, `BUSUC`, `BUID`, `UDIAT`).

2. **Level 1: Priority UI Dashboard (Optional PLUS A — Architectural Specification)**
   - Triage system for unannotated clinical cases, ranking images by uncertainty score ($U$) to prioritize difficult segmentations for human sonographer review.

3. **Level 2: Multimodal Similarity Search (Optional PLUS B — Architectural Specification)**
   - Retrieval engine combining frozen CLIP visual embeddings with within-list block-summary predicate pruning in FAISS IVF.

---

## 2. Directory Structure

```text
uncertainty_rejection/
├── README.md                          # Contribution overview and reproduction guide
│
├── src/                               # Core uncertainty quantification package
│   └── uncertainty/
│       ├── __init__.py                # Exported API
│       ├── raw_entropy.py             # Binary entropy computation & .npy / PNG persistence
│       ├── aggregation.py             # Feature computation (U_mean, U_fg, U_std_fg, U_ratio, U_boundary)
│       ├── metrics.py                 # Spearman rho & failure AUROC
│       └── selective_prediction.py    # Tau calibration, risk-coverage curves & controls
│
├── research/                          # Level 0 scientific experiments
│   ├── README.md                      # Experimental pipeline overview
│   ├── raw_entropy/                   # Stage 01: Raw entropy specifications
│   ├── aggregation/                   # Stage 02: run_aggregation.py (correlation & AUROC)
│   ├── selective_prediction/          # Stage 03: run_selective_prediction.py (selective prediction)
│   ├── ood_evaluation/                # Stage 04: OOD domain shift protocol
│   └── future_work.md                 # Untested hypotheses and future research directions
│
├── systems/                           # Clinical application systems (Scaffolding)
│   ├── priority_ui/                   # Level 1 triage specifications
│   └── similarity_search/             # Level 2 similarity search specifications
│
├── notebooks/                         # Interactive analysis
│   └── evaluate.ipynb                 # Interactive evaluation notebook
│
├── scripts/                           # Execution scripts
│   ├── run_aggregation.sh            # Stage 02 execution runner
│   ├── run_selective_prediction.sh   # Stage 03 execution runner
│   └── run_tests.sh                  # Automated test suite runner
│
└── tests/                             # Automated test suite
    ├── test_imports.py                # Package import validation
    ├── test_uncertainty.py            # Mathematical bounds and selective logic tests
    └── test_config_loading.py         # Configuration file validation
```

---

## 3. Implementation Attribution

- **Shared Baseline:** Builds upon the frozen MedCLIPSeg baseline located in `shared/medclipseg/`.
- **Contribution Scope:**
  - Raw `float32` `.npy` array persistence added to the inference loop (`shared/medclipseg/test.py`).
  - The entire `uncertainty_rejection/src/uncertainty/` package.
  - All experimental runners (`run_aggregation.py`, `run_selective_prediction.py`).
  - All test suites in `uncertainty_rejection/tests/`.

---

## 4. Execution & Reproduction

From the repository root:

```bash
# 1. Run automated test suite
bash uncertainty_rejection/scripts/run_tests.sh

# 2. Run Stage 02 aggregation and error correlation analysis
bash uncertainty_rejection/scripts/run_aggregation.sh BUSI 666

# 3. Run Stage 03 selective prediction and control comparisons
bash uncertainty_rejection/scripts/run_selective_prediction.sh BUSI 666 U_fg
```
