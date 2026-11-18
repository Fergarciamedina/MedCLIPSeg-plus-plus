# Uncertainty-Guided Selective Segmentation

A self-contained scientific research module for post-hoc predictive uncertainty quantification, error correlation analysis, and validation-calibrated selective prediction in breast ultrasound image segmentation.

Part of the **[MedCLIPSeg++](https://github.com/Fergarciamedina/MedCLIPSeg-plus-plus)** research platform. For the complete scientific block diagrams (Figures R1, R2, R3), mathematical formulations, and cross-domain empirical results, see [docs/architecture/SELECTIVE_SEGMENTATION_ARCHITECTURE.md](../docs/architecture/SELECTIVE_SEGMENTATION_ARCHITECTURE.md).

---

## 1. Methodology Overview

This module provides the complete scientific implementation for:

1. **Pixel-Level Entropy Extraction:** Extracts raw `float32` pixel-wise Shannon entropy arrays from MedCLIPSeg's Monte Carlo variational attention prior to visualization colormapping.
2. **Mask-Conditioned Aggregation:** Computes 5 image-level uncertainty metrics: $U_{\text{fg}}$ (foreground mean), $U_{\text{std\_fg}}$ (foreground standard deviation), $U_{\text{boundary}}$ (morphological boundary band), $U_{\text{mean}}$ (global background grid), and $U_{\text{ratio}}$.
3. **Statistical Validation:** Evaluates Spearman rank correlation ($\rho$) against Dice Similarity Coefficient (DSC) and Normalized Surface Distance (NSD), and failure discrimination AUROC ($\text{DSC} < 0.5$).
4. **Selective Prediction:** Calibrates a deployable rejection threshold $\tau^*$ exclusively on in-distribution validation data, generates Risk-Coverage curves and Area Under Risk-Coverage (AURC), and benchmarks against Random, Naive Confidence, and Oracle controls.
5. **Domain Shift Evaluation:** Evaluates the frozen model and frozen threshold across 4 external out-of-distribution (OOD) cohorts: `BUSBRA`, `BUSUC`, `BUID`, and `UDIAT` ($N=464$ external scans).

---

## 2. Directory Structure

```text
uncertainty_rejection/
├── README.md                          # Module overview and reproduction guide
│
├── src/                               # Core Python package
│   └── uncertainty/
│       ├── __init__.py                # Exported API
│       ├── raw_entropy.py             # Binary Shannon entropy calculation & persistence
│       ├── aggregation.py             # Spatial feature aggregation (U_fg, U_mean, etc.)
│       ├── metrics.py                 # Spearman rho & failure AUROC computation
│       └── selective_prediction.py    # Tau calibration, risk-coverage curves & controls
│
├── research/                          # Canonical scientific records & experiment scripts
│   ├── EXPERIMENTAL_PROTOCOL.md       # Formal experimental design & evaluation protocol
│   ├── THEORY_AND_METRICS.md          # Mathematical formulas, metrics & conventions
│   ├── SCIENTIFIC_EVIDENCE_AND_CLAIMS.md # Empirical claims ledger with full data tables
│   ├── RESEARCH_DECISIONS_AND_LIMITATIONS.md # Methodological decisions & known boundary conditions
│   ├── PUBLISHED_BASELINE_COMPARISON.md  # Detailed comparison with Koleilat et al. baseline
│   ├── aggregation/                   # Aggregation feature computation scripts
│   ├── inference/                     # Validation inference scripts
│   ├── selective_prediction/          # Selective prediction & locked test evaluation scripts
│   └── statistical_analysis/          # Validation statistics & bootstrap CI runners
│
├── scripts/                           # Reproduction shell runners
│   ├── run_aggregation.sh            # Aggregation & correlation runner
│   ├── run_selective_prediction.sh   # Selective prediction runner
│   └── run_tests.sh                  # Regression test runner (31 tests)
│
└── tests/                             # Automated test suite
    ├── test_imports.py                # Package imports validation
    ├── test_uncertainty.py            # Mathematical bounds, aggregation & selective logic
    └── test_config_loading.py         # Config files & frozen parameter loading
```

---

## 3. Reproduction & Execution

From `MedCLIPSeg-plus-plus/`:

```bash
# 1. Run full 31-test regression suite
bash uncertainty_rejection/scripts/run_tests.sh

# 2. Run aggregation and correlation analysis on BUSI
bash uncertainty_rejection/scripts/run_aggregation.sh BUSI 666

# 3. Run selective prediction evaluation
bash uncertainty_rejection/scripts/run_selective_prediction.sh BUSI 666 U_fg
```
