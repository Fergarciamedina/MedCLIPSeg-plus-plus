# Collaborator Integration Guide: Uncertainty-Based Selective Prediction

## 1. Executive Summary

This package provides a post-hoc selective prediction and quality triage module for MedCLIPSeg. It evaluates test-time predictions without modifying network weights, retraining, or test-time adaptation.

By aggregating dense predictive entropy maps strictly over the predicted lesion ($U_{\mathrm{fg}}$), the module mitigates the background area dilution effect inherent to global spatial averages ($U_{\mathrm{mean}}$), enabling reliable quality filtering under cross-domain distribution shifts.

### Frozen Operating Point
- **Primary Aggregator:** $U_{\mathrm{fg}}$ (Foreground Mean Entropy)
- **Calibrated Decision Threshold:** $\tau = 0.064058324$ (calibrated on BUSI validation at nominal 80% coverage)
- **Screening Empty Fallback:** $\tau_{\mathrm{neg}} = 10^{-4}$ ($U_{\mathrm{mean}}$ gate for empty predicted masks)

---

## 2. Package Architecture

```
uncertainty_selective_evaluation/
├── __init__.py                       # Package exports
├── metrics.py                        # Self-contained aggregators, DSC, NSD, RC curves, AURC
├── model_adapter.py                  # Standardized inference adapter contracts
├── evaluate.py                       # CLI evaluation tool (supports CSVs and demo mode)
├── test_reproducibility.py           # Clean-room verification suite
├── requirements.txt                  # Minimal dependencies (numpy, scipy, pandas, matplotlib)
├── README.md                         # Detailed documentation and API reference
└── COLLABORATOR_INTEGRATION_GUIDE.md # This integration document
```

---

## 3. How to Integrate Into MedCLIPSeg Test Pipelines

### Option A: Direct Post-Hoc Evaluation on Existing CSVs
If your evaluation scripts already dump per-image metrics to CSV (e.g. columns `image_id`, `DSC`, `u_fg`):
```bash
python research/stage_4/uncertainty_selective_evaluation/evaluate.py \
  --input_csv /path/to/test_predictions.csv \
  --threshold 0.064058324 \
  --output_dir ./selective_results
```

### Option B: Seamless Integration in Python Evaluation Loops
```python
from uncertainty_selective_evaluation import (
    compute_u_fg,
    compute_dice,
    apply_selective_rejection,
    compute_risk_coverage_curve,
    compute_aurc,
)

# During test evaluation:
# entropy_map: 2D numpy array of predictive entropy from MC passes
# pred_mask: 2D binary numpy array of model prediction
# gt_mask: 2D binary numpy array of ground truth

u_fg = compute_u_fg(entropy_map, pred_mask)
dsc = compute_dice(pred_mask, gt_mask)

# Selective triage decision
tau = 0.064058324
is_accepted = u_fg <= tau  # True = Retain for autonomous review, False = Refer
```

---

## 4. Verification and Clean-Room Testing

To verify that the evaluation toolkit functions correctly on your local machine:
```bash
python research/stage_4/uncertainty_selective_evaluation/test_reproducibility.py
```
This runs synthetic verification and matches all published multi-center cross-domain benchmarks.

---

See the [Stage 4 protocol](../EXPERIMENTAL_PROTOCOL.md) for calibration and provenance.
