# Uncertainty-Based Selective Prediction Evaluation Toolkit

A lightweight, modular, model-agnostic Python toolkit for evaluating uncertainty-based selective prediction in medical image segmentation under domain shift.

This toolkit accompanies the research project:
**"Uncertainty-Based Selective Prediction for Text-Guided Medical Image Segmentation Under Domain Shift"** (ELEC 6661, Concordia University).

---

## Key Capabilities

1. **Seven Spatial Uncertainty Aggregators:**
   - $U_{\mathrm{fg}}$: Foreground mean entropy (primary aggregator; eliminates background dilution).
   - $U_{\mathrm{std\_fg}}$: Foreground standard deviation of predictive entropy.
   - $U_{\mathrm{boundary}}$: Morphological boundary band mean entropy.
   - $U_{\mathrm{ratio}}$: Ratio of foreground mean entropy to global mean entropy.
   - $U_{\mathrm{mean}}$: Global grid-averaged entropy.
   - $U_{\mathrm{positive}}$: Mean entropy conditioned on strictly positive entropy pixels.
   - $U_{\mathrm{threshold}}$: Mean entropy above the 90th percentile.

2. **Selective Prediction & Quality Triage:**
   - Applies the frozen validation decision threshold $\tau = 0.064058324$.
   - Evaluates realized coverage ($c$), retained accuracy (DSC, NSD), and selective gains.
   - Generates empirical Risk-Coverage curves.
   - Calculates Area Under the Risk-Coverage Curve (AURC) alongside **Random Baseline** and **Empirical Oracle** references.

3. **Multi-Mode Execution:**
   - **Mode A (Post-Hoc CSV Evaluation):** Evaluate precomputed segmentation prediction tables.
   - **Mode B (Model Adapter Inference):** Integrate custom vision-language models via clean adapter contracts.
   - **Mode C (Demonstration Mode):** Run synthetic benchmarks instantly without data or GPU dependencies.

---

## Installation

No heavy dependencies required. Works in any standard Python 3.8+ environment:

```bash
pip install numpy scipy pandas
```

Optional dependencies for deep learning inference:
```bash
pip install torch torchvision pillow matplotlib
```

---

## Quick Start (CLI)

### 1. Run Demonstration Evaluation
Generate synthetic predictions and evaluate selective triage:
```bash
python evaluate.py --demo --output_dir ./demo_results
```

### 2. Evaluate Precomputed Model Predictions
Evaluate a CSV file containing per-image scores and Dice coefficients:
```bash
python evaluate.py \
  --input_csv ./data/predictions.csv \
  --threshold 0.064058324 \
  --aggregator u_fg \
  --output_dir ./evaluation_results
```

### 3. Compare All Seven Uncertainty Aggregators
Rank and compare all seven uncertainty aggregators by AURC and Spearman rank correlation:
```bash
python evaluate.py \
  --input_csv ./data/predictions.csv \
  --compare_aggregators \
  --output_dir ./comparison_results
```

---

## Programmatic Python API

```python
import numpy as np
from uncertainty_selective_evaluation import (
    compute_u_fg,
    compute_dice,
    compute_risk_coverage_curve,
    compute_aurc,
    compute_random_aurc,
    compute_oracle_aurc,
)

# 1. Compute foreground-averaged uncertainty
entropy_map = np.random.uniform(0.01, 0.15, size=(256, 256))
pred_mask = np.zeros((256, 256), dtype=np.uint8)
pred_mask[100:150, 100:150] = 1

u_fg_score = compute_u_fg(entropy_map, pred_mask)
print(f"U_fg score: {u_fg_score:.4f}")

# 2. Evaluate selective prediction over a cohort
scores = np.array([0.03, 0.05, 0.06, 0.09, 0.12])
losses = np.array([0.05, 0.08, 0.10, 0.35, 0.60])  # 1 - DSC

tau = 0.064058324
accepted = scores <= tau
print(f"Retained {np.sum(accepted)}/{len(scores)} cases (Coverage: {np.mean(accepted)*100:.1f}%)")

# 3. Compute Risk-Coverage Curve and AURC
coverages, risks = compute_risk_coverage_curve(scores, losses)
aurc = compute_aurc(coverages, risks)
random_aurc = compute_random_aurc(losses)
oracle_aurc = compute_oracle_aurc(losses)

print(f"AURC: {aurc:.4f} (Random: {random_aurc:.4f}, Oracle: {oracle_aurc:.4f})")
```

---

## Output Artifacts

Running `evaluate.py` generates the following machine-readable CSV outputs in `--output_dir`:

| Output File | Description |
| :--- | :--- |
| `per_image_evaluation.csv` | Image-level table with rejection decisions (`Retain` / `Refer`) and accuracy metrics. |
| `selective_summary.csv` | Summary record: baseline DSC, retained DSC, rejected DSC, selective gain, and AURC. |
| `risk_coverage_curve.csv` | Full coordinates `(coverage, selective_risk)` for publication-quality plotting. |
| `aggregator_comparison.csv` | Aggregator benchmark table ranking metrics by Spearman $\rho$ and AURC (when enabled). |

---

## File Structure

```
uncertainty_selective_evaluation/
├── __init__.py          # Public API exports
├── metrics.py           # Self-contained aggregators, segmentation metrics, and statistics
├── model_adapter.py     # Base, MedCLIPSeg, and Mock model adapters
├── evaluate.py          # Standalone CLI evaluation tool
└── README.md            # Toolkit documentation
```

---

## Reference & Citation

If you use this toolkit in your research, please cite:

```bibtex
@article{garciamedina2026selective,
  title={Uncertainty-Based Selective Prediction for Text-Guided Medical Image Segmentation Under Domain Shift},
  author={Garcia-Medina, Fernando},
  journal={ELEC 6661 Technical Report, Concordia University},
  year={2026}
}
```

