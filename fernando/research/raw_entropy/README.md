# Stage 01: Raw Entropy Extraction

**Status:** In Progress (Technical enhancement pending verification in code)
**Related Scripts:**
- Mac MPS Runner: [`repo/mac/src/test.py`](../../../mac/src/test.py)
- Colab CUDA Runner: [`repo/colab/src/test.py`](../../../colab/src/test.py)
- Shell Workflow: [`repo/mac/run.sh`](../../../mac/run.sh)

---

## 1. Goal and Description

The objective of Stage 01 is to extract uncompressed, pixel-wise predictive uncertainty from MedCLIPSeg's Monte Carlo variational Value attention and persist it as **raw `float32` arrays (`.npy`)**, saved **strictly before** any per-image min-max normalization (`normalize(u_map)`) or colormap application.

Simultaneously, existing RGB colorized `.png` uncertainty maps must continue to be saved to ensure backwards compatibility with visual inspection workflows.

---

## 2. Technical Specification

### Inputs
- Preprocessed ultrasound image tensor: $\mathbf{x} \in \mathbb{R}^{B \times 3 \times H \times W}$ (default $H = W = 352$).
- Clinical text prompt string: e.g., `"ultrasound breast cancer segmentation"`.
- Frozen model checkpoint: UniMedCLIP ViT-B/16 checkpoint (`best_dice.pth` or released Hugging Face weights).
- Number of Monte Carlo passes: $S = \text{cfg.TEST.NUM\_SAMPLES}$ (default: 10).

### Mathematical Operation
1. Collect $S$ stochastic forward passes of sigmoid logits: $p^{(s)} = \sigma(f_\theta(\mathbf{x}, \mathbf{t}))$.
2. Compute predictive mean: $\bar{p} = \frac{1}{S} \sum_{s=1}^S p^{(s)}$.
3. Compute binary pixel entropy (predictive uncertainty):
   $$H(p) = - \left[ \bar{p} \log(\bar{p} + \epsilon) + (1 - \bar{p}) \log(1 - \bar{p} + \epsilon) \right]$$
4. Output predicted binary segmentation mask via thresholding: $\hat{\mathbf{y}} = \mathbb{I}(\bar{p} > 0.5)$.

### Outputs
- Predicted binary masks: `data/outputs/<DATASET>/seg_results/seed<S>/.../<filename>.png` ($H \times W$, binary uint8 $\{0, 255\}$).
- **Target Raw Uncertainty:** `data/outputs/<DATASET>/unc_results/seed<S>/.../<filename>.npy` ($H \times W$, float32 entropy values in $[0, \ln 2]$).
- Visualization Uncertainty: `data/outputs/<DATASET>/unc_results/seed<S>/.../<filename>.png` ($H \times W \times 3$, normalized 8-bit nipy_spectral colormap).

---

## 3. Validation Requirement

Before proceeding to metric aggregation in Stage 02:
1. Verify that raw `.npy` arrays contain unclipped floating-point numbers whose maximum theoretical value is bounded by $\ln(2) \approx 0.6931$.
2. Confirm array shapes match the predicted binary mask dimensions $(352, 352)$.
3. Ensure no disk I/O race conditions occur during batch generation.
