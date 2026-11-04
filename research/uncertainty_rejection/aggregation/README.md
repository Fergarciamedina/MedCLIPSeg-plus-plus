# Stage 02: Aggregation Metrics & Correlation Analysis

**Status:** In Progress
**Related Scripts:**
- Analysis Script: [`repo/mac/src/step1_matrix_trends.py`](../../../mac/src/step1_matrix_trends.py)
- Metrics Computation: [`repo/mac/src/utils/eval.py`](../../../mac/src/utils/eval.py), [`repo/mac/src/utils/SurfaceDice.py`](../../../mac/src/utils/SurfaceDice.py)

---

## 1. Goal and Description

Stage 02 aggregates dense $H \times W$ pixel-level uncertainty maps into scalar image-level uncertainty scores $U$, and evaluates their capacity to predict true segmentation errors measured by Dice Similarity Coefficient (DSC) and Normalized Surface Distance (NSD).

---

## 2. Summary Uncertainty Metrics

For each image, given raw uncertainty map $\mathbf{U} \in \mathbb{R}^{H \times W}$ and predicted foreground mask $\hat{\mathbf{M}} \in \{0, 1\}^{H \times W}$:

1. **$U_{\text{mean}}$ (Global Mean Entropy):**
   $$U_{\text{mean}} = \frac{1}{H \cdot W} \sum_{i, j} \mathbf{U}_{i, j}$$
2. **$U_{\text{positive}}$ / $U_{\text{fg}}$ (Foreground Lesion Mean):**
   Mean entropy strictly over predicted positive pixels $\{ (i, j) : \hat{\mathbf{M}}_{i, j} = 1 \}$.
   $$U_{\text{fg}} = \frac{1}{\sum \hat{\mathbf{M}}} \sum_{i, j} \mathbf{U}_{i, j} \cdot \hat{\mathbf{M}}_{i, j}$$
3. **$U_{\text{std\_fg}}$ (Foreground Standard Deviation):**
   Standard deviation of entropy within the predicted lesion, capturing spatial variability and boundary ambiguity.
4. **$U_{\text{ratio}}$ (Foreground-to-Global Ratio):**
   $$U_{\text{ratio}} = \frac{U_{\text{fg}}}{U_{\text{mean}} + \epsilon}$$
5. **$U_{\text{boundary}}$ (Lesion Boundary Band Uncertainty):**
   Mean entropy computed within a narrow morphological boundary band (e.g., 2-pixel dilation minus erosion of $\hat{\mathbf{M}}$).

---

## 3. Robustness and Edge Case Handling

- **Empty Masks ($\sum \hat{\mathbf{M}} = 0$):** In cases with no predicted lesion (e.g., true normal breast ultrasound images or complete false negatives), $U_{\text{fg}}$, $U_{\text{std\_fg}}$, and $U_{\text{boundary}}$ are analytically assigned $0.0$, and the case is evaluated on $U_{\text{mean}}$.
- **Missing or Corrupted Files:** Exception handling must ensure execution terminates gracefully with informative diagnostic logging rather than silent omission of images.

---

## 4. Evaluation and Statistical Validation

- **Spearman Rank Correlation ($\rho$):** Measures monotonic relationship between uncertainty $U$ and segmentation quality (DSC, NSD).
- **Failure AUROC:** Evaluates ability of $U$ to discriminate failure images from successful segmentations, with failure defined a priori as:
  $$\text{Failure} = \mathbb{I}(\text{DSC} < 0.5) \quad \text{and} \quad \mathbb{I}(\text{NSD} < 0.5)$$

> [!NOTE]
> The current [`step1_matrix_trends.py`](../../../mac/src/step1_matrix_trends.py) script implements $U_{\text{mean}}$, $U_{\text{fg}}$, $U_{\text{ratio}}$, and $U_{\text{std\_fg}}$. It will be updated to consume raw float `.npy` inputs once Stage 01 completes verification.
