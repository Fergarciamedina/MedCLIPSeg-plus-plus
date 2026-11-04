# Stage 03: Selective Prediction & Risk-Coverage Analysis

**Status:** Planned
**Related Components:**
- Calibration Dataset: `project/data/raw/BUSI/Val_Folder/`
- Test Evaluation Set: `project/data/raw/BUSI/Test_Folder/`

---

## 1. Goal and Methodology

Selective prediction allows the segmentation system to abstain from making autonomous predictions on high-uncertainty cases, routing them to expert clinical review. Stage 03 evaluates whether this abstention rule systematically improves the quality (mean DSC) of accepted predictions.

---

## 2. Threshold Calibration vs. Descriptive Ranking

To maintain strict scientific integrity, two distinct evaluations are conducted:

1. **Deployable Threshold Calibration ($\tau$ on Validation Only):**
   - The numerical threshold $\tau_c$ corresponding to a target coverage $c \in \{100\%, 90\%, 80\%, 70\%\}$ is **strictly calibrated on the BUSI validation set (`Val_Folder`)**:
     $$\tau_c = \text{Quantile}_{c}\left(\{ U_i \}_{i \in \text{Val}}\right)$$
   - This fixed numerical threshold $\tau_c$ is subsequently applied frozen to the BUSI test set and all OOD sets.
   - **Rule:** Never tune, calibrate, or select $\tau$ on the test set.

2. **Descriptive In-Sample Ranking (Test Set Analysis):**
   - Directly rejecting the top 10%, 20%, 30% highest-uncertainty images within the test set itself is reported **strictly as a descriptive upper-performance benchmark**, clearly distinguished from a deployable calibrated rule.

---

## 3. Control Baselines

To prove that the rejection gain is non-trivial, selective prediction using $U$ is compared against three controls under identical coverage budgets:

| Control | Description | Role |
| :--- | :--- | :--- |
| **No Rejection** | All predictions are retained (100% coverage). | Baseline zero point. |
| **Random Rejection** | Images rejected uniformly at random (averaged over 1,000 bootstrap draws). | Proves uncertainty provides true error discrimination. |
| **Naive Confidence** | Rejection based on mean foreground sigmoid probability / max-probability without Monte Carlo sampling. | Tests whether Monte Carlo variational entropy outperforms simple probability. |
| **Oracle Rejection** | Images rejected strictly by lowest ground-truth DSC first. | Theoretical performance ceiling. |

---

## 4. Evaluation Metrics

- **Accepted-Set Mean DSC / NSD:** Evaluated at fixed coverages (e.g., 100%, 90%, 80%, 70%).
- **Risk-Coverage Curve:** Plots empirical risk $R = 1 - \text{DSC}$ against coverage $C \in (0, 1]$.
- **AURC (Area Under the Risk-Coverage Curve):** Summary metric for selective classification/segmentation; lower AURC indicates superior selective performance.
- **Statistical Significance:** 95% bootstrap confidence intervals computed over 2,000 resamples.
