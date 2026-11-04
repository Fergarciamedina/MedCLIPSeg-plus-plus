# Stage 04: Out-Of-Distribution (OOD) Domain Shift Evaluation

**Status:** Planned
**Target Datasets:**
- In-Distribution Source: `BUSI` (Al-Dhabyani et al.)
- Out-Of-Distribution Targets: `BUSBRA`, `BUSUC`, `BUID`, `UDIAT`

---

## 1. Goal and Research Significance

The ultimate objective of Level 0 is to test whether uncertainty-based rejection remains reliable under domain shift. In clinical deployment, ultrasound acquisitions vary across scanner manufacturers, acoustic probe frequencies, operator skill, and patient populations.

Stage 04 tests the hypothesis:
*Does the relationship between uncertainty and segmentation error weaken under domain shift, and can the BUSI-validation-calibrated rejection threshold $\tau$ still safeguard against failures in unseen hospital datasets?*

---

## 2. Experimental Protocol

1. **Frozen Checkpoint:**
   The UniMedCLIP ViT-B/16 checkpoint trained on BUSI (`seed 666`) is evaluated without any fine-tuning, test-time adaptation, or parameter update.
2. **Frozen Threshold ($\tau_{\text{Val}}$):**
   The exact numerical threshold $\tau_c$ determined on the BUSI validation split is applied directly to filter predictions across each OOD dataset:
   $$\hat{y}_{\text{decision}} = \begin{cases} \text{Accept}, & U(\mathbf{x}) \le \tau_{\text{Val}} \\ \text{Reject / Flag}, & U(\mathbf{x}) > \tau_{\text{Val}} \end{cases}$$
3. **Reported Quantities per Domain:**
   - **Effective Realized Coverage:** Percentage of images accepted under the fixed threshold $\tau$.
   - **Accepted-Set Mean DSC & NSD:** Improvement in segmentation quality over the uncurated full set.
   - **OOD Failure AUROC:** Discriminative capacity for detecting low-quality segmentations ($\text{DSC} < 0.5$) under domain shift.
   - **Generalization Degradation Gap:** Difference in Spearman $\rho$ and AURC between ID (BUSI) and OOD targets.

---

## 3. Dataset Availability and Compatibility

Execution of Stage 04 will proceed as dataset archives and compatible ground-truth segmentation masks are downloaded and placed into `project/data/raw/<DATASET>/`:
- **BUSBRA:** Multi-center breast ultrasound dataset.
- **BUSUC:** Clinical breast ultrasound collection with diverse scanner artifacts.
- **BUID:** Breast Ultrasound Images Dataset.
- **UDIAT:** Benchmark ultrasound dataset from UDIAT Diagnostic Centre.
