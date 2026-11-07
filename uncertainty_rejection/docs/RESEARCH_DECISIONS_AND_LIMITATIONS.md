# ELEC 6661 — Scientific Research Decisions & Methodological Limitations

**Document:** `RESEARCH_DECISIONS_AND_LIMITATIONS.md`  
**Location:** `uncertainty_rejection/docs/`  
**Date:** 2026-11-07  
**Scope:** Comprehensive record of core research design decisions, architectural choices, and disclosed clinical/statistical limitations across in-distribution and out-of-distribution evaluations.

---

## 1. Core Research Design Decisions

### 1.1. Training-Free Formulation vs. Fine-Tuning
- **Decision:** Evaluate predictive uncertainty quantification and selective rejection strictly without retraining, fine-tuning, or modifying the baseline MedCLIPSeg weights.
- **Scientific Justification:** Clinical deployment often requires integrating automated quality control wrappers around frozen, vendor-provided or regulatory-cleared foundation models. Requiring task-specific fine-tuning or retraining degrades generalization, risks catastrophic forgetting of pretraining visual-language representations, and introduces severe compute barriers in clinical facilities.
- **Impact:** Demonstrates that standard stochastic inference yields powerful quality signals without parameter updates.

### 1.2. Monte Carlo Dropout Sampling ($T = 30$)
- **Decision:** Utilize $T = 30$ stochastic forward passes per image with active dropout layers.
- **Scientific Justification:** Koleilat et al. (CVPR 2026) established that $T=30$ provides an optimal tradeoff between entropy variance stabilization and computational throughput. In our pilot testing, $T < 20$ yielded noisy boundary entropy estimates, whereas $T > 30$ scaled inference latency linearly without meaningful improvement in Spearman rank correlation against DSC.

### 1.3. Foreground-Normalized Entropy ($U_{\mathrm{fg}}$) vs. Global Averaging ($U_{\mathrm{mean}}$)
- **Decision:** Reject global full-image mean entropy ($U_{\mathrm{mean}}$) and standardize on foreground-normalized predictive entropy ($U_{\mathrm{fg}}$) as the primary rejection metric.
- **Scientific Justification:** Global averaging integrates over $224 \times 224 = 50,176$ pixels. Because background ultrasound speckle has lower mean entropy than ambiguous lesion margins, any image with a larger lesion contains a larger absolute count of high-entropy pixels. Consequently, $U_{\mathrm{mean}}$ exhibits an artificial positive correlation with lesion footprint ($\rho = +0.57$ to $+0.90$, $p < 10^{-3}$), rejecting large, clinically critical tumors even when segmented accurately. $U_{\mathrm{fg}}$ normalizes by $|\Omega_{\mathrm{fg}}|$, decoupling confidence from tumor dimensions.

### 1.4. Validation-Only Threshold Calibration ($\tau = 0.064058324$)
- **Decision:** Calibrate all numerical decision thresholds strictly on the BUSI validation set ($N=65$ lesion cohort) targeting 80% coverage, and freeze them permanently for in-distribution and out-of-distribution test evaluation.
- **Scientific Justification:** Zero-leakage protocol. To test genuine zero-shot domain transfer, rejection thresholds cannot be re-tuned on target hospital data, which would simulate unrealistic supervised recalibration.

### 1.5. Two-Stage Empty-Mask Deployment Policy
- **Decision:** When the predicted mask is completely empty ($|\Omega_{\mathrm{fg}}| = 0$), $U_{\mathrm{fg}}$ is mathematically undefined ($\frac{0}{0}$). We implement a two-stage triage policy:
  - If $|\Omega_{\mathrm{fg}}| == 0$ and background entropy $U_{\mathrm{mean}} \le \tau_{\mathrm{neg}}$ ($10^{-4}$): accept as confident negative scan.
  - If $|\Omega_{\mathrm{fg}}| == 0$ and $U_{\mathrm{mean}} > \tau_{\mathrm{neg}}$: abstain with $+ \infty$ uncertainty to trigger manual sonographer review (suspicious missed lesion).
  - If $|\Omega_{\mathrm{fg}}| > 0$: score $= U_{\mathrm{fg}}$.
- **Scientific Justification:** Provides complete coverage for clinical screening populations containing true normal scans (e.g. BUSI) without requiring ground-truth knowledge at inference time.

### 1.6. Breast-Only Ultrasound OOD Scope
- **Decision:** Restrict out-of-distribution benchmarks to external breast ultrasound datasets (UDIAT, BUID, BUSUC, BUSBRA) rather than cross-organ benchmarks (e.g., DDTI thyroid ultrasound).
- **Scientific Justification:** Cross-organ transfer introduces organ-specific semantic shifts (thyroid parenchyma vs. breast glandular tissue) and text prompt misalignment that confound acquisition hardware shifts with anatomical domain change. Breast ultrasound benchmarks isolate scanner hardware, transducer frequency, and center demographics.

---

## 2. Disclosed Methodological & Clinical Limitations

### 2.1. Residual Correlation with Lesion Footprint
- **Observation:** While $U_{\mathrm{fg}}$ eliminates the severe positive confounding of $U_{\mathrm{mean}}$ ($\rho = +0.57$ to $+0.90$), it exhibits a mild-to-moderate **negative** correlation with lesion area in external benchmarks:
  - BUSI: $\rho = -0.033$ ($p = 0.795$, ns)
  - UDIAT: $\rho = +0.074$ ($p = 0.726$, ns)
  - BUID: $\rho = -0.360$ ($p = 0.034$, \*)
  - BUSUC: $\rho = -0.312$ ($p = 4.7 \times 10^{-4}$, \*\*\*)
  - BUSBRA: $\rho = -0.316$ ($p = 5.6 \times 10^{-8}$, \*\*\*)
- **Mechanistic Cause:** Small lesions contain relatively few interior pixels, making their average foreground entropy more sensitive to perimeter boundary speckle.
- **Scientific Limitation:** We do not claim zero confounding. The method prevents systematic penalization of large tumors, but exhibits modest residual sensitivity to tiny lesions.

### 2.2. False Certainty Failures
- **Observation:** Selective rejection is **not guaranteed error detection**. In BUSBRA, MedCLIPSeg produced a small number of confident segmentation failures where $U_{\mathrm{fg}} \le \tau$ but $\text{DSC} < 0.50$.
- **Mechanistic Cause:** High acoustic attenuation, posterior shadowing, or rib artifacts that MedCLIPSeg confidently misidentified as solid lesion tissue with low internal entropy variance.
- **Clinical Implication:** Automated selective prediction must function as a workload prioritizer in a hybrid human-in-the-loop workflow, not as an autonomous diagnostic authority.

### 2.3. Patient-Level Clustering in Public Datasets
- **Observation:** Public benchmarks generally provide sequential image filenames rather than DICOM patient identifiers.
  - In BUSBRA, audit of 282 test files identified 262 unique patient prefixes, with 20 patients contributing 2 scans each ($7.1\%$ clustering rate).
  - In UDIAT, BUID, and BUSUC, patient identifiers are absent from public releases.
- **Statistical Limitation:** Standard benchmark practice treats images as conditionally independent observations. True patient-level multi-level clustering cannot be modeled without proprietary institutional metadata.

### 2.4. Pixel-Grid Coordinate Assumption in Surface Dice (NSD)
- **Observation:** `SurfaceDice.py` is evaluated with `spacing_mm = [1, 1, 1]` on 2D images resized to $224 \times 224$, with boundary tolerance $\tau_{\mathrm{surf}} = 2$.
- **Technical Limitation:** The tolerance represents 2 pixel-grid units, not 2 physical millimeters. True physical millimeter tolerances require uncropped DICOM pixel spacing attributes that are stripped from public research PNGs.

### 2.5. Realized Coverage Variation Under Domain Shift
- **Observation:** A fixed threshold $\tau = 0.064058324$ calibrated for $80\%$ validation coverage does not produce $80\%$ coverage under domain shift:
  - BUSI Test: $72.3\%$
  - UDIAT: $56.0\%$
  - BUID: $57.1\%$
  - BUSUC: $63.9\%$
  - BUSBRA: $44.7\%$
- **Scientific Insight:** Increased clinical noise, acoustic speckle, and scanner contrast shifts naturally increase foreground predictive entropy, triggering greater automated abstention ($55.3\%$ abstention on BUSBRA). This represents desirable conservative behavior, but means target coverage cannot be guaranteed a priori without localized site calibration.

