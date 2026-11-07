# ELEC 6661 — Step 11A: External Dataset Readiness and Domain Shift Protocol

**Date:** 2026-11-07  
**Stage:** Step 11A — External Dataset Readiness Audit & Pre-Registered OOD Evaluation Protocol  
**Auditor:** Automated Scientific Pair Programmer  
**Readiness Recommendation for Step 11B (Inference):** **NO-GO (Datasets Not Staged Locally)**  
**Readiness Recommendation for Protocol Specification:** **GO (Protocol Fully Frozen & Audited)**  

---

## 1. Executive Summary & Readiness Verdict

Following the completion and verification of Step 10 (Locked BUSI In-Domain Test Evaluation), Step 11A audits the readiness of external datasets and establishes a pre-registered evaluation protocol for out-of-distribution (OOD) testing.

### Key Audit Findings:
1. **Local Storage Audit:** A recursive filesystem search confirmed that only the in-distribution dataset (`BUSI`) is currently present in `project/data/raw/BUSI/`. Neither `UDIAT` (breast ultrasound) nor `DDTI` (thyroid ultrasound) is stored locally on disk.
2. **Readiness Gate Verdict:** **NO-GO for Step 11B inference execution** until dataset images, ground-truth masks, and pre-specified prompt files are downloaded, verified, and placed into canonical paths.
3. **Compatibility Verification:** The frozen MedCLIPSeg inference pipeline (`shared/medclipseg/test.py` via `--source_dataset BUSI`) supports external datasets through modular YAML configurations without modifying model weights.
4. **Prompt Protocol Pre-Specification:** Prompt transfer must be strictly separated from image domain shift. Fixed anatomical prompt templates are defined prior to any external data observation.
5. **Frozen Calibration Lock:** The primary metric ($U_{\mathrm{fg}}$), prespecified comparator ($U_{\mathrm{std\_fg}}$), rejection threshold ($\tau = 0.064058324$), and two-stage empty-mask cutoff ($\tau_{\mathrm{neg}} = 10^{-4}$) remain permanently frozen. No post-hoc recalibration is permitted on external datasets.

---

## 2. Audit of Locally Available Datasets

We executed an exhaustive scan of `project/data/raw/` and surrounding project directories:

```
project/data/raw/
└── BUSI/
    ├── Train_Folder/ (img, label)
    ├── Val_Folder/   (img, label)
    ├── Test_Folder/  (img, label)
    └── Prompts_Folder/ (Excel prompt files)
```

| Dataset Name | Target Anatomy | Local Raw Data Present? | Config Present? | Prompt File Present? | Readiness Status |
| :--- | :--- | :---: | :---: | :---: | :---: |
| **BUSI** | Breast | **YES** ($N=780$) | `BUSI.yaml` | `Test_text_original.xlsx` | Evaluated (Steps 6–10) |
| **UDIAT** | Breast | **NO** | `UDIAT.yaml` | **NO** | Blocked (Requires Data Staging) |
| **DDTI** | Thyroid | **NO** | **NO** | **NO** | Blocked (Requires Config & Staging) |
| **BUSBRA** | Breast | **NO** | `BUSBRA.yaml` | **NO** | Blocked (Secondary Alternative) |
| **BUID** | Breast | **NO** | `BUID.yaml` | **NO** | Blocked (Secondary Alternative) |
| **BUSUC** | Breast | **NO** | `BUSUC.yaml` | **NO** | Blocked (Secondary Alternative) |

**Conclusion:** External inference cannot proceed immediately. Step 11B must remain on hold until the physical image arrays and ground-truth masks are staged.

---

## 3. Dataset Profiles & Scientific Source Audit

### 3.1 Primary Candidate 1: UDIAT (Near-Domain Acquisition Shift)
- **Scientific Source:** Yap, M. H., Pons, G., Martí, J., Ganau, S., Sentís, M., Zwiggelaar, R., Davison, A. K., & Martí, R. (2017). *"Automated breast ultrasound lesions detection using convolutional neural networks."* IEEE Journal of Biomedical and Health Informatics, 22(4), 1218–1226.
- **Clinical Source:** UDIAT Diagnostic Centre of Parc Taulí Corporation, Sabadell, Spain.
- **Anatomy & Target:** Female breast; focal breast lesion segmentation (hypoechoic masses, benign fibroadenomas/cysts, invasive ductal carcinomas).
- **Cohort Size:** 163 images from 163 distinct female patients (110 benign, 53 malignant).
- **Scanner Hardware:** Siemens Elegra ultrasound system with a 7.5 MHz linear transducer.
- **Format:** Grayscale PNG/BMP images with binary ground-truth segmentation masks.
- **Licensing & Permitted Use:** Public academic research benchmark (commonly distributed as Breast Ultrasound Dataset B). Non-commercial educational use permitted.
- **Overlap Risk with BUSI:** **Zero overlap**. BUSI was collected at Baheya Hospital, Cairo, Egypt using GE LOGIQ E9 and LOGIQ 7 systems; UDIAT was collected in Spain with Siemens hardware.
- **Nature of Domain Shift:**
  - **In-Domain Anatomy:** Female breast ultrasound.
  - **Out-of-Domain Scanner & Demographics:** Cross-institution, cross-country, and cross-manufacturer acoustic speckle and gain shift.

### 3.2 Primary Candidate 2: DDTI (Far-Domain Anatomical Shift)
- **Scientific Source:** Pedraza, L., Vargas, C., Narváez, F., Durán, O., Muñoz, E., & Romero, E. (2015). *"An open access thyroid ultrasound-image database."* In 10th International Symposium on Medical Information Processing and Analysis (Vol. 9287, pp. 188–193). SPIE.
- **Clinical Source:** Universidad Nacional de Colombia and Hospital Universitario San Ignacio, Bogotá, Colombia.
- **Anatomy & Target:** **Thyroid gland**; thyroid nodule segmentation (solid, cystic, mixed nodules, thyroiditis).
- **Cohort Size:** ~637 B-mode ultrasound images across ~298 patients (~400 cases with full expert segmentation contours).
- **Scanner Hardware:** Toshiba Nemio 30, Toshiba Nemio MX with 7.5–12 MHz linear probes.
- **Format:** JPEG/PNG ultrasound frames with XML contour annotations or converted binary masks.
- **Licensing & Permitted Use:** Open-access academic repository. Free for non-commercial scientific research with attribution.
- **Overlap Risk with BUSI:** **Zero overlap** (completely distinct anatomical region, country, and scanner).
- **Nature of Domain Shift:**
  - **Cross-Anatomical Shift (Far OOD):** Thyroid vs. Breast tissue architecture. Neck ultrasound contains thyroid parenchyma, trachea, carotid artery, jugular vein, and strap muscles, contrasting sharply with breast fibroglandular parenchyma and subcutaneous adipose tissue.
  - **Acoustic & Frequency Shift:** Higher frequency linear probes ($10\text{–}12\text{ MHz}$) with distinct speckle attenuation.

---

## 4. MedCLIPSeg Compatibility Matrix

We audited the technical pipeline requirements in `shared/medclipseg/`:

| Dimension | MedCLIPSeg Expectation | UDIAT Compatibility | DDTI Compatibility | Remediation / Handling |
| :--- | :--- | :--- | :--- | :--- |
| **Input Spatial Size** | $224 \times 224$ pixels | Varies ($\approx 760 \times 570$) | Varies ($\approx 500 \times 400$) | Bicubic resize to $224 \times 224$ in `ValGenerator` |
| **Channels** | 3-channel RGB (or 1-ch expanded) | 1-channel grayscale / RGB | 1-channel grayscale / RGB | `correct_dims` expands 1D $\to$ 3D |
| **Normalization** | ImageNet / CLIP stats: $\mu=[0.485, 0.456, 0.406]$, $\sigma=[0.229, 0.224, 0.225]$ | Grayscale converted to $[0, 1]$ then normalized | Grayscale converted to $[0, 1]$ then normalized | Handled natively by `CLIP_NORMALIZE` |
| **Mask Encoding** | Binary $\{0, 1\}$; resized with `INTER_NEAREST` | Binary $\{0, 255\}$ | Binary $\{0, 255\}$ or XML | Threshold at $127 \to \{0, 1\}$; `INTER_NEAREST` |
| **Model Weights** | `MedCLIPSeg_unimedclip_ViT-B-16_latest.pth` | Same frozen weights | Same frozen weights | Pass `--source_dataset BUSI` |
| **Inference Mode** | MC Dropout, $T=30$ forward passes | $T=30$ forward passes | $T=30$ forward passes | Preserve exact MC dropout setting |
| **Prompt Format** | Text string in sample dictionary | Text string required | Text string required | Requires dedicated prompt table |

---

## 5. Pre-Inference Text Prompt Protocol

A major methodological hazard in text-guided segmentation is **confounding image domain shift with prompt mismatch**. If an external breast dataset is evaluated with a generic prompt ("breast ultrasound") while BUSI was evaluated with fine-grained descriptive prompts ("One small circle-shaped regular tumor at the center..."), any performance drop could be driven by prompt degradation rather than acoustic domain shift.

### Pre-Specified Prompting Design:
To isolate these phenomena, the OOD evaluation protocol will test two pre-registered prompt regimes:

1. **Protocol P1 (Domain-Aligned Canonical Template):**
   - For Breast Ultrasound (UDIAT):
     ```
     "A breast ultrasound image showing a tumor."
     ```
     *(Applied uniformly across all lesion cases; captures baseline prompt transfer).*
   - For Thyroid Ultrasound (DDTI):
     ```
     "A thyroid ultrasound image showing a nodule."
     ```
     *(Anatomically aligned for the thyroid domain).*
2. **Protocol P2 (Class-Conditioned Prompt):**
   - For Benign Cases: `"A breast ultrasound image showing a benign tumor."` (or thyroid nodule)
   - For Malignant Cases: `"A breast ultrasound image showing a malignant tumor."` (or thyroid nodule)

**Strict Rule:** No prompt engineering, post-hoc optimization, or text search will be conducted based on external segmentation outcomes.

---

## 6. Pre-Registered OOD Evaluation Specification

The OOD evaluation in Step 11B will execute under the exact mathematical rules verified in Step 9 and Step 10:

1. **Uncertainty Aggregators:** All seven metrics ($U_{\mathrm{fg}}$, $U_{\mathrm{std\_fg}}$, $U_{\mathrm{mean}}$, $U_{\mathrm{boundary}}$, $U_{\mathrm{threshold}}$, $U_{\mathrm{positive}}$, $U_{\mathrm{ratio}}$) computed from raw float32 predictive entropy arrays.
2. **Primary Rejection Metric:** $U_{\mathrm{fg}}$ (Foreground Mean Predictive Entropy).
3. **Prespecified Comparator:** $U_{\mathrm{std\_fg}}$ (Foreground Entropy Population Standard Deviation).
4. **Primary Rejection Threshold:** $\tau = 0.064058324$ (Locked from BUSI validation 80% coverage calibration).
5. **Secondary Operating Profile:** The 5 additional validation thresholds ($\tau_{95}=0.10193$, $\tau_{90}=0.08346$, $\tau_{85}=0.07270$, $\tau_{75}=0.05969$, $\tau_{70}=0.05621$) evaluated as secondary points without recalibration.
6. **Setting A Reporting (Lesion-Present Cohort):**
   - Evaluated strictly on scans with predicted foreground ($|\Omega_{\mathrm{fg}}| > 0$).
   - Report realized coverage, accepted DSC, accepted NSD, absolute gains, and 95% bootstrap CIs.
   - Compute Risk-Coverage curves and AURC (1-DSC and 1-NSD) on common empirical coordinates.
7. **Setting B Reporting (Full Screening Cohort):**
   - Evaluates all available external scans using the frozen `two_stage` policy ($\tau_{\mathrm{neg}} = 10^{-4}$).
   - If $|\Omega_{\mathrm{fg}}| = 0$ and $U_{\mathrm{mean}} \le 10^{-4} \implies$ accepted negative; otherwise rejected.
8. **Hypothesis Testing:** Paired bootstrap $\Delta\text{AURC}$ with $B=1000$ paired resamples against $U_{\mathrm{fg}}$.

---

## 7. Audit of Scientific Claims & Precise Terminology

To maintain scientific integrity in all project documents and papers, the following claims from Step 10 are audited and refined:

| Step 10 Statement | Scientific Critique | Mandatory Precise Formulation |
| :--- | :--- | :--- |
| *"Zero area confounding for $U_{\mathrm{fg}}$"* | A low Spearman rank correlation ($\rho = -0.0328$) proves absence of monotonic association, but does not rule out complex nonlinear or shape-dependent interactions. | *"Near-zero rank correlation with lesion area ($\rho = -0.0328$, $p=0.795$), indicating resilience against simple volume dilution."* |
| *"Improves model segmentation quality"* | Selective prediction does not alter model weights or improve raw pixel predictions on individual images. | *"Improves the mean segmentation quality of the retained cohort via selective rejection of low-confidence predictions."* |
| *"Oracle is a benchmark method"* | Oracle requires ground-truth masks ($1 - \text{DSC}$), which are unavailable at inference time. | *"Oracle serves solely as an unachievable empirical upper bound for evaluating aggregation efficiency."* |
| *"Target coverage was 80%"* | Validation target coverage does not guarantee test realized coverage due to empirical sampling variance. | *"Validation-calibrated target coverage was 80.0%, yielding 72.3% realized coverage on the test cohort."* |
| *"Eliminates segmentation failures"* | Case `116_malignant.png` ($\text{DSC}=0.2193$) had $U_{\mathrm{fg}} = 0.06279 < \tau$, leaking through as an accepted false certainty failure. | *"Successfully triaged 4 of the 5 worst failures, though isolated cases of model overconfidence (e.g. `116_malignant.png`) escaped rejection."* |
| *"Clinically safe deployment"* | Ultrasound selective prediction has not undergone clinical prospective trials or regulatory validation. | *"Research-grade automated quality triage intended as an assistive decision-support filter, not an autonomous diagnostic."* |

---

## 8. Provenance Manifest & Documentation Update Plan

This protocol directly anchors future updates to the repository's foundational documentation:

- **`RESEARCH_DECISIONS_AND_LIMITATIONS.md`:**
  - Record the decision to benchmark UDIAT (near-domain acquisition shift) and DDTI (far-domain anatomical shift).
  - Explicitly document the vulnerability to false certainty (`116_malignant.png`) and the non-guarantee of realized coverage.
- **`THEORY_AND_METRICS.md`:**
  - Formalize the mathematical definitions of prompt transfer vs. image domain shift.
  - Document the empirical interval integration convention for AURC ($c \in [1/N, 1.0]$ vs. $[0, 1.0]$).
- **`EXPERIMENTAL_PROTOCOL.md`:**
  - Record the frozen operating point $\tau = 0.064058324$ and the pre-specified prompt templates (P1, P2).
- **`SCIENTIFIC_EVIDENCE_AND_CLAIMS.md`:**
  - Update with the locked Step 10 numerical results: Baseline DSC $0.8286 \to$ Accepted DSC $0.9031$ ($+7.45\%$ gain), paired bootstrap $p < 0.001$ over $U_{\mathrm{mean}}$ and $U_{\mathrm{boundary}}$.

---

## 9. Final Recommendation: NO-GO for Step 11B Inference

### Recommendation Summary:
- **Protocol & Mathematical Formulation:** **APPROVED (GO)**
- **Code Infrastructure & Checkpoint:** **APPROVED (GO)**
- **Physical Dataset Readiness:** **BLOCKED (NO-GO)**

### Required Prerequisite Actions Before Step 11B:
1. Download or stage raw image archives for `UDIAT` (and optionally `DDTI`) into `project/data/raw/<DATASET>/`.
2. Format ground-truth masks into standard binary PNG files under `<DATASET>/Test_Folder/label/`.
3. Construct the canonical prompt tables (`Test_text_original.xlsx` or `.csv`) using the pre-registered protocol templates.
4. Execute an automated integrity and split verification script before initiating inference.

