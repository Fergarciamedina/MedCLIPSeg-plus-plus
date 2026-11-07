# ELEC 6661 — Step 11B.1: Staging Report for Four Breast Ultrasound OOD Datasets

**Date:** 2026-11-07  
**Stage:** Step 11B.1 — Staging, Integrity Verification & Prompt Protocol Freeze for External Breast Ultrasound Benchmarks  
**Auditor:** Automated Scientific Assistant  
**Domain Scope:** Breast-Only Ultrasound Domain Generalization (Zero-Shot OOD Transfer from BUSI)  
**Readiness Verdict for Step 11B.2 (Inference):** **GO (All Datasets Staged, Verified & Configured)**  

---

## 1. Executive Summary & Staging Verdict

In accordance with the project plan and the original MedCLIPSeg paper (Koleilat et al., CVPR 2026), the four official breast ultrasound out-of-distribution (OOD) benchmarks have been successfully located, licensed, downloaded, staged outside Git under `project/data/raw/`, and verified.

### Key Milestones Achieved:
1. **Official Sources Acquired:** All four official dataset packages (`UDIAT.zip`, `BUID.zip`, `BUSUC.zip`, `BUSBRA.zip`) were retrieved directly from the official MedCLIPSeg Hugging Face repository (`TahaKoleilat/MedCLIPSeg`).
2. **Open-Access Licensing:** Distributed under Creative Commons Attribution-NonCommercial 4.0 International (**CC-BY-NC-4.0**). Publicly accessible without authentication barriers.
3. **Data Integrity & 1-to-1 Pairing:** Verified $100\%$ match between ultrasound images and binary segmentation masks across all official splits (Train, Val, Test). Zero missing or corrupted files.
4. **Pre-Registered Prompt Freeze:** Staged both official author prompts (`Test_text_original.xlsx`) containing pathology labels and fair clinical screening prompts (`Test_text_neutral.xlsx`) with diagnostic labels strictly removed to eliminate label leakage during inference.
5. **Frozen Infrastructure:** Modular YAML configurations created under `project/MedCLIPSeg-plus-plus/shared/configs/`. Frozen BUSI weights, primary metric ($U_{\mathrm{fg}}$), primary operating point ($\tau = 0.064058324$), comparator ($U_{\mathrm{std\_fg}}$), and empty-mask cutoff ($\tau_{\mathrm{neg}} = 10^{-4}$) remain permanently locked.

**Final Verdict:** **GO for Step 11B.2 OOD Inference Execution.**

---

## 2. Staged Dataset Inventory & Download Manifest

All archives were downloaded to `project/data/raw/downloads/` and unpacked into canonical subdirectories under `project/data/raw/<DATASET>/`:

| Dataset | Modality & Organ | Archive Size | Archive SHA256 Checksum | License | Local Staging Path |
| :--- | :--- | :---: | :--- | :--- | :--- |
| **UDIAT** | Breast Ultrasound | 19.4 MB | `1e5ba4c97e7d7a284067965d2333cac276583f80ca7000bc5268dbc4bbcb92d0` | CC-BY-NC-4.0 | `project/data/raw/UDIAT/` |
| **BUID** | Breast Ultrasound | 57.0 MB | `03c6850dfb5726054c411c5941e542de345298dfe48a17a7992c6b9d9340b24a` | CC-BY-NC-4.0 | `project/data/raw/BUID/` |
| **BUSUC** | Breast Ultrasound | 46.5 MB | `1b93c48a34678f17643fa4fa6808c66b983644106d192b83d4e62f1c8282e849` | CC-BY-NC-4.0 | `project/data/raw/BUSUC/` |
| **BUSBRA** | Breast Ultrasound | 128.0 MB | `c177b7f693f6e8f0f686599d500fa2a550055e88d536f49d9893432920e7f569` | CC-BY-NC-4.0 | `project/data/raw/BUSBRA/` |

---

## 3. Cohort Split Sizes & Ground Truth Composition

Each dataset was verified against the official MedCLIPSeg partition definitions:

| Dataset | Total Images | Train Count | Val Count | Test Count | Test Benign | Test Malignant | Test Normal | Image/Mask 1-to-1 Pairing |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **UDIAT** | 163 | 113 | 25 | **25** | 17 (68.0%) | 8 (32.0%) | 0 (0.0%) | **VERIFIED (25/25)** |
| **BUID** | 232 | 162 | 35 | **35** | 16 (45.7%) | 19 (54.3%) | 0 (0.0%) | **VERIFIED (35/35)** |
| **BUSUC** | 811 | 567 | 122 | **122** | 54 (44.3%) | 68 (55.7%) | 0 (0.0%) | **VERIFIED (122/122)** |
| **BUSBRA** | 1,875 | 1,311 | 282 | **282** | 191 (67.7%) | 91 (32.3%) | 0 (0.0%) | **VERIFIED (282/282)** |
| **TOTAL OOD** | **3,081** | **2,153** | **464** | **464** | **278 (59.9%)** | **186 (40.1%)** | **0 (0.0%)** | **VERIFIED (464/464)** |

### Scientific Finding on Cohort Prevalence:
Unlike `BUSI` (which contains 133 normal scans with empty masks, of which 13 are in the test set), all four external breast ultrasound datasets represent **lesion-only diagnostic cohorts ($100\%$ lesion prevalence)**.
- **Setting A (Lesion Cohort):** Evaluates all 464 test cases where a lesion is predicted.
- **Setting B (Clinical Triage):** Serves to catch **confidently missed lesions** ($|\Omega_{\mathrm{fg}}|=0$ on a true positive scan). If MedCLIPSeg produces an empty mask on an OOD image, Setting B tests whether background entropy $U_{\mathrm{mean}} > \tau_{\mathrm{neg}}$ successfully flags the missed tumor for expert review.

---

## 4. Technical Characteristics & Acquisition Shift Matrix

| Dataset | Scanner Manufacturer & Transducer | Clinical Center / Country | Spatial Resolution Range | Mask Format | Nature of Domain Shift vs. BUSI |
| :--- | :--- | :--- | :--- | :--- | :--- |
| **BUSI (ID)** | GE LOGIQ E9 & LOGIQ 7 (linear) | Baheya Hospital, Cairo, Egypt | $500 \times 500$ avg | Binary $\{0, 255\}$ | In-Distribution Baseline |
| **UDIAT** | Siemens Elegra (7.5 MHz linear) | UDIAT Diagnostic Centre, Sabadell, Spain | $233 \times 344$ to $596 \times 631$ | Binary $\{0, 255\}$ | Scanner shift (Siemens vs. GE), demographic shift (Spain vs. Egypt) |
| **BUID** | Unknown clinical scanner | Public academic repository | $380 \times 430$ to $664 \times 650$ | Binary $\{0, 255\}$ | Center shift, probe variation, variable image contrast |
| **BUSUC** | Multi-hospital clinical registry | Multi-center | $250 \times 256$ to $256 \times 256$ | Anti-aliased grayscale $\{0\dots 255\}$ | Heavy compression artifacts, pre-cropped aspect ratios |
| **BUSBRA** | Multi-scanner (GE, Philips, Toshiba) | Multi-center, Brazil | $295 \times 270$ to $473 \times 571$ | Binary $\{0, 255\}$ | Heterogeneous multi-scanner acoustic speckle, Brazilian demographic |

---

## 5. Pre-Inference Prompting Protocol

To address the critical requirement of **preventing diagnostic label leakage**, two prompt configurations are frozen before running inference:

### 1. Protocol P_orig: Official MedCLIPSeg Author Baseline (Oracle Diagnostic Prompting)
- Uses the official `Test_text_original.xlsx` provided by the MedCLIPSeg authors.
- **Characteristics:** Contains explicit pathology labels (`benign tumor`, `malignant tumor`, `mass`).
- **Purpose:** Enables direct numerical comparison against the published MedCLIPSeg CVPR 2026 baseline tables.
- **Labeling:** Must be explicitly reported as *Oracle-Pathology Prompting* due to ground-truth label inclusion.

### 2. Protocol P_neutral: Fair Clinical Screening Protocol (Label-Free Deployment Prompting)
- Staged in `Test_text_neutral.xlsx` for each dataset.
- **Characteristics:** All diagnostic adjectives (`benign`, `malignant`) are strictly removed while preserving the identical sentence structure and spatial indicators.
  - *Example Original:* `"A benign tumor is present in the center region of the breast."`
  - *Example Neutral:* `"A tumor is present in the center region of the breast."`
  - *Example Original:* `"The breast shows a malignant tumor positioned in the upper."`
  - *Example Neutral:* `"The breast shows a tumor positioned in the upper."`
- **Purpose:** Represents genuine clinical screening conditions where lesion malignancy is unknown prior to segmentation.

---

## 6. Preprocessing & Inference Configuration

All dataset configuration files in `project/MedCLIPSeg-plus-plus/shared/configs/` (`UDIAT.yaml`, `BUID.yaml`, `BUSUC.yaml`, `BUSBRA.yaml`) were updated to use dynamic path resolution:

```yaml
DATASET:
  NAME: "<DATASET>"
  TRAIN_PATH: "${MEDCLIPSEG_DATA}/<DATASET>/Train_Folder/"
  VAL_PATH: "${MEDCLIPSEG_DATA}/<DATASET>/Val_Folder/"
  TEST_PATH: "${MEDCLIPSEG_DATA}/<DATASET>/Test_Folder/"
  TEXT_PROMPT_PATH: "${MEDCLIPSEG_DATA}/<DATASET>/Prompts_Folder/"
  SIZE: 224

TEST:
  NUM_SAMPLES: 30
  USE_LATEST: True

MODEL:
  CLIP_MODEL: unimedclip
  BACKBONE: "ViT-B/16"
  ADAPTER_DIM: 256
  NUM_UPSCALE: 2
  BETA: 2.35
  GATE_INIT: 0.0
  LAYERS: [1, 2, 3, 4, 5, 6, 7, 8, 9, 10]
  TEMPERATURE: 0.2
  DEVICE: "mps"
```

### Preprocessing Safeguards Verified:
1. **Mask Binarization:** In `shared/medclipseg/datasets/dataloader.py` and `eval.py`, all masks are binarized using `threshold >= 127`. This correctly handles BUSUC's anti-aliased $\{0\dots 255\}$ masks.
2. **Channel Expansion:** Grayscale images are expanded to 3 channels via `correct_dims`.
3. **Resizing:** Bicubic interpolation to $224 \times 224$ for images; nearest-neighbor interpolation for masks.
4. **Checkpoint Isolation:** Checkpoint loading uses `--source_dataset BUSI`, ensuring that the frozen weights trained strictly on in-domain BUSI (`seed 666`) are loaded across all four external domains without retraining or fine-tuning.

---

## 7. Frozen Selective Prediction Parameters (Permanent Lock)

The following parameters derived strictly on BUSI validation remain permanently frozen and cannot be adjusted on any external dataset:
- **Primary Uncertainty Metric:** $U_{\mathrm{fg}} = \frac{1}{|\Omega_{\mathrm{fg}}|} \sum_{i \in \Omega_{\mathrm{fg}}} H(p_i)$
- **Prespecified Comparator:** $U_{\mathrm{std\_fg}} = \sqrt{\frac{1}{|\Omega_{\mathrm{fg}}|} \sum_{i \in \Omega_{\mathrm{fg}}} (H(p_i) - U_{\mathrm{fg}})^2}$
- **Primary Rejection Cutoff:** $\tau = 0.064058324$ (Calibrated for 80% coverage on BUSI validation)
- **Two-Stage Negative Threshold:** $\tau_{\mathrm{neg}} = 0.0001$ ($1.0 \times 10^{-4}$)
- **Secondary Operating Cutoffs:**
  - $\tau_{95} = 0.101927518$
  - $\tau_{90} = 0.083463000$
  - $\tau_{85} = 0.072696542$
  - $\tau_{75} = 0.059689360$
  - $\tau_{70} = 0.056208188$
- **Monte Carlo Samples:** $T = 30$ forward passes per image

---

## 8. Final Recommendation: GO for Step 11B.2 Inference Execution

All prerequisites for Stage 11B.2 are fully satisfied:
- [x] All 4 datasets downloaded from official Hugging Face repository.
- [x] SHA256 checksums verified and documented.
- [x] Unpacked outside Git under `project/data/raw/<DATASET>/`.
- [x] 1-to-1 image-mask correspondence confirmed (464 test samples total).
- [x] YAML configurations updated with environment variable path resolution.
- [x] Dual prompt protocols (Oracle vs. Neutral) generated and frozen.
- [x] Full regression test suite passing (27 / 27 tests).

The system is declared **READY FOR STEP 11B.2 (FROZEN OOD INFERENCE AND EVALUATION)**.

