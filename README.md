# MedCLIPSeg++: Uncertainty-Based Selective Prediction for Medical Image Segmentation

**Course:** ELEC 6661 — Medical Image Processing (Fall 2026), Concordia University\
**Lab:** IMPACT Lab (Image Processing And Characterization of Tissue)\
**Supervisor / Instructor:** Prof. Hassan Rivaz, Omid Nejati Manzari\
**Author:** Fernando Garcia-Medina (`fernando.garciamedina@concordia.ca`)\
**Baseline Upstream Repository:** [HealthX-Lab/MedCLIPSeg](https://github.com/HealthX-Lab/MedCLIPSeg) (CVPR 2026, Koleilat et al., Apache-2.0)\
**Reference Paper:** *MedCLIPSeg: Data-Efficient Medical Image Segmentation with Frozen Vision-Language Foundation Models* (CVPR 2026)

---

## 1. Project Overview & Deliverables

This repository contains the complete implementation, experimental evaluation, scientific documentation, and publication-ready manuscript for:

> **"Uncertainty-Based Selective Prediction for Text-Guided Medical Image Segmentation Under Domain Shift"**

### Primary Deliverables:
- **Scientific Manuscript (10 pages, IEEE Computer Society format):** [`uncertainty_rejection/paper/main.pdf`](uncertainty_rejection/paper/main.pdf) (LaTeX sources in [`uncertainty_rejection/paper/main.tex`](uncertainty_rejection/paper/main.tex)).
- **Canonical Scientific Documentation (7 Records):** Located under [`uncertainty_rejection/research/`](uncertainty_rejection/research/).
- **Reusable Evaluation Toolkit:** Self-contained, model-independent package under [`uncertainty_selective_evaluation/`](uncertainty_selective_evaluation/).
- **Google Colab Notebook:** Interactive, 1-click evaluation notebook in [`notebooks/uncertainty_selective_evaluation.ipynb`](notebooks/uncertainty_selective_evaluation.ipynb).
- **Automated Regression Suite:** 31/31 unit, integration, and statistical reproducibility tests passing.

---

## 2. Key Scientific Findings & Benchmark Results

We evaluate a frozen MedCLIPSeg foundation model across five independent breast ultrasound benchmarks ($N=542$ total test scans) using foreground-averaged predictive entropy ($U_{\mathrm{fg}}$) with a frozen decision threshold ($\tau = 0.064058324$) calibrated on in-distribution validation data:

| Benchmark Dataset | Clinical Origin | Cohort Size ($N$) | Baseline DSC | Retained DSC | Selective Gain | Realized Coverage | Primary AURC ($U_{\mathrm{fg}}$) | Random AURC |
| :--- | :--- | :---: | :---: | :---: | :---: | :---: | :---: |
| **BUSI Test (Setting A)** | Cairo, Egypt (In-Domain) | 65 | 0.8286 | **0.9031** | **+0.0745** | 72.3% | **0.0790** | 0.1687 |
| **BUSI Test (Setting B)** | Cairo, Egypt (Full Screening) | 78 | 0.8572 | **0.9241** | **+0.0669** | 76.9% | --- | --- |
| **UDIAT** | Sabadell, Spain (OOD) | 25 | 0.7467 | **0.8747** | **+0.1281** | 56.0% | **0.1133** | 0.2533 |
| **BUID** | Academic Benchmark (OOD) | 35 | 0.7900 | **0.8818** | **+0.0919** | 57.1% | **0.1094** | 0.2100 |
| **BUSUC** | Multi-center Registry (OOD) | 122 | 0.8430 | **0.8982** | **+0.0551** | 63.9% | **0.0916** | 0.1570 |
| **BUSBRA** | Multi-center (Brazil, OOD) | 282 | 0.7511 | **0.8536** | **+0.1025** | 44.7% | **0.1529** | 0.2489 |
| **Pooled External OOD** | 4 Combined Centers | **464** | **0.7780** | **0.8718** | **+0.0939** | **51.3%** | **0.1267** | **0.2216** |

*All results verified against canonical per-image CSV outputs; 95% bootstrap CI for pooled OOD gain: $[+0.0775, +0.1113]$, $p < 0.001$.*

---

## 3. Repository Architecture

```text
MedCLIPSeg-plus-plus/
├── uncertainty_rejection/             # Uncertainty Quantification & Selective Prediction
│   ├── paper/                         # 10-page IEEEtran manuscript, figures & verification
│   ├── research/                      # Canonical scientific records & experiment scripts
│   ├── src/uncertainty/               # Production uncertainty extraction & metrics
│   ├── scripts/                       # Reproduction shell scripts & test runner
│   ├── tests/                         # Unit, integration & regression tests (31 tests)
│   └── docs/                          # Historical step audits and documentation
│
├── uncertainty_selective_evaluation/  # Lightweight, reusable evaluation package
│   ├── metrics.py                     # Self-contained aggregators & statistical metrics
│   ├── model_adapter.py               # Model adapter contracts & mock engine
│   ├── evaluate.py                    # Standalone CLI evaluation tool
│   ├── test_reproducibility.py        # Clean-room verification suite
│   ├── COLLABORATOR_INTEGRATION_GUIDE.md # Integration instructions for EhsanSl/MedCLIPSeg
│   └── notebooks/                     # Colab evaluation notebook
│
├── notebooks/                         # Google Colab notebooks
│   └── uncertainty_selective_evaluation.ipynb
│
├── shared/                            # Upstream MedCLIPSeg baseline & configs
│   ├── medclipseg/                    # Model architecture & variational attention adapters
│   └── configs/                       # Dataset configuration YAMLs
│
└── requirements.txt                   # Consolidated Python dependencies
```

---

## 4. Quick Start & Reproduction

### Prerequisites
```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

### Run Automated Tests
```bash
# Run full 31-test regression suite
bash uncertainty_rejection/scripts/run_tests.sh

# Run clean-room evaluation toolkit tests
python uncertainty_selective_evaluation/test_reproducibility.py
```

### Evaluate Custom Predictions via CLI
```bash
python uncertainty_selective_evaluation/evaluate.py \
  --input_csv /path/to/predictions.csv \
  --threshold 0.064058324 \
  --output_dir ./results
```

---

## 5. Canonical Scientific Documentation

The project maintains seven authoritative records under `uncertainty_rejection/research/`:
1. [`FINAL_RESULTS_INDEX.md`](uncertainty_rejection/research/FINAL_RESULTS_INDEX.md) — Single authoritative numerical anchor for all datasets.
2. [`SCIENTIFIC_EVIDENCE_AND_CLAIMS.md`](uncertainty_rejection/research/SCIENTIFIC_EVIDENCE_AND_CLAIMS.md) — Formal answers to RQs 1–5 with empirical evidence.
3. [`THEORY_AND_METRICS.md`](uncertainty_rejection/research/THEORY_AND_METRICS.md) — Mathematical derivations and aggregation operator definitions.
4. [`EXPERIMENTAL_PROTOCOL.md`](uncertainty_rejection/research/EXPERIMENTAL_PROTOCOL.md) — Fair evaluation, frozen calibration, and prompt protocols.
5. [`RESEARCH_DECISIONS_AND_LIMITATIONS.md`](uncertainty_rejection/research/RESEARCH_DECISIONS_AND_LIMITATIONS.md) — Comprehensive decision ledger (D-001 through D-163).
6. [`PUBLISHED_BASELINE_COMPARISON.md`](uncertainty_rejection/research/PUBLISHED_BASELINE_COMPARISON.md) — Rigorous audit against CVPR 2026 published tables.
7. [`PROJECT_EXECUTION_HISTORY.md`](uncertainty_rejection/research/PROJECT_EXECUTION_HISTORY.md) — Chronological execution history of all project steps.

---

## 6. License & Attribution

This project is licensed under the Apache-2.0 License. It builds upon [MedCLIPSeg](https://github.com/HealthX-Lab/MedCLIPSeg) (CVPR 2026, Koleilat et al.). See [`ATTRIBUTION.md`](ATTRIBUTION.md) and [`LICENSE`](LICENSE) for details.
| :--- | :--- | :--- | :--- | :--- |
| **Uncertainty-Based Selective Prediction** | `uncertainty_rejection/` | Raw pixel-level entropy extraction, 5 aggregation metrics, validation threshold calibration, risk-coverage curves, control benchmarks, and domain shift evaluation. | `feature/uncertainty-rejection` | [`uncertainty_rejection/README.md`](uncertainty_rejection/README.md) |
| **Text-Prompt Robustness** | `prompt_robustness/` | Prompt perturbation analysis, linguistic sensitivity, and cross-center phrasing evaluation. | `feature/prompt-robustness` | [`prompt_robustness/README.md`](prompt_robustness/README.md) |
| **Common Baseline** | `shared/` | MedCLIPSeg neural network, datasets, consolidated configs, and common execution scripts. | Coordinated via PR | [`shared/README.md`](shared/README.md) |

---

## 3. Development & Branching Workflow

To maintain clean separation between research tracks:

```text
  feature/uncertainty-rejection  ──────────────────────────────┐
                                                                ▼
                                       PR + Review ───►  main (integrated)
                                                                ▲
  feature/prompt-robustness      ──────────────────────────────┘
```

1. **`feature/uncertainty-rejection`**: All commits regarding uncertainty quantification, selective prediction, aggregation metrics, and optional systems.
2. **`feature/prompt-robustness`**: All commits regarding text-prompt design, linguistic perturbations, and prompt sensitivity evaluations.
3. **`main`**: Production branch containing reviewed, integrated code. Direct pushes to `main` are restricted to integrated releases.
4. **Shared Core Governance**: Any change to `shared/medclipseg/`, `shared/configs/`, or `shared/scripts/` requires mutual coordination and review.

---

## 4. Shared Baseline Enhancements

Although `shared/medclipseg/` serves as the common baseline, specific enhancements were integrated for the research workflows:
- Raw `float32` `.npy` uncertainty extraction in `shared/medclipseg/test.py` (saving uncompressed entropy arrays prior to visualization normalization).
- Dynamic `${MEDCLIPSEG_DATA}` environment variable resolution.
- Apple Silicon MPS device support and automated device fallback.

---

## 5. Reproduction & Execution

### Prerequisites
```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

### Run Tests & Experiments
```bash
# 1. Run all tests
bash uncertainty_rejection/scripts/run_tests.sh

# 2. Run baseline inference (shared engine)
bash shared/scripts/run_inference.sh BUSI 666

# 3. Run Stage 02 uncertainty aggregation & correlation analysis
bash uncertainty_rejection/scripts/run_aggregation.sh BUSI 666

# 4. Run Stage 03 selective prediction & control comparison
bash uncertainty_rejection/scripts/run_selective_prediction.sh BUSI 666 U_fg
```

---

## 6. License & Attribution

This project is licensed under the Apache-2.0 License. It builds upon [MedCLIPSeg](https://github.com/HealthX-Lab/MedCLIPSeg) (CVPR 2026, Koleilat et al.). See [`ATTRIBUTION.md`](ATTRIBUTION.md) and [`LICENSE`](LICENSE) for details.
