# MedCLIPSeg++: Reliability-Aware Segmentation and Case Retrieval

An open-source research platform for post-hoc uncertainty quantification, validation-calibrated selective prediction, and uncertainty-aware case retrieval in breast ultrasound image segmentation, building upon [MedCLIPSeg](https://github.com/HealthX-Lab/MedCLIPSeg) (CVPR 2026, Koleilat et al.).

[![Python 3.9+](https://img.shields.io/badge/python-3.9+-blue.svg)](https://www.python.org/downloads/)
[![License: Apache 2.0](https://img.shields.io/badge/License-Apache%202.0-blue.svg)](LICENSE)
[![Tests Passing](https://img.shields.io/badge/tests-31%2F31%20passing-brightgreen.svg)]()
[![Multi-Center: 5 Cohorts](https://img.shields.io/badge/multi--center-5%20cohorts%20%28N%3D542%29-orange.svg)]()

### Complete System Architecture (Figure S1)

<p align="center">
  <img src="docs/architecture/figures/block_diagram/complete_system_architecture.svg" alt="MedCLIPSeg++ Complete System Architecture" width="100%"/>
</p>

<p align="center"><em>Figure S1 — Six-stage end-to-end modular block diagram: ① Inputs (X, R) → ② Intent Router f_intent(R) → ③ MedCLIPSeg Segmentation (M, U) → ④ Selective Prediction / Triage (gate g) → ⑤ Similarity Retrieval (FAISS-IVF + PABP, gated by g∧r) → ⑥ System Output. Detailed specifications: <a href="docs/architecture/SYSTEM_ARCHITECTURE.md">SYSTEM_ARCHITECTURE.md</a>.</em></p>

---

## 1. System Modules & Core Deliverables

The platform is structured into three self-contained modules, each with dedicated documentation:

| Component | Directory | Documentation | Core Functionality |
| :--- | :--- | :--- | :--- |
| **Component A: Scientific Research** | [`uncertainty_rejection/`](uncertainty_rejection/) | [Module README](uncertainty_rejection/README.md) • [Scientific Architecture](docs/architecture/SELECTIVE_SEGMENTATION_ARCHITECTURE.md) | Pixel-level Shannon entropy extraction, mask-conditioned foreground aggregation ($U_{\mathrm{fg}}$), frozen threshold calibration ($\tau^* = 0.064058324$), and multi-center out-of-distribution evaluation ($N=542$). |
| **Component B: Clinical Application** | [`applications/uncertainty_aware_retrieval/`](applications/uncertainty_aware_retrieval/) | [Application README](applications/uncertainty_aware_retrieval/README.md) • [System Architecture](docs/architecture/SYSTEM_ARCHITECTURE.md) | Interactive Streamlit clinical interface with 3-step workflow, FAISS-IVF indexing with Predicate-Aware Block Pruning (PABP), dynamic contour overlays, and triage telemetry. |
| **Medical Dashboard** | [`applications/medical_dashboard/`](applications/medical_dashboard/) | [Dashboard README](applications/medical_dashboard/README.md) | Local React + FastAPI dashboard. Send a prompt with up to 20 images, inspect matched benchmark segmentation and top-K similar cases, and browse the benchmark gallery. |
| **Evaluation Toolkit** | [`uncertainty_selective_evaluation/`](uncertainty_selective_evaluation/) | [Toolkit README](uncertainty_selective_evaluation/README.md) • [Integration Guide](uncertainty_selective_evaluation/COLLABORATOR_INTEGRATION_GUIDE.md) | Model-agnostic Python package and Google Colab notebook for benchmarking arbitrary segmentation models against our 7 uncertainty operators and frozen decision policies. |

---

## 2. Multi-Center Benchmark Results

Evaluating a frozen MedCLIPSeg foundation backbone (`UniMedCLIP ViT-B/16`) across 5 independent breast ultrasound benchmarks with mask-conditioned foreground entropy ($U_{\mathrm{fg}}$) and frozen threshold $\tau^* = 0.064058324$:

| Benchmark Dataset | Clinical Origin | Cohort Size ($N$) | Baseline DSC | Retained DSC | Selective Gain | Realized Coverage | Primary AURC ($U_{\mathrm{fg}}$) | Random AURC |
| :--- | :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **BUSI Test (Setting A)** | Cairo, Egypt (In-Domain) | 65 | 0.8286 | **0.9031** | **+0.0745** | 72.3% | **0.0790** | 0.1687 |
| **BUSI Test (Setting B)** | Cairo, Egypt (Full Screening) | 78 | 0.8572 | **0.9241** | **+0.0669** | 76.9% | — | — |
| **UDIAT** | Sabadell, Spain (OOD) | 25 | 0.7467 | **0.8747** | **+0.1281** | 56.0% | **0.1133** | 0.2533 |
| **BUID** | Academic Benchmark (OOD) | 35 | 0.7900 | **0.8818** | **+0.0919** | 57.1% | **0.1094** | 0.2100 |
| **BUSUC** | Multi-center Registry (OOD) | 122 | 0.8430 | **0.8982** | **+0.0551** | 63.9% | **0.0916** | 0.1570 |
| **BUSBRA** | Multi-center (Brazil, OOD) | 282 | 0.7511 | **0.8536** | **+0.1025** | 44.7% | **0.1529** | 0.2489 |
| **Pooled External OOD** | 4 Combined Centers | **464** | **0.7780** | **0.8718** | **+0.0939** | **51.3%** | **0.1267** | **0.2216** |

*All results verified against canonical per-image CSV outputs; 95% bootstrap CI for pooled OOD gain: $[+0.0775, +0.1113]$, $p < 0.001$.*

---

## 3. Data Governance & Benchmark Provenance

To ensure scientific reproducibility, patient privacy, and strict separation between calibration and evaluation data, datasets are governed under the following protocol:

| Dataset | Clinical Source | Scanner / Equipment | Cohort Size | Role in Study | Access & License |
| :--- | :--- | :--- | :---: | :--- | :--- |
| **BUSI** | Baheya Hospital, Cairo, Egypt | LOGIQ E9 & LOGIQ 7 (GE Healthcare) | 780 scans (130 Val, 78 Test) | **In-Domain (ID)**: Split 80/20 train/test. $\tau^*$ calibrated strictly on Val; never on Test. | Open-access (CC BY 4.0) |
| **UDIAT** | Parc Taulí Hospital, Sabadell, Spain | Siemens ACUSON Sequoia 512 | 25 scans | **Out-of-Distribution (OOD)**: Blind evaluation under hospital & scanner shift. | Academic research |
| **BUID** | Multi-institutional Repository | Mixed transducers (7.5–12 MHz) | 35 scans | **Out-of-Distribution (OOD)**: Blind evaluation under variable resolution. | Open-access |
| **BUSUC** | Multi-center Registry | Mixed linear probes | 122 scans | **Out-of-Distribution (OOD)**: Multi-site protocol shift. | Academic research |
| **BUSBRA** | Multi-center Consortium, Brazil | Mixed modern ultrasound units | 282 scans | **Out-of-Distribution (OOD)**: Large-scale demographic & geographic shift. | Open-access (CC BY 4.0) |

### Data Organization & Path Resolution

Datasets are maintained in a clean hierarchical structure outside the Git version tree:

```text
data/
├── raw/                               # De-identified clinical scans & expert ground-truth masks
│   ├── BUSI/                          # (Train / Val / Test_Folder)
│   ├── UDIAT/
│   ├── BUID/
│   ├── BUSUC/
│   └── BUSBRA/
└── outputs/                           # Canonical per-image scientific evaluation outputs
    ├── BUSI_Test/                     # Predictions, float32 entropy maps, embeddings, CSVs
    ├── UDIAT/
    ├── BUID/
    ├── BUSUC/
    └── BUSBRA/
```

- **Environment Variable:** Set `export MEDCLIPSEG_DATA="/path/to/data"` to customize the dataset root. If unset, loaders automatically fallback to `../data` relative to the repository.
- **Config Files:** Individual dataset properties (image resolutions, normalization constants, split paths) are declared declaratively in [`shared/configs/`](shared/configs/).

---

## 4. Technical Architecture & Visual Documentation

The repository features five publication-grade architectural diagrams available in `.svg`, `.drawio`, and `.pdf` formats:

| ID | Title | Scope | Technical Highlights | Formats |
|:---|:------|:------|:---------------------|:--------|
| **S1** | **Complete System Architecture** | 6-Stage Pipeline | 6-stage modular pipeline: Inputs → Intent Router → MedCLIPSeg → Triage → PABP Retrieval → System Output | [SVG](docs/architecture/figures/block_diagram/complete_system_architecture.svg) • [Draw.io](docs/architecture/figures/block_diagram/complete_system_architecture.drawio) • [PDF](docs/architecture/figures/block_diagram/complete_system_architecture.pdf) |
| **S2** | **Application Operational Flowchart** | Component B | User input validation, caching, PABP retrieval, 3 terminal exits | [SVG](docs/architecture/figures/flowcharts/application_operational_flowchart.svg) • [Draw.io](docs/architecture/figures/flowcharts/application_operational_flowchart.drawio) • [PDF](docs/architecture/figures/flowcharts/application_operational_flowchart.pdf) |
| **R1** | **Selective Segmentation Architecture** | Component A | 4-stage scientific block diagram: Inputs $\to$ Backbone $\to$ Triage $\to$ Output | [SVG](docs/architecture/figures/block_diagram/selective_segmentation_architecture.svg) • [Draw.io](docs/architecture/figures/block_diagram/selective_segmentation_architecture.drawio) • [PDF](docs/architecture/figures/block_diagram/selective_segmentation_architecture.pdf) |
| **R2** | **Selective Decision Flowchart** | Component A | Spatial stencil $U_{\mathrm{fg}}$, dual branching, empty-mask fallback $\tau_{\mathrm{neg}}$ | [SVG](docs/architecture/figures/flowcharts/selective_decision_flowchart.svg) • [Draw.io](docs/architecture/figures/flowcharts/selective_decision_flowchart.drawio) • [PDF](docs/architecture/figures/flowcharts/selective_decision_flowchart.pdf) |
| **R3** | **Experimental Validation Framework** | Component A | 5 multi-center cohorts, 7 uncertainty operators, 3-phase pipeline | [SVG](docs/architecture/figures/framework/experimental_validation_framework.svg) • [Draw.io](docs/architecture/figures/framework/experimental_validation_framework.drawio) • [PDF](docs/architecture/figures/framework/experimental_validation_framework.pdf) |

### Detailed Architecture Specifications:
- [**SYSTEM_ARCHITECTURE.md**](docs/architecture/SYSTEM_ARCHITECTURE.md): Complete system specification detailing the L0–L3 hierarchy, interface contracts for all 6 L1 modules (`MOD-INTENT-INTERPRET`, `MOD-SEG-BACKBONE`, `MOD-SELECTIVE-DECISION`, `MOD-IMAGE-EMBED`, `MOD-SIMILARITY-RETRIEVAL`, `MOD-CLINICAL-PRESENT`), extensibility matrix, and clinical safety invariants.
- [**SELECTIVE_SEGMENTATION_ARCHITECTURE.md**](docs/architecture/SELECTIVE_SEGMENTATION_ARCHITECTURE.md): Rigorous mathematical exposition of Monte Carlo variational attention, spatial entropy stencils, empty-mask degeneracy handling, 10 clinical casuísticas, and the seven uncertainty operators comparison table.

---

## 5. Repository Structure

```text
MedCLIPSeg-plus-plus/
├── docs/                              # Technical Architecture & Scientific Specifications
│   └── architecture/
│       ├── SYSTEM_ARCHITECTURE.md     # Component B: System specification & L0-L3 contracts
│       ├── SELECTIVE_SEGMENTATION_ARCHITECTURE.md # Component A: Scientific theory & empirical tables
│       └── figures/                   # Vector SVGs, editable Draw.io sources, and PDF exports
│
├── stages/                            # Modular Pipeline Stages (LEGO Architecture)
│   ├── stage1_inputs/                 # [①] Image X & Clinical Request R ingestion
│   ├── stage2_intent_router/          # [②] f_intent(R) → Execution Plan (s, r, K)
│   ├── stage3_segmentation_backbone/  # [③] MedCLIPSeg (Ev, Et, PVL, Dseg) → M, U
│   ├── stage4_selective_prediction/   # [④] Triage: U_fg ≤ τ*, gate g ∈ {0,1}
│   ├── stage5_similarity_retrieval/   # [⑤] FAISS-IVF + PABP, gated by g∧r
│   └── stage6_system_output/          # [⑥] Adaptive multi-panel clinical output
│
├── applications/                      # Engineering Applications
│   ├── medical_dashboard/              # Local React + FastAPI case review dashboard
│   │   ├── README.md                   # Dashboard setup, local launcher, and API
│   │   └── run_local.sh                # Starts local services on 127.0.0.1
│   └── uncertainty_aware_retrieval/   # Clinical Ultrasound Retrieval & PABP Engine
│       ├── README.md                  # Application guide & clinical workflow documentation
│       ├── app.py                     # Interactive 3-step Streamlit clinical UI
│       ├── cli.py                     # Headless batch inspection CLI
│       ├── uncertainty_aware_retrieval/ # PABP index, FAISS engine, and triage router
│       └── tests/                     # 25 application unit tests
│
├── uncertainty_rejection/             # Core Scientific Research Module
│   ├── README.md                      # Scientific overview & reproduction guide
│   ├── src/uncertainty/               # Production uncertainty extraction & metrics
│   ├── research/                      # Canonical scientific records & experiment scripts
│   ├── scripts/                       # Shell runners (run_tests.sh, run_aggregation.sh)
│   └── tests/                         # Full automated regression suite (31 tests)
│
├── uncertainty_selective_evaluation/  # Lightweight Evaluation Toolkit
│   ├── README.md                      # Toolkit documentation & quickstart
│   ├── COLLABORATOR_INTEGRATION_GUIDE.md # Integration instructions for external models
│   ├── metrics.py                     # Self-contained aggregators & statistical metrics
│   ├── model_adapter.py               # Model adapter contracts & mock engine
│   └── evaluate.py                    # Standalone CLI evaluation tool
│
├── shared/                            # Vision-Language Foundation & Configs
│   ├── medclipseg/                    # Frozen vision-language model & variational adapters
│   └── configs/                       # Dataset configuration YAMLs (BUSI, BUSBRA, etc.)
│
├── notebooks/                         # Interactive Research Notebooks
│   ├── 01_complete_system_pipeline.ipynb # End-to-end 6-stage execution pipeline (headless)
│   └── 02_scientific_benchmarking.ipynb  # Scientific evaluation, 7 operators & PABP vector benchmark
│
└── requirements.txt                   # Consolidated Python dependencies
```

---

## 6. Quick Start & Reproduction

### Installation
```bash
git clone https://github.com/Fergarciamedina/MedCLIPSeg-plus-plus.git
cd MedCLIPSeg-plus-plus
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

### Run Full Test Suites
```bash
# 1. Run scientific regression test suite (31 tests)
bash uncertainty_rejection/scripts/run_tests.sh

# 2. Run retrieval application test suite (25 tests)
python3 -m unittest discover -s applications/uncertainty_aware_retrieval/tests -v

# 3. Run clean-room evaluation reproducibility test
python3 uncertainty_selective_evaluation/test_reproducibility.py
```

### Launch Clinical Retrieval Application
```bash
streamlit run applications/uncertainty_aware_retrieval/app.py
```

### Launch the Medical Dashboard (macOS)
From the repository root, run:
```bash
bash applications/medical_dashboard/run_local.sh
```
The script starts the API and web interface on `127.0.0.1`, opens <http://127.0.0.1:5173>, and stops both services with `Ctrl+C`. Local dependencies must already be installed. Uploaded images remain request-scoped; exact benchmark matches display their stored segmentation and prompt-requested similar cases. New images remain unevaluated until live inference is connected. See the [dashboard README](applications/medical_dashboard/README.md) for requirements and setup.

### Evaluate Custom Predictions via CLI
```bash
python3 uncertainty_selective_evaluation/evaluate.py \
  --input_csv /path/to/predictions.csv \
  --threshold 0.064058324 \
  --output_dir ./results
```

---

## 7. Attribution & Upstream Baseline

- **Baseline Codebase:** [HealthX-Lab/MedCLIPSeg](https://github.com/HealthX-Lab/MedCLIPSeg) (CVPR 2026, Apache-2.0 License).
- **Foundation Paper:** *MedCLIPSeg: Data-Efficient Medical Image Segmentation with Frozen Vision-Language Foundation Models* (Koleilat et al., CVPR 2026, [arXiv:2602.20423](https://arxiv.org/abs/2602.20423)).
- **License:** Apache License 2.0 (see [LICENSE](LICENSE) and [ATTRIBUTION.md](ATTRIBUTION.md)).
