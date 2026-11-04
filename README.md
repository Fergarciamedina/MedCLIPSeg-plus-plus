# MedCLIPSeg++: Multi-Track Medical Image Segmentation Research

**Course:** ELEC 6661 — Medical Image Processing (Fall 2026), Concordia University\
**Lab:** IMPACT Lab (Image Processing And Characterization of Tissue)\
**Supervisor / Instructor:** Prof. Hassan Rivaz, Omid Nejati Manzari\
**Baseline Upstream Repository:** [HealthX-Lab/MedCLIPSeg](https://github.com/HealthX-Lab/MedCLIPSeg) (CVPR 2026, Apache-2.0)\
**Reference Paper:** *MedCLIPSeg: Data-Efficient Medical Image Segmentation with Frozen Vision-Language Foundation Models* (arXiv:2602.20423)

---

## 1. Project Overview & Research Architecture

This project extends the MedCLIPSeg baseline into two independent, complementary research workstreams within the IMPACT Lab framework (**BASELINE + ONE CHANGE + FAIR EVALUATION**).

The repository is organized strictly by **research contribution**:

```text
MedCLIPSeg-plus-plus/
├── uncertainty_rejection/             # Uncertainty Quantification & Selective Prediction
│   ├── src/uncertainty/               # Uncertainty extraction, aggregation & rejection library
│   ├── research/                      # Level 0 experiments (Stages 01-04)
│   ├── systems/                       # Level 1 (Priority UI) & Level 2 (Similarity Search)
│   ├── notebooks/                     # Interactive evaluation notebook
│   ├── scripts/                       # Stage 02/03 runner scripts & test runner
│   └── tests/                         # Automated unit & integration tests
│
├── prompt_robustness/                 # Text-Prompt Robustness & Perturbation Analysis
│   └── research/                      # Text-prompt perturbation & sensitivity experiments
│
├── shared/                            # Minimal Shared Baseline & Infrastructure
│   ├── medclipseg/                    # Frozen baseline model (encoders, adapters, trainers)
│   ├── configs/                       # Consolidated dataset YAML configs (MPS / CUDA)
│   └── scripts/                       # Common inference, eval & checkpoint download scripts
│
├── docs/                              # Project-Wide Documentation (GitHub Pages)
│   ├── index.md                       # Public presentation storyline
│   ├── architecture.md                # Multi-workstream system architecture & Mermaid diagrams
│   └── assets/                        # Figures & regression plots
│
├── requirements.txt                   # Consolidated Python dependencies
├── .env.example                       # Environment variable template
├── LICENSE                            # Apache-2.0 license
└── ATTRIBUTION.md                     # Upstream attribution notices
```

---

## 2. Research Contributions & Team Credits

| Research Contribution | Module Path | Focus & Scope | Dedicated Branch | Module Guide | Contributor |
| :--- | :--- | :--- | :--- | :--- | :--- |
| **Uncertainty-Based Selective Prediction** | `uncertainty_rejection/` | Raw pixel-level entropy extraction, 5 aggregation metrics, validation threshold calibration, risk-coverage curves, control benchmarks, and domain shift evaluation. | `feature/uncertainty-rejection` | [`uncertainty_rejection/README.md`](uncertainty_rejection/README.md) | Fernando Garcia Medina |
| **Text-Prompt Robustness** | `prompt_robustness/` | Prompt perturbation analysis, linguistic sensitivity, and cross-center phrasing evaluation. | `feature/prompt-robustness` | [`prompt_robustness/README.md`](prompt_robustness/README.md) | Ehsan |
| **Common Baseline** | `shared/` | MedCLIPSeg neural network, datasets, consolidated configs, and common execution scripts. | Coordinated via PR | [`shared/README.md`](shared/README.md) | Team Shared |

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

## 4. Attribution of Baseline Enhancements

Although `shared/medclipseg/` serves as the common baseline, specific enhancements were authored by team members:
- **Fernando Garcia Medina** implemented the raw `float32` `.npy` uncertainty extraction in `shared/medclipseg/test.py` (saving uncompressed entropy arrays prior to visualization normalization), dynamic `${MEDCLIPSEG_DATA}` resolution, and Apple Silicon MPS device support.

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
