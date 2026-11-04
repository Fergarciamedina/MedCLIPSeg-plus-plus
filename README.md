# MedCLIPSeg++: Uncertainty-Based Selective Prediction Under Domain Shift

**Course:** ELEC 6661 — Medical Image Processing (Fall 2026), Concordia University\
**Lab:** IMPACT Lab (Image Processing And Characterization of Tissue)\
**Supervisor / Instructor:** Prof. Hassan Rivaz, Omid Nejati Manzari\
**Baseline Upstream Repository:** [HealthX-Lab/MedCLIPSeg](https://github.com/HealthX-Lab/MedCLIPSeg) (CVPR 2026, Apache-2.0)\
**Reference Paper:** *MedCLIPSeg: Data-Efficient Medical Image Segmentation with Frozen Vision-Language Foundation Models* (arXiv:2602.20423)

---

## 1. Project Philosophy & Core Research Scope

Built following the IMPACT Lab course methodology (**BASELINE + ONE CHANGE + FAIR EVALUATION**), this project investigates whether raw pixel-level predictive uncertainty from frozen probabilistic adapters can reliably predict segmentation failures and guide selective prediction under domain shift.

- **Baseline:** MedCLIPSeg probabilistic vision-language architecture (`UniMedCLIP ViT-B/16` frozen backbone, variational Key/Value attention, and Monte Carlo Value sampling).
- **The One Change:** Post-hoc **Uncertainty-Based Selective Prediction (Rejection Rule)** with **zero model retraining**. A rejection threshold ($\tau$) is calibrated strictly on In-Distribution (BUSI) validation data to abstain from making autonomous predictions on low-confidence cases, evaluating the accepted-set quality across out-of-distribution ultrasound domains.
- **Fair Evaluation:** Evaluation on breast ultrasound benchmarks (`BUSI` as In-Distribution; `BUSBRA`, `BUSUC`, `BUID`, `UDIAT` as Out-Of-Distribution) using identical preprocessing, splits, and metrics (DSC, NSD, AURC).

---

## 2. Three-Level System Architecture

The repository is organized around three modular levels, accompanied by an independent parallel research track on prompt sensitivity:

```text
MedCLIPSeg-plus-plus/
├── README.md                          # Project documentation and reproduction guide
├── LICENSE                            # Apache-2.0 open-source license
├── ATTRIBUTION.md                     # Upstream attribution to HealthX-Lab
├── requirements.txt                   # Consolidated Python dependencies
├── .gitignore                         # Strict exclusion of raw datasets, weights & arrays
├── .env.example                       # Canonical environment variable template
│
├── src/                               # Shared reusable core library
│   ├── medclipseg/                    # MedCLIPSeg vision-language segmentation engine
│   │   ├── test.py                    # Inference with raw .npy & visualization extraction
│   │   ├── datasets/                  # Dataloaders and preprocessing generators
│   │   ├── trainers/                  # Variational adapters & model builders
│   │   ├── clip/                      # CLIP encoders
│   │   ├── biomedclip/                # BiomedCLIP text and vision modules
│   │   ├── open_clip_lib/             # OpenCLIP transformer backbones
│   │   └── utils/                     # Evaluator (eval.py), metrics, checkpoint downloader
│   └── uncertainty/                   # Reusable uncertainty quantification library
│       ├── raw_entropy.py             # Binary pixel entropy calculation & artifact persistence
│       ├── aggregation.py             # Feature computation (U_mean, U_fg, U_std_fg, U_ratio, U_boundary)
│       ├── metrics.py                 # Spearman rank correlation & failure AUROC
│       └── selective_prediction.py    # Tau calibration, risk-coverage curves & controls
│
├── research/                          # Scientific experiments and investigative tracks
│   ├── uncertainty_rejection/         # [LEVEL 0: Mandatory Course Scope]
│   │   ├── raw_entropy/               # Stage 01: Raw float32 array extraction specifications
│   │   ├── aggregation/               # Stage 02: run_aggregation.py (Spearman rho & AUROC)
│   │   ├── selective_prediction/      # Stage 03: run_selective_prediction.py (Tau, Risk-Coverage, AURC)
│   │   ├── ood_evaluation/            # Stage 04: Domain shift evaluation across BUSBRA/BUSUC/BUID/UDIAT
│   │   └── future_work.md             # Untested extensions and future research ideas
│   └── prompt_robustness/             # Track: Text prompt perturbation & sensitivity (Ehsan)
│
├── systems/                           # Clinical applications and retrieval systems
│   ├── priority_ui/                   # [LEVEL 1: Optional PLUS A] Triage dashboard for human review queue
│   └── similarity_search/             # [LEVEL 2: Optional PLUS B] FAISS IVF retrieval with within-list block pruning
│
├── configs/                           # Dataset configuration YAMLs (BUSI, BUSBRA, UDIAT, etc.)
├── scripts/                           # Shell workflow entry points (run_inference, download_ckpt, eval)
├── notebooks/                         # Interactive notebooks (evaluate.ipynb)
├── tests/                             # Automated test suite (imports, uncertainty, configs)
└── docs/                              # Project web documentation & architecture diagrams
    ├── index.md                       # GitHub Pages presentation storyline
    ├── architecture.md                # System diagrams & technical data contracts
    └── assets/                        # Figures & regression plots
```

---

## 3. Implementation & Validation Status Matrix

| Level / Component | Objective | Status | Primary Entry Point |
| :--- | :--- | :---: | :--- |
| **MedCLIPSeg Core** | Frozen probabilistic inference and mask prediction | `completed` | [`src/medclipseg/test.py`](src/medclipseg/test.py) |
| **Raw Entropy Persistence** | Save float32 `.npy` arrays strictly before visualization normalization | `completed` | [`src/uncertainty/raw_entropy.py`](src/uncertainty/raw_entropy.py) |
| **Aggregation Metrics** | Extract $U_{\text{mean}}, U_{\text{fg}}, U_{\text{std\_fg}}, U_{\text{ratio}}, U_{\text{boundary}}$ | `completed` | [`src/uncertainty/aggregation.py`](src/uncertainty/aggregation.py) |
| **Correlation & AUROC** | Spearman $\rho$ and failure detection AUROC ($\text{DSC} < 0.5$) | `completed` | [`research/uncertainty_rejection/aggregation/run_aggregation.py`](research/uncertainty_rejection/aggregation/run_aggregation.py) |
| **Selective Prediction** | Validation $\tau$ calibration, Risk-Coverage curve, and controls comparison | `completed` | [`research/uncertainty_rejection/selective_prediction/run_selective_prediction.py`](research/uncertainty_rejection/selective_prediction/run_selective_prediction.py) |
| **OOD Evaluation** | Evaluate frozen $\tau$ on BUSBRA, BUSUC, BUID, and UDIAT | `planned` | [`research/uncertainty_rejection/ood_evaluation/README.md`](research/uncertainty_rejection/ood_evaluation/README.md) |
| **Level 1: Priority UI** | Triage dashboard prioritizing unannotated cases for expert review | `planned` | [`systems/priority_ui/README.md`](systems/priority_ui/README.md) |
| **Level 2: Similarity Search** | Inverted list index with within-list block-summary predicate pruning | `planned` | [`systems/similarity_search/README.md`](systems/similarity_search/README.md) |
| **Prompt Robustness** | Prompt sensitivity and linguistic perturbation track (Ehsan) | `in progress` | [`research/prompt_robustness/README.md`](research/prompt_robustness/README.md) |

---

## 4. Benchmark Results on BUSI (In-Distribution Test Set)

Running Stage 02 and Stage 03 on the official BUSI test set yields the following verified metrics:

### Correlation with Segmentation Quality & Failure Detection
| Uncertainty Feature | Spearman $\rho$ (DSC) | Spearman $\rho$ (NSD) | AUROC ($\text{DSC} < 0.5$) | Diagnostic Assessment |
| :--- | :---: | :---: | :---: | :--- |
| **$U_{\text{std\_fg}}$ (Lesion Std Dev)** | **-0.9022** ($p=1.8\times 10^{-29}$) | **-0.8882** ($p=2.2\times 10^{-27}$) | **0.9560** | Strongest overall error predictor |
| **$U_{\text{fg}}$ (Lesion Mean)** | **-0.9016** ($p=2.2\times 10^{-29}$) | **-0.8963** ($p=1.5\times 10^{-28}$) | **0.9630** | Near-perfect failure discriminator |
| **$U_{\text{boundary}}$ (Boundary Band)** | -0.6596 ($p=5.1\times 10^{-11}$) | -0.7593 ($p=7.8\times 10^{-16}$) | 0.8750 | Sensitive to surface boundary errors |
| **$U_{\text{mean}}$ (Global Mean)** | -0.5523 ($p=1.6\times 10^{-07}$) | -0.6695 ($p=2.1\times 10^{-11}$) | 0.7917 | Moderate (diluted by background pixels) |
| **$U_{\text{ratio}}$ (Foreground Ratio)** | -0.4982 ($p=3.5\times 10^{-06}$) | -0.3662 ($p=9.8\times 10^{-04}$) | 0.6088 | Weakest individual predictor |

### Selective Prediction vs. Control Baselines (Feature: $U_{\text{fg}}$)
| Target Coverage | Realized Coverage | Uncertainty Rejection (DSC) | Random Rejection (95% CI) | Oracle Rejection (DSC) |
| :---: | :---: | :---: | :---: | :---: |
| **100%** | 100% | 0.8572 | 0.8572 [0.857 – 0.857] | 0.8572 |
| **90%** | 90% | **0.9038** | 0.8573 [0.847 – 0.874] | 0.9175 |
| **80%** | 79% | **0.9202** | 0.8574 [0.839 – 0.884] | 0.9333 |
| **70%** | 71% | **0.9386** | 0.8586 [0.831 – 0.889] | 0.9431 |

* **Area Under Risk-Coverage Curve (AURC):** **0.0464** (lower indicates superior selective performance).

> [!NOTE]
> Values above reflect current evaluation on BUSI test data. Definitive thesis conclusions will compare these against validation-calibrated fixed $\tau$ runs and raw `.npy` arrays across all out-of-distribution targets.

---

## 5. Quickstart & Reproduction Guide

### Prerequisites
- Python 3.9+ with Apple Silicon (MPS) or NVIDIA GPU (CUDA).
- Canonical data directory structured under `../data/raw/` and `../data/outputs/`.

```bash
# 1. Setup virtual environment
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt

# 2. Run automated test suite
PYTHONPATH=. python -c "
import tests.test_imports as t1; t1.test_import_medclipseg(); t1.test_import_uncertainty()
import tests.test_uncertainty as t2; t2.test_pixel_entropy_bounds(); t2.test_aggregation_features()
import tests.test_config_loading as t3; t3.test_yaml_syntax()
print('All tests passed!')
"

# 3. Execute inference and generate uncertainty maps (Apple Silicon MPS default)
bash scripts/run_inference.sh BUSI 666

# 4. Run Stage 02 aggregation and correlation analysis
python research/uncertainty_rejection/aggregation/run_aggregation.py --dataset BUSI --seed 666

# 5. Run Stage 03 selective prediction and control comparison
python research/uncertainty_rejection/selective_prediction/run_selective_prediction.py --dataset BUSI --seed 666 --feature U_fg
```

---

## 6. License & Attribution

This codebase is licensed under the Apache-2.0 License. It builds directly upon [MedCLIPSeg](https://github.com/HealthX-Lab/MedCLIPSeg) developed by the HealthX-Lab and IMPACT Lab at Concordia University. Full notices and citations are documented in [`ATTRIBUTION.md`](ATTRIBUTION.md) and [`LICENSE`](LICENSE).
