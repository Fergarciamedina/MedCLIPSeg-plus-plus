# Shared Baseline: MedCLIPSeg & Common Infrastructure

**Baseline Foundation:** [HealthX-Lab/MedCLIPSeg](https://github.com/HealthX-Lab/MedCLIPSeg) (CVPR 2026, Apache-2.0)\
**Reference Paper:** arXiv:2602.20423

---

## 1. Overview & Principles

The `shared/` directory contains the minimal, common foundation used by both team workstreams (**Fernando** and **Ehsan**). It strictly houses:
1. **`shared/medclipseg/`**: The core vision-language neural network modules (CLIP, BiomedCLIP, OpenCLIP, variational adapters, dataloaders, and evaluation routines).
2. **`shared/configs/`**: Consolidated dataset configuration YAML files (supporting Apple Silicon MPS and NVIDIA CUDA).
3. **`shared/scripts/`**: Common workflow scripts for checkpoint downloading, baseline inference, and standard Dice/NSD ground-truth evaluation.

---

## 2. Shared Directory Layout

```text
shared/
├── README.md                          # Shared infrastructure documentation and guidelines
│
├── medclipseg/                        # Core MedCLIPSeg implementation
│   ├── test.py                        # Unified inference runner
│   ├── datasets/                      # Dataset loaders and preprocessing
│   ├── trainers/                      # Variational adapter architectures & builders
│   ├── clip/                          # CLIP text and vision modules
│   ├── biomedclip/                    # BiomedCLIP modules
│   ├── open_clip_lib/                 # OpenCLIP transformer backbones
│   └── utils/                         # Evaluator (eval.py), metrics (SurfaceDice), etc.
│
├── configs/                           # Dataset configuration YAMLs
│   ├── BUSI.yaml, BUSBRA.yaml, BUSUC.yaml, BUID.yaml, UDIAT.yaml, etc.
│
└── scripts/                           # Common execution entry points
    ├── run_inference.sh               # Run baseline inference
    ├── download_checkpoint.sh         # Fetch weights from Hugging Face
    └── run_evaluation.sh              # Compute ground-truth DSC and NSD
```

---

## 3. Team Coordination & Governance

> [!IMPORTANT]
> **Coordination Rule:**
> The code in `shared/` is shared across both research tracks. Any modifications to `shared/medclipseg/`, `shared/configs/`, or `shared/scripts/` must be coordinated between team members through a pull request and code review before merging into `main`.

---

## 4. Attribution of Shared Modifications

While `shared/medclipseg/` serves as the common baseline, specific architectural enhancements were contributed by team members:
- **Raw Uncertainty Persistence:** The modification to `shared/medclipseg/test.py` that serializes uncompressed `float32` `.npy` predictive entropy arrays strictly before per-image min-max normalization was designed and implemented by **Fernando** to support the Level 0 uncertainty rejection research.
- **Environment Resolution:** Support for dynamic `${MEDCLIPSEG_DATA}` environment variable resolution and MPS device fallback was implemented by **Fernando**.
