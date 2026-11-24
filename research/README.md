# Research

The [six-stage architecture](../docs/architecture/figures/block_diagram/complete_system_architecture.drawio) traces the system from the image and request to its visible results. Stage 4 is the research contribution evaluated in this repository. The other stages establish the data, model, retrieval and output context for that experiment.

| Stage | Role |
| --- | --- |
| [1 · Inputs](stage_1/README.md) | Breast ultrasound cohorts, text prompts and test split provenance |
| [2 · Intent routing](stage_2/README.md) | Reused request routing and its role in the system |
| [3 · Segmentation](stage_3/README.md) | Frozen MedCLIPSeg backbone and artifact provenance |
| [4 · Selective prediction](stage_4/README.md) | Definitions, validation-only selection, locked test, external evaluation, and reproducibility |
| [5 · Similarity retrieval](stage_5/README.md) | PABP upstream and its use in the system |
| [6 · System output](stage_6/README.md) | Case status, overlays and reference cases |

## Notebooks

- [Scientific benchmarking](notebooks/02_scientific_benchmarking.ipynb) reads the bundled aggregate outputs. Optional local per-image artifacts enable the qualitative examples and additional plots.
- [Six-stage software walkthrough](notebooks/01_complete_system_pipeline.ipynb) takes a folder of images and uses a synthetic segmentation adapter. The walkthrough demonstrates orchestration. Stage 4 evidence comes from the saved experimental results and analysis scripts.

Run the following commands from the repository root to install the notebook dependencies and start Jupyter.

```bash
python3 -m venv .venv
source .venv/bin/activate
python -m pip install -r requirements.txt -r application/uncertainty_aware_retrieval/requirements.txt jupyterlab
python -m jupyterlab research/notebooks
```

For the software walkthrough, set `INPUT_FOLDER` in the configuration cell to a folder of PNG or JPEG images. The benchmarking notebook runs its aggregate sections using the checkout.
