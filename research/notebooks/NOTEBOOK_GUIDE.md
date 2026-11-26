# Research and software notebooks

Location `research/notebooks/` · [Up to research/](../RESEARCH_GUIDE.md)

Start with a local clone of [MedCLIPSeg++ on GitHub](https://github.com/Fergarciamedina/MedCLIPSeg-plus-plus). Run the setup for the notebook you want inside the clone. Both notebooks run independently of the dashboard and keep generated outputs local.

## Complete system pipeline notebook

[Open 01_complete_system_pipeline.ipynb](01_complete_system_pipeline.ipynb) to follow the six software interfaces. It accepts a directory of PNG or JPEG images. The synthetic segmentation adapter generates a mask and entropy map to demonstrate the handoffs, policy and display. The retrieval example can use a configured gallery or demonstration data. Use the [dashboard](../../application/APPLICATION_GUIDE.md) to run the application with real precomputed outputs and vectors.

```bash
python3 -m venv .venv
.venv/bin/python -m pip install -r application/uncertainty_aware_retrieval/requirements.txt jupyterlab
export MEDCLIPSEG_NOTEBOOK_IMAGES="/absolute/path/to/images"
.venv/bin/python -m jupyterlab research/notebooks/01_complete_system_pipeline.ipynb
```

You can also set `INPUT_FOLDER` in the notebook. Follow the [gallery contract](../../application/APPLICATION_GUIDE.md#gallery-catalog) if supplying gallery metadata and embeddings. The notebook reports that its segmentation and review demonstration use mock outputs.

## Scientific benchmarking notebook

[Open 02_scientific_benchmarking.ipynb](02_scientific_benchmarking.ipynb) to reproduce the plots and inspect the archived results. This mode uses the aggregate CSVs in the repository.

```bash
python3 -m venv .venv
.venv/bin/python -m pip install jupyterlab numpy pandas matplotlib pillow scipy
.venv/bin/python -m jupyterlab research/notebooks/02_scientific_benchmarking.ipynb
```

For local case examples, set `MEDCLIPSEG_OUTPUT` to the experiment artifacts. To rerun the protocol from that notebook, install the [research requirements](../../requirements.txt), configure the [study inputs and paths](../stage_4_selective_prediction/SELECTIVE_PREDICTION_STUDY.md#reproduction), and select `scores` or `full` in the first code cell. The same launcher runs from Bash. The optional rerun displays a separate table of regenerated results and writes all outputs to the configured directory.
