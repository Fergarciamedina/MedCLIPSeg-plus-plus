# Evaluate saved segmentation scores

Location `research/stage_4_selective_prediction/uncertainty_selective_evaluation/` · [Up to stage_4_selective_prediction/](../SELECTIVE_PREDICTION_STUDY.md)

This standalone toolkit measures selective prediction from a CSV of per-image uncertainty and segmentation quality. It can report how often a score retains predictions, the mean quality of those predictions, and risk across a range of coverage. For the complete Stage 4 experiment and its reported tables, start with the [Stage 4 study guide](../SELECTIVE_PREDICTION_STUDY.md).

## Files in this folder

| File | Role |
| --- | --- |
| [evaluate.py](evaluate.py) | CLI entry point for synthetic data or a per-image score CSV. |
| [metrics.py](metrics.py) | Implements score, uncertainty and selective prediction helpers used by the toolkit. |
| [model_adapter.py](model_adapter.py) | Defines a mask-and-entropy interface for integrating an evaluator with a segmentation model and includes a synthetic adapter. |
| [__init__.py](__init__.py) | Exposes the toolkit's Python interface. |
| [test_reproducibility.py](test_reproducibility.py) | Exercises the standalone evaluation with reproducible inputs. |
| [requirements.txt](requirements.txt) | Defines the dependencies for running the toolkit independently. |
| [COLLABORATOR_INTEGRATION_GUIDE.md](COLLABORATOR_INTEGRATION_GUIDE.md) | Explains how to connect code that already produces masks and uncertainty. |

## Quick start

Run these commands from the repository root with your Python environment active.

```bash
python -m pip install -r research/stage_4_selective_prediction/uncertainty_selective_evaluation/requirements.txt
python research/stage_4_selective_prediction/uncertainty_selective_evaluation/evaluate.py \
  --demo --output_dir ./demo_results
```

The demo generates synthetic scores and metrics so you can inspect the output format without downloading images or a segmentation model.

## Evaluate a CSV

Supply one row per image, including a score column such as `U_fg` and a reference quality column such as `DSC`. Lower uncertainty values are retained. If the input includes other recognized uncertainty columns, `--compare_aggregators` compares the available columns. Calibrate the cutoff on your evaluation protocol's validation set before applying it to a separate test set.

```bash
python research/stage_4_selective_prediction/uncertainty_selective_evaluation/evaluate.py \
  --input_csv /path/to/predictions.csv \
  --aggregator U_fg --metric_col DSC --threshold 0.064058324 \
  --compare_aggregators --output_dir ./evaluation_results
```

The example cutoff is the one fitted on BUSI in this repository. The optional `--two_stage` branch uses `U_mean` for empty predictions. See the [frozen policy](../experiments/RESULTS_GUIDE.md#how-to-read-the-frozen-yaml) for scope and thresholds.

| Output | How to use it |
| --- | --- |
| `per_image_evaluation.csv` | Inspect which cases are accepted or flagged for review. |
| `selective_summary.csv` | Compare baseline and retained quality alongside coverage and AURC. |
| `risk_coverage_curve.csv` | Plot retained loss against retained fraction. |
| `aggregator_comparison.csv` | Compare rank association and AURC for the score columns supplied. |

The [integration guide](COLLABORATOR_INTEGRATION_GUIDE.md) shows how to use the Python helpers with existing predictions. The [upstream attribution](../../../ATTRIBUTION.md) documents sources used by MedCLIPSeg++.
