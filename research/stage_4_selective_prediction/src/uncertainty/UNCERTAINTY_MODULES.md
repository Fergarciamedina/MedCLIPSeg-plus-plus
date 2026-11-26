# Uncertainty and selective prediction modules

Location `research/stage_4_selective_prediction/src/uncertainty/` · [Up to src/](../RESEARCH_SOURCE.md)

This package provides the reusable calculations called by the Stage 4 experiment scripts. Its mathematical definitions and evaluation conventions are in [Theory and metrics](../../study_reference/THEORY_AND_METRICS.md).

| File | Responsibility |
| --- | --- |
| [__init__.py](__init__.py) | Package initializer for the uncertainty modules. |
| [raw_entropy.py](raw_entropy.py) | Converts sampled logits into pixelwise predictive entropy and writes numeric and visualization artifacts. |
| [aggregation.py](aggregation.py) | Reduces a mask and entropy map to scan-level scores, including foreground and boundary summaries. |
| [metrics.py](metrics.py) | Computes score–quality Spearman correlations, bootstrap intervals and failure AUROC. |
| [selective_prediction.py](selective_prediction.py) | Calibrates retention thresholds, ranks cases, computes risk–coverage and AURC, and compares empty-mask policies and controls. |

[Research source index](../RESEARCH_SOURCE.md) · [Study verification](../../tests/STUDY_TESTS.md)
