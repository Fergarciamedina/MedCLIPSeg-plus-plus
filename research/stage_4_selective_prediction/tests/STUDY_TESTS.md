# Stage 4 study verification

Location `research/stage_4_selective_prediction/tests/` · [Up to stage_4_selective_prediction/](../SELECTIVE_PREDICTION_STUDY.md)

This folder checks the experiment implementation and the expected patterns in archived results. The [study protocol](../study_reference/EXPERIMENTAL_PROTOCOL.md) explains the separate validation and held-out evaluation.

| File | What it checks |
| --- | --- |
| [run_tests.sh](run_tests.sh) | Runs the import, configuration and uncertainty suites from the repository environment. |
| [test_imports.py](test_imports.py) | Imports the frozen segmentation backbone and the research uncertainty modules. |
| [test_config_loading.py](test_config_loading.py) | Checks YAML syntax for dataset profiles. |
| [test_uncertainty.py](test_uncertainty.py) | Checks entropy bounds, geometric and degenerate mask cases, score statistics, threshold reproducibility and the expected structure of BUSI and external results. |

From the repository root, with the [research environment](../SELECTIVE_PREDICTION_STUDY.md#reproduction) configured:

```bash
bash research/stage_4_selective_prediction/tests/run_tests.sh
```
