# Uncertainty-aware retrieval services

Location `application/uncertainty_aware_retrieval/` · [Up to application/](../APPLICATION_GUIDE.md)

The dashboard backend, CLI and Streamlit view share the gallery loader, selective prediction service and PABP index. See the [application guide](../APPLICATION_GUIDE.md) to configure a collection, prepare vectors and interpret application outputs.

| File or folder | Role |
| --- | --- |
| [cli.py](cli.py) | Prepares the gallery vectors and runs batch retrieval demonstrations from the terminal. |
| [app.py](app.py) | Opens the Streamlit view for interactive inspection of the case and retrieval results. |
| [requirements.txt](requirements.txt) | Declares dependencies for these services and entry points. |
| [uncertainty_aware_retrieval/](uncertainty_aware_retrieval/RETRIEVAL_MODULES.md) | Implements the services shared by the three interfaces. |
| [tests/](tests/RETRIEVAL_TESTS.md) | Checks the individual services and filtered search behavior. |

Activate the application environment and run these commands from the repository root.

```bash
python application/uncertainty_aware_retrieval/cli.py --prepare
python application/uncertainty_aware_retrieval/cli.py --demo
streamlit run application/uncertainty_aware_retrieval/app.py
```

The CLI demonstration partitions a set of queries from the configured historical gallery. The canonical gallery has enough cases for ten representative queries. The [retrieval stage](../../research/stage_5_similarity_retrieval/RETRIEVAL_IN_CONTEXT.md) attributes the PABP implementation and the [research guide](../../research/stage_4_selective_prediction/SELECTIVE_PREDICTION_STUDY.md) describes the frozen uncertainty rule.

The following service tests use the original local data layout.

```bash
python -m unittest discover -s application/uncertainty_aware_retrieval/tests -v
```
