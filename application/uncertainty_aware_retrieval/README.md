# Uncertainty-aware retrieval services

The dashboard backend, CLI and Streamlit view share the gallery loader, selective prediction service and PABP index. See the [application guide](../README.md) to configure a collection, prepare vectors and interpret application outputs.

Activate the application environment and run these commands from the repository root.

```bash
python application/uncertainty_aware_retrieval/cli.py --prepare
python application/uncertainty_aware_retrieval/cli.py --demo
streamlit run application/uncertainty_aware_retrieval/app.py
```

The CLI demonstration partitions a set of queries from the configured historical gallery. The canonical gallery has enough cases for ten representative queries. The [retrieval stage](../../research/stage_5/README.md) attributes the PABP implementation and the [research guide](../../research/stage_4/README.md) describes the frozen uncertainty rule.

The following service tests use the original local data layout.

```bash
python -m unittest discover -s application/uncertainty_aware_retrieval/tests -v
```
