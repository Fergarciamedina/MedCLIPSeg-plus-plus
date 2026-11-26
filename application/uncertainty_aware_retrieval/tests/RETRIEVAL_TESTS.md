# Retrieval service tests

Location `application/uncertainty_aware_retrieval/tests/` · [Up to uncertainty_aware_retrieval/](../RETRIEVAL_SERVICES.md)

Use this folder to verify contracts shared by the CLI, Streamlit, and dashboard. The [service guide](../RETRIEVAL_SERVICES.md) gives the test command and dataset requirements.

| File | What it checks |
| --- | --- |
| [test_intent_router.py](test_intent_router.py) | Supported operations, count extraction and ambiguous requests. |
| [test_manifest.py](test_manifest.py) | Canonical gallery construction and stable case identity. |
| [test_pabp_engine.py](test_pabp_engine.py) | Candidate eligibility, self-exclusion, search counters and comparison with exact filtered search at full probe depth. |
| [test_selective_prediction.py](test_selective_prediction.py) | Foreground and empty-mask decisions and the decision metadata used downstream. |
| [test_visualizer.py](test_visualizer.py) | Overlay rendering and optional uncertainty display artifacts. |
