# Stage 2 · Intent routing

The application reuses the project's existing rule-based [intent router](../../application/uncertainty_aware_retrieval/uncertainty_aware_retrieval/intent_router.py) to translate requests into segmentation, retrieval, or both, with an optional number of matches. The [six-stage walkthrough](../notebooks/01_complete_system_pipeline.ipynb) illustrates its role. The [Stage 4 experiment](../stage_4/README.md) works from saved segmentation outputs.
