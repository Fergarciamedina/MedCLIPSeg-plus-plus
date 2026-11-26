# Runtime retrieval modules

Location `application/uncertainty_aware_retrieval/uncertainty_aware_retrieval/` · [Up to uncertainty_aware_retrieval/](../RETRIEVAL_SERVICES.md)

This Python package is the shared implementation used by the dashboard backend, the CLI and the Streamlit view. It runs over an image catalog and saved segmentation artifacts.

| Module | Responsibility |
| --- | --- |
| [__init__.py](__init__.py) | Exposes the service classes and helper functions used by the entry points. |
| [config.py](config.py) | Centralizes data routes, artifact settings, uncertainty cutoffs and search defaults. |
| [manifest_builder.py](manifest_builder.py) | Builds the canonical gallery or validates and loads a custom case CSV. |
| [intent_router.py](intent_router.py) | Maps a web request to its operation and requested number of references. |
| [segmentation_service.py](segmentation_service.py) | Loads stored masks and associated uncertainty artifacts for catalog cases. |
| [selective_prediction.py](selective_prediction.py) | Applies the foreground or empty-mask cutoff to produce the review decision used in runtime search. |
| [embedding_extractor.py](embedding_extractor.py) | Loads keyed precomputed vectors or encodes images with UniMedCLIP and caches their representations. |
| [pabp_engine.py](pabp_engine.py) | Builds the IVF index and runs eligibility-filtered similarity search, tracking pruned blocks and scored candidates. |
| [visualizer.py](visualizer.py) | Composes image overlays and loads uncertainty visualizations for presentation. |

[Run the services](../RETRIEVAL_SERVICES.md) · [Service tests](../tests/RETRIEVAL_TESTS.md) · [Stage 5 search rationale](../../stages/stage5_similarity_retrieval/SIMILARITY_SEARCH.md)
