# Stage 5 · Similarity retrieval in the system

Location `research/stage_5/` · [Up to research/](../RESEARCH_GUIDE.md)

Once Stage 4 accepts a matched query, the dashboard can search for similar historical cases if the user requests them. The search uses precomputed 512-dimensional visual embeddings and an eligibility flag on each reference case. The PABP method comes from [pabp-filtered-ivf](https://github.com/Fergarciamedina/pabp-filtered-ivf). The [application integration](../../application/uncertainty_aware_retrieval/uncertainty_aware_retrieval/pabp_engine.py) uses FAISS for the coarse search and block metadata for eligibility filtering.

The [retrieval tests](../../application/uncertainty_aware_retrieval/tests/test_pabp_engine.py) check candidate filtering, self-exclusion and comparison with exact filtered search on sampled queries at full probe depth. These are software checks. The Stage 4 study reports segmentation quality and coverage, independently of retrieval relevance. See the [application Stage 5 guide](../../application/stages/stage5_similarity_retrieval/SIMILARITY_SEARCH.md) for the gallery configuration, defaults and performance limits.

[Research overview](../RESEARCH_GUIDE.md) · [Next · Outputs](../stage_6/STUDY_OUTPUTS.md)
