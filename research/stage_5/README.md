# Stage 5 · Similarity retrieval

The source project for Predicate-Aware Block Pruning (PABP) is [pabp-filtered-ivf](https://github.com/Fergarciamedina/pabp-filtered-ivf). MedCLIPSeg++ uses a local [PABP integration](../../application/uncertainty_aware_retrieval/uncertainty_aware_retrieval/pabp_engine.py) with FAISS and precomputed visual vectors. Stage 4 determines query eligibility and the gallery search retains eligible reference cases. See the [application guide](../../application/README.md) to prepare or supply embeddings.
