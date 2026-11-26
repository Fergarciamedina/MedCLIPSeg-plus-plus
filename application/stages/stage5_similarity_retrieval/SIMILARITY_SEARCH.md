# Stage 5 · Search eligible reference cases

Location `application/stages/stage5_similarity_retrieval/` · [Up to stages/](../APPLICATION_STAGES.md)

The backend searches only when the request includes retrieval and the query passes the uncertainty rule. The query and reference scans use the same visual embedding space. Retrieval compares normalized embeddings by inner product, which is cosine similarity on unit vectors. Eligible neighbors are cases that also passed the policy. The query's own case key is excluded.

| Setting | Application default |
| --- | --- |
| Embeddings | 512-dimensional float32 visual vectors. The prepared encoder is UniMedCLIP ViT-B/16. Supply a keyed archive or precompute with the documented encoder workflow. |
| Coarse search | FAISS with eight requested inverted lists and four probes per query. The engine limits the effective list count for small collections. |
| Predicate pruning | PABP groups vectors in blocks of up to 16. A block with zero eligible cases can be skipped before computing candidate similarities. |
| Returned matches | Up to the Stage 2 limit, which defaults to five. The result can be shorter when fewer eligible references are found in the probed lists. |

PABP is attributed to [pabp-filtered-ivf](https://github.com/Fergarciamedina/pabp-filtered-ivf). The local [search engine](../../uncertainty_aware_retrieval/uncertainty_aware_retrieval/pabp_engine.py) uses eligibility summaries to avoid unnecessary comparisons. Probing a subset of lists limits the search work and can miss eligible neighbors in other lists. The [vector-search rationale](../../../docs/architecture/DESIGN_RATIONALE.md#vector-search-and-predicate-aware-block-pruning) explains that trade-off.

UniMedCLIP supplies the visual feature representation. FAISS supplies the indexing and coarse-search primitives. The source PABP project supplies the predicate-pruning approach. The application defines acceptance as the eligibility predicate and uses its own default search settings. Embedded candidate blocks and item-level filters also make the number of computed similarities observable in tests. [UniMedCLIP paper](https://arxiv.org/abs/2412.10372) · [Douze et al., *The Faiss Library*](https://arxiv.org/abs/2401.08281) · [PABP source](https://github.com/Fergarciamedina/pabp-filtered-ivf) · [Search defaults in code](../../uncertainty_aware_retrieval/uncertainty_aware_retrieval/config.py).

The [six engine tests](../../uncertainty_aware_retrieval/tests/test_pabp_engine.py) exercise index construction, block summaries and skips, acceptance filtering, self-exclusion, and comparison with exact filtered search at full probe depth. The Stage 4 study measures segmentation-quality selection, and these tests measure software behavior.

## Notebook files in this folder

| File | Role |
| --- | --- |
| [gallery_manifest.py](gallery_manifest.py) | Loads the historical gallery or creates demonstration metadata when the configured gallery is unavailable. |
| [visual_embedding_extractor.py](visual_embedding_extractor.py) | Wraps the shared encoder and provides synthetic vectors for the walkthrough when encoder assets are unavailable. |
| [pabp_engine.py](pabp_engine.py) | Makes the shared PABP engine available through the Stage 5 interface. |
| [vector_search_benchmarking.py](vector_search_benchmarking.py) | Compares exact prefiltering, ANN postfiltering, IVF bitmap filtering and PABP in the software walkthrough. |
| [__init__.py](__init__.py) | Exports the search, vector and benchmark interfaces. |

[Application stage map](../APPLICATION_STAGES.md) · [Prepare or connect embeddings](../../APPLICATION_GUIDE.md#embeddings-and-search-index) · [Next · Dashboard output](../stage6_system_output/DASHBOARD_OUTPUTS.md)
