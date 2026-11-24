"""Stage 5: Similarity Retrieval (FAISS-IVF + PABP) Package."""
from .pabp_engine import (
    PABPRetrievalEngine,
    PABPBlock,
    RetrievalMetricRecord,
    RetrievedNeighbor,
)
from .gallery_manifest import (
    load_or_build_gallery_manifest,
)
from .visual_embedding_extractor import (
    VisualEmbeddingExtractor,
)
from .vector_search_benchmarking import (
    VectorSearchBenchmarkSuite,
    benchmark_vector_search_algorithms,
)

__all__ = [
    "PABPRetrievalEngine",
    "PABPBlock",
    "RetrievalMetricRecord",
    "RetrievedNeighbor",
    "load_or_build_gallery_manifest",
    "VisualEmbeddingExtractor",
    "VectorSearchBenchmarkSuite",
    "benchmark_vector_search_algorithms",
]
