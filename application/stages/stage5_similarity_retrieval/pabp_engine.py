"""Stage 4: Predicate-Aware Block Pruning (PABP) Engine (MOD-SIMILARITY-RETRIEVAL / L1.5).

Implements vector similarity search with IVF clustering and block-level predicate
pruning to skip non-eligible blocks before vector distance computations.
"""

from application.uncertainty_aware_retrieval.uncertainty_aware_retrieval.pabp_engine import (
    PABPRetrievalEngine,
    PABPBlock,
    RetrievalMetricRecord,
    RetrievedNeighbor,
)

__all__ = [
    "PABPRetrievalEngine",
    "PABPBlock",
    "RetrievalMetricRecord",
    "RetrievedNeighbor",
]

