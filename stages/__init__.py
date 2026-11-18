"""MedCLIPSeg++: 6-Stage Modular Architecture Package.

Provides high-level modular LEGO building blocks for the complete clinical pipeline:
- Stage 1 (Inputs): Clinical input ingestion and validation (Ultrasound X & Prompt R)
- Stage 2 (Intent Router): Clinical intent parsing f_intent(R) -> ExecutionPlan(s, r, K)
- Stage 3 (MedCLIPSeg): Multi-modal segmentation backbone (Ev, Et, PVL Adapters, Dseg -> M, U)
- Stage 4 (Selective Prediction): Calibrated triage policy (U_fg <= tau_fg, U_mean <= tau_neg -> Gate g)
- Stage 5 (Similarity Retrieval): FAISS-IVF candidate search + PABP block pruning (g AND r)
- Stage 6 (System Output): Multi-panel clinical output composer and triage audit tables
"""

# Stage 1: Inputs
from stages.stage1_inputs import (
    ClinicalInputs,
    load_clinical_inputs,
)

# Stage 2: Intent Router
from stages.stage2_intent_router import (
    ExecutionPlan,
    parse_clinical_intent,
)

# Stage 3: Segmentation Backbone
from stages.stage3_segmentation_backbone import (
    BaseSegmentationModelAdapter,
    MedCLIPSegAdapter,
    MockModelAdapter,
)

# Stage 4: Selective Prediction (Triage)
from stages.stage4_selective_prediction import (
    DecisionOutcome,
    evaluate_decision_policy,
    FROZEN_TAU_STAR,
    FROZEN_TAU_NEG,
    compute_foreground_uncertainty,
    compute_mean_uncertainty,
    compute_boundary_uncertainty,
    compute_all_uncertainty_metrics,
    extract_all_aggregators,
)

# Stage 5: Similarity Retrieval (FAISS-IVF + PABP)
from stages.stage5_similarity_retrieval import (
    PABPRetrievalEngine,
    PABPBlock,
    RetrievalMetricRecord,
    RetrievedNeighbor,
    load_or_build_gallery_manifest,
    VisualEmbeddingExtractor,
    VectorSearchBenchmarkSuite,
    benchmark_vector_search_algorithms,
)

# Stage 6: System Output
from stages.stage6_system_output import (
    compose_system_output,
    generate_triage_summary_report,
)

__all__ = [
    # Stage 1
    "ClinicalInputs",
    "load_clinical_inputs",
    # Stage 2
    "ExecutionPlan",
    "parse_clinical_intent",
    # Stage 3
    "BaseSegmentationModelAdapter",
    "MedCLIPSegAdapter",
    "MockModelAdapter",
    # Stage 4
    "DecisionOutcome",
    "evaluate_decision_policy",
    "FROZEN_TAU_STAR",
    "FROZEN_TAU_NEG",
    "compute_foreground_uncertainty",
    "compute_mean_uncertainty",
    "compute_boundary_uncertainty",
    "compute_all_uncertainty_metrics",
    "extract_all_aggregators",
    # Stage 5
    "PABPRetrievalEngine",
    "PABPBlock",
    "RetrievalMetricRecord",
    "RetrievedNeighbor",
    "load_or_build_gallery_manifest",
    "VisualEmbeddingExtractor",
    "VectorSearchBenchmarkSuite",
    "benchmark_vector_search_algorithms",
    # Stage 6
    "compose_system_output",
    "format_triage_audit_table",
]
