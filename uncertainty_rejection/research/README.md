# Track: Uncertainty-Based Selective Prediction

**Track Lead:** Fernando Garcia Medina
**Course:** ELEC 6661 — Medical Image Processing (Fall 2026), Concordia University
**Supervisor / Lab:** IMPACT Lab (Prof. Hassan Rivaz, Omid Nejati Manzari)
**Baseline:** MedCLIPSeg (CVPR 2026, Koleilat et al., arXiv:2602.20423)

---

## 1. Track Overview and Architectural Levels

This track investigates post-hoc failure detection and selective prediction for text-guided medical image segmentation using MedCLIPSeg's Monte Carlo variational Value uncertainty maps. The architecture is modularly separated into three distinct levels:

| Level | Name | Purpose | Course Status | Implementation Status |
| :--- | :--- | :--- | :--- | :--- |
| **Level 0** | **Research / Experimental Evaluation** | Scientific evaluation of raw uncertainty calibration, failure AUROC, selective prediction, and domain shift across BUS datasets without model retraining. | **Mandatory Course Scope** | **In Progress** |
| **Level 1** | **Priority UI / Simple System** | Interactive triage dashboard prioritizing unannotated clinical cases by uncertainty score to flag high-risk segmentations for human specialist review. | Optional PLUS A | Planned (Scaffold only) |
| **Level 2** | **Similarity Search / Advanced System** | Multimodal retrieval engine indexing historical cases via frozen CLIP embeddings with within-list block-summary predicate pruning in FAISS IVF. | Optional PLUS B | Planned (Scaffold only) |

---

## 2. Dependency Rules and Code Reuse Principles

To maintain strict scientific rigor and software maintainability, the following non-negotiable architectural rules apply:

```text
[ shared/medclipseg/ ]       <-- Shared Core Inference Engine (test.py)
          │
          ▼
 [ Level 0: Research ]      <-- Mandatory Scientific Deliverable (ELEC 6661)
          │ (Aggregators, Calibrated Tau, Raw Uncertainty Arrays)
    ┌──────┴──────────────┐
    ▼                     ▼
[ Level 1: UI ]      [ Level 2: Search ]  <-- Strictly Optional Extensions
```

1. **Zero Code Duplication:**
   - Neither Level 1 nor Level 2 implements independent copies of the MedCLIPSeg neural network, inference loops, or metric calculators.
   - All experimental execution relies on the unified platform backend in `shared/medclipseg/` (supporting Apple Silicon MPS and NVIDIA CUDA).
2. **Strict Independence of Level 0:**
   - Level 0 is completely self-contained. Progress on the course deliverable must **never** be delayed or blocked by work on Level 1 or Level 2.
3. **Calibrated Thresholds as Boundary:**
   - Level 1 and Level 2 consume only the **validation-calibrated rejection threshold ($\tau$)** and aggregation functions established and validated in Level 0.
4. **No Premature Implementation:**
   - Optional levels remain documented scaffolds with defined interface boundaries until Level 0 validation is fully executed with raw numerical uncertainty data.
