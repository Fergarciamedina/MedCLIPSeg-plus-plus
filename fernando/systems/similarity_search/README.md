# Level 2 — Similarity Search & Metadata Pruning (Advanced Conceptual System)

**Status:** Optional Extension (PLUS B) — Planned Architectural Scaffold
**Design Phase:** Algorithmic Specification and Pipeline Boundary Definition

---

## 1. System Motivation and Scope

When clinical sonographers review flagged, ambiguous, or rare ultrasound presentations, finding historically confirmed cases with similar acoustic signatures and pathology can aid clinical decision support.

Level 2 specifies a conceptual multimodal retrieval engine that couples frozen CLIP visual embeddings with metadata-predicate search, accelerating search using **within-list block-summary pruning** inside an inverted file index (e.g., FAISS IVF).

> [!IMPORTANT]
> **Scientific and Diagnostic Disclaimers:**
> - The presence of clinically discriminative image embeddings in frozen UniMedCLIP encoders, retrieval relevance, and search speedups are **research hypotheses that require rigorous empirical benchmarking**.
> - Metadata predicates (e.g., pathology: `benign`, `malignant`, `normal`; patient age; scanner model) **must originate strictly from verified clinical or dataset annotations**, NEVER inferred or imputed from uncertainty scores alone.
> - Level 2 is strictly optional and must never create dependencies that delay or compromise Level 0 course deliverables.

---

## 2. Documented System Pipelines

Level 2 consists of two strictly decoupled pipelines:

### Pipeline A: Offline Indexing

```text
[ Authorized, Deidentified Image Archive + Clinical Metadata ]
                             │
                             ▼
     [ Frozen CLIP Vision Encoder (UniMedCLIP ViT-B/16) ]
                             │
                             ▼
           [ L2-Normalized Image Embeddings (d = 512) ]
                             │
                             ▼
                 [ Inverted File Index (FAISS IVF) ]
                (Quantization into Voronoi Centroids)
                             │
                             ▼
       [ Partition Inverted Posting Lists into Fixed Blocks ]
                             │
                             ▼
           [ Precompute Block Metadata Summaries ]
          (Bitsets / Min-Max ranges for metadata predicates)
                             │
                             ▼
      [ Persist IVF Vector Index + Block Summary Metadata ]
```

1. **Embedding Generation:** Historical ultrasound images pass through the frozen CLIP vision backbone; feature vectors are normalized ($\|v\|_2 = 1$).
2. **IVF Partitioning:** Vectors are assigned to coarse Voronoi cells ($k$-means centroids).
3. **Block-Summary Construction:** Inverted posting lists within each cell are divided into contiguous blocks (e.g., 64 or 128 vectors). For each block, metadata predicate summaries (e.g., min/max age, boolean bitmasks for pathology classes) are precomputed.
4. **Persistence:** Serialized to disk along with metadata lookup tables.

---

### Pipeline B: Online Querying & Within-List Block Pruning

```text
[ Query Image + Optional Text Prompt + Metadata Filter Predicates ]
                             │
                             ▼
  [ Shared Core Inference: CLIP Embedding + Level 0 Uncertainty ]
                             │
                             ▼
          [ Coarse Centroid Selection (Probe top-n lists) ]
                             │
                             ▼
       [ WITHIN-LIST BLOCK-SUMMARY PRUNING (Posting Level) ]
        ┌───────────────────────────────────────────────┐
        │  For each block in probed inverted list:      │
        │  Check block summary against query predicate. │
        │  - If NO vector matches: SKIP ENTIRE BLOCK    │
        │  - If MATCH possible: DECOMPRESS & SCORE      │
        └───────────────────────────────────────────────┘
                             │
                             ▼
              [ Top-K Nearest Neighbor Ranking ]
                             │
                             ▼
        [ Display Reference Cases + Validated Metadata ]
```

---

## 3. Algorithmic Distinction: Within-List Pruning vs. Pre/Post-Filtering

It is essential to distinguish the three filtering paradigms:

1. **Pre-Filtering:** Metadata predicates are evaluated first over the entire database, building an ad-hoc index or restricting vector search to a small ID set. *Drawback:* Degrades vector index cache locality and breaks coarse IVF quantization when predicate selectivity is high.
2. **Post-Filtering:** Vector similarity is computed over top-$M$ nearest neighbors, after which metadata predicates reject non-matching items. *Drawback:* May return zero results if matching metadata cases fall outside the top-$M$ vector sphere.
3. **Within-List Block-Summary Pruning (Specified Architecture):** The index traverses the probed inverted posting lists of selected IVF centroids. Before computing vector dot products for a block, its precomputed summary is evaluated against the query predicate. If the block contains zero qualifying items, the entire block of distance computations is bypassed without vector decompression.

---

## 4. Implementation Boundary

- In accordance with the project reorganization rules, no placeholder Python modules (such as `retrieval_engine.py`) are created at this time.
- Implementation will remain deferred until Level 0 raw evaluation is fully completed.
