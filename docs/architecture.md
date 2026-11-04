# System Architecture & Technical Specifications

**Course:** ELEC 6661 — Medical Image Processing (Fall 2026), Concordia University\
**Project:** MedCLIPSeg++ Uncertainty-Based Selective Prediction Under Domain Shift

---

## 1. Multi-Track & Multi-Level System Architecture

The repository enforces strict separation between foundational scientific research (Level 0) and optional conceptual systems (Levels 1 and 2), with parallel support for prompt perturbation research:

```mermaid
graph TD
    classDef core fill:#2b5c8f,stroke:#1a365d,color:#ffffff,stroke-width:2px;
    classDef track fill:#4a5568,stroke:#2d3748,color:#ffffff,stroke-width:1.5px;
    classDef opt fill:#718096,stroke:#4a5568,color:#ffffff,stroke-dasharray: 5 5;
    classDef run fill:#2c7a7b,stroke:#234e52,color:#ffffff,stroke-width:1.5px;

    CoreLib["Shared Core Libraries<br>(src/medclipseg/ & src/uncertainty/)"]:::run

    subgraph Tracks ["Research Tracks"]
        TrackPR["Prompt Robustness Track<br>(research/prompt_robustness/)"]:::track
        TrackUR["Level 0: Uncertainty Rejection Research<br>(research/uncertainty_rejection/)"]:::core
    end

    subgraph ClinicalSystems ["Application Systems"]
        SysL1["Level 1: Priority UI Dashboard<br>(systems/priority_ui/)"]:::opt
        SysL2["Level 2: Similarity Retrieval Engine<br>(systems/similarity_search/)"]:::opt
    end

    CoreLib --> TrackPR
    CoreLib --> TrackUR
    TrackUR -->|"Calibrated Tau & Summary Metrics"| SysL1
    TrackUR -->|"Shared CLIP Feature Embeddings"| SysL2
```

---

## 2. Level 0: Scientific Research Pipeline

Level 0 is the mandatory core deliverable of the project. It calibrates selective prediction on In-Distribution validation data and tests its robustness under domain shift without model retraining.

```mermaid
flowchart TD
    classDef input fill:#4a5568,stroke:#2d3748,color:#ffffff;
    classDef proc fill:#2b5c8f,stroke:#1a365d,color:#ffffff;
    classDef dec fill:#d69e2e,stroke:#b7791f,color:#ffffff;
    classDef out fill:#2c7a7b,stroke:#234e52,color:#ffffff;

    subgraph DataSplits ["Breast Ultrasound Benchmarks"]
        BUSI_Val["BUSI Validation Split<br>(data/raw/BUSI/Val_Folder/)"]:::input
        BUSI_Test["BUSI Test Split<br>(data/raw/BUSI/Test_Folder/)"]:::input
        OOD_Sets["OOD Datasets<br>(BUSBRA, BUSUC, BUID, UDIAT)"]:::input
    end

    subgraph Stage01 ["Stage 01: Raw Entropy Extraction"]
        Infer["Frozen UniMedCLIP Inference<br>(src/medclipseg/test.py)"]:::proc
        SaveRaw["Save Raw float32 .npy<br>(src/uncertainty/raw_entropy.py)"]:::proc
    end

    subgraph Stage02 ["Stage 02: Aggregation & Correlation"]
        Agg["Compute Summary Metrics<br>(src/uncertainty/aggregation.py)"]:::proc
        Metrics["Spearman rho & AUROC<br>(run_aggregation.py)"]:::proc
    end

    subgraph Stage03 ["Stage 03: Selective Prediction"]
        Calib["Calibrate Fixed Tau_c on Val<br>(calibrate_rejection_threshold)"]:::dec
        RiskCov["Risk-Coverage Curves & AURC<br>(run_selective_prediction.py)"]:::proc
        Controls["Evaluate Controls<br>(Random, Naive Confidence, Oracle)"]:::proc
    end

    subgraph Stage04 ["Stage 04: Domain Shift Evaluation"]
        ApplyFrozen["Apply Fixed Tau_c to OOD<br>(Zero Retraining)"]:::dec
        OODMetrics["Evaluate Realized Coverage,<br>Accepted DSC & Degradation Gap"]:::out
    end

    BUSI_Val --> Infer
    BUSI_Test --> Infer
    OOD_Sets --> Infer

    Infer --> SaveRaw
    SaveRaw --> Agg
    Agg --> Metrics

    Agg --> Calib
    Calib -->|"Fixed Tau"| ApplyFrozen
    Agg --> RiskCov
    RiskCov --> Controls

    ApplyFrozen --> OODMetrics
```

---

## 3. Level 1: Operational Triage & Priority Flow

Level 1 specifies an interactive triage dashboard for new unannotated ultrasound acquisitions where ground truth is not available.

```mermaid
sequenceDiagram
    autonumber
    actor Clinician as Sonographer / Specialist
    participant UI as Priority Triage Dashboard
    participant Backend as Shared Inference Runner
    participant Filter as Validation Calibrated Rejection Rule

    Clinician->>UI: Upload N Unlabeled Ultrasound Images + Text Prompt
    UI->>Backend: Dispatch batch forward pass
    Backend->>Backend: Monte Carlo Value attention sampling
    Backend->>Backend: Compute predicted mask & raw entropy (U_fg)
    Backend->>Filter: Evaluate against fixed validation threshold Tau
    alt U_fg <= Tau
        Filter-->>UI: Tag: Routine Confidence (Low Review Priority)
    else U_fg > Tau
        Filter-->>UI: Tag: Needs Specialist Review (High Priority)
    end
    UI-->>Clinician: Prioritized audit queue with mask & uncertainty overlay
```

---

## 4. Level 2: Similarity Search & Predicate Pruning

Level 2 decouples vector indexing from online querying, utilizing precomputed block metadata summaries to prune non-matching IVF posting lists during vector distance computation.

### A. Offline Indexing Pipeline

```mermaid
flowchart LR
    classDef proc fill:#2b5c8f,stroke:#1a365d,color:#ffffff;
    classDef store fill:#4a5568,stroke:#2d3748,color:#ffffff;

    RawData["Deidentified Images +<br>Clinical Metadata"] --> CLIPEnc["Frozen CLIP Encoder<br>(L2 Normalized Vectors)"]:::proc
    CLIPEnc --> IVF["FAISS IVF Partitioning<br>(Voronoi Centroids)"]:::proc
    IVF --> BlockPart["Partition Inverted Lists<br>into Contiguous Blocks"]:::proc
    BlockPart --> Summaries["Precompute Metadata<br>Summaries per Block"]:::proc
    Summaries --> PersistIndex[("Persisted IVF Index +<br>Block Summaries")]:::store
```

### B. Online Query Pipeline with Within-List Block Pruning

```mermaid
flowchart TD
    classDef query fill:#4a5568,stroke:#2d3748,color:#ffffff;
    classDef decision fill:#d69e2e,stroke:#b7791f,color:#ffffff;
    classDef proc fill:#2b5c8f,stroke:#1a365d,color:#ffffff;
    classDef out fill:#2c7a7b,stroke:#234e52,color:#ffffff;

    Q["Query Image + Prompt + Metadata Predicate<br>(e.g., Pathology = Malignant)"]:::query --> QEnc["Compute Query Vector &<br>Level 0 Uncertainty Score"]:::proc
    QEnc --> IVFProbe["Select Top-N Nearest IVF Centroids"]:::proc
    IVFProbe --> ListTraverse["Traverse Posting Lists"]:::proc

    subgraph PruningBlock ["Within-List Block-Summary Pruning"]
        CheckSummary{"Does Block Summary<br>Satisfy Predicate?"}:::decision
        SkipBlock["Bypass Entire Block<br>(Zero Distance Computations)"]:::out
        ScoreBlock["Decompress Vectors &<br>Compute Dot Products"]:::proc
    end

    ListTraverse --> CheckSummary
    CheckSummary -- No --> SkipBlock
    CheckSummary -- Yes --> ScoreBlock

    ScoreBlock --> TopK["Select Top-K Similar References"]:::proc
    TopK --> Results["Display Diagnostic Matches +<br>Verified Dataset Metadata"]:::out
```
