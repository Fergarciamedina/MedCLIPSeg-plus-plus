# System Architecture & Technical Specifications

**Course:** ELEC 6661 — Medical Image Processing (Fall 2026), Concordia University\
**Project:** MedCLIPSeg++ Multi-Track Research

---

## 1. Research Contributions & Architecture

The repository is organized by **research contribution**, providing clear separation of experimental tracks while anchoring both in a minimal shared baseline:

```mermaid
graph TD
    classDef shared fill:#2c7a7b,stroke:#234e52,color:#ffffff,stroke-width:2px;
    classDef uq fill:#2b5c8f,stroke:#1a365d,color:#ffffff,stroke-width:2px;
    classDef prompt fill:#4a5568,stroke:#2d3748,color:#ffffff,stroke-width:1.5px;
    classDef opt fill:#718096,stroke:#4a5568,color:#ffffff,stroke-dasharray: 5 5;

    SharedBaseline["shared/medclipseg/<br>(Frozen Foundation & Adapters)"]:::shared

    subgraph UncertaintyContribution ["Uncertainty Quantification & Rejection (uncertainty_rejection/)"]
        UQ_Uncertainty["uncertainty_rejection/src/uncertainty/<br>(Entropy, Aggregation, Selective Rejection)"]:::uq
        UQ_Research["uncertainty_rejection/research/<br>(Level 0: Experiments 01-04)"]:::uq
        UQ_Sys1["uncertainty_rejection/systems/priority_ui/<br>(Level 1: Triage Dashboard)"]:::opt
        UQ_Sys2["uncertainty_rejection/systems/similarity_search/<br>(Level 2: IVF Block Pruning)"]:::opt
    end

    subgraph PromptRobustnessContribution ["Prompt Robustness & Perturbations (prompt_robustness/)"]
        PR_Research["prompt_robustness/research/<br>(Prompt Robustness & Perturbations)"]:::prompt
    end

    SharedBaseline --> UQ_Uncertainty
    SharedBaseline --> PR_Research
    UQ_Uncertainty --> UQ_Research
    UQ_Research -->|"Calibrated Tau"| UQ_Sys1
    SharedBaseline -->|"CLIP Embeddings"| UQ_Sys2
```

---

## 2. Development & Branching Workflow

```mermaid
gitGraph
   commit id: "baseline"
   branch feature/uncertainty-rejection
   checkout feature/uncertainty-rejection
   commit id: "raw-npy-save"
   commit id: "aggregation-metrics"
   commit id: "selective-prediction"
   checkout main
   branch feature/prompt-robustness
   checkout feature/prompt-robustness
   commit id: "prompt-scenarios"
   commit id: "perturbation-eval"
   checkout main
   merge feature/uncertainty-rejection id: "pr-uq-merge"
   merge feature/prompt-robustness id: "pr-pr-merge"
```

---

## 3. Level 0: Scientific Uncertainty & Rejection Pipeline

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
        Infer["Frozen UniMedCLIP Inference<br>(shared/medclipseg/test.py)"]:::proc
        SaveRaw["Save Raw float32 .npy<br>(uncertainty_rejection/src/uncertainty/raw_entropy.py)"]:::proc
    end

    subgraph Stage02 ["Stage 02: Aggregation & Correlation"]
        Agg["Compute Summary Metrics<br>(uncertainty_rejection/src/uncertainty/aggregation.py)"]:::proc
        Metrics["Spearman rho & AUROC<br>(uncertainty_rejection/research/aggregation/)"]:::proc
    end

    subgraph Stage03 ["Stage 03: Selective Prediction"]
        Calib["Calibrate Fixed Tau_c on Val<br>(calibrate_rejection_threshold)"]:::dec
        RiskCov["Risk-Coverage Curves & AURC<br>(uncertainty_rejection/research/selective_prediction/)"]:::proc
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

## 4. Level 1: Operational Triage & Priority Flow

Level 1 specifies an interactive triage dashboard for new unannotated ultrasound acquisitions where ground truth is not available.

```mermaid
sequenceDiagram
    autonumber
    actor Clinician as Sonographer / Specialist
    participant UI as Priority Triage Dashboard (uncertainty_rejection/systems/priority_ui/)
    participant Backend as Shared Inference Runner (shared/medclipseg/)
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

## 5. Level 2: Similarity Search & Predicate Pruning

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

    Q["Query Image + Prompt + Metadata Predicate<br>(e.g., Pathology = Malignant)"]:::query --> QEnc["Compute Query Vector &<br>Uncertainty Score"]:::proc
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
