# Level 1 — Priority UI / Triage System (Conceptual System Architecture)

**Status:** Optional Extension (PLUS A) — Planned Architectural Scaffold
**Design Phase:** Specification and Interface Boundary Definition

---

## 1. System Motivation and Operational Objective

In clinical screening workflows, radiologists and sonographers face substantial image volumes. When deploying automated segmentation models, unannotated clinical images have **no ground-truth masks**; therefore, Dice Similarity Coefficient (DSC) and Normalized Surface Distance (NSD) cannot be computed.

Level 1 specifies a conceptual, lightweight triage dashboard that uses Level 0's validated uncertainty aggregation metric ($U$) and fixed validation-calibrated threshold ($\tau$) to categorize incoming acquisitions by **clinical review priority**.

> [!CAUTION]
> **Clinical Boundary Disclaimer:**
> The proposed system estimates **review priority**, NOT verified segmentation correctness. A low uncertainty score does not guarantee diagnostic accuracy, and an elevated score does not establish pathology. This specification describes an academic research prototype, not an approved medical device.

---

## 2. Operational Pipeline Flow

```text
[ N New Ultrasound Images ] + [ Standardized Text Prompt ]
               │
               ▼
[ Shared Frozen MedCLIPSeg Inference (mac/src/ or colab/src/) ]
               │
               ├────────────────────────┬────────────────────────┐
               ▼                        ▼                        ▼
     [ Predicted Mask ]       [ Raw Float Entropy ]    [ Colormap Overlay ]
               │                        │
               └───────────┬────────────┘
                           ▼
            [ Validated Aggregation (U_fg) ]
                           │
                           ▼
             [ Comparison with Fixed Tau ]
              (Tau calibrated in Level 0)
                           │
             ┌─────────────┴─────────────┐
             ▼                           ▼
    [ Low Review Priority ]    [ High Review Priority ]
     (U <= Tau: Expected        (U > Tau: Flagged for
      routine confidence)        expert human audit)
             │                           │
             └─────────────┬─────────────┘
                           ▼
          [ Interactive Triage Dashboard ]
```

---

## 3. Interface and Data Contract

### Inputs
- Unlabeled ultrasound images ($\text{PNG}/\text{DICOM}$).
- Text prompt string (e.g., `"ultrasound breast cancer segmentation"`).
- Pre-calibrated threshold $\tau_c$ and aggregation function imported directly from Level 0.

### Outputs & Visual Controls
- **Triage Queue:** Tabular list of cases sorted descending by uncertainty score $U$.
- **Side-by-Side Diagnostic Viewer:**
  1. Original B-mode ultrasound image.
  2. Predicted binary segmentation boundary overlay.
  3. Continuous pixel-level uncertainty map.
- **Triage Badges:**
  - `Routine Confidence` ($U \le \tau$)
  - `Needs Human Review` ($U > \tau$)

---

## 4. Implementation Boundary

- To preserve architectural cleanliness and prevent distraction from core research, **no empty placeholder UI scripts** (e.g., `app.py`) are created at this time.
- Implementation will proceed only if Level 0 research goals are fully achieved and explicit user instruction is provided.
