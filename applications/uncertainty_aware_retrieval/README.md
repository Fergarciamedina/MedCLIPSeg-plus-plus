# Uncertainty-Aware Medical Image Retrieval & Triage

An interactive clinical decision-support application combining post-hoc predictive uncertainty triage, validation-calibrated selective prediction, and Predicate-Aware Block Pruning (PABP) for multimodal breast ultrasound retrieval.

Part of the **[MedCLIPSeg++](https://github.com/Fergarciamedina/MedCLIPSeg-plus-plus)** research platform. Detailed system specifications and flowcharts are provided in [docs/architecture/SYSTEM_ARCHITECTURE.md](../../docs/architecture/SYSTEM_ARCHITECTURE.md).

---

## 1. Clinical Workflow & Architecture

The application is structured around a 3-step clinical workflow:

```
┌────────────────────────────────────────────────────────────────────────┐
│  STEP 1: SELECT IMAGES                                                 │
│  - Demonstration Cohort (10 cases: 6 low uncertainty, 4 review)       │
│  - Clinical Benchmark Selection (BUSI, UDIAT, BUID, BUSUC, BUSBRA)     │
│  - Upload External Ultrasound Scans                                    │
└───────────────────────────────────┬────────────────────────────────────┘
                                    ▼
┌────────────────────────────────────────────────────────────────────────┐
│  STEP 2: ANALYZE                                                       │
│  - Deterministic Natural-Language Intent Router                        │
│  - Post-Hoc Selective Triage (tau = 0.064058, empty fallback = 1e-4)    │
│  - FAISS-IVF + Predicate-Aware Block Pruning (PABP) Retrieval          │
└───────────────────────────────────┬────────────────────────────────────┘
                                    ▼
┌────────────────────────────────────────────────────────────────────────┐
│  STEP 3: REVIEW RESULTS                                                │
│  - Summary Metrics (Processed, Accepted by Policy, Needs Review)       │
│  - Case-by-Case Inspection (Interactive Dropdown)                     │
│  - Side-by-Side View: Original Ultrasound vs AI-Predicted Contour     │
│  - Interactive Opacity Control Slider (0% to 100%)                     │
│  - Optional Predictive Entropy Heatmap Toggle                          │
│  - Top-K Retrieved Historical Matches (Compact Visual Cards)           │
│  - Separate Tab for Review-Required Scans                              │
│  - Collapsed Research Diagnostics (PABP Profiling & Ground-Truth DSC)  │
└────────────────────────────────────────────────────────────────────────┘
```

---

## 2. Root Cause of Previous Crash & Stabilization Architecture

### The Problem:
On macOS Apple Silicon (M-series), switching Streamlit input modes previously triggered intermittent native crashes (SIGSEGV / connection errors).

### Root Cause Analysis:
1. **Unnecessary Native Model Execution on UI Reruns:** On every widget change (e.g., clicking a radio button or moving a slider), the application was re-training `faiss.Clustering.train()` and invoking PyTorch ViT forward passes concurrently while the previous Streamlit worker thread was being interrupted.
2. **OpenMP Multi-Threading Contention:** PyTorch, FAISS, and Accelerate load separate OpenMP runtimes on macOS. When multiple threads executed native C++ loops simultaneously during rapid Streamlit script reloads, thread-pool collisions resulted in segmentation faults.
3. **Eager Execution:** The analysis pipeline ran unconditionally on every render turn rather than waiting for an explicit user trigger.

### The Solution:
1. **Index Built and Cached ONCE:** The full 542-scan FAISS-IVF PABP index is constructed once at launch using `@st.cache_resource`. Query searches use `excluded_ids` for $O(1)$ query exclusion without rebuilding the index.
2. **Precomputed Embedding Lookup:** For benchmark and demonstration scans, 512-dim visual embeddings are retrieved via $O(1)$ table lookup from precomputed arrays, avoiding redundant neural network passes during UI interactions.
3. **Thread Pool Serialization:** Single-threaded execution is enforced across runtimes (`faiss.omp_set_num_threads(1)` and environment variable limits for OpenMP/MKL/Accelerate).
4. **Decoupled Workflow with Session State:** Step 1 (selection), Step 2 (explicit "Analyze Images" trigger), and Step 3 (viewing and opacity controls) are cleanly decoupled through `st.session_state`. Changing display settings never reruns inference or retrieval.

---

## 3. Scientific Correctness & Counting Reconciliation

### Discrepancy Resolution:
- **Prior Issue:** The manifest reported 6 Accepted and 4 Rejected demo cases, while the triage dashboard showed 5 Accepted and 5 Rejected.
- **Root Cause:** In `selective_prediction.py`, the condition `if is_unseen or u_fg is None:` was evaluated before checking `count_fg == 0` (empty mask). For normal scan `103_normal.png`, the mask is empty, meaning $U_{\text{fg}}$ is `None` by definition. The check erroneously routed it to rejection before evaluating the Level 0 empty-mask fallback ($\tau_{\text{neg}} = 10^{-4}$ on $U_{\text{mean}}$).
- **Correction:** The empty-mask check is now evaluated first. Because `103_normal.png` has $U_{\text{mean}} = 0.000000 \le 10^{-4}$, it is correctly classified as **Accepted by Policy**.
- **Result:** The demonstration cohort evaluates to exactly **6 Accepted by Policy** and **4 Needs Review**, matching the canonical Level 0 benchmark truth table.

### Triage Status Distinctions:
1. **Accepted by Policy:** Low predictive boundary uncertainty ($U_{\text{fg}} \le 0.064058$ or $U_{\text{mean}} \le 10^{-4}$ for empty masks). Proceeds to similarity retrieval.
2. **Needs Review:** Elevated predictive boundary uncertainty ($U_{\text{fg}} > 0.064058$ or suspicious empty mask). Autonomous retrieval suppressed; referred for sonographer inspection.
3. **Not Evaluated:** External uploads or scans lacking calibrated predictive entropy maps. Clearly distinguished from policy rejections.

---

## 4. PABP Algorithmic Diagnostics & Profiling

The application strictly separates block-level pruning from per-item metadata filtering:

$$\text{Total Candidates Probed} = N_{\text{pruned\_blocks}} + N_{\text{filtered\_in\_scanned}} + N_{\text{excluded\_query}} + N_{\text{distances\_computed}}$$

- **Blocks Pruned by PABP:** Inverted list blocks containing zero eligible candidates (`any_eligible == False`) that are **skipped entirely before calculating vector inner products**.
- **Candidates Filtered in Scanned Blocks:** Ineligible candidates skipped individually inside retained blocks.
- **Vector Distances Computed:** Exact floating-point inner products calculated only for eligible candidates.

*All PABP parameters and detailed counters are isolated in the collapsed Research Diagnostics section to keep the clinical interface uncluttered.*

---

## 5. Execution Commands

### Run the Application:
```bash
streamlit run applications/uncertainty_aware_retrieval/app.py
```

### Run the CLI Batch Demonstration ($M=10$ Scans):
```bash
python3 applications/uncertainty_aware_retrieval/cli.py --demo
```

### Run Full Application Test Suite (25 Tests):
```bash
python3 -m unittest discover -s applications/uncertainty_aware_retrieval/tests -v
```

### Run Level 0 Scientific Regression Tests (31 Tests):
```bash
bash uncertainty_rejection/scripts/run_tests.sh
```
