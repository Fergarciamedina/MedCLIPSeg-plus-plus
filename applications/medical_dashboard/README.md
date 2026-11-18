# MedCLIPSeg++ Research Dashboard (Phase 1)

An additive React + TypeScript dashboard and FastAPI API for reviewing the repository's precomputed benchmark cases. The existing Streamlit application remains available and unchanged.

Phase 1 supports case selection, original/mask/uncertainty image display, the existing selective triage policy and threshold values, and accepted-case PABP retrieval. Uploaded images are labeled **Not Evaluated**; live MedCLIPSeg inference is not connected in this phase. This is a research prototype, not a diagnostic product.

## Requirements

- macOS with Python 3.9+ and Node.js 20+
- The repository's benchmark files under `../data/raw` and `../data/outputs`, or set `MEDCLIPSEG_DATA` to the parent directory containing `raw/` and `outputs/`
- The existing UniMedCLIP checkpoint under `../data/checkpoints/` (or at the path expected by `applications/uncertainty_aware_retrieval/uncertainty_aware_retrieval/config.py`)
- The project dependencies in `applications/uncertainty_aware_retrieval/requirements.txt`
- A cached `gallery_embeddings.npy` avoids recomputing embeddings; absent that cache, the configured checkpoint is needed to compute them

## Launch on Mac

From the repository root, create and activate an environment if you do not already have one:

```bash
python3 -m venv .venv
source .venv/bin/activate
```

Install the Python dependencies, including the API server:

```bash
python -m pip install -r applications/medical_dashboard/backend/requirements.txt
```

Start the API in Terminal 1:

```bash
cd applications/medical_dashboard/backend
python -m uvicorn main:app --host 127.0.0.1 --port 8000
```

Install and start the frontend in Terminal 2:

```bash
cd applications/medical_dashboard/frontend
npm install
npm run dev
```

Open <http://localhost:5173>. API health and interactive documentation are at <http://localhost:8000/api/health> and <http://localhost:8000/docs>.

To use a non-default data directory, set its parent before starting the API:

```bash
export MEDCLIPSEG_DATA="/absolute/path/to/data"
```

The dashboard loads and builds the existing manifest and PABP gallery on its first data request. Keep the API process running after initialization; its in-memory index is shared across requests. The existing Streamlit application can still be started separately with:

```bash
streamlit run applications/uncertainty_aware_retrieval/app.py
```

## Phase 1 API surface

- `GET /api/health` — process health
- `GET /api/summary` — benchmark totals and frozen policy thresholds
- `GET /api/cases` — filterable precomputed manifest
- `GET /api/cases/{dataset}/{image_id}/analyze` — triage and gated retrieval for one manifest case
- `GET /api/image?dataset=...&image_id=...&kind=original|mask|uncertainty` — manifest-checked image response
- `POST /api/uploads/evaluate` — returns HTTP 501 with **Not Evaluated** until live inference is integrated

## Verification

```bash
cd applications/medical_dashboard/frontend
npm run build
```

```bash
cd applications/medical_dashboard/backend
python -m unittest discover -s tests -v
```

## Phase 2: live inference integration

The repository currently lacks an inference service that matches the apparent `MedCLIPSegAdapter` interface: the existing application `SegmentationService` does precomputed lookups and has no compatible `predict` method. Phase 2 should first identify the authoritative model loader, checkpoint, prompt format, preprocessing and MC sampling implementation. Then implement an adapter that returns the existing mask and dense entropy-map outputs, call it from an explicit asynchronous analysis job, persist temporary results safely, and pass its outputs through the unchanged selective-prediction service. The API should keep live uploads **Not Evaluated** until both outputs exist. No model, algorithm, metric, threshold, or research protocol changes are needed for the UI/API integration itself.
