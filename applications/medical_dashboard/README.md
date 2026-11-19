# MedCLIPSeg++ Research Dashboard (Phase 1)

An additive React + TypeScript dashboard and FastAPI API for reviewing the repository's precomputed benchmark cases. The existing Streamlit application remains available and unchanged.

The dashboard accepts a clinical prompt plus one to twenty images per request through a single **Send** action. It routes the prompt through the existing deterministic `IntentRouter`; for an uploaded scan that exactly matches a canonical benchmark image by decoded pixels, it returns the stored segmentation, frozen uncertainty decision, and any prompt-authorized PABP matches. In **Current request**, the clinician selects one submitted image at a time and sees its original image, segmentation overlay, policy status, and requested similar cases. The uncertainty map is not displayed in this view. A genuinely new image remains **Not Evaluated** because live MedCLIPSeg inference is not connected. Results are request-scoped and not persisted. This is a research prototype, not a diagnostic product.

Benchmark metadata is sourced from the cohort aggregation CSVs and cached in `applications/uncertainty_aware_retrieval/runtime/gallery_manifest.json`; embeddings are cached in `applications/uncertainty_aware_retrieval/runtime/gallery_embeddings.npy`. Original scans, masks, uncertainty maps and source metric CSVs remain in the configured `data/raw/` and `data/outputs/` tree. There is no SQL database in this application. The API builds the PABP index in memory at startup.

## Requirements

- macOS with Python 3.9+ and Node.js 20+
- The repository's benchmark files under `../data/raw` and `../data/outputs`, or set `MEDCLIPSEG_DATA` to the parent directory containing `raw/` and `outputs/`
- The existing UniMedCLIP checkpoint under `../data/checkpoints/` (or at the path expected by `applications/uncertainty_aware_retrieval/uncertainty_aware_retrieval/config.py`)
- The project dependencies in `applications/uncertainty_aware_retrieval/requirements.txt`
- A cached `gallery_embeddings.npy` avoids recomputing embeddings; absent that cache, the configured checkpoint is needed to compute them

## Launch on Mac

The simplest local launch is one command from the repository root:

```bash
bash applications/medical_dashboard/run_local.sh
```

It checks the local dependencies, starts both services on `127.0.0.1`, opens
`http://127.0.0.1:5173`, and stops both services when you press `Ctrl+C`. It
does not upload data, install packages, or make the dashboard reachable from
another device.

### Manual launch

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
- `GET /api/image?dataset=...&image_id=...&kind=original|mask|uncertainty` — manifest-checked image response
- `GET /api/cases/{dataset}/{image_id}/analyze?prompt=...` — parse the request through the existing intent router and run the requested supported actions over precomputed case outputs
- `POST /api/uploads/evaluate` — multipart `files` (1–20) + prompt. Exact canonical matches return their precomputed segmentation, triage, and prompt-authorized retrieval results; new scans return an explicit **Not Evaluated** state until live inference exists.

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

The repository currently lacks an inference service that matches the apparent `MedCLIPSegAdapter` interface: the existing application `SegmentationService` does precomputed lookups and has no compatible `predict` method. Phase 2 should first identify the authoritative model loader, checkpoint, prompt format, preprocessing and MC sampling implementation. Then implement an adapter that returns the existing mask and dense entropy-map outputs, call it from an explicit asynchronous analysis job, handle temporary results safely, and pass its outputs through the unchanged selective-prediction service. The API should keep live uploads **Not Evaluated** until both outputs exist. No model, algorithm, metric, threshold, or research protocol changes are needed for the UI/API integration itself.
