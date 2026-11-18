"""FastAPI interface for the precomputed MedCLIPSeg++ benchmark gallery."""

from __future__ import annotations

import os
import sys
import hashlib
from functools import lru_cache
from pathlib import Path
from typing import Any, Optional

from fastapi import FastAPI, HTTPException, Query
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse

APP_DIR = Path(__file__).resolve().parents[2] / "uncertainty_aware_retrieval"
REPO_ROOT = Path(__file__).resolve().parents[3]
for path in (str(APP_DIR), str(REPO_ROOT)):
    if path not in sys.path:
        sys.path.insert(0, path)

app = FastAPI(title="MedCLIPSeg++ Research Dashboard API", version="1.0.0")
app.add_middleware(
    CORSMiddleware,
    allow_origins=os.getenv("DASHBOARD_CORS_ORIGINS", "http://localhost:5173").split(","),
    allow_credentials=False,
    allow_methods=["GET", "POST"],
    allow_headers=["*"],
)


@lru_cache(maxsize=1)
def get_system() -> dict[str, Any]:
    """Initialize the exact manifest, triage service, and PABP index used by Streamlit."""
    try:
        from uncertainty_aware_retrieval import (
            DEFAULT_BLOCK_SIZE,
            DEFAULT_NLIST,
            DEFAULT_NPROBE,
            EmbeddingExtractor,
            ManifestBuilder,
            PABPRetrievalEngine,
            SegmentationService,
            SelectivePredictionService,
        )

        custom_data_root = os.getenv("MEDCLIPSEG_DATA")
        if custom_data_root:
            data_root = Path(custom_data_root).expanduser()
            cache_key = hashlib.sha256(str(data_root.resolve()).encode("utf-8")).hexdigest()[:12]
            builder = ManifestBuilder(
                raw_dir=data_root / "raw",
                outputs_dir=data_root / "outputs",
                cache_path=APP_DIR / "runtime" / f"gallery_manifest_{cache_key}.json",
            )
        else:
            builder = ManifestBuilder()
        records = builder.build_or_load_manifest()
        extractor = EmbeddingExtractor()
        embeddings = extractor.extract_or_load_gallery_embeddings(records)
        triage = SelectivePredictionService()
        segmentation = SegmentationService(manifest=records)
        retrieval = PABPRetrievalEngine(
            nlist=DEFAULT_NLIST,
            nprobe=DEFAULT_NPROBE,
            block_size=DEFAULT_BLOCK_SIZE,
        )
        retrieval.build_index(records, embeddings)
        return {
            "records": records,
            "embeddings": embeddings,
            "extractor": extractor,
            "triage": triage,
            "segmentation": segmentation,
            "retrieval": retrieval,
            "nprobe": DEFAULT_NPROBE,
            "key_to_idx": {
                record["image_id"] + "@" + record["dataset"]: i
                for i, record in enumerate(records)
            },
        }
    except Exception as exc:
        raise RuntimeError(
            "Could not initialize the benchmark services. Install the repository requirements, "
            "and verify MEDCLIPSEG_DATA plus the checkpoint/embedding cache. "
            f"Underlying error: {exc}"
        ) from exc


def _system_or_503() -> dict[str, Any]:
    try:
        return get_system()
    except Exception as exc:
        raise HTTPException(status_code=503, detail=str(exc)) from exc


def _record(records: list[dict[str, Any]], dataset: str, image_id: str) -> dict[str, Any]:
    for item in records:
        if item["dataset"] == dataset and item["image_id"] == image_id:
            return item
    raise HTTPException(status_code=404, detail="Case not found in the benchmark manifest.")


@app.get("/api/health")
def health() -> dict[str, str]:
    return {"status": "ok", "service": "MedCLIPSeg++ Research Dashboard"}


@app.get("/api/summary")
def summary() -> dict[str, Any]:
    system = _system_or_503()
    records = system["records"]
    datasets = sorted({r["dataset"] for r in records})
    return {
        "case_count": len(records),
        "datasets": datasets,
        "accepted_count": sum(r["status"] == "Accepted" for r in records),
        "review_count": sum(r["status"] == "Rejected" for r in records),
        "threshold": system["triage"].threshold,
        "empty_mask_threshold": system["triage"].empty_fallback_threshold,
    }


@app.get("/api/cases")
def cases(
    dataset: Optional[str] = None,
    status: Optional[str] = None,
    search: str = "",
    limit: int = Query(default=600, ge=1, le=1000),
) -> dict[str, Any]:
    system = _system_or_503()
    selected = system["records"]
    if dataset:
        selected = [r for r in selected if r["dataset"] == dataset]
    if status:
        selected = [r for r in selected if r["status"] == status]
    if search.strip():
        term = search.casefold().strip()
        selected = [r for r in selected if term in r["image_id"].casefold()]
    return {
        "items": [
            {
                "image_id": r["image_id"],
                "dataset": r["dataset"],
                "status": r["status"],
                "u_fg": r.get("u_fg"),
                "u_mean": r.get("u_mean"),
                "count_fg": r.get("count_fg"),
                "dsc": r.get("dsc"),
                "nsd": r.get("nsd"),
            }
            for r in selected[:limit]
        ],
        "total": len(selected),
    }


@app.get("/api/cases/{dataset}/{image_id}/analyze")
def analyze_case(dataset: str, image_id: str) -> dict[str, Any]:
    system = _system_or_503()
    record = _record(system["records"], dataset, image_id)
    seg = system["segmentation"].get_segmentation_for_query(
        record["image_path"], image_name=record["image_id"]
    )
    triage = system["triage"].evaluate_case(
        u_fg=seg["u_fg"],
        u_mean=seg["u_mean"],
        count_fg=seg["count_fg"],
        image_id=seg["image_id"],
        dataset=seg["dataset"],
        is_unseen=seg.get("is_unseen", False),
    )
    key = image_id + "@" + dataset
    neighbors: list[dict[str, Any]] = []
    retrieval_metrics = None
    if triage.status == "Accepted":
        query_vector = system["embeddings"][system["key_to_idx"][key]]
        found, metrics = system["retrieval"].search_top_k(
            query_vector=query_vector,
            k=5,
            nprobe=system["nprobe"],
            filter_status="Accepted",
            excluded_ids={key},
        )
        neighbors = [
            {
                "image_id": item.image_id,
                "dataset": item.dataset,
                "similarity": float(item.similarity),
                "u_fg": item.u_fg,
                "dsc": item.dsc,
            }
            for item in found
        ]
        retrieval_metrics = metrics.__dict__

    score = triage.u_fg if triage.u_fg is not None else triage.u_mean
    return {
        "case": {
            "image_id": record["image_id"],
            "dataset": record["dataset"],
            "status": triage.status,
            "u_fg": triage.u_fg,
            "u_mean": triage.u_mean,
            "score": score,
            "threshold": triage.threshold,
            "margin": (triage.threshold - score) if score is not None else None,
            "count_fg": record.get("count_fg"),
            "dsc": record.get("dsc"),
            "nsd": record.get("nsd"),
            "reason": triage.reason,
            "clinical_note": triage.clinical_note,
            "source": "Precomputed benchmark result",
        },
        "images": {
            kind: f"/api/image?dataset={dataset}&image_id={image_id}&kind={kind}"
            for kind in ("original", "mask", "uncertainty")
        },
        "neighbors": neighbors,
        "retrieval_metrics": retrieval_metrics,
    }


@app.get("/api/image")
def image(
    dataset: str,
    image_id: str,
    kind: str = Query(pattern="^(original|mask|uncertainty)$"),
) -> FileResponse:
    system = _system_or_503()
    record = _record(system["records"], dataset, image_id)
    path_key = {"original": "image_path", "mask": "seg_mask_path", "uncertainty": "unc_png_path"}[kind]
    path = Path(record[path_key]).resolve()
    if not path.is_file():
        raise HTTPException(status_code=404, detail=f"Precomputed {kind} image is unavailable.")
    media_type = "image/png" if path.suffix.lower() == ".png" else "image/jpeg"
    return FileResponse(path, media_type=media_type)


@app.post("/api/uploads/evaluate")
def uploads_evaluate() -> None:
    raise HTTPException(
        status_code=501,
        detail="Not Evaluated: new-image inference is not available in Phase 1.",
    )
