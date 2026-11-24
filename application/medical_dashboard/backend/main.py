"""FastAPI interface for the precomputed MedCLIPSeg++ benchmark gallery."""

from __future__ import annotations

import os
import sys
import hashlib
import base64
from io import BytesIO
from functools import lru_cache
from pathlib import Path
from threading import Lock
from typing import Any, Optional
from urllib.parse import urlencode

from fastapi import FastAPI, File, Form, HTTPException, Query, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse, Response

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


def _parse_prompt(prompt: str, image_count: int = 1) -> dict[str, Any]:
    """Use the research application's existing deterministic prompt router."""
    from uncertainty_aware_retrieval import IntentRouter

    parsed = IntentRouter().parse(prompt, images=[None] * image_count)
    return {
        "raw_instruction": parsed.raw_instruction,
        "intent": parsed.intent,
        "top_k": parsed.top_k,
        "is_ambiguous": parsed.is_ambiguous,
        "explanation": parsed.explanation,
    }


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

        records = ManifestBuilder().build_or_load_manifest()
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
            "Could not initialize the gallery services. Install the application requirements, "
            "and verify the configured catalog and embeddings or checkpoint. "
            f"Underlying error: {exc}"
        ) from exc


_system_lock = Lock()


def _system_or_503() -> dict[str, Any]:
    try:
        with _system_lock:
            return get_system()
    except Exception as exc:
        raise HTTPException(status_code=503, detail=str(exc)) from exc


def _record(records: list[dict[str, Any]], dataset: str, image_id: str) -> dict[str, Any]:
    for item in records:
        if item["dataset"] == dataset and item["image_id"] == image_id:
            return item
    raise HTTPException(status_code=404, detail="Case not found in the gallery.")


def _image_fingerprint(image: Any) -> str:
    """Hash decoded pixels, not the upload filename or PNG/JPEG container bytes."""
    rgb = image.convert("RGB")
    digest = hashlib.sha256()
    digest.update(f"{rgb.width}x{rgb.height}:RGB".encode("ascii"))
    digest.update(rgb.tobytes())
    return digest.hexdigest()


def _find_matching_record(system: dict[str, Any], image: Any) -> Optional[dict[str, Any]]:
    """Return the canonical benchmark case when an uploaded scan is byte-for-byte equal in pixels."""
    fingerprints = system.setdefault("image_fingerprints", {})
    if not fingerprints:
        from PIL import Image

        for record in system["records"]:
            try:
                with Image.open(record["image_path"]) as canonical:
                    fingerprints.setdefault(_image_fingerprint(canonical), record)
            except (OSError, ValueError):
                # A corrupt/missing gallery asset must not stop other uploads from being evaluated.
                continue
    return fingerprints.get(_image_fingerprint(image))


def _analyze_record(system: dict[str, Any], record: dict[str, Any], prompt: str) -> dict[str, Any]:
    """Run the established prompt, triage, and retrieval flow over one canonical case."""
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
    key = record["image_id"] + "@" + record["dataset"]
    parsed = _parse_prompt(prompt)
    neighbors: list[dict[str, Any]] = []
    retrieval_metrics = None
    if triage.status == "Accepted" and parsed["intent"] in ("retrieve", "segment_retrieve"):
        query_vector = system["embeddings"][system["key_to_idx"][key]]
        found, metrics = system["retrieval"].search_top_k(
            query_vector=query_vector,
            k=parsed["top_k"],
            nprobe=system["nprobe"],
            filter_status="Accepted",
            excluded_ids={key},
        )
        neighbors = [
            {"image_id": item.image_id, "dataset": item.dataset, "similarity": float(item.similarity), "u_fg": item.u_fg, "dsc": item.dsc}
            for item in found
        ]
        retrieval_metrics = metrics.__dict__

    score = triage.u_fg if triage.u_fg is not None else triage.u_mean
    dataset, image_id = record["dataset"], record["image_id"]
    images = {
        kind: "/api/image?" + urlencode({"dataset": dataset, "image_id": image_id, "kind": kind})
        for kind in ("original", "mask", "uncertainty")
        if kind != "uncertainty" or record.get("unc_png_path") and Path(record["unc_png_path"]).is_file()
    }
    return {
        "case": {
            "image_id": image_id, "dataset": dataset, "status": triage.status,
            "u_fg": triage.u_fg, "u_mean": triage.u_mean, "score": score,
            "threshold": triage.threshold,
            "margin": (triage.threshold - score) if score is not None else None,
            "count_fg": record.get("count_fg"), "dsc": record.get("dsc"), "nsd": record.get("nsd"),
            "reason": triage.reason, "clinical_note": triage.clinical_note,
            "source": record.get("source", "Precomputed benchmark result"),
        },
        "images": images,
        "neighbors": neighbors,
        "retrieval_metrics": retrieval_metrics,
        "parsed_prompt": parsed,
    }


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
    offset: int = Query(default=0, ge=0),
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
            for r in selected[offset:offset + limit]
        ],
        "total": len(selected),
    }


@app.get("/api/cases/{dataset}/{image_id}/analyze")
def analyze_case(
    dataset: str,
    image_id: str,
    prompt: str = Query(default="Segment lesions and retrieve similar historical cases"),
) -> dict[str, Any]:
    system = _system_or_503()
    record = _record(system["records"], dataset, image_id)
    return _analyze_record(system, record, prompt)


@app.get("/api/image")
def image(
    dataset: str,
    image_id: str,
    kind: str = Query(pattern="^(original|mask|uncertainty)$"),
) -> Response:
    system = _system_or_503()
    record = _record(system["records"], dataset, image_id)
    path_key = {"original": "image_path", "mask": "seg_mask_path", "uncertainty": "unc_png_path"}[kind]
    path = Path(record[path_key]) if record.get(path_key) else None
    if path is None or not path.is_file():
        raise HTTPException(status_code=404, detail=f"Precomputed {kind} image is unavailable.")
    if kind == "mask":
        from PIL import Image

        buffer = BytesIO()
        with Image.open(path) as mask:
            binary = mask.convert("L").point(lambda value: 255 if value else 0)
            if Path(record["image_path"]).is_file():
                with Image.open(record["image_path"]) as original:
                    if binary.size != original.size:
                        binary = binary.resize(original.size, Image.Resampling.NEAREST)
            binary.save(buffer, format="PNG")
        return Response(buffer.getvalue(), media_type="image/png")
    media_type = "image/png" if path.suffix.lower() == ".png" else "image/jpeg"
    return FileResponse(path, media_type=media_type)


@app.post("/api/uploads/evaluate")
async def uploads_evaluate(
    files: list[UploadFile] = File(...),
    prompt: str = Form("Segment lesions and retrieve similar historical cases"),
) -> dict[str, Any]:
    """Evaluate one or more uploads against the canonical precomputed cohort.

    A matching upload is evaluated through the exact frozen benchmark flow. A truly
    new scan is kept request-scoped and reported honestly until the live inference
    adapter supplies a mask and dense entropy map.
    """
    from PIL import Image, UnidentifiedImageError

    if not files:
        raise HTTPException(status_code=400, detail="Select at least one PNG or JPEG ultrasound image.")
    if len(files) > 20:
        raise HTTPException(status_code=400, detail="A request can contain at most 20 images.")
    system = _system_or_503()
    parsed = _parse_prompt(prompt, len(files))
    items: list[dict[str, Any]] = []
    for file in files:
        if file.content_type not in {"image/png", "image/jpeg", "image/jpg"}:
            raise HTTPException(status_code=415, detail=f"{file.filename or 'Upload'} is not a PNG or JPEG image.")
        content = await file.read(20 * 1024 * 1024 + 1)
        if not content or len(content) > 20 * 1024 * 1024:
            raise HTTPException(status_code=413, detail=f"{file.filename or 'Image'} must be non-empty and no larger than 20 MB.")
        try:
            with Image.open(BytesIO(content)) as image_data:
                image_data.verify()
            with Image.open(BytesIO(content)) as image_data:
                image_data = image_data.convert("RGB")
                matched = _find_matching_record(system, image_data)
                preview_image = image_data.copy()
                preview_image.thumbnail((1600, 1600))
                preview = BytesIO()
                preview_image.save(preview, format="JPEG", quality=88)
        except (UnidentifiedImageError, OSError, ValueError) as exc:
            raise HTTPException(status_code=400, detail=f"{file.filename or 'Upload'} is not a readable image.") from exc

        preview_url = "data:image/jpeg;base64," + base64.b64encode(preview.getvalue()).decode("ascii")
        if matched:
            analysis = _analyze_record(system, matched, prompt)
            items.append({
                "upload_name": file.filename or "uploaded-ultrasound", "matched": True,
                "preview_data_url": preview_url, "analysis": analysis,
                "status": analysis["case"]["status"],
                "message": f"Matched canonical case {matched['image_id']} from {matched['dataset']} by decoded image content.",
            })
        else:
            items.append({
                "upload_name": file.filename or "uploaded-ultrasound", "matched": False,
                "preview_data_url": preview_url, "status": "Not Evaluated", "analysis": None,
                "message": "This is not an exact image in the precomputed cohort. Live MedCLIPSeg inference is not connected, so no segmentation, uncertainty score, triage, or retrieval can be claimed yet.",
            })
    return {
        "prompt": parsed,
        "items": items,
        "evaluated_count": sum(item["matched"] for item in items),
        "not_evaluated_count": sum(not item["matched"] for item in items),
    }
