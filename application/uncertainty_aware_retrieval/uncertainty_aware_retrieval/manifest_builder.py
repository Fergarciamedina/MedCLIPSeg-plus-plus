"""Load a precomputed case catalog or assemble the original benchmark gallery."""

import csv
import math
import os
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple
import numpy as np
import pandas as pd
from PIL import Image

from .selective_prediction import SelectivePredictionService

from .config import (
    CANONICAL_DATASETS,
    DEFAULT_OUTPUTS_DIR,
    DEFAULT_RAW_DATA_DIR,
    EMPTY_MASK_FALLBACK_THRESHOLD,
    FROZEN_THRESHOLD,
)


class ManifestBuilder:
    """Builds and caches the verified historical case manifest."""

    def __init__(
        self,
        raw_dir: Optional[Path] = None,
        outputs_dir: Optional[Path] = None,
        gallery_path: Optional[Path] = None,
    ):
        self.raw_dir = Path(raw_dir or DEFAULT_RAW_DATA_DIR)
        self.outputs_dir = Path(outputs_dir or DEFAULT_OUTPUTS_DIR)
        configured_gallery = gallery_path or os.getenv("MEDCLIPSEG_GALLERY")
        self.gallery_path = Path(configured_gallery).expanduser().resolve() if configured_gallery else None

    def build_or_load_manifest(self) -> List[Dict[str, Any]]:
        if self.gallery_path:
            return self.load_gallery()
        return self.build_manifest()

    def load_gallery(self) -> List[Dict[str, Any]]:
        """Read one case per CSV row and compute missing triage scores from raw entropy."""
        required = {"dataset", "image_id", "image_path", "seg_mask_path"}
        records = []
        case_keys = set()
        triage = SelectivePredictionService()
        with self.gallery_path.open(newline="", encoding="utf-8-sig") as handle:
            reader = csv.DictReader(handle)
            if not required.issubset(reader.fieldnames or []):
                raise ValueError(f"Gallery CSV requires columns: {', '.join(sorted(required))}")
            for line, raw in enumerate(reader, 2):
                row = {k: (v or "").strip() for k, v in raw.items() if k is not None}
                try:
                    dataset, image_id = row["dataset"], row["image_id"]
                    if not all(
                        item and item not in {".", ".."} and not set(item).intersection("@/\\")
                        for item in (dataset, image_id)
                    ):
                        raise ValueError("dataset and image_id must be valid nonempty case identifiers")
                    key = (dataset, image_id)
                    if key in case_keys:
                        raise ValueError(f"duplicate case {dataset}/{image_id}")
                    case_keys.add(key)
                    artifact_paths = {
                        field: self._artifact_path(row.get(field, ""), required=field in required)
                        for field in ("image_path", "seg_mask_path", "unc_png_path", "unc_npy_path")
                    }
                    count_fg, u_fg, u_mean = self._triage_scores(row, artifact_paths)
                    status = triage.evaluate_case(u_fg, u_mean, count_fg, image_id, dataset).status
                    records.append({
                        "dataset": dataset, "image_id": image_id, **artifact_paths,
                        "source": "Precomputed gallery result",
                        "count_fg": count_fg, "u_fg": u_fg, "u_mean": u_mean, "status": status,
                        "dsc": self._number(row, "dsc", 1.0),
                        "nsd": self._number(row, "nsd", 1.0),
                    })
                except (OSError, ValueError) as exc:
                    raise ValueError(f"{self.gallery_path.name}, line {line}: {exc}") from exc
        if not records:
            raise ValueError("Gallery CSV must include at least one case")
        return records

    def _artifact_path(self, value: str, required: bool) -> Optional[str]:
        if not value and not required:
            return None
        path = (self.gallery_path.parent / value).resolve()
        if not value or not path.is_file():
            raise ValueError(f"artifact is missing: {value or '(empty path)'}")
        return str(path)

    @staticmethod
    def _number(row, field, maximum):
        value = row.get(field, "")
        if not value:
            return None
        number = float(value)
        if not math.isfinite(number) or not 0 <= number <= maximum:
            raise ValueError(f"{field} must be in [0, {maximum}]")
        return number

    def _triage_scores(self, row, paths):
        with Image.open(paths["image_path"]) as image, Image.open(paths["seg_mask_path"]) as mask_image:
            if image.format not in {"PNG", "JPEG"} or mask_image.format != "PNG":
                raise ValueError("use a PNG/JPEG original and a binary PNG mask")
            mask_values = np.asarray(mask_image.convert("L"))
        if not np.isin(mask_values, (0, 1, 255)).all():
            raise ValueError("seg_mask_path must contain a binary mask (0/1 or 0/255)")
        mask = mask_values > 0
        count_fg = int(mask.sum())
        u_fg = self._number(row, "u_fg", math.log(2) + 1e-6)
        u_mean = self._number(row, "u_mean", math.log(2) + 1e-6)
        if paths["unc_npy_path"]:
            entropy = np.load(paths["unc_npy_path"], allow_pickle=False)
            if not (
                isinstance(entropy, np.ndarray) and entropy.shape == mask.shape
                and np.issubdtype(entropy.dtype, np.floating) and np.isfinite(entropy).all()
                and entropy.min() >= 0 and entropy.max() <= math.log(2) + 1e-6
            ):
                raise ValueError("unc_npy_path must contain finite floating-point entropy in nats with mask shape")
            if u_mean is None:
                u_mean = float(entropy.mean())
            if count_fg and u_fg is None:
                u_fg = float(entropy[mask].mean())
        if (count_fg and u_fg is None) or (not count_fg and u_mean is None):
            raise ValueError("provide the applicable u_fg/u_mean score or a raw entropy array")
        return count_fg, u_fg if count_fg else None, u_mean

    def build_manifest(self) -> List[Dict[str, Any]]:
        """Scan directories and authoritative CSVs to build the complete 542-scan manifest."""
        records: List[Dict[str, Any]] = []

        for dataset in CANONICAL_DATASETS:
            csv_path = (
                self.outputs_dir
                / dataset
                / "aggregation_results"
                / "seed666"
                / f"uncertainty_scores_{dataset}_test_seed666.csv"
            )
            if not csv_path.exists():
                raise FileNotFoundError(f"Authoritative CSV not found: {csv_path}")

            df = pd.read_csv(csv_path)
            img_dir = self.raw_dir / dataset / "Test_Folder" / "img"
            seg_dir = (
                self.outputs_dir
                / dataset
                / "seg_results"
                / "seed666"
                / "MedCLIPSeg_unimedclip_ViT-B-16_Prompt-original"
            )
            unc_dir = (
                self.outputs_dir
                / dataset
                / "unc_results"
                / "seed666"
                / "MedCLIPSeg_unimedclip_ViT-B-16_Prompt-original"
            )

            for _, row in df.iterrows():
                image_id = str(row["image_id"])
                img_path = img_dir / image_id
                seg_path = seg_dir / image_id
                unc_png_path = unc_dir / image_id
                base_name = Path(image_id).stem
                unc_npy_path = unc_dir / f"{base_name}.npy"

                u_fg = float(row["U_fg"]) if pd.notna(row["U_fg"]) else None
                u_mean = float(row["U_mean"]) if pd.notna(row["U_mean"]) else 0.0
                count_fg = int(row["count_fg"]) if pd.notna(row["count_fg"]) else 0
                dsc = float(row["DSC"]) if pd.notna(row["DSC"]) else 0.0
                nsd = float(row["NSD"]) if pd.notna(row["NSD"]) else 0.0

                # Level 0 Selective Prediction Rule
                if count_fg > 0:
                    status = "Accepted" if (u_fg is not None and u_fg <= FROZEN_THRESHOLD) else "Rejected"
                else:
                    # Empty predicted mask policy (Setting B fallback)
                    status = "Accepted" if (u_mean <= EMPTY_MASK_FALLBACK_THRESHOLD) else "Rejected"

                records.append({
                    "image_id": image_id,
                    "dataset": dataset,
                    "image_path": str(img_path.resolve()) if img_path.exists() else str(img_path),
                    "seg_mask_path": str(seg_path.resolve()) if seg_path.exists() else str(seg_path),
                    "unc_png_path": str(unc_png_path.resolve()) if unc_png_path.exists() else str(unc_png_path),
                    "unc_npy_path": str(unc_npy_path.resolve()) if unc_npy_path.exists() else str(unc_npy_path),
                    "u_fg": u_fg,
                    "u_mean": u_mean,
                    "count_fg": count_fg,
                    "dsc": dsc,
                    "nsd": nsd,
                    "status": status,
                })

        return records

    @staticmethod
    def partition_demo_queries(
        records: List[Dict[str, Any]], m: int = 10
    ) -> Tuple[List[Dict[str, Any]], List[Dict[str, Any]]]:
        """Select M demonstration query images (spanning Accepted and Rejected cases)

        and partition into (demo_queries, historical_gallery).
        Guarantees query images are excluded from historical gallery.
        """
        # Ensure diverse selection across datasets and statuses
        accepted_candidates = [r for r in records if r["status"] == "Accepted"]
        rejected_candidates = [r for r in records if r["status"] == "Rejected"]

        # Select representative cases (6 Accepted, 4 Rejected)
        queries: List[Dict[str, Any]] = []
        n_acc = min(6, m, len(accepted_candidates))
        n_rej = min(m - n_acc, len(rejected_candidates))

        # Sample deterministically across different datasets
        step_acc = max(1, len(accepted_candidates) // max(1, n_acc))
        for i in range(n_acc):
            queries.append(accepted_candidates[i * step_acc])

        step_rej = max(1, len(rejected_candidates) // max(1, n_rej))
        for i in range(n_rej):
            queries.append(rejected_candidates[i * step_rej])

        query_ids = {q["image_id"] + "@" + q["dataset"] for q in queries}
        gallery = [
            r for r in records if (r["image_id"] + "@" + r["dataset"]) not in query_ids
        ]

        return queries, gallery
