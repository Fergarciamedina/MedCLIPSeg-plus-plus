"""Historical Gallery Manifest Builder.

Constructs and manages the authoritative metadata manifest for the 542 evaluated
ultrasound cases across the five canonical benchmarks (BUSI, UDIAT, BUID, BUSUC, BUSBRA).
"""

import json
import os
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple
import pandas as pd

from .config import (
    CANONICAL_DATASETS,
    DEFAULT_OUTPUTS_DIR,
    DEFAULT_RAW_DATA_DIR,
    EMPTY_MASK_FALLBACK_THRESHOLD,
    FROZEN_THRESHOLD,
    MANIFEST_CACHE_PATH,
)


class ManifestBuilder:
    """Builds and caches the verified historical case manifest."""

    def __init__(
        self,
        raw_dir: Optional[Path] = None,
        outputs_dir: Optional[Path] = None,
        cache_path: Optional[Path] = None,
    ):
        self.raw_dir = Path(raw_dir or DEFAULT_RAW_DATA_DIR)
        self.outputs_dir = Path(outputs_dir or DEFAULT_OUTPUTS_DIR)
        self.cache_path = Path(cache_path or MANIFEST_CACHE_PATH)

    def build_or_load_manifest(self, force_rebuild: bool = False) -> List[Dict[str, Any]]:
        """Load manifest from cache if available, or build and cache it."""
        if not force_rebuild and self.cache_path.exists():
            try:
                with open(self.cache_path, "r", encoding="utf-8") as f:
                    data = json.load(f)
                if len(data) == 542:
                    return data
            except Exception:
                pass

        records = self.build_manifest()
        self.save_manifest(records)
        return records

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

    def save_manifest(self, records: List[Dict[str, Any]]) -> None:
        """Persist manifest records to cache file."""
        self.cache_path.parent.mkdir(parents=True, exist_ok=True)
        with open(self.cache_path, "w", encoding="utf-8") as f:
            json.dump(records, f, indent=2)

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
        n_acc = min(6, len(accepted_candidates))
        n_rej = min(m - n_acc, len(rejected_candidates))

        # Sample deterministically across different datasets
        step_acc = max(1, len(accepted_candidates) // n_acc)
        for i in range(n_acc):
            queries.append(accepted_candidates[i * step_acc])

        step_rej = max(1, len(rejected_candidates) // n_rej)
        for i in range(n_rej):
            queries.append(rejected_candidates[i * step_rej])

        query_ids = {q["image_id"] + "@" + q["dataset"] for q in queries}
        gallery = [
            r for r in records if (r["image_id"] + "@" + r["dataset"]) not in query_ids
        ]

        return queries, gallery

