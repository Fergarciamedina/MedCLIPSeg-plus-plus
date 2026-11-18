"""Configuration parameters for MedCLIPSeg++ Uncertainty-Aware Medical Image Retrieval."""

import os
import sys
from pathlib import Path

# Thread and Native Runtime Stabilization (Apple Silicon / macOS)
# Restricts OpenMP / BLAS / intra-op parallelism to 1 thread to prevent thread-pool contention crashes
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"
os.environ["OMP_NUM_THREADS"] = "1"
os.environ["MKL_NUM_THREADS"] = "1"
os.environ["OPENBLAS_NUM_THREADS"] = "1"
os.environ["VECLIB_MAXIMUM_THREADS"] = "1"
os.environ["NUMEXPR_NUM_THREADS"] = "1"
os.environ["HF_HUB_OFFLINE"] = "1"
os.environ["TRANSFORMERS_OFFLINE"] = "1"

# Application Directory Structure
APP_ROOT = Path(__file__).resolve().parent.parent
RUNTIME_DIR = APP_ROOT / "runtime"
RUNTIME_DIR.mkdir(parents=True, exist_ok=True)

# Project and Data Paths
REPO_ROOT = APP_ROOT.parent.parent
DEFAULT_RAW_DATA_DIR = REPO_ROOT.parent / "data" / "raw"
DEFAULT_OUTPUTS_DIR = REPO_ROOT.parent / "data" / "outputs"
CHECKPOINT_PATH = REPO_ROOT / "checkpoints" / "unimed_clip_vit_b16.pt"

# Evaluated Ultrasound Cohorts
CANONICAL_DATASETS = ["BUSI", "UDIAT", "BUID", "BUSUC", "BUSBRA"]

# Level 0 Selective Prediction Frozen Parameters
# Calibrated on BUSI Validation Setting A lesion cohort (N=65)
FROZEN_THRESHOLD = 0.064058324
EMPTY_MASK_FALLBACK_THRESHOLD = 1e-4
PRIMARY_SCORE_COL = "U_fg"
FALLBACK_SCORE_COL = "U_mean"

# PABP & Retrieval Default Parameters
DEFAULT_K = 5
DEFAULT_NLIST = 8
DEFAULT_NPROBE = 4
DEFAULT_BLOCK_SIZE = 16
EMBEDDING_DIM = 512

# Cache File Locations
MANIFEST_CACHE_PATH = RUNTIME_DIR / "gallery_manifest.json"
EMBEDDINGS_CACHE_PATH = RUNTIME_DIR / "gallery_embeddings.npy"
INDEX_METADATA_PATH = RUNTIME_DIR / "pabp_index_meta.json"
