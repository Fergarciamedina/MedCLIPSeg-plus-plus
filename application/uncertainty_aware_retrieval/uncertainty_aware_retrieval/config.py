"""Configuration parameters for MedCLIPSeg++ Uncertainty-Aware Medical Image Retrieval."""

import os
from pathlib import Path

# Thread and Native Runtime Stabilization (Apple Silicon / macOS)
# Restricts OpenMP / BLAS / intra-op parallelism to 1 thread to prevent thread-pool contention crashes
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"
os.environ["OMP_NUM_THREADS"] = "1"
os.environ["MKL_NUM_THREADS"] = "1"
os.environ["OPENBLAS_NUM_THREADS"] = "1"
os.environ["VECLIB_MAXIMUM_THREADS"] = "1"
os.environ["NUMEXPR_NUM_THREADS"] = "1"
os.environ.setdefault("HF_HUB_OFFLINE", "1")
os.environ.setdefault("TRANSFORMERS_OFFLINE", "1")

# Application Directory Structure
APP_ROOT = Path(__file__).resolve().parent.parent
RUNTIME_DIR = Path(os.getenv("MEDCLIPSEG_RUNTIME", str(APP_ROOT / "runtime"))).expanduser()
REPO_ROOT = APP_ROOT.parent.parent
PROJECT_DATA_DIR = REPO_ROOT.parent / "data"
DATA_PATH = Path(os.getenv("MEDCLIPSEG_DATA", str(PROJECT_DATA_DIR))).expanduser()
# The dashboard historically took the data parent. Research scripts use the raw directory.
IS_DATA_PARENT = (DATA_PATH / "raw").is_dir() or DATA_PATH == PROJECT_DATA_DIR
DEFAULT_RAW_DATA_DIR = DATA_PATH / "raw" if IS_DATA_PARENT else DATA_PATH
DEFAULT_OUTPUTS_DIR = Path(os.getenv(
    "MEDCLIPSEG_OUTPUT",
    str((DATA_PATH if IS_DATA_PARENT else DATA_PATH.parent) / "outputs"),
)).expanduser()
CHECKPOINT_PATH = Path(os.getenv(
    "MEDCLIPSEG_CHECKPOINT", str(REPO_ROOT / "checkpoints" / "unimed_clip_vit_b16.pt"),
)).expanduser()

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
EMBEDDINGS_CACHE_PATH = RUNTIME_DIR / "gallery_embeddings.npz"
INDEX_METADATA_PATH = RUNTIME_DIR / "pabp_index_meta.json"
