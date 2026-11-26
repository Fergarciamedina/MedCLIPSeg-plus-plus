#!/usr/bin/env bash
# ---------------------------------------------------------------------------
# run_compute_uncertainty_csv.sh
# Stage 04 — Per-image uncertainty CSV pipeline
#
# Usage:
#   bash research/stage_4/experiments/prepare_uncertainty_scores/run_compute_uncertainty_csv.sh [DATASET] [SPLIT] [SEED]
#
# Defaults:
#   DATASET = BUSI
#   SPLIT   = test
#   SEED    = 666
#
# This launcher prepares per-image input scores. Validation selection runs separately.
# ---------------------------------------------------------------------------
set -euo pipefail

DATASET="${1:-BUSI}"
SPLIT="${2:-test}"
SEED="${3:-666}"

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../../../.." && pwd)"
VENV_PYTHON="${REPO_ROOT}/.venv/bin/python3"
SCRIPT="${REPO_ROOT}/research/stage_4/experiments/prepare_uncertainty_scores/compute_uncertainty_csv.py"

# Default output root (override with MEDCLIPSEG_OUTPUT env var)
export MEDCLIPSEG_OUTPUT="${MEDCLIPSEG_OUTPUT:-${REPO_ROOT}/../data/outputs}"
export MPLCONFIGDIR=/tmp/matplotlib_cache

echo "====================================================================="
echo "  Stage 04 — Per-Image Uncertainty CSV"
echo "  Dataset: ${DATASET} | Split: ${SPLIT} | Seed: ${SEED}"
echo "  Output root: ${MEDCLIPSEG_OUTPUT}"
echo "====================================================================="

"${VENV_PYTHON}" "${SCRIPT}" \
    --dataset "${DATASET}" \
    --split "${SPLIT}" \
    --seed "${SEED}" \
    --boundary-width 2 \
    --threshold-t 0.1 \
    --eps 1e-6

