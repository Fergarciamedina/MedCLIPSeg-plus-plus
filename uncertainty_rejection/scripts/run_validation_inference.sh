#!/usr/bin/env bash
# ---------------------------------------------------------------------------
# run_validation_inference.sh
# Stage 06 — BUSI Validation Inference & Uncertainty Extraction Runner
#
# Usage:
#   bash uncertainty_rejection/scripts/run_validation_inference.sh [SEED] [DEVICE]
#
# Defaults:
#   SEED   = 666
#   DEVICE = auto (mps if available, else cpu)
# ---------------------------------------------------------------------------
set -euo pipefail

SEED="${1:-666}"
DEVICE="${2:-mps}"

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
VENV_PYTHON="${REPO_ROOT}/.venv/bin/python3"
SCRIPT="${REPO_ROOT}/uncertainty_rejection/research/inference/run_validation_inference.py"

export MEDCLIPSEG_DATA="${MEDCLIPSEG_DATA:-${REPO_ROOT}/../data/raw}"
export MEDCLIPSEG_OUTPUT="${MEDCLIPSEG_OUTPUT:-${REPO_ROOT}/../data/outputs}"
export MPLCONFIGDIR=/tmp/matplotlib_cache
export HF_HUB_OFFLINE=1
export TRANSFORMERS_OFFLINE=1
export HF_DATASETS_OFFLINE=1

echo "====================================================================="
echo "  Executing Stage 06 — BUSI Validation Inference"
echo "  Seed: ${SEED} | Device: ${DEVICE}"
echo "  Data: ${MEDCLIPSEG_DATA}"
echo "  Out:  ${MEDCLIPSEG_OUTPUT}"
echo "====================================================================="

"${VENV_PYTHON}" "${SCRIPT}" \
    --dataset "BUSI" \
    --seed "${SEED}" \
    --prompt-design "original" \
    --batch-size 8 \
    --num-samples 30 \
    --device "${DEVICE}"

