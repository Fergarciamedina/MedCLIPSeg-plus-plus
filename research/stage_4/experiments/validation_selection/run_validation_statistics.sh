#!/usr/bin/env bash
# ---------------------------------------------------------------------------
# run_validation_statistics.sh
# Stage 08 — BUSI Validation Statistical Analysis Runner
#
# Usage:
#   bash research/stage_4/experiments/validation_selection/run_validation_statistics.sh [SEED]
#
# Defaults:
#   SEED = 666
# ---------------------------------------------------------------------------
set -euo pipefail

SEED="${1:-666}"

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../../../.." && pwd)"
VENV_PYTHON="${REPO_ROOT}/.venv/bin/python3"
SCRIPT="${REPO_ROOT}/research/stage_4/experiments/validation_selection/run_validation_statistics.py"

export MEDCLIPSEG_DATA="${MEDCLIPSEG_DATA:-${REPO_ROOT}/../data/raw}"
export MEDCLIPSEG_OUTPUT="${MEDCLIPSEG_OUTPUT:-${REPO_ROOT}/../data/outputs}"
export MPLCONFIGDIR=/tmp/matplotlib_cache
mkdir -p "$MPLCONFIGDIR"

echo "====================================================================="
echo "  Executing Stage 08 — BUSI Validation Statistical Analysis"
echo "  Seed: ${SEED}"
echo "  Data: ${MEDCLIPSEG_DATA}"
echo "  Out:  ${MEDCLIPSEG_OUTPUT}"
echo "====================================================================="

"${VENV_PYTHON}" "${SCRIPT}" \
    --dataset "BUSI" \
    --seed "${SEED}" \
    --prompt-design "original" \
    --n-bootstrap 1000

