#!/usr/bin/env bash
# ---------------------------------------------------------------------------
# run_visualize_aggregation.sh
# Stage 05 — Aggregation visualization suite
#
# Usage:
#   bash research/stage_4/experiments/prepare_uncertainty_scores/run_visualize_aggregation.sh [DATASET] [SPLIT] [SEED]
#
# Defaults:
#   DATASET = BUSI
#   SPLIT   = test
#   SEED    = 666
#
# EXPLORATORY DIAGNOSTICS ONLY — Not for method selection or threshold tuning.
# ---------------------------------------------------------------------------
set -euo pipefail

DATASET="${1:-BUSI}"
SPLIT="${2:-test}"
SEED="${3:-666}"

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../../../.." && pwd)"
VENV_PYTHON="${REPO_ROOT}/.venv/bin/python3"
SCRIPT="${REPO_ROOT}/research/stage_4/experiments/prepare_uncertainty_scores/visualize_aggregation.py"

# Default output root (override with MEDCLIPSEG_OUTPUT env var)
export MEDCLIPSEG_OUTPUT="${MEDCLIPSEG_OUTPUT:-${REPO_ROOT}/../data/outputs}"
export MPLCONFIGDIR=/tmp/matplotlib_cache

echo "====================================================================="
echo "  Stage 05 — Aggregation Visualization Suite"
echo "  Dataset: ${DATASET} | Split: ${SPLIT} | Seed: ${SEED}"
echo "  Output root: ${MEDCLIPSEG_OUTPUT}"
echo "====================================================================="

"${VENV_PYTHON}" "${SCRIPT}" \
    --dataset "${DATASET}" \
    --split "${SPLIT}" \
    --seed "${SEED}" \
    --n-qc-samples 6

