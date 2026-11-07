#!/usr/bin/env bash
# ==============================================================================
# ELEC 6661 — Stage 10: Locked BUSI Test Evaluation Script
# ==============================================================================
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/../.." && pwd)"

VENV_PYTHON="${REPO_ROOT}/.venv/bin/python3"
if [ ! -f "${VENV_PYTHON}" ]; then
    VENV_PYTHON="python3"
fi

DATASET="${1:-BUSI}"
SEED="${2:-666}"
N_BOOTSTRAP="${3:-1000}"

echo "=========================================================="
echo " Running ELEC 6661 Locked Test Evaluation (Stage 10)"
echo " Dataset     : ${DATASET}"
echo " Seed        : ${SEED}"
echo " Resamples   : ${N_BOOTSTRAP}"
echo " Python      : ${VENV_PYTHON}"
echo "=========================================================="

export PYTHONPATH="${REPO_ROOT}:${REPO_ROOT}/uncertainty_rejection/src:${REPO_ROOT}/shared:${PYTHONPATH:-}"

"${VENV_PYTHON}" "${REPO_ROOT}/uncertainty_rejection/research/selective_prediction/run_locked_test_evaluation.py" \
    --dataset "${DATASET}" \
    --seed "${SEED}" \
    --n-bootstrap "${N_BOOTSTRAP}"

echo "=========================================================="
echo " Locked Test Evaluation Completed Successfully!"
echo "=========================================================="

