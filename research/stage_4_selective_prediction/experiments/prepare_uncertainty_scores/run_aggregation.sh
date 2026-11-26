#!/bin/bash
set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "$SCRIPT_DIR/../../../.." && pwd)"

DATASET="${1:-BUSI}"
SEED="${2:-666}"

PYTHON_EXEC="python3"
if [ -f "$REPO_ROOT/.venv/bin/python" ]; then
    PYTHON_EXEC="$REPO_ROOT/.venv/bin/python"
fi

export MPLCONFIGDIR="${MPLCONFIGDIR:-/tmp/matplotlib_cache}"
mkdir -p "$MPLCONFIGDIR"

PYTHONPATH="$REPO_ROOT/research/stage_4_selective_prediction/src:$REPO_ROOT/shared:$REPO_ROOT" $PYTHON_EXEC \
    "$REPO_ROOT/research/stage_4_selective_prediction/experiments/prepare_uncertainty_scores/run_aggregation.py" \
    --dataset "$DATASET" \
    --seed "$SEED"
