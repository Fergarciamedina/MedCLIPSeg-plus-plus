#!/bin/bash
set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "$SCRIPT_DIR/../.." && pwd)"

DATASET="${1:-BUSI}"
SEED="${2:-666}"
FEATURE="${3:-U_fg}"

PYTHON_EXEC="python3"
if [ -f "$REPO_ROOT/.venv/bin/python" ]; then
    PYTHON_EXEC="$REPO_ROOT/.venv/bin/python"
fi

export MPLCONFIGDIR="${MPLCONFIGDIR:-/tmp/matplotlib_cache}"
mkdir -p "$MPLCONFIGDIR"

PYTHONPATH="$REPO_ROOT/fernando/src:$REPO_ROOT/shared:$REPO_ROOT" $PYTHON_EXEC \
    "$REPO_ROOT/fernando/research/selective_prediction/run_selective_prediction.py" \
    --dataset "$DATASET" \
    --seed "$SEED" \
    --feature "$FEATURE"
