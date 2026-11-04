#!/bin/bash
# ==============================================================================
# Run Standard Segmentation Metrics (DSC, NSD) on Test Predictions
# ==============================================================================

set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "$SCRIPT_DIR/../.." && pwd)"

DATASET="${1:-BUSI}"
SEED="${2:-666}"
CONFIG_FILE="${3:-$REPO_ROOT/shared/configs/${DATASET}.yaml}"

export MEDCLIPSEG_DATA="${MEDCLIPSEG_DATA:-$REPO_ROOT/../data/raw}"
export MEDCLIPSEG_OUTPUT="${MEDCLIPSEG_OUTPUT:-$REPO_ROOT/../data/outputs}"

PYTHON_EXEC="python3"
if [ -f "$REPO_ROOT/.venv/bin/python" ]; then
    PYTHON_EXEC="$REPO_ROOT/.venv/bin/python"
fi

echo "Running ground-truth evaluation for $DATASET..."

cd "$REPO_ROOT/shared/medclipseg"
$PYTHON_EXEC utils/eval.py \
    --config-file "$CONFIG_FILE" \
    --output-dir "$MEDCLIPSEG_OUTPUT" \
    --data_percentage 100 \
    --seed "$SEED"

echo "✅ Evaluation CSV saved to: $MEDCLIPSEG_OUTPUT/$DATASET/seg_results/seed$SEED/test_Prompt-original.csv"
