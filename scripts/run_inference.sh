#!/bin/bash
# ==============================================================================
# MedCLIPSeg Unified Inference Runner
# Supports Apple Silicon (MPS), NVIDIA GPU (CUDA), and CPU execution.
# ==============================================================================

set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"

DATASET="${1:-BUSI}"
SEED="${2:-666}"
CONFIG_FILE="${3:-$REPO_ROOT/configs/${DATASET}.yaml}"

# Set default data paths relative to canonical project root if not already defined
export MEDCLIPSEG_DATA="${MEDCLIPSEG_DATA:-$REPO_ROOT/../data/raw}"
export MEDCLIPSEG_OUTPUT="${MEDCLIPSEG_OUTPUT:-$REPO_ROOT/../data/outputs}"
export MPLCONFIGDIR="${MPLCONFIGDIR:-/tmp/matplotlib_cache}"
mkdir -p "$MPLCONFIGDIR"

echo "=========================================================="
echo "  MedCLIPSeg Inference Execution"
echo "  Dataset:       $DATASET"
echo "  Seed:          $SEED"
echo "  Config:        $CONFIG_FILE"
echo "  Data Raw:      $MEDCLIPSEG_DATA"
echo "  Output Dir:    $MEDCLIPSEG_OUTPUT"
echo "=========================================================="

if [ ! -f "$CONFIG_FILE" ]; then
    echo "Error: Config file not found: $CONFIG_FILE"
    exit 1
fi

PYTHON_EXEC="python3"
if [ -f "$REPO_ROOT/.venv/bin/python" ]; then
    PYTHON_EXEC="$REPO_ROOT/.venv/bin/python"
fi

# Run inference
PYTHONPATH="$REPO_ROOT/src:$REPO_ROOT" $PYTHON_EXEC "$REPO_ROOT/src/medclipseg/test.py" \
    --config-file "$CONFIG_FILE" \
    --output-dir "$MEDCLIPSEG_OUTPUT" \
    --source_dataset "$DATASET" \
    --data_percentage 100 \
    --seed "$SEED"

echo "✅ Inference complete. Outputs saved in: $MEDCLIPSEG_OUTPUT/$DATASET"
