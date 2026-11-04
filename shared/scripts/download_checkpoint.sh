#!/bin/bash
# ==============================================================================
# Download Pretrained MedCLIPSeg Checkpoints from Hugging Face
# ==============================================================================

set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "$SCRIPT_DIR/../.." && pwd)"

DATASET="${1:-BUSI}"
SEED="${2:-666}"
PERCENTAGE="${3:-100}"

export MEDCLIPSEG_OUTPUT="${MEDCLIPSEG_OUTPUT:-$REPO_ROOT/../data/outputs}"

PYTHON_EXEC="python3"
if [ -f "$REPO_ROOT/.venv/bin/python" ]; then
    PYTHON_EXEC="$REPO_ROOT/.venv/bin/python"
fi

echo "Downloading checkpoint for $DATASET (seed $SEED, $PERCENTAGE%)..."

cd "$REPO_ROOT/shared/medclipseg"
$PYTHON_EXEC utils/download_ckpt.py --dataset "$DATASET" --data_percentage "$PERCENTAGE" --seed "$SEED"

# Relocate from temporary local folder to canonical MEDCLIPSEG_OUTPUT
if [ -d "outputs_medclipseg" ]; then
    find outputs_medclipseg -type f -exec bash -c '
        for f; do
            dest="$1/${f#outputs_medclipseg/}"
            mkdir -p "$(dirname "$dest")"
            mv "$f" "$dest"
        done
    ' _ "$MEDCLIPSEG_OUTPUT" {} +
    rm -rf outputs_medclipseg
fi

echo "✅ Checkpoint saved to: $MEDCLIPSEG_OUTPUT/$DATASET/trained_models/seed$SEED"
