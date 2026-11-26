#!/usr/bin/env bash
# ==============================================================================
# run_external_inference_and_aggregation.sh
# Executes frozen MedCLIPSeg inference, DSC/NSD evaluation, and uncertainty
# score aggregation across external breast ultrasound datasets (UDIAT, BUID, BUSUC, BUSBRA).
# ==============================================================================
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "$SCRIPT_DIR/../../../.." && pwd)"
cd "$REPO_ROOT"

export MEDCLIPSEG_DATA="${MEDCLIPSEG_DATA:-$REPO_ROOT/../data/raw}"
export MEDCLIPSEG_OUTPUT="${MEDCLIPSEG_OUTPUT:-$REPO_ROOT/../data/outputs}"
export MPLCONFIGDIR="/tmp/matplotlib_cache"
mkdir -p "$MPLCONFIGDIR"
export HF_HUB_OFFLINE="${HF_HUB_OFFLINE:-1}"
export TRANSFORMERS_OFFLINE="${TRANSFORMERS_OFFLINE:-1}"
export HF_DATASETS_OFFLINE="${HF_DATASETS_OFFLINE:-1}"

VENV_PYTHON="$REPO_ROOT/.venv/bin/python3"

DATASETS=("$@")
if [[ ${#DATASETS[@]} -eq 0 ]]; then
    DATASETS=(UDIAT BUID BUSUC BUSBRA)
fi

echo "====================================================================="
echo "  Executing Step 11B.2 — Frozen OOD Pipeline Execution"
echo "  Datasets       : ${DATASETS[*]}"
echo "  Frozen Source  : BUSI (seed 666)"
echo "  Prompt Protocol: original (P_orig)"
echo "  Data Root      : ${MEDCLIPSEG_DATA}"
echo "  Output Root    : ${MEDCLIPSEG_OUTPUT}"
echo "====================================================================="

for DSET in "${DATASETS[@]}"; do
    echo ""
    echo "====================================================================="
    echo "  [1/3] Running Frozen MedCLIPSeg Inference on ${DSET} (Test Split)"
    echo "====================================================================="
    "$VENV_PYTHON" "$REPO_ROOT/shared/medclipseg/test.py" \
        --config-file "$REPO_ROOT/shared/configs/${DSET}.yaml" \
        --output-dir "$MEDCLIPSEG_OUTPUT" \
        --source_dataset BUSI \
        --data_percentage 100 \
        --seed 666 \
        --prompt_design original

    echo ""
    echo "====================================================================="
    echo "  [2/3] Running Official Segmentation Evaluation on ${DSET}"
    echo "====================================================================="
    "$VENV_PYTHON" "$REPO_ROOT/shared/medclipseg/utils/eval.py" \
        --config-file "$REPO_ROOT/shared/configs/${DSET}.yaml" \
        --output-dir "$MEDCLIPSEG_OUTPUT" \
        --seed 666 \
        --prompt_design original

    echo ""
    echo "====================================================================="
    echo "  [3/3] Computing Uncertainty Aggregation Metrics on ${DSET}"
    echo "====================================================================="
    "$VENV_PYTHON" "$REPO_ROOT/research/stage_4_selective_prediction/experiments/prepare_uncertainty_scores/compute_uncertainty_csv.py" \
        --dataset "${DSET}" \
        --split test \
        --seed 666 \
        --prompt-design original \
        --output-dir "$MEDCLIPSEG_OUTPUT"

    echo ""
    echo ">>> Successfully completed processing for ${DSET}."
done

echo ""
echo "====================================================================="
echo "  All requested OOD datasets processed successfully!"
echo "====================================================================="
