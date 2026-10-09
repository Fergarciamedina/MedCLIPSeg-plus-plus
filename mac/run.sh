#!/bin/bash
if [[ "$VIRTUAL_ENV" == "" ]]; then
    echo "Warning: No virtual environment activated. Please activate it first."
    echo "Run: source .venv/bin/activate"
    exit 1
fi

export MEDCLIPSEG_DATA="$(cd "$(dirname "$0")/../.." && pwd)/data/raw"
export MEDCLIPSEG_OUTPUT="$(cd "$(dirname "$0")/../.." && pwd)/data/outputs"

echo "MEDCLIPSEG_DATA resolves to: $MEDCLIPSEG_DATA"
echo "MEDCLIPSEG_OUTPUT resolves to: $MEDCLIPSEG_OUTPUT"

cd "$(dirname "$0")/src" || exit 1

echo "===== Step A: Download checkpoint ====="
python utils/download_ckpt.py --dataset BUSI --data_percentage 100 --seed 666
[ -d "outputs_medclipseg" ] && find outputs_medclipseg -type f -exec bash -c 'for f; do dest="$1/${f#outputs_medclipseg/}"; mkdir -p "$(dirname "$dest")"; mv "$f" "$dest"; done' _ "$MEDCLIPSEG_OUTPUT" {} + && rm -rf outputs_medclipseg || true

echo "===== Step B: Run inference ====="
python test.py --config-file ../configs/BUSI_mac.yaml \
    --output-dir "$MEDCLIPSEG_OUTPUT" \
    --source_dataset BUSI \
    --data_percentage 100 \
    --seed 666

echo "===== Step C: Run evaluation ====="
python utils/eval.py --config-file ../configs/BUSI_mac.yaml \
    --output-dir "$MEDCLIPSEG_OUTPUT" \
    --data_percentage 100 \
    --seed 666

echo "===== Done. Check results in $MEDCLIPSEG_OUTPUT ====="