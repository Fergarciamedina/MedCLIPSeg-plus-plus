#!/usr/bin/env bash
# Run the frozen validation and test protocol using the existing phase runners.
set -euo pipefail

study_dir="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
repo_dir="$(cd "$study_dir/../.." && pwd)"
experiments="$study_dir/experiments"
python="$repo_dir/.venv/bin/python"
seed=666

usage() {
  cat <<'HELP'
Usage: bash research/stage_4_selective_prediction/reproduce_study.sh [--from-scores]

Set MEDCLIPSEG_OUTPUT to a writable output folder. Full reproduction also
requires MEDCLIPSEG_DATA and the released BUSI checkpoint in the output tree.
Set MEDCLIPSEG_DEVICE to cpu (default), mps, or cuda for inference.
Use --from-scores to rerun selection and testing from saved per-image scores.
HELP
}

mode="${1:-full}"
if [[ "$mode" == "--help" || "$mode" == "-h" ]]; then
  usage
  exit 0
fi
if [[ $# -gt 1 || ( $# -gt 0 && "$mode" != "--from-scores" ) ]]; then
  usage >&2
  exit 2
fi

: "${MEDCLIPSEG_OUTPUT:?Set MEDCLIPSEG_OUTPUT to the local experiment output folder.}"
mkdir -p "$MEDCLIPSEG_OUTPUT"
export MEDCLIPSEG_OUTPUT="$(cd "$MEDCLIPSEG_OUTPUT" && pwd)"
export MEDCLIPSEG_DEVICE="${MEDCLIPSEG_DEVICE:-cpu}"
if [[ ! -x "$python" ]]; then
  echo "Install requirements.txt into $repo_dir/.venv first." >&2
  exit 1
fi

require_file() {
  if [[ ! -f "$1" ]]; then
    echo "Missing input: $1" >&2
    exit 1
  fi
}

if [[ "$mode" == "full" ]]; then
  : "${MEDCLIPSEG_DATA:?Set MEDCLIPSEG_DATA to the folder containing the five prepared cohorts.}"
  export MEDCLIPSEG_DATA="$(cd "$MEDCLIPSEG_DATA" && pwd)"
  require_file "$MEDCLIPSEG_OUTPUT/BUSI/trained_models/seed$seed/MedCLIPSeg_unimedclip_ViT-B-16_latest.pth"
  case "$MEDCLIPSEG_DEVICE" in
    cpu|mps|cuda) ;;
    *) echo "Choose cpu, mps, or cuda for MEDCLIPSEG_DEVICE." >&2; exit 2 ;;
  esac
else
  require_file "$MEDCLIPSEG_OUTPUT/BUSI/validation/seed$seed/aggregation_results/uncertainty_scores_BUSI_val_seed$seed.csv"
  for cohort in BUSI UDIAT BUID BUSUC BUSBRA; do
    require_file "$MEDCLIPSEG_OUTPUT/$cohort/aggregation_results/seed$seed/uncertainty_scores_${cohort}_test_seed$seed.csv"
  done
fi

cd "$repo_dir"
if [[ "$mode" == "full" ]]; then
  bash "$experiments/validation_selection/run_validation_inference.sh" "$seed" "$MEDCLIPSEG_DEVICE"
  bash "$experiments/prepare_uncertainty_scores/run_compute_uncertainty_csv.sh" BUSI val "$seed"
  bash "$experiments/validation_selection/run_validation_statistics.sh" "$seed"
else
  bash "$experiments/validation_selection/run_validation_statistics.sh" "$seed" --scores-only
fi

bash "$experiments/validation_selection/run_validation_selection.sh" "$seed"

if [[ "$mode" == "full" ]]; then
  bash shared/scripts/run_inference.sh BUSI "$seed"
  bash shared/scripts/run_evaluation.sh BUSI "$seed"
  bash "$experiments/prepare_uncertainty_scores/run_compute_uncertainty_csv.sh" BUSI test "$seed"
fi
bash "$experiments/locked_busi_test/run_locked_test_evaluation.sh" BUSI "$seed"

if [[ "$mode" == "full" ]]; then
  bash "$experiments/external_test/run_external_inference_and_aggregation.sh"
fi
"$python" "$experiments/external_test/run_external_test_evaluation.py" --seed "$seed"
