#!/bin/bash
set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "$SCRIPT_DIR/../../.." && pwd)"

PYTHON_EXEC="python3"
if [ -f "$REPO_ROOT/.venv/bin/python3" ]; then
    PYTHON_EXEC="$REPO_ROOT/.venv/bin/python3"
elif [ -f "$REPO_ROOT/.venv/bin/python" ]; then
    PYTHON_EXEC="$REPO_ROOT/.venv/bin/python"
fi

export MPLCONFIGDIR="${MPLCONFIGDIR:-/tmp/matplotlib_cache}"
mkdir -p "$MPLCONFIGDIR"

echo "=========================================================="
echo " Running Uncertainty Rejection Test Suite"
echo " Python Executable: $PYTHON_EXEC"
echo "=========================================================="

PYTHONPATH="$REPO_ROOT/research/stage_4/src:$REPO_ROOT/shared:$REPO_ROOT" $PYTHON_EXEC -c "
import sys
import research.stage_4.tests.test_imports as t1
import research.stage_4.tests.test_uncertainty as t2
import research.stage_4.tests.test_config_loading as t3

test_cases = [
    ('test_import_medclipseg', t1.test_import_medclipseg),
    ('test_import_uncertainty', t1.test_import_uncertainty),
    ('test_pixel_entropy_bounds', t2.test_pixel_entropy_bounds),
    ('test_empty_foreground', t2.test_empty_foreground),
    ('test_all_zero_uncertainty', t2.test_all_zero_uncertainty),
    ('test_uniform_uncertainty', t2.test_uniform_uncertainty),
    ('test_border_lesion', t2.test_border_lesion),
    ('test_nonrectangular_foreground', t2.test_nonrectangular_foreground),
    ('test_positive_vs_mean_difference', t2.test_positive_vs_mean_difference),
    ('test_boundary_band_geometry', t2.test_boundary_band_geometry),
    ('test_configurable_parameters', t2.test_configurable_parameters),
    ('test_error_handling', t2.test_error_handling),
    ('test_single_pixel_foreground', t2.test_single_pixel_foreground),
    ('test_selective_prediction_and_controls', t2.test_selective_prediction_and_controls),
    ('test_statistical_ties_handling', t2.test_statistical_ties_handling),
    ('test_statistical_nan_handling', t2.test_statistical_nan_handling),
    ('test_statistical_ranking_direction', t2.test_statistical_ranking_direction),
    ('test_statistical_degenerate_cases', t2.test_statistical_degenerate_cases),
    ('test_spearman_bootstrap_ci', t2.test_spearman_bootstrap_ci),
    ('test_paired_bootstrap_aurc_ci', t2.test_paired_bootstrap_aurc_ci),
    ('test_empty_mask_policies', t2.test_empty_mask_policies),
    ('test_threshold_calibration_reproducibility', t2.test_threshold_calibration_reproducibility),
    ('test_frozen_config_loading', t2.test_frozen_config_loading),
    ('test_time_no_leakage', t2.test_time_no_leakage),
    ('test_deterministic_rejection', t2.test_deterministic_rejection),
    ('test_result_reproducibility', t2.test_result_reproducibility),
    ('test_ood_cohort_integrity', t2.test_ood_cohort_integrity),
    ('test_ood_selective_gain_positive', t2.test_ood_selective_gain_positive),
    ('test_ood_ufg_beats_umean_aurc', t2.test_ood_ufg_beats_umean_aurc),
    ('test_ood_umean_confounding_persists', t2.test_ood_umean_confounding_persists),
    ('test_yaml_syntax', t3.test_yaml_syntax),
]

passed = 0
failed = 0

for name, test_fn in test_cases:
    try:
        test_fn()
        print(f'  [PASS] {name}')
        passed += 1
    except Exception as e:
        print(f'  [FAIL] {name}: {e}')
        failed += 1

print('----------------------------------------------------------')
print(f'Tests completed: {passed} passed, {failed} failed out of {len(test_cases)} total.')
if failed > 0:
    sys.exit(1)
print('✅ All unit and integration tests passed successfully!')
"
