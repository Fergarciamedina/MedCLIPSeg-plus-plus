#!/bin/bash
set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "$SCRIPT_DIR/../.." && pwd)"

PYTHON_EXEC="python3"
if [ -f "$REPO_ROOT/.venv/bin/python" ]; then
    PYTHON_EXEC="$REPO_ROOT/.venv/bin/python"
fi

export MPLCONFIGDIR="${MPLCONFIGDIR:-/tmp/matplotlib_cache}"
mkdir -p "$MPLCONFIGDIR"

echo "Running uncertainty & selective rejection test suite..."
PYTHONPATH="$REPO_ROOT/uncertainty_rejection/src:$REPO_ROOT/shared:$REPO_ROOT" $PYTHON_EXEC -c "
import uncertainty_rejection.tests.test_imports as t1
import uncertainty_rejection.tests.test_uncertainty as t2
import uncertainty_rejection.tests.test_config_loading as t3

t1.test_import_medclipseg()
t1.test_import_uncertainty()
print('✅ Module imports test passed')

t2.test_pixel_entropy_bounds()
t2.test_aggregation_features()
t2.test_empty_mask_handling()
t2.test_selective_prediction_and_controls()
print('✅ Uncertainty quantification & rejection test passed')

t3.test_yaml_syntax()
print('✅ Shared configuration syntax test passed')
"
echo "All tests passed successfully!"
