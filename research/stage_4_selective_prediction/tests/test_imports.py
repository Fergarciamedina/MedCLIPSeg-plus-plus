"""Test module imports across the codebase."""

import os
import sys

REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", ".."))
sys.path.insert(0, REPO_ROOT)
sys.path.insert(0, os.path.join(REPO_ROOT, "research", "stage_4_selective_prediction", "src"))
sys.path.insert(0, os.path.join(REPO_ROOT, "shared"))


def test_import_medclipseg():
    import medclipseg
    assert medclipseg.__version__ == "0.1.0"


def test_import_uncertainty():
    from uncertainty import (
        compute_pixel_entropy,
        save_uncertainty_artifacts,
        compute_u_mean,
        compute_u_positive,
        compute_u_fg,
        compute_boundary_band_mask,
        compute_u_boundary,
        compute_u_threshold,
        compute_u_std_fg,
        compute_u_ratio,
        compute_uncertainty_features,
        extract_image_features,
        compute_spearman_correlations,
        compute_failure_auroc,
        calibrate_rejection_threshold,
        apply_selective_rejection,
        compute_risk_coverage_curve,
        compute_aurc,
        evaluate_rejection_controls,
    )
    assert callable(compute_pixel_entropy)
    assert callable(compute_u_mean)
    assert callable(compute_u_positive)
    assert callable(compute_u_fg)
    assert callable(compute_boundary_band_mask)
    assert callable(compute_u_boundary)
    assert callable(compute_u_threshold)
    assert callable(compute_u_std_fg)
    assert callable(compute_u_ratio)
    assert callable(compute_uncertainty_features)
    assert callable(extract_image_features)
    assert callable(calibrate_rejection_threshold)
