"""Test module imports across the codebase."""


def test_import_medclipseg():
    import src.medclipseg
    assert src.medclipseg.__version__ == "0.1.0"


def test_import_uncertainty():
    from src.uncertainty import (
        compute_pixel_entropy,
        save_uncertainty_artifacts,
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
    assert callable(compute_uncertainty_features)
    assert callable(calibrate_rejection_threshold)
