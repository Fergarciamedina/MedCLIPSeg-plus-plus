"""Unit tests for uncertainty quantification logic."""

import numpy as np
import pandas as pd
from src.uncertainty.raw_entropy import compute_pixel_entropy
from src.uncertainty.aggregation import compute_uncertainty_features
from src.uncertainty.metrics import compute_spearman_correlations, compute_failure_auroc
from src.uncertainty.selective_prediction import (
    calibrate_rejection_threshold,
    apply_selective_rejection,
    compute_risk_coverage_curve,
    compute_aurc,
    evaluate_rejection_controls,
)


def test_pixel_entropy_bounds():
    # 5 samples of shape (32, 32) with varying probabilities
    samples = np.random.uniform(0.1, 0.9, size=(5, 32, 32))
    mean_prob, entropy = compute_pixel_entropy(samples)
    assert mean_prob.shape == (32, 32)
    assert entropy.shape == (32, 32)
    assert np.all(entropy >= 0.0)
    assert np.all(entropy <= np.log(2.0) + 1e-4)


def test_aggregation_features():
    unc = np.ones((64, 64), dtype=np.float32) * 0.2
    mask = np.zeros((64, 64), dtype=np.uint8)
    mask[20:40, 20:40] = 1
    unc[20:40, 20:40] = 0.8  # elevated uncertainty inside lesion

    feats = compute_uncertainty_features(unc, mask)
    assert np.isclose(feats["U_fg"], 0.8, atol=1e-4)
    assert feats["U_ratio"] > 1.0
    assert feats["U_boundary"] > 0.0


def test_empty_mask_handling():
    unc = np.ones((32, 32), dtype=np.float32) * 0.1
    mask = np.zeros((32, 32), dtype=np.uint8)  # empty mask

    feats = compute_uncertainty_features(unc, mask)
    assert feats["U_fg"] == 0.0
    assert feats["U_std_fg"] == 0.0
    assert feats["U_boundary"] == 0.0
    assert np.isclose(feats["U_mean"], 0.1, atol=1e-4)


def test_selective_prediction_and_controls():
    # Synthetic dataset with negative correlation between uncertainty and DSC
    rng = np.random.default_rng(42)
    n = 100
    dsc = rng.uniform(0.2, 1.0, size=n)
    # Higher DSC -> lower uncertainty
    u_fg = (1.0 - dsc) + rng.normal(0, 0.05, size=n)
    u_fg = np.clip(u_fg, 0.01, 1.0)

    df = pd.DataFrame({"DSC": dsc, "U_fg": u_fg, "NSD": dsc * 0.95})

    # Validation calibration
    tau = calibrate_rejection_threshold(df, "U_fg", target_coverage=0.80)
    assert tau > 0

    rejection_res = apply_selective_rejection(df, "U_fg", tau)
    assert rejection_res["realized_coverage"] > 0
    assert rejection_res["accepted_mean_dsc"] >= rejection_res["rejected_mean_dsc"]

    # Risk-coverage
    covs, risks = compute_risk_coverage_curve(df, "U_fg", metric_col="DSC")
    aurc = compute_aurc(covs, risks)
    assert 0.0 <= aurc <= 1.0

    # Controls
    controls_df = evaluate_rejection_controls(df, "U_fg", target_coverages=[1.0, 0.80, 0.60])
    assert len(controls_df) == 3
    # Check that Oracle >= Random at 80% coverage
    row_80 = controls_df[controls_df["Target_Coverage"] == 0.80].iloc[0]
    assert row_80["Oracle_DSC"] >= row_80["Random_DSC"]
