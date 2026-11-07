"""
Unit tests for uncertainty quantification and aggregation logic.

Comprehensive synthetic unit tests with manually calculated and verifiable ground truth:
- Empty foreground mask.
- All-zero uncertainty map.
- Uniform non-zero uncertainty map.
- Small lesion touching an image border.
- Non-rectangular foreground (L-shaped).
- Cases where U_positive is strictly different from U_mean.
- Exact boundary band morphological pixel selection.
- Configurable parameters (threshold_t, boundary_width, eps).
- Error handling (shape mismatch, non-finite values, dimension checks).
- Downstream selective prediction and control comparisons.
"""

import os
import sys

REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
sys.path.insert(0, os.path.join(REPO_ROOT, "uncertainty_rejection", "src"))

import cv2
import json
import numpy as np
import pandas as pd


class AssertRaises:
    """Context manager to test exception raising without external test dependencies."""
    def __init__(self, exc_type, match=None):
        self.exc_type = exc_type
        self.match = match

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        if exc_type is None:
            raise AssertionError(f"Expected {self.exc_type.__name__} was not raised")
        if not issubclass(exc_type, self.exc_type):
            return False
        if self.match and self.match not in str(exc_val):
            raise AssertionError(f"Exception message '{exc_val}' did not contain expected match '{self.match}'")
        return True

from uncertainty.raw_entropy import compute_pixel_entropy
from uncertainty.aggregation import (
    compute_u_mean,
    compute_u_positive,
    compute_u_fg,
    compute_boundary_band_mask,
    compute_u_boundary,
    compute_u_threshold,
    compute_u_std_fg,
    compute_u_ratio,
    compute_uncertainty_features,
)
from uncertainty.metrics import (
    compute_spearman_correlations,
    compute_failure_auroc,
    compute_spearman_bootstrap_ci,
)
from uncertainty.selective_prediction import (
    calibrate_rejection_threshold,
    apply_selective_rejection,
    compute_risk_coverage_curve,
    compute_aurc,
    evaluate_rejection_controls,
    compute_paired_bootstrap_aurc_ci,
    apply_empty_mask_policy,
)


def test_pixel_entropy_bounds():
    """Verify Monte Carlo Shannon entropy satisfies theoretical bounds [0, ln 2]."""
    samples = np.random.uniform(0.1, 0.9, size=(5, 32, 32))
    mean_prob, entropy = compute_pixel_entropy(samples)
    assert mean_prob.shape == (32, 32)
    assert entropy.shape == (32, 32)
    assert np.all(entropy >= 0.0)
    assert np.all(entropy <= np.log(2.0) + 1e-5)


def test_empty_foreground():
    """
    Test empty foreground (e.g. normal ultrasound case where no lesion is predicted).
    Foreground-dependent metrics must evaluate to NaN with valid=False.
    """
    unc = np.array([
        [0.1, 0.2, 0.3, 0.4],
        [0.1, 0.2, 0.3, 0.4],
        [0.1, 0.2, 0.3, 0.4],
        [0.1, 0.2, 0.3, 0.4],
    ], dtype=np.float32)
    mask = np.zeros((4, 4), dtype=np.uint8)

    feats = compute_uncertainty_features(unc, mask, boundary_width=1, threshold_t=0.2, eps=1e-6)

    # U_mean: (0.1 + 0.2 + 0.3 + 0.4) / 4 = 0.25
    assert np.isclose(feats["U_mean"], 0.25)
    assert feats["valid_mean"] is True
    assert feats["count_total"] == 16

    # U_positive: all 16 pixels are > 0, so equals U_mean
    assert np.isclose(feats["U_positive"], 0.25)
    assert feats["valid_pos"] is True
    assert feats["count_pos"] == 16

    # U_fg: empty foreground -> NaN, valid=False, count=0
    assert np.isnan(feats["U_fg"])
    assert feats["valid_fg"] is False
    assert feats["count_fg"] == 0

    # U_boundary: empty boundary -> NaN, valid=False, count=0
    assert np.isnan(feats["U_boundary"])
    assert feats["valid_boundary"] is False
    assert feats["count_boundary"] == 0

    # U_threshold at 0.2: 8 pixels with 0.3 and 0.4 -> mean = 0.35
    assert np.isclose(feats["U_threshold"], 0.35)
    assert feats["valid_threshold"] is True
    assert feats["count_threshold"] == 8

    # U_std_fg and U_ratio: undefined when foreground is empty -> NaN, valid=False
    assert np.isnan(feats["U_std_fg"])
    assert feats["valid_std_fg"] is False
    assert np.isnan(feats["U_ratio"])
    assert feats["valid_ratio"] is False


def test_all_zero_uncertainty():
    """
    Test completely confident model (uncertainty identically 0 everywhere).
    U_positive and U_threshold must evaluate to NaN with valid=False.
    """
    unc = np.zeros((4, 4), dtype=np.float32)
    mask = np.zeros((4, 4), dtype=np.uint8)
    mask[1:3, 1:3] = 1  # 4 foreground pixels

    feats = compute_uncertainty_features(unc, mask, boundary_width=1, threshold_t=0.1)

    assert feats["U_mean"] == 0.0
    assert feats["valid_mean"] is True

    # No positive pixels
    assert np.isnan(feats["U_positive"])
    assert feats["valid_pos"] is False
    assert feats["count_pos"] == 0

    # Foreground pixels are all 0
    assert feats["U_fg"] == 0.0
    assert feats["valid_fg"] is True
    assert feats["count_fg"] == 4

    # Boundary band pixels are all 0
    assert feats["U_boundary"] == 0.0
    assert feats["valid_boundary"] is True
    assert feats["count_boundary"] > 0

    # No pixels exceed threshold t=0.1
    assert np.isnan(feats["U_threshold"])
    assert feats["valid_threshold"] is False
    assert feats["count_threshold"] == 0

    # Standard deviation of 4 identical zero pixels is 0.0
    assert feats["U_std_fg"] == 0.0
    assert feats["valid_std_fg"] is True

    # Ratio: 0.0 / (0.0 + eps) = 0.0
    assert np.isclose(feats["U_ratio"], 0.0)
    assert feats["valid_ratio"] is True


def test_uniform_uncertainty():
    """
    Test uniform non-zero uncertainty (constant c = 0.5 everywhere).
    All spatial aggregations equal 0.5, U_std_fg is 0.0, and U_ratio is ~1.0.
    """
    c = 0.5
    unc = np.full((6, 6), c, dtype=np.float32)
    mask = np.zeros((6, 6), dtype=np.uint8)
    mask[2:5, 2:5] = 1  # 3x3 foreground block = 9 pixels

    feats = compute_uncertainty_features(unc, mask, boundary_width=1, threshold_t=0.4, eps=1e-6)

    assert np.isclose(feats["U_mean"], c)
    assert np.isclose(feats["U_positive"], c)
    assert np.isclose(feats["U_fg"], c)
    assert np.isclose(feats["U_boundary"], c)
    assert np.isclose(feats["U_threshold"], c)
    assert feats["count_threshold"] == 36

    # Population standard deviation of constant array is exactly 0.0
    assert feats["U_std_fg"] == 0.0
    assert feats["valid_std_fg"] is True

    # U_ratio: 0.5 / (0.5 + 1e-6) ~= 1.0
    assert np.isclose(feats["U_ratio"], c / (c + 1e-6), atol=1e-5)

    # When threshold > c, threshold metric is NaN
    u_th_high, valid_th_high, count_high = compute_u_threshold(unc, threshold_t=0.6)
    assert np.isnan(u_th_high)
    assert valid_th_high is False
    assert count_high == 0


def test_border_lesion():
    """
    Test lesion touching the top-left boundary of the image.
    Verifies morphological boundary dilation/erosion with image borders.
    """
    unc = np.zeros((6, 6), dtype=np.float32)
    mask = np.zeros((6, 6), dtype=np.uint8)

    # 2x2 lesion at top-left corner: rows 0..1, cols 0..1
    mask[0:2, 0:2] = 1
    unc[0:2, 0:2] = 0.6  # 4 lesion pixels at 0.6

    # Outside neighborhood in boundary band: 5 pixels at 0.2
    # Rows 0..1, col 2 and Row 2, cols 0..2
    unc[0:2, 2] = 0.2
    unc[2, 0:3] = 0.2

    feats = compute_uncertainty_features(unc, mask, boundary_width=1)

    # Total pixels with uncertainty: 4*0.6 + 5*0.2 = 2.4 + 1.0 = 3.4
    # U_mean: 3.4 / 36 = 0.094444...
    assert np.isclose(feats["U_mean"], 3.4 / 36.0)

    # U_fg: 4 pixels at 0.6 -> mean = 0.6
    assert np.isclose(feats["U_fg"], 0.6)
    assert feats["count_fg"] == 4

    # Boundary band with w=1 around 2x2 corner in OpenCV morphology:
    # Dilated 3x3 region at corner: 9 pixels (rows 0..2, cols 0..2).
    # Eroded region: in cv2.erode, default morphology border value is max (1), so pixel (0,0) remains 1.
    # Therefore boundary band contains 9 - 1 = 8 pixels.
    # 3 boundary pixels inside lesion at 0.6: (0,1), (1,0), (1,1) -> 3 * 0.6 = 1.8
    # 5 boundary pixels outside lesion at 0.2: (0,2), (1,2), (2,0), (2,1), (2,2) -> 5 * 0.2 = 1.0
    # Boundary mean: (1.8 + 1.0) / 8 = 2.8 / 8 = 0.35
    assert feats["count_boundary"] == 8
    assert np.isclose(feats["U_boundary"], 0.35)


def test_nonrectangular_foreground():
    """
    Test non-rectangular lesion (L-shape) with hand-computed population standard deviation.
    """
    unc = np.zeros((5, 5), dtype=np.float32)
    mask = np.zeros((5, 5), dtype=np.uint8)

    # L-shaped lesion: 5 pixels
    # (1, 1), (1, 2), (1, 3), (2, 1), (3, 1)
    l_coords = [(1, 1), (1, 2), (1, 3), (2, 1), (3, 1)]
    l_values = [0.1, 0.2, 0.3, 0.4, 0.5]

    for (r, c), val in zip(l_coords, l_values):
        mask[r, c] = 1
        unc[r, c] = val

    feats = compute_uncertainty_features(unc, mask, boundary_width=1)

    # Hand calculation:
    # Mean = (0.1 + 0.2 + 0.3 + 0.4 + 0.5) / 5 = 1.5 / 5 = 0.3
    assert np.isclose(feats["U_fg"], 0.3)
    assert feats["count_fg"] == 5

    # Population variance (ddof=0):
    # diffs = [-0.2, -0.1, 0.0, 0.1, 0.2]
    # sum of squares = 0.04 + 0.01 + 0 + 0.01 + 0.04 = 0.10
    # variance = 0.10 / 5 = 0.02
    # std = sqrt(0.02) ~= 0.141421356
    expected_std = np.sqrt(0.02)
    assert np.isclose(feats["U_std_fg"], expected_std)
    assert feats["valid_std_fg"] is True


def test_positive_vs_mean_difference():
    """
    Test scenario where U_positive is strictly distinct from U_mean.
    90 pixels are zero, 10 pixels are 0.5 -> U_pos is 10x higher than U_mean.
    """
    unc = np.zeros((10, 10), dtype=np.float32)
    unc[0, :] = 0.5  # 10 pixels at 0.5, remaining 90 at 0.0
    mask = np.zeros((10, 10), dtype=np.uint8)
    mask[0, :] = 1

    feats = compute_uncertainty_features(unc, mask)

    # U_mean: 5.0 / 100 = 0.05
    assert np.isclose(feats["U_mean"], 0.05)
    assert feats["count_total"] == 100

    # U_positive: 5.0 / 10 = 0.5
    assert np.isclose(feats["U_positive"], 0.5)
    assert feats["count_pos"] == 10
    assert feats["U_positive"] > feats["U_mean"] * 9.9


def test_boundary_band_geometry():
    """
    Verify exact morphological pixel selection for a 5x5 square centered in 9x9 grid.
    Interior center pixels must NOT belong to boundary band.
    """
    mask = np.zeros((9, 9), dtype=np.uint8)
    mask[2:7, 2:7] = 1  # 5x5 square (25 pixels)

    boundary_mask = compute_boundary_band_mask(mask, boundary_width=1, kernel_shape="rect")

    # Dilation with 3x3 kernel expands to 7x7 (rows 1..7, cols 1..7 = 49 pixels).
    # Erosion with 3x3 kernel shrinks to 3x3 (rows 3..5, cols 3..5 = 9 pixels).
    # Boundary band = 49 - 9 = 40 pixels.
    assert int(np.sum(boundary_mask)) == 40

    # Interior 3x3 core is NOT in boundary band
    assert not np.any(boundary_mask[3:6, 3:6])

    # Outer perimeter of the dilated square is in the boundary band
    assert np.all(boundary_mask[1, 1:8])
    assert np.all(boundary_mask[7, 1:8])


def test_configurable_parameters():
    """Verify that user-supplied thresholds, boundary widths, and epsilons are respected."""
    unc = np.linspace(0.0, 1.0, 100, dtype=np.float32).reshape(10, 10)
    mask = np.zeros((10, 10), dtype=np.uint8)
    mask[3:7, 3:7] = 1

    # Vary threshold_t
    u_t1, _, c1 = compute_u_threshold(unc, threshold_t=0.2)
    u_t2, _, c2 = compute_u_threshold(unc, threshold_t=0.8)
    assert u_t2 > u_t1
    assert c1 > c2

    # Vary boundary_width
    b_mask1 = compute_boundary_band_mask(mask, boundary_width=1)
    b_mask2 = compute_boundary_band_mask(mask, boundary_width=2)
    assert np.sum(b_mask2) > np.sum(b_mask1)

    # Vary eps
    u_r1, _, _ = compute_u_ratio(unc, mask, eps=1e-6)
    u_r2, _, _ = compute_u_ratio(unc, mask, eps=1.0)
    assert u_r1 > u_r2


def test_error_handling():
    """Verify robust exceptions for shape mismatches, non-finite values, and invalid configs."""
    unc = np.ones((10, 10), dtype=np.float32)
    mask = np.ones((10, 10), dtype=np.uint8)

    # Shape mismatch
    with AssertRaises(ValueError, match="Shape mismatch"):
        compute_u_fg(unc, np.ones((8, 8), dtype=np.uint8))

    # 3D uncertainty map
    with AssertRaises(ValueError, match="Expected 2D"):
        compute_u_mean(np.ones((10, 10, 3), dtype=np.float32))

    # Non-finite values (NaN)
    unc_nan = unc.copy()
    unc_nan[2, 2] = np.nan
    with AssertRaises(ValueError, match="non-finite"):
        compute_u_mean(unc_nan)

    # Non-finite values (Inf)
    unc_inf = unc.copy()
    unc_inf[2, 2] = np.inf
    with AssertRaises(ValueError, match="non-finite"):
        compute_u_positive(unc_inf)

    # Invalid boundary_width
    with AssertRaises(ValueError, match="boundary_width must be >= 1"):
        compute_boundary_band_mask(mask, boundary_width=0)

    # Negative epsilon
    with AssertRaises(ValueError, match="epsilon must be non-negative"):
        compute_u_ratio(unc, mask, eps=-1e-5)


def test_single_pixel_foreground():
    """Verify single-pixel foreground computes std as 0.0 with valid=True."""
    unc = np.full((5, 5), 0.4, dtype=np.float32)
    mask = np.zeros((5, 5), dtype=np.uint8)
    mask[2, 2] = 1

    std_val, valid, count = compute_u_std_fg(unc, mask)
    assert std_val == 0.0
    assert valid is True
    assert count == 1


def test_selective_prediction_and_controls():
    """Integration test verifying downstream selective prediction pipelines remain functional."""
    rng = np.random.default_rng(42)
    n = 100
    dsc = rng.uniform(0.2, 1.0, size=n)
    u_fg = (1.0 - dsc) + rng.normal(0, 0.05, size=n)
    u_fg = np.clip(u_fg, 0.01, 1.0)

    df = pd.DataFrame({"DSC": dsc, "U_fg": u_fg, "NSD": dsc * 0.95})

    tau = calibrate_rejection_threshold(df, "U_fg", target_coverage=0.80)
    assert tau > 0

    rejection_res = apply_selective_rejection(df, "U_fg", tau)
    assert rejection_res["realized_coverage"] > 0
    assert rejection_res["accepted_mean_dsc"] >= rejection_res["rejected_mean_dsc"]

    covs, risks = compute_risk_coverage_curve(df, "U_fg", metric_col="DSC")
    aurc = compute_aurc(covs, risks)
    assert 0.0 <= aurc <= 1.0

    controls_df = evaluate_rejection_controls(df, "U_fg", target_coverages=[1.0, 0.80, 0.60])
    assert len(controls_df) == 3
    row_80 = controls_df[controls_df["Target_Coverage"] == 0.80].iloc[0]
    assert row_80["Oracle_DSC"] >= row_80["Random_DSC"]


def test_statistical_ties_handling():
    """Verify that tied scores produce identical deterministic risk-coverage steps regardless of permutation."""
    # Group 1: 3 items with score=0.1, DSC=[0.9, 0.8, 0.7] -> mean=0.8, risk=0.2
    # Group 2: 2 items with score=0.5, DSC=[0.4, 0.2] -> combined mean=(0.9+0.8+0.7+0.4+0.2)/5 = 0.6, risk=0.4
    df_order_a = pd.DataFrame({
        "score": [0.1, 0.1, 0.1, 0.5, 0.5],
        "DSC": [0.9, 0.8, 0.7, 0.4, 0.2]
    })
    # Permute order of tied elements
    df_order_b = pd.DataFrame({
        "score": [0.1, 0.5, 0.1, 0.5, 0.1],
        "DSC": [0.7, 0.2, 0.9, 0.4, 0.8]
    })

    cov_a, risk_a = compute_risk_coverage_curve(df_order_a, "score", "DSC", handle_ties=True)
    cov_b, risk_b = compute_risk_coverage_curve(df_order_b, "score", "DSC", handle_ties=True)

    assert len(cov_a) == 2, f"Expected 2 unique threshold cutoffs, got {len(cov_a)}"
    assert np.allclose(cov_a, cov_b)
    assert np.allclose(risk_a, risk_b)
    assert np.isclose(cov_a[0], 3 / 5)
    assert np.isclose(risk_a[0], 1.0 - 0.8)  # 0.2
    assert np.isclose(cov_a[1], 1.0)
    assert np.isclose(risk_a[1], 1.0 - 0.6)  # 0.4

    aurc_a = compute_aurc(cov_a, risk_a)
    aurc_b = compute_aurc(cov_b, risk_b)
    assert np.isclose(aurc_a, aurc_b)


def test_statistical_nan_handling():
    """Verify statistical metrics filter NaNs cleanly and do not crash or propagate NaNs."""
    df_with_nan = pd.DataFrame({
        "U_clean": [0.1, 0.2, 0.3, 0.4, 0.5],
        "U_nan": [0.1, np.nan, 0.3, np.nan, 0.5],
        "DSC": [0.9, 0.8, np.nan, 0.6, 0.5]
    })

    # compute_risk_coverage_curve should drop NaNs automatically
    cov, risk = compute_risk_coverage_curve(df_with_nan, "U_nan", "DSC")
    assert len(cov) == 2  # indices 0 and 4 are valid
    assert not np.any(np.isnan(cov))
    assert not np.any(np.isnan(risk))

    # compute_spearman_correlations should handle NaNs
    corr_df, summary = compute_spearman_correlations(df_with_nan, ["U_clean", "U_nan"], ["DSC"])
    row_nan = corr_df[corr_df["Feature"] == "U_nan"].iloc[0]
    assert row_nan["DSC_n"] == 2
    assert not np.isnan(row_nan["DSC_rho"])

    # compute_failure_auroc handles all-NaN or partial NaN
    df_all_nan = pd.DataFrame({"U_empty": [np.nan, np.nan], "DSC": [0.2, 0.8]})
    auroc_df = compute_failure_auroc(df_all_nan, ["U_empty"], failure_threshold=0.5)
    assert np.isnan(auroc_df.iloc[0]["AUROC_DSC"])


def test_statistical_ranking_direction():
    """Verify uncertainty sorts ascending (lowest uncertainty first) and oracle sorts descending."""
    df = pd.DataFrame({
        "uncertainty": [0.9, 0.1, 0.4],
        "DSC": [0.2, 0.95, 0.6]
    })

    # Uncertainty ranking: ascending uncertainty -> items 1 (unc=0.1, dsc=0.95), then 2, then 0
    cov_unc, risk_unc = compute_risk_coverage_curve(df, "uncertainty", "DSC", reverse_sort=False, handle_ties=False)
    assert np.isclose(risk_unc[0], 1.0 - 0.95)  # lowest risk accepted first

    # Oracle ranking: descending DSC -> items 1 (dsc=0.95), 2 (dsc=0.6), 0 (dsc=0.2)
    cov_ora, risk_ora = compute_risk_coverage_curve(df, "DSC", "DSC", reverse_sort=True, handle_ties=False)
    assert np.isclose(risk_ora[0], 1.0 - 0.95)
    assert np.all(risk_unc >= risk_ora - 1e-6)  # Oracle risk is lower or equal everywhere


def test_statistical_degenerate_cases():
    """Verify graceful handling of edge cases: empty DataFrame, single row, constant metrics."""
    # Empty DataFrame
    df_empty = pd.DataFrame({"score": [], "DSC": []})
    cov_e, risk_e = compute_risk_coverage_curve(df_empty, "score", "DSC")
    assert len(cov_e) == 0 and len(risk_e) == 0
    assert np.isnan(compute_aurc(cov_e, risk_e))

    # Single-element DataFrame
    df_single = pd.DataFrame({"score": [0.3], "DSC": [0.85]})
    cov_s, risk_s = compute_risk_coverage_curve(df_single, "score", "DSC")
    assert len(cov_s) == 1
    assert np.isclose(risk_s[0], 0.15)
    assert np.isclose(compute_aurc(cov_s, risk_s), 0.15)

    # Constant metric (zero variance in metric)
    df_const = pd.DataFrame({"score": [0.1, 0.2, 0.3], "DSC": [0.8, 0.8, 0.8]})
    cov_c, risk_c = compute_risk_coverage_curve(df_const, "score", "DSC")
    assert np.allclose(risk_c, 0.2)
    expected_aurc = 0.2 * (cov_c[-1] - cov_c[0])
    assert np.isclose(compute_aurc(cov_c, risk_c), expected_aurc)


def test_spearman_bootstrap_ci():
    """Verify that bootstrap confidence intervals produce valid bounds containing empirical rho."""
    rng = np.random.default_rng(666)
    x = rng.normal(0, 1, size=50)
    y = -2.0 * x + rng.normal(0, 0.2, size=50)  # Strong negative correlation

    rho, ci_l, ci_u = compute_spearman_bootstrap_ci(x, y, n_bootstrap=500, ci=0.95, seed=666)
    assert rho < -0.8
    assert ci_l <= rho <= ci_u
    assert -1.0 <= ci_l < ci_u <= 1.0


def test_paired_bootstrap_aurc_ci():
    """Verify paired bootstrap difference between two scoring methods detects clear superiority."""
    rng = np.random.default_rng(42)
    n = 60
    dsc = rng.uniform(0.3, 1.0, size=n)

    # Method A is strongly aligned with risk: score = 1 - DSC + tiny noise
    score_a = (1.0 - dsc) + rng.normal(0, 0.02, size=n)
    # Method B is random noise
    score_b = rng.uniform(0, 1, size=n)

    df = pd.DataFrame({"DSC": dsc, "score_a": score_a, "score_b": score_b})
    delta, ci_l, ci_u, p_val = compute_paired_bootstrap_aurc_ci(
        df, "score_a", "score_b", metric_col="DSC", n_bootstrap=200, seed=42
    )

    # Method A should have strictly lower AURC, so delta = AURC(A) - AURC(B) < 0
    assert delta < 0
    assert ci_l <= delta <= ci_u
    assert ci_u < 0  # Confirms statistically significant lower risk at 95% confidence


def test_empty_mask_policies():
    """Verify that all inference-time empty-mask policies behave predictably on empty/non-empty inputs."""
    df = pd.DataFrame({
        "image_id": ["lesion_1.png", "normal_1.png", "lesion_2.png"],
        "valid_fg": [True, False, True],
        "count_fg": [1200, 0, 850],
        "U_fg": [0.08, np.nan, 0.12],
        "U_mean": [0.015, 1e-15, 0.025],
        "DSC": [0.92, 1.0, 0.85]
    })

    # Policy 1: drop_empty -> leaves only the 2 lesion cases
    res_drop = apply_empty_mask_policy(df, "U_fg", "U_mean", policy="drop_empty")
    assert len(res_drop) == 2
    assert list(res_drop["image_id"]) == ["lesion_1.png", "lesion_2.png"]
    assert np.all(res_drop["deployed_score"] > 0)

    # Policy 2: global_fallback -> empty mask receives U_mean (1e-15)
    res_fb = apply_empty_mask_policy(df, "U_fg", "U_mean", policy="global_fallback")
    assert len(res_fb) == 3
    assert np.isclose(res_fb.loc[1, "deployed_score"], 1e-15)
    assert res_fb.loc[1, "policy_action"] == "fallback_global_uncertainty"

    # Policy 3: conservative_abstention -> empty mask receives high uncertainty (always rejected)
    res_abs = apply_empty_mask_policy(df, "U_fg", "U_mean", policy="conservative_abstention")
    assert len(res_abs) == 3
    assert res_abs.loc[1, "deployed_score"] >= 1e8
    assert res_abs.loc[1, "policy_action"] == "conservative_abstain_empty"

    # Policy 4: two_stage -> low global uncertainty accepted as confident negative
    res_2s = apply_empty_mask_policy(df, "U_fg", "U_mean", policy="two_stage", two_stage_tau_neg=0.001)
    assert len(res_2s) == 3
    assert np.isclose(res_2s.loc[1, "deployed_score"], 1e-15)
    assert res_2s.loc[1, "policy_action"] == "two_stage_accept_negative"

    # Policy 4 with suspicious high global uncertainty -> abstains
    df_suspicious = df.copy()
    df_suspicious.loc[1, "U_mean"] = 0.05  # high global uncertainty on empty mask!
    res_2s_susp = apply_empty_mask_policy(df_suspicious, "U_fg", "U_mean", policy="two_stage", two_stage_tau_neg=0.001)
    assert res_2s_susp.loc[1, "deployed_score"] >= 1e8
    assert res_2s_susp.loc[1, "policy_action"] == "two_stage_abstain_suspicious_negative"


def test_threshold_calibration_reproducibility():
    """Verify that threshold calibration is deterministic, bounded, and monotonically strict."""
    df = pd.DataFrame({"score": [0.01, 0.02, 0.05, 0.10, 0.20, 0.50, 0.80, 0.90, 0.95, 1.00]})
    tau_80_first = calibrate_rejection_threshold(df, "score", target_coverage=0.80)
    tau_80_second = calibrate_rejection_threshold(df, "score", target_coverage=0.80)
    assert tau_80_first == tau_80_second

    tau_70 = calibrate_rejection_threshold(df, "score", target_coverage=0.70)
    tau_90 = calibrate_rejection_threshold(df, "score", target_coverage=0.90)
    assert tau_70 <= tau_80_first <= tau_90

    # Test error on invalid coverage
    with AssertRaises(ValueError):
        calibrate_rejection_threshold(df, "score", target_coverage=1.5)
    with AssertRaises(ValueError):
        calibrate_rejection_threshold(df, "score", target_coverage=0.0)


def test_frozen_config_loading():
    """Verify that the frozen selective prediction configuration loads identically from YAML and JSON."""
    import yaml
    import json
    
    cfg_dir = os.path.abspath(os.path.join(
        os.path.dirname(__file__), "..", "..", "..", "data", "outputs", "BUSI",
        "validation", "seed666", "selective_prediction"
    ))
    yaml_path = os.path.join(cfg_dir, "frozen_selective_prediction_config.yaml")
    json_path = os.path.join(cfg_dir, "frozen_selective_prediction_config.json")
    
    assert os.path.exists(yaml_path), f"Frozen YAML config missing: {yaml_path}"
    assert os.path.exists(json_path), f"Frozen JSON config missing: {json_path}"
    
    with open(yaml_path, "r") as f:
        y_cfg = yaml.safe_load(f)
    with open(json_path, "r") as f:
        j_cfg = json.load(f)
        
    assert y_cfg == j_cfg, "YAML and JSON frozen configs are not semantically identical!"
    assert y_cfg["primary_method"]["name"] == "U_fg"
    assert y_cfg["prespecified_comparator"]["name"] == "U_std_fg"
    assert y_cfg["primary_operating_point"]["target_coverage"] == 0.80
    assert np.isclose(y_cfg["primary_operating_point"]["tau"], 0.064058324)
    assert len(y_cfg["calibrated_abstention_thresholds_setting_A"]) == 6


def test_time_no_leakage():
    """Verify that test-time rejection operates strictly without ground-truth columns (DSC, NSD, label)."""
    # DataFrame strictly containing inference-time model outputs
    df_no_gt = pd.DataFrame({
        "image_id": ["scan_01.png", "scan_02.png", "scan_03.png"],
        "valid_fg": [True, True, False],
        "count_fg": [1500, 250, 0],
        "U_fg": [0.045, 0.085, np.nan],
        "U_mean": [0.005, 0.012, 1e-6],
    })
    
    # Ensure neither DSC nor NSD nor labels exist in the DataFrame
    assert "DSC" not in df_no_gt.columns
    assert "NSD" not in df_no_gt.columns
    assert "label" not in df_no_gt.columns
    
    # 1. Setting A: Rejection strictly based on U_fg and fixed tau
    res_a = apply_selective_rejection(df_no_gt.dropna(subset=["U_fg"]), "U_fg", tau=0.064058324)
    assert res_a["n_accepted"] == 1
    assert res_a["n_rejected"] == 1
    assert "accepted_mean_dsc" not in res_a  # No ground truth evaluated
    
    # 2. Setting B: Two-stage policy strictly based on mask emptiness and U_mean
    res_b = apply_empty_mask_policy(df_no_gt, "U_fg", "U_mean", policy="two_stage", two_stage_tau_neg=1e-4)
    assert len(res_b) == 3
    assert res_b.loc[2, "policy_action"] == "two_stage_accept_negative"
    assert np.isclose(res_b.loc[2, "deployed_score"], 1e-6)


def test_deterministic_rejection():
    """Verify that threshold rejection produces 100% bitwise identical results across repeated executions."""
    df = pd.DataFrame({
        "image_id": [f"img_{i}.png" for i in range(20)],
        "U_fg": np.linspace(0.01, 0.15, 20),
        "DSC": np.linspace(0.95, 0.50, 20),
    })
    tau = 0.064058324
    
    res1 = apply_selective_rejection(df, "U_fg", tau)
    res2 = apply_selective_rejection(df, "U_fg", tau)
    
    assert res1["n_accepted"] == res2["n_accepted"]
    assert res1["n_rejected"] == res2["n_rejected"]
    assert res1["realized_coverage"] == res2["realized_coverage"]
    assert res1["accepted_df"].index.tolist() == res2["accepted_df"].index.tolist()
    assert res1["rejected_df"].index.tolist() == res2["rejected_df"].index.tolist()
    assert np.isclose(res1["accepted_mean_dsc"], res2["accepted_mean_dsc"])


def test_result_reproducibility():
    """Verify that bootstrap confidence intervals are reproducible under a fixed seed."""
    df_synth = pd.DataFrame({
        "m_a": [0.02, 0.04, 0.06, 0.08, 0.10, 0.12, 0.14, 0.16, 0.18, 0.20],
        "m_b": [0.01, 0.05, 0.03, 0.09, 0.11, 0.15, 0.13, 0.17, 0.19, 0.22],
        "DSC": [0.95, 0.90, 0.88, 0.85, 0.80, 0.78, 0.75, 0.70, 0.65, 0.60],
    })
    
    # Run twice with the exact same seed
    res1 = compute_paired_bootstrap_aurc_ci(df_synth, "m_a", "m_b", "DSC", n_bootstrap=200, seed=666)
    res2 = compute_paired_bootstrap_aurc_ci(df_synth, "m_a", "m_b", "DSC", n_bootstrap=200, seed=666)
    
    assert np.isclose(res1[0], res2[0])  # point delta
    assert np.isclose(res1[1], res2[1])  # ci lower
    assert np.isclose(res1[2], res2[2])  # ci upper
    assert np.isclose(res1[3], res2[3])  # p-value


def test_ood_cohort_integrity():
    """Verify that all four OOD datasets have 100% processed test samples."""
    base_out = os.environ.get("MEDCLIPSEG_OUTPUT", os.path.abspath(os.path.join(REPO_ROOT, "..", "data", "outputs")))
    expected_counts = {"UDIAT": 25, "BUID": 35, "BUSUC": 122, "BUSBRA": 282}
    for dset, expected_n in expected_counts.items():
        csv_path = os.path.join(base_out, dset, "aggregation_results", "seed666", f"uncertainty_scores_{dset}_test_seed666.csv")
        assert os.path.exists(csv_path), f"Missing CSV for {dset}: {csv_path}"
        df = pd.read_csv(csv_path)
        assert len(df) == expected_n, f"Expected {expected_n} samples for {dset}, got {len(df)}"
        assert df["U_fg"].isna().sum() == 0, f"Found unexpected NaN U_fg in {dset}"


def test_ood_selective_gain_positive():
    """Verify that frozen U_fg threshold tau achieves strictly positive DSC gain across all 4 OOD datasets."""
    base_out = os.environ.get("MEDCLIPSEG_OUTPUT", os.path.abspath(os.path.join(REPO_ROOT, "..", "data", "outputs")))
    tau = 0.064058324
    for dset in ["UDIAT", "BUID", "BUSUC", "BUSBRA"]:
        summary_path = os.path.join(base_out, dset, "test_evaluation", "seed666", f"summary_evaluation_{dset}.json")
        assert os.path.exists(summary_path), f"Missing summary for {dset}"
        with open(summary_path, "r") as f:
            data = json.load(f)
        prim = data["primary_evaluation"]
        assert prim["dsc_gain"] > 0, f"Expected positive DSC gain on {dset}, got {prim['dsc_gain']}"
        assert prim["accepted_dsc"] > data["baseline_stats"]["mean_dsc"]


def test_ood_ufg_beats_umean_aurc():
    """Verify that U_fg achieves lower AURC than U_mean on all 4 OOD datasets."""
    base_out = os.environ.get("MEDCLIPSEG_OUTPUT", os.path.abspath(os.path.join(REPO_ROOT, "..", "data", "outputs")))
    for dset in ["UDIAT", "BUID", "BUSUC", "BUSBRA"]:
        summary_path = os.path.join(base_out, dset, "test_evaluation", "seed666", f"summary_evaluation_{dset}.json")
        with open(summary_path, "r") as f:
            data = json.load(f)
        aurc_fg = data["aurc_dsc"]["U_fg"]
        aurc_mean = data["aurc_dsc"]["U_mean"]
        assert aurc_fg < aurc_mean, f"Expected U_fg AURC < U_mean AURC on {dset}, got {aurc_fg} vs {aurc_mean}"


def test_ood_umean_confounding_persists():
    """Verify that U_mean suffers strong positive area confounding (rho > 0.5) across all OOD datasets."""
    base_out = os.environ.get("MEDCLIPSEG_OUTPUT", os.path.abspath(os.path.join(REPO_ROOT, "..", "data", "outputs")))
    for dset in ["UDIAT", "BUID", "BUSUC", "BUSBRA"]:
        summary_path = os.path.join(base_out, dset, "test_evaluation", "seed666", f"summary_evaluation_{dset}.json")
        with open(summary_path, "r") as f:
            data = json.load(f)
        rho_mean = data["confounding_spearman"]["U_mean"]["rho"]
        assert rho_mean > 0.50, f"Expected strong positive area confounding for U_mean on {dset}, got rho={rho_mean}"


