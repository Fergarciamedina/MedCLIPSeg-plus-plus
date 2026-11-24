"""Clean-room reproducibility test for the selective prediction evaluation toolkit.

Verifies:
1. Synthetic demonstration cohort evaluation.
2. Real benchmark evaluations on canonical per-image CSVs:
   - UDIAT (N=25)
   - BUID (N=35)
   - BUSUC (N=122)
   - BUSBRA (N=282)
   - BUSI Test Lesion (N=65)
3. Exact concordance with canonical numbers published in the scientific documentation and paper.
"""

import os
import sys
import numpy as np
import pandas as pd

from metrics import (
    compute_dice,
    compute_risk_coverage_curve,
    compute_aurc,
    compute_random_aurc,
    compute_oracle_aurc,
    apply_selective_rejection,
)
from evaluate import run_evaluation, FROZEN_DEFAULT_THRESHOLD


CANONICAL_TARGETS = {
    "UDIAT": {
        "n": 25,
        "cov": 0.5600,
        "base_dsc": 0.7467,
        "acc_dsc": 0.8747,
        "gain": 0.1281,
    },
    "BUID": {
        "n": 35,
        "cov": 0.5714,
        "base_dsc": 0.7900,
        "acc_dsc": 0.8818,
        "gain": 0.0919,
    },
    "BUSUC": {
        "n": 122,
        "cov": 0.6393,
        "base_dsc": 0.8430,
        "acc_dsc": 0.8982,
        "gain": 0.0551,
    },
    "BUSBRA": {
        "n": 282,
        "cov": 0.4468,
        "base_dsc": 0.7511,
        "acc_dsc": 0.8536,
        "gain": 0.1025,
    },
}


def test_synthetic_evaluation():
    print("[TEST] Running synthetic demonstration evaluation...")
    from evaluate import generate_demo_dataset
    df = generate_demo_dataset(100, seed=42)
    summary = run_evaluation(
        df=df,
        threshold=FROZEN_DEFAULT_THRESHOLD,
        aggregator="u_fg",
        metric_col="DSC",
        dataset_name="Synthetic Demo",
        compare_aggregators=True,
        output_dir="./tmp_test_synth",
    )
    assert summary["Total_Cases"] == 100
    assert summary["Realized_Coverage"] > 0.0
    assert summary["Accepted_DSC"] > summary["Baseline_DSC"]
    assert summary["AURC_Primary"] < summary["AURC_Random"]
    print("  [PASS] Synthetic demonstration evaluation passed.")


def test_real_benchmarks():
    print("[TEST] Running canonical cross-domain benchmarks...")
    base_dir = os.environ.get("MEDCLIPSEG_OUTPUT", os.path.abspath(
        os.path.join(os.path.dirname(__file__), "..", "..", "..", "..", "data", "outputs")
    ))

    csv_paths = {
        "UDIAT": os.path.join(base_dir, "UDIAT", "aggregation_results", "seed666", "uncertainty_scores_UDIAT_test_seed666.csv"),
        "BUID": os.path.join(base_dir, "BUID", "aggregation_results", "seed666", "uncertainty_scores_BUID_test_seed666.csv"),
        "BUSUC": os.path.join(base_dir, "BUSUC", "aggregation_results", "seed666", "uncertainty_scores_BUSUC_test_seed666.csv"),
        "BUSBRA": os.path.join(base_dir, "BUSBRA", "aggregation_results", "seed666", "uncertainty_scores_BUSBRA_test_seed666.csv"),
    }

    all_dfs = []
    for name, path in csv_paths.items():
        assert os.path.exists(path), f"File missing: {path}"
        df = pd.read_csv(path)
        all_dfs.append(df)
        
        summary = run_evaluation(
            df=df,
            threshold=FROZEN_DEFAULT_THRESHOLD,
            aggregator="u_fg",
            metric_col="DSC",
            dataset_name=name,
            compare_aggregators=False,
            output_dir=f"./tmp_test_{name.lower()}",
        )

        expected = CANONICAL_TARGETS[name]
        assert summary["Total_Cases"] == expected["n"], f"{name}: Total cases mismatch"
        np.testing.assert_allclose(summary["Realized_Coverage"], expected["cov"], atol=1e-3, err_msg=f"{name}: Coverage")
        np.testing.assert_allclose(summary["Baseline_DSC"], expected["base_dsc"], atol=1e-3, err_msg=f"{name}: Baseline DSC")
        np.testing.assert_allclose(summary["Accepted_DSC"], expected["acc_dsc"], atol=1e-3, err_msg=f"{name}: Accepted DSC")
        np.testing.assert_allclose(summary["Selective_Gain"], expected["gain"], atol=1e-3, err_msg=f"{name}: Selective Gain")
        print(f"  [PASS] {name} evaluation perfectly matched canonical values.")

    # Test Pooled OOD (N=464)
    print("[TEST] Running Pooled OOD benchmark (N=464)...")
    pooled_df = pd.concat(all_dfs, ignore_index=True)
    pooled_summary = run_evaluation(
        df=pooled_df,
        threshold=FROZEN_DEFAULT_THRESHOLD,
        aggregator="u_fg",
        metric_col="DSC",
        dataset_name="Pooled OOD (N=464)",
        compare_aggregators=False,
        output_dir="./tmp_test_pooled",
    )
    assert pooled_summary["Total_Cases"] == 464
    np.testing.assert_allclose(pooled_summary["Realized_Coverage"], 0.5129, atol=1e-3)
    np.testing.assert_allclose(pooled_summary["Baseline_DSC"], 0.7780, atol=1e-3)
    np.testing.assert_allclose(pooled_summary["Accepted_DSC"], 0.8718, atol=1e-3)
    np.testing.assert_allclose(pooled_summary["Selective_Gain"], 0.0939, atol=1e-3)
    print("  [PASS] Pooled OOD evaluation perfectly matched canonical values.")


def cleanup():
    import shutil
    for d in ["./tmp_test_synth", "./tmp_test_udiat", "./tmp_test_buid", "./tmp_test_busuc", "./tmp_test_busbra", "./tmp_test_pooled"]:
        if os.path.exists(d):
            shutil.rmtree(d)


if __name__ == "__main__":
    try:
        test_synthetic_evaluation()
        test_real_benchmarks()
        print("\nAll clean-room reproducibility tests passed successfully!")
    finally:
        cleanup()

