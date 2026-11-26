"""Test configuration files in shared/configs/."""

import os
import glob
import yaml


def test_yaml_syntax():
    config_dir = os.path.join(os.path.dirname(__file__), "..", "..", "..", "shared", "configs")
    yaml_files = glob.glob(os.path.join(config_dir, "*.yaml"))
    assert len(yaml_files) > 0, "No yaml configs found in shared/configs/"

    for yf in yaml_files:
        with open(yf, "r") as f:
            content = yaml.safe_load(f)
            assert isinstance(content, dict), f"Failed to parse dict from {yf}"
            assert "DATASET" in content, f"Missing DATASET section in {yf}"
            assert "MODEL" in content, f"Missing MODEL section in {yf}"
