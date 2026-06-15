#!/usr/bin/env python3
"""Convenience wrapper for deep torch sweeps on target_h5."""

from __future__ import annotations

import subprocess
import sys
from pathlib import Path


def main() -> None:
    root = Path(__file__).resolve().parents[2]
    cmd = [
        sys.executable,
        str(root / "src/experiments/run_deep_torch_experiments.py"),
        "--target",
        "target_h5",
        "--results-csv",
        str(root / "reports/deep_torch_model_results_h5.csv"),
        "--summary-json",
        str(root / "reports/deep_torch_model_summary_h5.json"),
        "--summary-md",
        str(root / "reports/deep_torch_model_summary_h5.md"),
    ]
    subprocess.run(cmd, check=True)


if __name__ == "__main__":
    main()
