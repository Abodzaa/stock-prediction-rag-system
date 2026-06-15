#!/usr/bin/env python3
"""Convenience wrapper: run deep-model sweeps for target_h5 outputs."""

from __future__ import annotations

import subprocess
import sys
from pathlib import Path


def main() -> None:
    root = Path(__file__).resolve().parents[2]
    cmd = [
        sys.executable,
        str(root / "src/experiments/run_deep_model_sweeps.py"),
        "--target",
        "target_h5",
        "--report-csv",
        str(root / "reports/deep_model_sweep_results_h5.csv"),
        "--summary-json",
        str(root / "reports/deep_model_sweep_summary_h5.json"),
        "--summary-md",
        str(root / "reports/deep_model_sweep_summary_h5.md"),
    ]
    subprocess.run(cmd, check=True)


if __name__ == "__main__":
    main()
