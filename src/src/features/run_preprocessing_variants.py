#!/usr/bin/env python3
"""Run all preprocessing variant scripts and materialize prepared CSV datasets."""

from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]

SCRIPTS = [
    "src/features/preprocess_panel_v1_winsor_robustz.py",
    "src/features/preprocess_panel_v2_ema_standardz.py",
    "src/features/preprocess_panel_v3_rollmed_minmax.py",
    "src/features/preprocess_panel_v4_hampel_ranknorm.py",
    "src/features/preprocess_panel_v5_iqrclip_log_cs_z.py",
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Build all panel preprocessing variants.")
    parser.add_argument("--continue-on-error", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    failures: list[tuple[str, int]] = []
    for script in SCRIPTS:
        cmd = [sys.executable, script]
        proc = subprocess.run(cmd, cwd=ROOT)
        if proc.returncode != 0:
            failures.append((script, proc.returncode))
            if not args.continue_on_error:
                break

    if failures:
        print("Preprocessing variants finished with failures:")
        for script, code in failures:
            print(f"- {script}: exit_code={code}")
        raise SystemExit(1)

    print("All preprocessing variants generated successfully.")


if __name__ == "__main__":
    main()
