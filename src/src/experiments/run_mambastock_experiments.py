#!/usr/bin/env python3
from __future__ import annotations
import subprocess, sys
from pathlib import Path

def main() -> None:
    root = Path(__file__).resolve().parents[2]
    cmd = [sys.executable, str(root / 'src/experiments/run_deep_torch_experiments.py'), '--models', 'MambaStock', *sys.argv[1:]]
    subprocess.run(cmd, check=True)

if __name__ == '__main__':
    main()
