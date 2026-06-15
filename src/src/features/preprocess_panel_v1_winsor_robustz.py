#!/usr/bin/env python3
"""Preprocessing variant v1: symbol winsorization + robust z-score normalization."""

from __future__ import annotations

import argparse
from pathlib import Path

from preprocess_panel_variants_common import (
    feature_columns,
    fill_numeric_na,
    load_panel,
    robust_zscore_by_symbol,
    winsorize_by_symbol,
    write_outputs,
)

DEFAULT_INPUT = Path("data/processed/features/research_features_panel.csv")
DEFAULT_OUTPUT = Path("data/processed/features/research_features_panel_pre_v1.csv")
DEFAULT_META = Path("data/metadata/research_features_panel_pre_v1.json")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Build preprocessing variant v1.")
    parser.add_argument("--input-csv", type=Path, default=DEFAULT_INPUT)
    parser.add_argument("--output-csv", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--metadata-json", type=Path, default=DEFAULT_META)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    df = load_panel(args.input_csv)
    cols = feature_columns(df)

    out = winsorize_by_symbol(df, cols=cols, low_q=0.01, high_q=0.99)
    out = robust_zscore_by_symbol(out, cols=cols)
    out = fill_numeric_na(out, cols=cols, value=0.0)

    write_outputs(
        df=out,
        output_csv=args.output_csv,
        metadata_json=args.metadata_json,
        method_name="v1_winsor_robustz",
        denoising_steps=["Per-symbol winsorization at 1st/99th percentiles"],
        normalization_steps=["Per-symbol robust z-score using median/IQR"],
    )

    print("Built preprocessing variant: v1_winsor_robustz")
    print(f"Output CSV: {args.output_csv}")


if __name__ == "__main__":
    main()
