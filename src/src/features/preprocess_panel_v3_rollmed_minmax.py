#!/usr/bin/env python3
"""Preprocessing variant v3: rolling-median denoising + per-symbol min-max normalization."""

from __future__ import annotations

import argparse
from pathlib import Path

from preprocess_panel_variants_common import (
    feature_columns,
    fill_numeric_na,
    load_panel,
    minmax_by_symbol,
    rolling_median_by_symbol,
    write_outputs,
)

DEFAULT_INPUT = Path("data/processed/features/research_features_panel.csv")
DEFAULT_OUTPUT = Path("data/processed/features/research_features_panel_pre_v3.csv")
DEFAULT_META = Path("data/metadata/research_features_panel_pre_v3.json")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Build preprocessing variant v3.")
    parser.add_argument("--input-csv", type=Path, default=DEFAULT_INPUT)
    parser.add_argument("--output-csv", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--metadata-json", type=Path, default=DEFAULT_META)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    df = load_panel(args.input_csv)
    cols = feature_columns(df)

    out = rolling_median_by_symbol(df, cols=cols, window=5)
    out = minmax_by_symbol(out, cols=cols)
    out = fill_numeric_na(out, cols=cols, value=0.0)

    write_outputs(
        df=out,
        output_csv=args.output_csv,
        metadata_json=args.metadata_json,
        method_name="v3_rollmed_minmax",
        denoising_steps=["Per-symbol rolling median denoising (window=5)"],
        normalization_steps=["Per-symbol min-max normalization to [-1, 1]"],
    )

    print("Built preprocessing variant: v3_rollmed_minmax")
    print(f"Output CSV: {args.output_csv}")


if __name__ == "__main__":
    main()
