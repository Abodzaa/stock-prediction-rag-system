#!/usr/bin/env python3
"""Preprocessing variant v4: Hampel denoising + cross-sectional rank normalization."""

from __future__ import annotations

import argparse
from pathlib import Path

from preprocess_panel_variants_common import (
    cross_sectional_rank_norm,
    feature_columns,
    fill_numeric_na,
    hampel_filter_by_symbol,
    load_panel,
    write_outputs,
)

DEFAULT_INPUT = Path("data/processed/features/research_features_panel.csv")
DEFAULT_OUTPUT = Path("data/processed/features/research_features_panel_pre_v4.csv")
DEFAULT_META = Path("data/metadata/research_features_panel_pre_v4.json")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Build preprocessing variant v4.")
    parser.add_argument("--input-csv", type=Path, default=DEFAULT_INPUT)
    parser.add_argument("--output-csv", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--metadata-json", type=Path, default=DEFAULT_META)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    df = load_panel(args.input_csv)
    cols = feature_columns(df)

    out = hampel_filter_by_symbol(df, cols=cols, window=7, n_sigma=3.0)
    out = cross_sectional_rank_norm(out, cols=cols)
    out = fill_numeric_na(out, cols=cols, value=0.0)

    write_outputs(
        df=out,
        output_csv=args.output_csv,
        metadata_json=args.metadata_json,
        method_name="v4_hampel_ranknorm",
        denoising_steps=["Per-symbol Hampel filter (window=7, n_sigma=3.0)"],
        normalization_steps=["Daily cross-sectional percentile rank scaled to [-1, 1]"],
    )

    print("Built preprocessing variant: v4_hampel_ranknorm")
    print(f"Output CSV: {args.output_csv}")


if __name__ == "__main__":
    main()
