#!/usr/bin/env python3
"""Preprocessing variant v2: EMA smoothing + per-symbol standard z-score normalization."""

from __future__ import annotations

import argparse
from pathlib import Path

from preprocess_panel_variants_common import (
    ema_smooth_by_symbol,
    feature_columns,
    fill_numeric_na,
    load_panel,
    standard_zscore_by_symbol,
    write_outputs,
)

DEFAULT_INPUT = Path("data/processed/features/research_features_panel.csv")
DEFAULT_OUTPUT = Path("data/processed/features/research_features_panel_pre_v2.csv")
DEFAULT_META = Path("data/metadata/research_features_panel_pre_v2.json")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Build preprocessing variant v2.")
    parser.add_argument("--input-csv", type=Path, default=DEFAULT_INPUT)
    parser.add_argument("--output-csv", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--metadata-json", type=Path, default=DEFAULT_META)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    df = load_panel(args.input_csv)
    cols = feature_columns(df)

    out = ema_smooth_by_symbol(df, cols=cols, span=5)
    out = standard_zscore_by_symbol(out, cols=cols)
    out = fill_numeric_na(out, cols=cols, value=0.0)

    write_outputs(
        df=out,
        output_csv=args.output_csv,
        metadata_json=args.metadata_json,
        method_name="v2_ema_standardz",
        denoising_steps=["Per-symbol EMA smoothing (span=5)"],
        normalization_steps=["Per-symbol standard z-score using mean/std"],
    )

    print("Built preprocessing variant: v2_ema_standardz")
    print(f"Output CSV: {args.output_csv}")


if __name__ == "__main__":
    main()
