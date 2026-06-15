#!/usr/bin/env python3
"""Preprocessing variant v5: IQR clipping + signed-log transform + cross-sectional z-score."""

from __future__ import annotations

import argparse
from pathlib import Path

from preprocess_panel_variants_common import (
    cross_sectional_zscore,
    feature_columns,
    fill_numeric_na,
    iqr_clip_by_symbol,
    load_panel,
    signed_log1p,
    write_outputs,
)

DEFAULT_INPUT = Path("data/processed/features/research_features_panel.csv")
DEFAULT_OUTPUT = Path("data/processed/features/research_features_panel_pre_v5.csv")
DEFAULT_META = Path("data/metadata/research_features_panel_pre_v5.json")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Build preprocessing variant v5.")
    parser.add_argument("--input-csv", type=Path, default=DEFAULT_INPUT)
    parser.add_argument("--output-csv", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--metadata-json", type=Path, default=DEFAULT_META)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    df = load_panel(args.input_csv)
    cols = feature_columns(df)

    out = iqr_clip_by_symbol(df, cols=cols, k=2.5)
    out = signed_log1p(out, cols=cols)
    out = cross_sectional_zscore(out, cols=cols)
    out = fill_numeric_na(out, cols=cols, value=0.0)

    write_outputs(
        df=out,
        output_csv=args.output_csv,
        metadata_json=args.metadata_json,
        method_name="v5_iqrclip_log_cs_z",
        denoising_steps=["Per-symbol IQR clipping with k=2.5"],
        normalization_steps=[
            "Signed log1p normalization for heavy tails",
            "Daily cross-sectional z-score normalization",
        ],
    )

    print("Built preprocessing variant: v5_iqrclip_log_cs_z")
    print(f"Output CSV: {args.output_csv}")


if __name__ == "__main__":
    main()
