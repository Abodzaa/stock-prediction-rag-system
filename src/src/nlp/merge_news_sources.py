#!/usr/bin/env python3
"""Merge multiple raw news CSV files into one deduplicated corpus."""

from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd


DEFAULT_OUTPUT = Path("data/raw_news/market_news_combined.csv")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Merge raw news CSV sources.")
    parser.add_argument("--inputs", required=True, help="Comma-separated input CSV paths.")
    parser.add_argument("--output-csv", type=Path, default=DEFAULT_OUTPUT)
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    input_paths = [Path(x.strip()) for x in args.inputs.split(",") if x.strip()]
    if not input_paths:
        raise ValueError("No input CSV paths provided.")

    frames = []
    for p in input_paths:
        if not p.exists():
            raise FileNotFoundError(f"Input CSV not found: {p}")
        d = pd.read_csv(p)
        d["_source_file"] = str(p)
        frames.append(d)

    out = pd.concat(frames, ignore_index=True)
    out["published_utc"] = pd.to_datetime(out["published_utc"], utc=True, errors="coerce")
    out = out.dropna(subset=["published_utc"]).copy()

    for col in ["news_id", "headline", "summary", "source", "url", "symbol"]:
        if col not in out.columns:
            out[col] = ""

    out["news_id"] = out["news_id"].astype(str)
    out["headline"] = out["headline"].astype(str)
    out["url"] = out["url"].astype(str)

    fallback = (
        out["published_utc"].dt.strftime("%Y%m%d%H%M%S")
        + "|"
        + out["url"].str.slice(0, 300)
        + "|"
        + out["headline"].str.slice(0, 120)
    )
    out["news_id"] = out["news_id"].mask(out["news_id"].str.len() == 0, fallback)

    out = out.drop_duplicates(subset=["news_id"], keep="first")
    out = out.sort_values("published_utc").reset_index(drop=True)

    args.output_csv.parent.mkdir(parents=True, exist_ok=True)
    out.to_csv(args.output_csv, index=False)

    print(f"Merged rows: {len(out)}")
    print(f"Range: {out['published_utc'].min()} -> {out['published_utc'].max()}")
    print(f"Output: {args.output_csv}")


if __name__ == "__main__":
    main()
