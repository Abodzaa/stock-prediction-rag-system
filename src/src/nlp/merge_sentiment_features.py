#!/usr/bin/env python3
"""Merge daily sentiment features into the main research feature matrix."""

from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd


DEFAULT_BASE_FEATURES = Path("data/processed/features/research_features_daily.csv")
DEFAULT_SENTIMENT_DAILY = Path("data/processed/features/news_sentiment_finbert_daily.csv")
DEFAULT_OUTPUT_FEATURES = Path("data/processed/features/research_features_with_sentiment.csv")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Merge sentiment daily features into base features.")
    parser.add_argument("--base-features", type=Path, default=DEFAULT_BASE_FEATURES)
    parser.add_argument("--sentiment-daily", type=Path, default=DEFAULT_SENTIMENT_DAILY)
    parser.add_argument("--output-features", type=Path, default=DEFAULT_OUTPUT_FEATURES)
    parser.add_argument("--fillna", type=float, default=0.0)
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    if not args.base_features.exists():
        raise FileNotFoundError(f"Base features CSV not found: {args.base_features}")
    if not args.sentiment_daily.exists():
        raise FileNotFoundError(f"Sentiment daily CSV not found: {args.sentiment_daily}")

    base = pd.read_csv(args.base_features)
    sent = pd.read_csv(args.sentiment_daily)

    base["Date"] = pd.to_datetime(base["Date"], errors="coerce")
    sent["Date"] = pd.to_datetime(sent["Date"], errors="coerce")

    base = base.dropna(subset=["Date"]).sort_values("Date")
    sent = sent.dropna(subset=["Date"]).sort_values("Date")

    sent_cols = [c for c in sent.columns if c != "Date"]
    merged = base.merge(sent, on="Date", how="left")

    if args.fillna is not None:
        merged[sent_cols] = merged[sent_cols].fillna(float(args.fillna))

    merged = merged.sort_values("Date").reset_index(drop=True)
    args.output_features.parent.mkdir(parents=True, exist_ok=True)
    merged.to_csv(args.output_features, index=False)

    print(f"Base rows: {len(base)}")
    print(f"Sentiment rows: {len(sent)}")
    print(f"Merged rows: {len(merged)}")
    print(f"Merged sentiment columns: {len(sent_cols)}")
    print(f"Output: {args.output_features}")


if __name__ == "__main__":
    main()
