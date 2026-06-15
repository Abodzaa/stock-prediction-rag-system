#!/usr/bin/env python3
"""Fetch market news from Yahoo Finance endpoints for S&P 500 symbols."""

from __future__ import annotations

import argparse
import time
from pathlib import Path

import pandas as pd
import yfinance as yf


DEFAULT_MAPPING_CSV = Path("data/metadata/sp500_constituent_file_map.csv")
DEFAULT_OUTPUT_CSV = Path("data/raw_news/yfinance_sp500_news.csv")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Fetch Yahoo Finance news for S&P 500 symbols.")
    parser.add_argument("--mapping-csv", type=Path, default=DEFAULT_MAPPING_CSV)
    parser.add_argument("--output-csv", type=Path, default=DEFAULT_OUTPUT_CSV)
    parser.add_argument("--max-symbols", type=int, default=80)
    parser.add_argument("--per-symbol-limit", type=int, default=30)
    parser.add_argument("--sleep-seconds", type=float, default=0.1)
    parser.add_argument("--include-index", action="store_true")
    return parser.parse_args()


def safe_get_news(symbol: str) -> list[dict]:
    try:
        ticker = yf.Ticker(symbol)
        items = ticker.news
        if isinstance(items, list):
            return items
        return []
    except Exception:
        return []


def item_to_row(symbol: str, item: dict) -> dict:
    content = item.get("content") if isinstance(item.get("content"), dict) else {}

    title = content.get("title") or item.get("title") or ""
    summary = content.get("summary") or content.get("description") or item.get("summary") or ""

    provider = content.get("provider") if isinstance(content.get("provider"), dict) else {}
    source = (
        provider.get("displayName")
        or content.get("providerDisplayName")
        or item.get("publisher")
        or item.get("provider")
        or "unknown"
    )

    canonical = content.get("canonicalUrl") if isinstance(content.get("canonicalUrl"), dict) else {}
    clickthrough = content.get("clickThroughUrl") if isinstance(content.get("clickThroughUrl"), dict) else {}
    link = (
        canonical.get("url")
        or clickthrough.get("url")
        or item.get("link")
        or ""
    )

    uuid = (
        content.get("id")
        or item.get("uuid")
        or item.get("id")
        or ""
    )

    ts_raw = item.get("providerPublishTime")
    if ts_raw is not None:
        published_utc = pd.to_datetime(ts_raw, unit="s", utc=True, errors="coerce")
    else:
        published_utc = pd.to_datetime(
            content.get("pubDate") or content.get("displayTime") or item.get("published"),
            utc=True,
            errors="coerce",
        )

    return {
        "symbol": symbol,
        "news_id": str(uuid),
        "published_utc": published_utc,
        "headline": str(title),
        "summary": str(summary),
        "source": str(source),
        "url": str(link),
    }


def main() -> None:
    args = parse_args()

    if not args.mapping_csv.exists():
        raise FileNotFoundError(f"Mapping CSV not found: {args.mapping_csv}")

    mapping = pd.read_csv(args.mapping_csv)
    mapping = mapping[mapping["status"].astype(str).str.lower() == "ok"].copy()
    symbols = mapping["symbol"].astype(str).dropna().tolist()[: args.max_symbols]
    if args.include_index:
        symbols = ["^GSPC"] + symbols

    rows: list[dict] = []
    for i, symbol in enumerate(symbols, start=1):
        items = safe_get_news(symbol)
        kept = 0
        for item in items:
            rows.append(item_to_row(symbol=symbol, item=item))
            kept += 1
            if kept >= args.per_symbol_limit:
                break

        if args.sleep_seconds > 0:
            time.sleep(args.sleep_seconds)

        if i % 10 == 0:
            print(f"Processed {i}/{len(symbols)} symbols; accumulated rows={len(rows)}")

    if not rows:
        raise RuntimeError("No news rows were fetched from Yahoo Finance.")

    out = pd.DataFrame(rows)
    out = out.dropna(subset=["published_utc"]).copy()
    out["published_utc"] = pd.to_datetime(out["published_utc"], utc=True, errors="coerce")
    out = out.dropna(subset=["published_utc"]).copy()

    # Dedupe by source identifier or stable URL/timestamp fallback.
    fallback_id = (
        out["symbol"].astype(str)
        + "|"
        + out["published_utc"].dt.strftime("%Y%m%d%H%M%S")
        + "|"
        + out["url"].astype(str)
        + "|"
        + out["headline"].astype(str).str.slice(0, 120)
    )
    out["news_id"] = out["news_id"].astype(str).mask(out["news_id"].astype(str).str.len() == 0, fallback_id)
    out = out.drop_duplicates(subset=["news_id"], keep="first")

    out = out.sort_values("published_utc").reset_index(drop=True)
    args.output_csv.parent.mkdir(parents=True, exist_ok=True)
    out.to_csv(args.output_csv, index=False)

    print(f"Fetched news rows: {len(out)}")
    print(f"Date range: {out['published_utc'].min()} -> {out['published_utc'].max()}")
    print(f"Output: {args.output_csv}")


if __name__ == "__main__":
    main()
