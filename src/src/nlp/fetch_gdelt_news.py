#!/usr/bin/env python3
"""Fetch historical finance news from GDELT Doc API (no API key required)."""

from __future__ import annotations

import argparse
import time
from datetime import timedelta
from pathlib import Path

import pandas as pd
import requests


DEFAULT_OUTPUT_CSV = Path("data/raw_news/gdelt_market_news_2015_2026.csv")
GDELT_URL = "https://api.gdeltproject.org/api/v2/doc/doc"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Fetch historical market news from GDELT API.")
    parser.add_argument("--output-csv", type=Path, default=DEFAULT_OUTPUT_CSV)
    parser.add_argument("--start-date", default="2015-01-01")
    parser.add_argument("--end-date", default="2026-04-11")
    parser.add_argument("--step-days", type=int, default=7)
    parser.add_argument("--max-records", type=int, default=250)
    parser.add_argument("--sleep-seconds", type=float, default=0.25)
    parser.add_argument("--max-retries", type=int, default=6)
    parser.add_argument("--backoff-base", type=float, default=1.5)
    parser.add_argument("--resume-from", default="")
    parser.add_argument(
        "--query",
        default='("S&P 500" OR "stock market" OR "earnings" OR "federal reserve" OR "inflation")',
    )
    return parser.parse_args()


def to_gdelt_dt(ts: pd.Timestamp) -> str:
    return ts.strftime("%Y%m%d%H%M%S")


def parse_seendate(s: str) -> pd.Timestamp:
    # Format from GDELT is commonly YYYYMMDDTHHMMSSZ.
    return pd.to_datetime(s, format="%Y%m%dT%H%M%SZ", utc=True, errors="coerce")


def fetch_window(
    query: str,
    start_ts: pd.Timestamp,
    end_ts: pd.Timestamp,
    max_records: int,
    max_retries: int,
    backoff_base: float,
) -> list[dict]:
    params = {
        "query": query,
        "mode": "artlist",
        "format": "json",
        "maxrecords": max_records,
        "startdatetime": to_gdelt_dt(start_ts),
        "enddatetime": to_gdelt_dt(end_ts),
        "sort": "DateAsc",
    }
    last_error: Exception | None = None
    for attempt in range(1, max_retries + 1):
        try:
            resp = requests.get(GDELT_URL, params=params, timeout=60)
            resp.raise_for_status()
            payload = resp.json()
            return payload.get("articles", []) if isinstance(payload, dict) else []
        except Exception as exc:
            last_error = exc
            if attempt == max_retries:
                break
            wait = backoff_base * (2 ** (attempt - 1))
            time.sleep(wait)

    raise RuntimeError(
        f"Failed window {start_ts} -> {end_ts} after {max_retries} retries: {last_error}"
    )


def article_to_row(item: dict) -> dict:
    title = str(item.get("title") or "")
    url = str(item.get("url") or "")
    source = str(item.get("domain") or item.get("sourcecountry") or "gdelt")
    lang = str(item.get("language") or "")
    seendate = str(item.get("seendate") or "")
    published_utc = parse_seendate(seendate)

    fallback_id = f"{published_utc}|{url}|{title[:120]}"
    return {
        "symbol": "SP500",
        "news_id": fallback_id,
        "published_utc": published_utc,
        "headline": title,
        "summary": str(item.get("socialimage") or ""),
        "source": source,
        "url": url,
        "language": lang,
    }


def main() -> None:
    args = parse_args()

    start = pd.Timestamp(args.start_date, tz="UTC")
    if args.resume_from:
        start = pd.Timestamp(args.resume_from, tz="UTC")
    end = pd.Timestamp(args.end_date, tz="UTC") + pd.Timedelta(days=1)
    if end <= start:
        raise ValueError("end-date must be after start-date")

    rows: list[dict] = []
    cursor = start
    step = timedelta(days=max(1, args.step_days))

    total_windows = 0
    while cursor < end:
        nxt = min(cursor + step, end)
        total_windows += 1
        try:
            arts = fetch_window(
                query=args.query,
                start_ts=cursor,
                end_ts=nxt,
                max_records=args.max_records,
                max_retries=args.max_retries,
                backoff_base=args.backoff_base,
            )
            for a in arts:
                rows.append(article_to_row(a))
        except Exception as exc:
            print(f"window failed {cursor} -> {nxt}: {exc}")

        cursor = nxt
        if args.sleep_seconds > 0:
            time.sleep(args.sleep_seconds)

        if total_windows % 20 == 0:
            print(f"processed windows={total_windows}, accumulated_rows={len(rows)}")

    if not rows:
        raise RuntimeError("No GDELT news rows fetched.")

    out = pd.DataFrame(rows)
    out = out.dropna(subset=["published_utc"]).copy()
    out["published_utc"] = pd.to_datetime(out["published_utc"], utc=True, errors="coerce")
    out = out.dropna(subset=["published_utc"]).copy()

    out = out.drop_duplicates(subset=["news_id", "url", "headline"], keep="first")
    out = out.sort_values("published_utc").reset_index(drop=True)

    args.output_csv.parent.mkdir(parents=True, exist_ok=True)
    out.to_csv(args.output_csv, index=False)

    print(f"GDELT rows: {len(out)}")
    print(f"Range: {out['published_utc'].min()} -> {out['published_utc'].max()}")
    print(f"Output: {args.output_csv}")


if __name__ == "__main__":
    main()
