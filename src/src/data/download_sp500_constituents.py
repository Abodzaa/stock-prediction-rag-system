#!/usr/bin/env python3
"""Download one CSV per S&P 500 constituent and create a mapping manifest CSV."""

from __future__ import annotations

import argparse
import concurrent.futures
import hashlib
import time
from io import StringIO
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import pandas as pd
import requests
import yfinance as yf


DEFAULT_START = "2006-01-01"
DEFAULT_END = "2026-01-01"
DEFAULT_TARGET_COUNT = 500
DEFAULT_OUTPUT_DIR = Path("data/raw_equities/sp500_constituents")
DEFAULT_MAPPING_CSV = Path("data/metadata/sp500_constituent_file_map.csv")
DEFAULT_CONSTITUENTS_SNAPSHOT = Path("data/metadata/sp500_constituents_snapshot.csv")
WIKI_SP500_URL = "https://en.wikipedia.org/wiki/List_of_S%26P_500_companies"
CRITICAL_COLUMNS = ["Open", "High", "Low", "Close", "Volume"]


@dataclass
class DownloadResult:
    symbol: str
    yahoo_symbol: str
    status: str
    file_path: str
    row_count: int
    date_min: str
    date_max: str
    sha256: str
    error: str


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Download S&P 500 constituent OHLCV into one CSV per stock."
    )
    parser.add_argument("--start", default=DEFAULT_START, help="Start date YYYY-MM-DD inclusive.")
    parser.add_argument("--end", default=DEFAULT_END, help="End date YYYY-MM-DD inclusive target date.")
    parser.add_argument(
        "--target-count",
        type=int,
        default=DEFAULT_TARGET_COUNT,
        help="Number of constituents to process from the current S&P table.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=DEFAULT_OUTPUT_DIR,
        help="Directory to save per-ticker CSV files.",
    )
    parser.add_argument(
        "--mapping-csv",
        type=Path,
        default=DEFAULT_MAPPING_CSV,
        help="Manifest CSV with ticker-to-file mapping and stats.",
    )
    parser.add_argument(
        "--constituents-snapshot",
        type=Path,
        default=DEFAULT_CONSTITUENTS_SNAPSHOT,
        help="CSV snapshot of constituents table used for this run.",
    )
    parser.add_argument("--max-workers", type=int, default=5, help="Concurrent download workers.")
    parser.add_argument("--max-retries", type=int, default=4, help="Retries per ticker.")
    parser.add_argument(
        "--base-wait-seconds",
        type=float,
        default=1.0,
        help="Base exponential backoff wait in seconds.",
    )
    return parser.parse_args()


def _to_download_end(end_date: str) -> str:
    parsed = pd.Timestamp(end_date)
    return (parsed + pd.Timedelta(days=1)).strftime("%Y-%m-%d")


def normalize_yahoo_symbol(symbol: str) -> str:
    # Yahoo uses '-' for classes like BRK.B -> BRK-B
    return symbol.replace(".", "-").strip()


def fetch_constituents(target_count: int) -> pd.DataFrame:
    headers = {
        "User-Agent": (
            "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) "
            "AppleWebKit/537.36 (KHTML, like Gecko) "
            "Chrome/125.0.0.0 Safari/537.36"
        )
    }
    response = requests.get(WIKI_SP500_URL, headers=headers, timeout=30)
    response.raise_for_status()

    tables = pd.read_html(StringIO(response.text))
    candidates = [t for t in tables if "Symbol" in t.columns and "Security" in t.columns]
    if not candidates:
        raise RuntimeError("Could not locate S&P 500 constituents table from Wikipedia.")

    df = candidates[0].copy()
    df["Symbol"] = df["Symbol"].astype(str).str.strip()
    df["YahooSymbol"] = df["Symbol"].map(normalize_yahoo_symbol)
    df = df.drop_duplicates(subset=["Symbol"]).reset_index(drop=True)

    if target_count <= 0:
        raise ValueError("target-count must be a positive integer.")

    if len(df) < target_count:
        raise ValueError(
            f"Constituents table has only {len(df)} unique symbols, cannot satisfy target-count={target_count}."
        )

    return df.head(target_count).copy()


def normalize_columns(df: pd.DataFrame) -> pd.DataFrame:
    if isinstance(df.columns, pd.MultiIndex):
        df.columns = [col[0] for col in df.columns]
    return df


def validate_price_frame(df: pd.DataFrame, start_date: str, end_date: str) -> None:
    if df.empty:
        raise ValueError("Downloaded data is empty.")

    missing_cols = [col for col in CRITICAL_COLUMNS if col not in df.columns]
    if missing_cols:
        raise ValueError(f"Missing required columns: {missing_cols}")

    if not df.index.is_monotonic_increasing:
        raise ValueError("Date index is not monotonic increasing.")

    dupes = int(df.index.duplicated().sum())
    if dupes > 0:
        raise ValueError(f"Found {dupes} duplicate timestamps.")

    start_ts = pd.Timestamp(start_date)
    end_ts = pd.Timestamp(end_date)
    min_ts = pd.Timestamp(df.index.min())
    max_ts = pd.Timestamp(df.index.max())

    if max_ts > end_ts:
        raise ValueError(f"Latest date {max_ts.date()} exceeds requested end {end_ts.date()}.")

    # Do not require full 2006 coverage for later-listed companies; just enforce a plausible lower bound.
    if min_ts > end_ts:
        raise ValueError(
            f"Earliest date {min_ts.date()} is after end date {end_ts.date()}, invalid series."
        )

    if min_ts < start_ts - pd.Timedelta(days=5):
        raise ValueError(
            f"Earliest date {min_ts.date()} is unexpectedly earlier than requested start {start_ts.date()}."
        )


def download_one_symbol(
    symbol: str,
    yahoo_symbol: str,
    start_date: str,
    end_date: str,
    output_dir: Path,
    max_retries: int,
    base_wait_seconds: float,
) -> DownloadResult:
    download_end = _to_download_end(end_date)
    output_file = output_dir / f"{yahoo_symbol}.csv"

    last_error: Exception | None = None
    for attempt in range(1, max_retries + 1):
        try:
            df = yf.download(
                yahoo_symbol,
                start=start_date,
                end=download_end,
                interval="1d",
                auto_adjust=False,
                actions=True,
                progress=False,
                threads=False,
            )
            df = normalize_columns(df)
            if df is None or df.empty:
                raise RuntimeError("Empty dataframe returned from yfinance.")

            df = df.sort_index()
            if df.index.tz is not None:
                df.index = df.index.tz_localize(None)
            df.index.name = "Date"
            validate_price_frame(df, start_date=start_date, end_date=end_date)

            output_file.parent.mkdir(parents=True, exist_ok=True)
            df.to_csv(output_file, index=True)
            sha256 = hashlib.sha256(output_file.read_bytes()).hexdigest()

            return DownloadResult(
                symbol=symbol,
                yahoo_symbol=yahoo_symbol,
                status="ok",
                file_path=str(output_file),
                row_count=int(len(df)),
                date_min=pd.Timestamp(df.index.min()).strftime("%Y-%m-%d"),
                date_max=pd.Timestamp(df.index.max()).strftime("%Y-%m-%d"),
                sha256=sha256,
                error="",
            )
        except Exception as exc:
            last_error = exc
            if attempt < max_retries:
                sleep_seconds = base_wait_seconds * (2 ** (attempt - 1))
                time.sleep(sleep_seconds)

    return DownloadResult(
        symbol=symbol,
        yahoo_symbol=yahoo_symbol,
        status="error",
        file_path="",
        row_count=0,
        date_min="",
        date_max="",
        sha256="",
        error=str(last_error) if last_error is not None else "unknown error",
    )


def build_mapping_row(constituent_row: pd.Series, result: DownloadResult) -> dict[str, Any]:
    mapping = {
        "symbol": result.symbol,
        "yahoo_symbol": result.yahoo_symbol,
        "security": constituent_row.get("Security", ""),
        "gics_sector": constituent_row.get("GICS Sector", ""),
        "gics_sub_industry": constituent_row.get("GICS Sub-Industry", ""),
        "headquarters_location": constituent_row.get("Headquarters Location", ""),
        "date_added": constituent_row.get("Date added", ""),
        "cik": constituent_row.get("CIK", ""),
        "founded": constituent_row.get("Founded", ""),
        "status": result.status,
        "file_path": result.file_path,
        "row_count": result.row_count,
        "date_min": result.date_min,
        "date_max": result.date_max,
        "sha256": result.sha256,
        "error": result.error,
    }
    return mapping


def main() -> None:
    args = parse_args()

    print(f"Loading S&P constituents table and selecting first {args.target_count} symbols...")
    constituents = fetch_constituents(target_count=args.target_count)

    args.output_dir.mkdir(parents=True, exist_ok=True)
    args.mapping_csv.parent.mkdir(parents=True, exist_ok=True)
    args.constituents_snapshot.parent.mkdir(parents=True, exist_ok=True)

    constituents.to_csv(args.constituents_snapshot, index=False)

    print(
        f"Downloading {len(constituents)} symbols from {args.start} to {args.end} "
        f"into {args.output_dir} with {args.max_workers} workers..."
    )

    rows = constituents.to_dict(orient="records")
    results: dict[str, DownloadResult] = {}

    with concurrent.futures.ThreadPoolExecutor(max_workers=args.max_workers) as executor:
        futures = {
            executor.submit(
                download_one_symbol,
                row["Symbol"],
                row["YahooSymbol"],
                args.start,
                args.end,
                args.output_dir,
                args.max_retries,
                args.base_wait_seconds,
            ): row["Symbol"]
            for row in rows
        }

        completed = 0
        total = len(futures)
        for future in concurrent.futures.as_completed(futures):
            symbol = futures[future]
            result = future.result()
            results[symbol] = result
            completed += 1
            if completed % 25 == 0 or completed == total:
                ok_count = sum(1 for r in results.values() if r.status == "ok")
                err_count = completed - ok_count
                print(f"Progress: {completed}/{total} complete | ok={ok_count} error={err_count}")

    mapping_rows: list[dict[str, Any]] = []
    for _, row in constituents.iterrows():
        result = results[row["Symbol"]]
        mapping_rows.append(build_mapping_row(row, result))

    mapping_df = pd.DataFrame(mapping_rows)
    mapping_df.to_csv(args.mapping_csv, index=False)

    ok_count = int((mapping_df["status"] == "ok").sum())
    err_count = int((mapping_df["status"] == "error").sum())
    print("Download job finished.")
    print(f"Constituents requested: {len(constituents)}")
    print(f"Successful files: {ok_count}")
    print(f"Failed files: {err_count}")
    print(f"Per-stock CSV folder: {args.output_dir}")
    print(f"Mapping CSV: {args.mapping_csv}")
    print(f"Constituents snapshot: {args.constituents_snapshot}")


if __name__ == "__main__":
    main()
