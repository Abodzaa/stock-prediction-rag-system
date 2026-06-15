#!/usr/bin/env python3
"""Download S&P 500 (^GSPC) OHLCV data with validation and reproducibility metadata."""

from __future__ import annotations

import argparse
import hashlib
import json
import time
from datetime import datetime, timezone
from pathlib import Path

import pandas as pd
import yfinance as yf


DEFAULT_TICKER = "^GSPC"
DEFAULT_START = "2006-01-01"
DEFAULT_END = "2026-01-01"
DEFAULT_OUTPUT = Path("data/raw_market/sp500_2006_2026.csv")
DEFAULT_METADATA = Path("data/metadata/sp500_2006_2026_metadata.json")
DEFAULT_CHECKSUM = Path("data/metadata/sp500_2006_2026_sha256.txt")
CRITICAL_COLUMNS = ["Open", "High", "Low", "Close", "Volume"]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Download historical S&P 500 OHLCV data.")
    parser.add_argument("--ticker", default=DEFAULT_TICKER, help="Ticker symbol to download.")
    parser.add_argument("--start", default=DEFAULT_START, help="Start date (YYYY-MM-DD), inclusive.")
    parser.add_argument("--end", default=DEFAULT_END, help="End date (YYYY-MM-DD), inclusive target date.")
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT, help="Output CSV path.")
    parser.add_argument("--metadata", type=Path, default=DEFAULT_METADATA, help="Output metadata JSON path.")
    parser.add_argument("--checksum", type=Path, default=DEFAULT_CHECKSUM, help="Output checksum TXT path.")
    parser.add_argument("--max-retries", type=int, default=5, help="Max retries for download.")
    parser.add_argument("--base-wait-seconds", type=float, default=1.5, help="Base wait for backoff.")
    return parser.parse_args()


def _to_download_end(end_date: str) -> str:
    """yfinance end is exclusive for daily data; shift by one day to target inclusive end."""
    parsed = pd.Timestamp(end_date)
    return (parsed + pd.Timedelta(days=1)).strftime("%Y-%m-%d")


def download_with_retry(
    ticker: str,
    start_date: str,
    end_date: str,
    max_retries: int,
    base_wait_seconds: float,
) -> pd.DataFrame:
    download_end = _to_download_end(end_date)
    last_error: Exception | None = None

    for attempt in range(1, max_retries + 1):
        try:
            df = yf.download(
                ticker,
                start=start_date,
                end=download_end,
                interval="1d",
                auto_adjust=False,
                actions=True,
                progress=False,
                threads=False,
            )
            if df is None or df.empty:
                raise RuntimeError("Downloaded dataframe is empty.")
            return df
        except Exception as exc:
            last_error = exc
            if attempt == max_retries:
                break
            wait_seconds = base_wait_seconds * (2 ** (attempt - 1))
            print(f"Attempt {attempt} failed: {exc}. Retrying in {wait_seconds:.1f}s...")
            time.sleep(wait_seconds)

    raise RuntimeError(f"Failed to download data after {max_retries} attempts.") from last_error


def normalize_columns(df: pd.DataFrame) -> pd.DataFrame:
    if isinstance(df.columns, pd.MultiIndex):
        df.columns = [col[0] for col in df.columns]
    return df


def validate_dataframe(df: pd.DataFrame, start_date: str, end_date: str) -> None:
    missing_cols = [col for col in CRITICAL_COLUMNS if col not in df.columns]
    if missing_cols:
        raise ValueError(f"Missing critical columns: {missing_cols}")

    if not df.index.is_monotonic_increasing:
        raise ValueError("Date index is not monotonic increasing.")

    duplicated = int(df.index.duplicated().sum())
    if duplicated > 0:
        raise ValueError(f"Found {duplicated} duplicate timestamps.")

    for col in CRITICAL_COLUMNS:
        if df[col].isna().any():
            missing = int(df[col].isna().sum())
            raise ValueError(f"Column '{col}' has {missing} missing values.")

    start_ts = pd.Timestamp(start_date)
    end_ts = pd.Timestamp(end_date)
    min_ts = pd.Timestamp(df.index.min())
    max_ts = pd.Timestamp(df.index.max())

    if min_ts > start_ts + pd.Timedelta(days=7):
        raise ValueError(
            f"Coverage issue: earliest row {min_ts.date()} is too late for start {start_ts.date()}."
        )

    if max_ts > end_ts:
        raise ValueError(
            f"Coverage issue: latest row {max_ts.date()} is after requested end {end_ts.date()}."
        )


def write_outputs(
    df: pd.DataFrame,
    ticker: str,
    start_date: str,
    end_date: str,
    output_path: Path,
    metadata_path: Path,
    checksum_path: Path,
) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    metadata_path.parent.mkdir(parents=True, exist_ok=True)
    checksum_path.parent.mkdir(parents=True, exist_ok=True)

    df.to_csv(output_path, index=True)

    checksum = hashlib.sha256(output_path.read_bytes()).hexdigest()
    checksum_path.write_text(f"{checksum}  {output_path.name}\n", encoding="utf-8")

    metadata = {
        "source": "yfinance",
        "ticker": ticker,
        "downloaded_at_utc": datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
        "requested_start": start_date,
        "requested_end": end_date,
        "row_count": int(len(df)),
        "date_min": pd.Timestamp(df.index.min()).strftime("%Y-%m-%d"),
        "date_max": pd.Timestamp(df.index.max()).strftime("%Y-%m-%d"),
        "columns": list(df.columns),
        "csv_sha256": checksum,
        "output_csv": str(output_path),
    }
    metadata_path.write_text(json.dumps(metadata, indent=2), encoding="utf-8")


def main() -> None:
    args = parse_args()
    print(
        f"Downloading {args.ticker} from {args.start} to {args.end} "
        f"into {args.output}..."
    )

    df = download_with_retry(
        ticker=args.ticker,
        start_date=args.start,
        end_date=args.end,
        max_retries=args.max_retries,
        base_wait_seconds=args.base_wait_seconds,
    )

    df = normalize_columns(df)
    df = df.sort_index()
    if df.index.tz is not None:
        df.index = df.index.tz_localize(None)
    df.index.name = "Date"

    validate_dataframe(df, start_date=args.start, end_date=args.end)

    write_outputs(
        df=df,
        ticker=args.ticker,
        start_date=args.start,
        end_date=args.end,
        output_path=args.output,
        metadata_path=args.metadata,
        checksum_path=args.checksum,
    )

    print("Download and validation complete.")
    print(f"Rows: {len(df)} | Range: {df.index.min().date()} -> {df.index.max().date()}")
    print(f"CSV: {args.output}")
    print(f"Metadata: {args.metadata}")
    print(f"Checksum: {args.checksum}")


if __name__ == "__main__":
    main()
