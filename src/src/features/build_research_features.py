#!/usr/bin/env python3
"""Build daily research feature matrix from index and constituent files."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd


DEFAULT_INDEX_CSV = Path("data/raw_market/sp500_2006_2026.csv")
DEFAULT_MAPPING_CSV = Path("data/metadata/sp500_constituent_file_map.csv")
DEFAULT_OUTPUT_CSV = Path("data/processed/features/research_features_daily.csv")
DEFAULT_METADATA_JSON = Path("data/metadata/research_features_metadata.json")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Build research-grade daily feature matrix.")
    parser.add_argument("--index-csv", type=Path, default=DEFAULT_INDEX_CSV)
    parser.add_argument("--mapping-csv", type=Path, default=DEFAULT_MAPPING_CSV)
    parser.add_argument("--output-csv", type=Path, default=DEFAULT_OUTPUT_CSV)
    parser.add_argument("--metadata-json", type=Path, default=DEFAULT_METADATA_JSON)
    parser.add_argument("--max-symbols", type=int, default=500)
    return parser.parse_args()


def load_index(index_csv: Path) -> pd.DataFrame:
    if not index_csv.exists():
        raise FileNotFoundError(f"Index CSV not found: {index_csv}")

    df = pd.read_csv(index_csv)
    required = {"Date", "Open", "High", "Low", "Close", "Volume"}
    missing = sorted(required.difference(df.columns))
    if missing:
        raise ValueError(f"Index CSV missing columns: {missing}")

    df["Date"] = pd.to_datetime(df["Date"], errors="coerce")
    df = df.dropna(subset=["Date"]).sort_values("Date").reset_index(drop=True)
    df = df.drop_duplicates(subset=["Date"], keep="last")
    return df


def compute_g1_features(df: pd.DataFrame) -> pd.DataFrame:
    out = pd.DataFrame({"Date": df["Date"]})

    close = df["Close"].astype(float)
    open_ = df["Open"].astype(float)
    high = df["High"].astype(float)
    low = df["Low"].astype(float)
    volume = df["Volume"].astype(float)

    log_close = np.log(close)
    out["g1_ret_1d"] = log_close.diff(1)
    out["g1_ret_5d"] = log_close.diff(5)
    out["g1_ret_10d"] = log_close.diff(10)
    out["g1_intraday_log_ret"] = np.log(close / open_)
    out["g1_overnight_log_ret"] = np.log(open_ / close.shift(1))
    out["g1_hl_range"] = (high - low) / close.replace(0, np.nan)
    out["g1_volume_log"] = np.log1p(volume)
    out["g1_volume_chg_1d"] = np.log1p(volume).diff(1)

    # Targets: next 1-day and 5-day close-to-close log returns.
    out["target_h1"] = log_close.shift(-1) - log_close
    out["target_h5"] = log_close.shift(-5) - log_close
    return out


def compute_g2_technical(df: pd.DataFrame) -> pd.DataFrame:
    out = pd.DataFrame({"Date": df["Date"]})

    close = df["Close"].astype(float)
    high = df["High"].astype(float)
    low = df["Low"].astype(float)
    volume = df["Volume"].astype(float)

    log_close = np.log(close)
    ret_1 = log_close.diff(1)

    out["g2_realized_vol_10"] = ret_1.rolling(10, min_periods=10).std()
    out["g2_realized_vol_20"] = ret_1.rolling(20, min_periods=20).std()

    tr_components = pd.concat(
        [
            (high - low).abs(),
            (high - close.shift(1)).abs(),
            (low - close.shift(1)).abs(),
        ],
        axis=1,
    )
    true_range = tr_components.max(axis=1)
    out["g2_atr_14"] = true_range.rolling(14, min_periods=14).mean()

    delta = close.diff(1)
    gain = delta.clip(lower=0)
    loss = -delta.clip(upper=0)
    avg_gain = gain.rolling(14, min_periods=14).mean()
    avg_loss = loss.rolling(14, min_periods=14).mean()
    rs = avg_gain / avg_loss.replace(0, np.nan)
    out["g2_rsi_14"] = 100 - (100 / (1 + rs))

    ema_12 = close.ewm(span=12, adjust=False).mean()
    ema_26 = close.ewm(span=26, adjust=False).mean()
    macd = ema_12 - ema_26
    macd_signal = macd.ewm(span=9, adjust=False).mean()
    out["g2_macd"] = macd
    out["g2_macd_signal"] = macd_signal
    out["g2_macd_hist"] = macd - macd_signal

    ma20 = close.rolling(20, min_periods=20).mean()
    std20 = close.rolling(20, min_periods=20).std()
    out["g2_bollinger_z20"] = (close - ma20) / std20.replace(0, np.nan)

    log_vol = np.log1p(volume)
    out["g2_volume_z20"] = (log_vol - log_vol.rolling(20, min_periods=20).mean()) / (
        log_vol.rolling(20, min_periods=20).std().replace(0, np.nan)
    )
    return out


def compute_g3_breadth(index_dates: pd.Series, mapping_csv: Path, max_symbols: int) -> pd.DataFrame:
    if not mapping_csv.exists():
        raise FileNotFoundError(f"Mapping CSV not found: {mapping_csv}")

    mapping = pd.read_csv(mapping_csv)
    mapping = mapping[mapping["status"].astype(str).str.lower() == "ok"].copy()
    mapping = mapping.head(max_symbols)

    calendar = pd.DatetimeIndex(index_dates)

    close_frames = []
    ret_frames = []
    above50_frames = []
    above200_frames = []

    for _, row in mapping.iterrows():
        symbol = str(row["symbol"])
        path = Path(str(row["file_path"]))
        if not path.exists():
            continue

        try:
            d = pd.read_csv(path, usecols=["Date", "Close"])
        except Exception:
            continue

        d["Date"] = pd.to_datetime(d["Date"], errors="coerce")
        d = d.dropna(subset=["Date"]).sort_values("Date")
        d = d.drop_duplicates(subset=["Date"], keep="last")
        d = d.set_index("Date")
        close = d["Close"].astype(float).reindex(calendar)

        ret = np.log(close).diff(1)
        ma50 = close.rolling(50, min_periods=50).mean()
        ma200 = close.rolling(200, min_periods=200).mean()

        close_frames.append(close.rename(symbol))
        ret_frames.append(ret.rename(symbol))
        above50_frames.append((close > ma50).rename(symbol))
        above200_frames.append((close > ma200).rename(symbol))

    if not close_frames:
        raise RuntimeError("No valid constituent close series loaded for breadth features.")

    close_df = pd.concat(close_frames, axis=1)
    ret_df = pd.concat(ret_frames, axis=1)
    above50_df = pd.concat(above50_frames, axis=1)
    above200_df = pd.concat(above200_frames, axis=1)

    active = close_df.notna()
    active_count = active.sum(axis=1).replace(0, np.nan)

    pct_above50 = above50_df.where(active).sum(axis=1) / active_count
    pct_above200 = above200_df.where(active).sum(axis=1) / active_count

    adv = (ret_df > 0).where(active).sum(axis=1)
    dec = (ret_df < 0).where(active).sum(axis=1)
    adv_dec_spread = adv - dec
    adv_dec_ratio = adv / dec.replace(0, np.nan)
    ad_line = adv_dec_spread.cumsum()

    eqw_ret = ret_df.where(active).mean(axis=1)

    out = pd.DataFrame(
        {
            "Date": calendar,
            "g3_active_constituents": active_count,
            "g3_pct_above_50dma": pct_above50,
            "g3_pct_above_200dma": pct_above200,
            "g3_adv_dec_spread": adv_dec_spread,
            "g3_adv_dec_ratio": adv_dec_ratio,
            "g3_ad_line": ad_line,
            "g3_equal_weight_ret_1d": eqw_ret,
        }
    )
    return out


def merge_feature_blocks(blocks: list[pd.DataFrame]) -> pd.DataFrame:
    out = blocks[0].copy()
    if "Date" in out.index.names:
        out = out.reset_index(drop=True)
    for b in blocks[1:]:
        if "Date" in b.index.names:
            b = b.reset_index(drop=True)
        out = out.merge(b, on="Date", how="left")
    out = out.sort_values("Date").reset_index(drop=True)
    return out


def main() -> None:
    args = parse_args()

    index_df = load_index(args.index_csv)

    g1 = compute_g1_features(index_df)
    g2 = compute_g2_technical(index_df)
    g3 = compute_g3_breadth(index_df["Date"], args.mapping_csv, max_symbols=args.max_symbols)

    features = merge_feature_blocks([g1, g2, g3])

    # The experiment runner performs train-only imputation and scaling; we only drop rows without target.
    features = features.dropna(subset=["target_h1"]).reset_index(drop=True)

    args.output_csv.parent.mkdir(parents=True, exist_ok=True)
    features.to_csv(args.output_csv, index=False)

    meta = {
        "rows": int(len(features)),
        "date_min": features["Date"].min().strftime("%Y-%m-%d"),
        "date_max": features["Date"].max().strftime("%Y-%m-%d"),
        "feature_counts": {
            "g1": int(sum(c.startswith("g1_") for c in features.columns)),
            "g2": int(sum(c.startswith("g2_") for c in features.columns)),
            "g3": int(sum(c.startswith("g3_") for c in features.columns)),
        },
        "target_columns": ["target_h1", "target_h5"],
        "output_csv": str(args.output_csv),
    }

    args.metadata_json.parent.mkdir(parents=True, exist_ok=True)
    args.metadata_json.write_text(json.dumps(meta, indent=2), encoding="utf-8")

    print("Feature build complete.")
    print(f"Rows: {meta['rows']} | Range: {meta['date_min']} -> {meta['date_max']}")
    print(
        "Feature columns: "
        f"g1={meta['feature_counts']['g1']} "
        f"g2={meta['feature_counts']['g2']} "
        f"g3={meta['feature_counts']['g3']}"
    )
    print(f"Output: {args.output_csv}")


if __name__ == "__main__":
    main()
