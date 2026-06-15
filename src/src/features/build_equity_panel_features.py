#!/usr/bin/env python3
"""Build panel (Date, Symbol) features from raw_equities constituent files."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd


DEFAULT_MAPPING_CSV = Path("data/metadata/sp500_constituent_file_map.csv")
DEFAULT_OUTPUT_CSV = Path("data/processed/features/research_features_panel.csv")
DEFAULT_METADATA_JSON = Path("data/metadata/research_features_panel_metadata.json")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Build panel features from raw_equities constituent files.")
    parser.add_argument("--mapping-csv", type=Path, default=DEFAULT_MAPPING_CSV)
    parser.add_argument("--output-csv", type=Path, default=DEFAULT_OUTPUT_CSV)
    parser.add_argument("--metadata-json", type=Path, default=DEFAULT_METADATA_JSON)
    parser.add_argument("--max-symbols", type=int, default=500)
    parser.add_argument("--winsor-lower", type=float, default=0.005)
    parser.add_argument("--winsor-upper", type=float, default=0.995)
    parser.add_argument("--min-history", type=int, default=260)
    return parser.parse_args()


def winsorize_series(s: pd.Series, low_q: float, high_q: float) -> pd.Series:
    if s.notna().sum() < 20:
        return s
    lo = s.quantile(low_q)
    hi = s.quantile(high_q)
    return s.clip(lower=lo, upper=hi)


def compute_symbol_features(df: pd.DataFrame) -> pd.DataFrame:
    out = pd.DataFrame(index=df.index)

    close = df["Close"].astype(float)
    open_ = df["Open"].astype(float)
    high = df["High"].astype(float)
    low = df["Low"].astype(float)
    volume = df["Volume"].astype(float)

    log_close = np.log(close.replace(0, np.nan))

    # G1: price/return fundamentals.
    out["g1_ret_1d"] = log_close.diff(1)
    out["g1_ret_5d"] = log_close.diff(5)
    out["g1_ret_10d"] = log_close.diff(10)
    out["g1_intraday_log_ret"] = np.log((close / open_).replace(0, np.nan))
    out["g1_overnight_log_ret"] = np.log((open_ / close.shift(1)).replace(0, np.nan))
    out["g1_hl_range"] = (high - low) / close.replace(0, np.nan)
    out["g1_volume_log"] = np.log1p(volume)
    out["g1_volume_chg_1d"] = np.log1p(volume).diff(1)

    # G2: technical indicators.
    ret_1 = out["g1_ret_1d"]
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

    # Targets per symbol.
    out["target_h1"] = log_close.shift(-1) - log_close
    out["target_h5"] = log_close.shift(-5) - log_close

    return out


def add_cross_sectional_features(panel: pd.DataFrame) -> pd.DataFrame:
    out = panel.copy()

    # G3 cross-sectional daily normalization features from the panel itself.
    for col in ["g1_ret_1d", "g2_realized_vol_20", "g1_volume_chg_1d"]:
        grp = out.groupby("Date")[col]
        mean = grp.transform("mean")
        std = grp.transform("std").replace(0, np.nan)
        out[f"g3_cs_{col}_z"] = (out[col] - mean) / std

    # Daily market breadth signal from panel (shared by all symbols on a day).
    adv = (out["g1_ret_1d"] > 0).groupby(out["Date"]).transform("sum")
    dec = (out["g1_ret_1d"] < 0).groupby(out["Date"]).transform("sum")
    out["g3_adv_dec_spread"] = adv - dec
    out["g3_adv_dec_ratio"] = adv / dec.replace(0, np.nan)

    return out


def main() -> None:
    args = parse_args()

    if not args.mapping_csv.exists():
        raise FileNotFoundError(f"Mapping CSV not found: {args.mapping_csv}")

    mapping = pd.read_csv(args.mapping_csv)
    mapping = mapping[mapping["status"].astype(str).str.lower() == "ok"].copy()
    mapping = mapping.head(args.max_symbols)

    frames: list[pd.DataFrame] = []
    per_symbol_rows = []

    for _, row in mapping.iterrows():
        symbol = str(row["symbol"])
        path = Path(str(row["file_path"]))
        if not path.exists():
            continue

        d = pd.read_csv(path, usecols=["Date", "Open", "High", "Low", "Close", "Volume"])
        d["Date"] = pd.to_datetime(d["Date"], errors="coerce")
        d = d.dropna(subset=["Date"]).sort_values("Date").drop_duplicates(subset=["Date"], keep="last")

        if len(d) < args.min_history:
            continue

        d = d.set_index("Date")
        feats = compute_symbol_features(d)
        feats = feats.reset_index().rename(columns={"index": "Date"})
        feats["Symbol"] = symbol

        # Denoising step 1: symbol-level winsorization for heavy-tail returns/targets.
        for col in ["g1_ret_1d", "g1_ret_5d", "g1_ret_10d", "target_h1", "target_h5"]:
            feats[col] = winsorize_series(feats[col], args.winsor_lower, args.winsor_upper)

        # Denoising step 2: soft clipping on extreme range/volume spikes.
        for col in ["g1_hl_range", "g2_volume_z20", "g2_bollinger_z20"]:
            feats[col] = feats[col].clip(lower=-10.0, upper=10.0)

        frames.append(feats)
        per_symbol_rows.append((symbol, len(feats)))

    if not frames:
        raise RuntimeError("No symbol files produced valid feature frames.")

    panel = pd.concat(frames, ignore_index=True)
    panel = panel.sort_values(["Date", "Symbol"]).reset_index(drop=True)
    panel = add_cross_sectional_features(panel)

    # Keep rows that still have valid targets; imputation/scaling remains train-only in experiment runner.
    panel = panel.dropna(subset=["target_h1", "target_h5"]).reset_index(drop=True)

    args.output_csv.parent.mkdir(parents=True, exist_ok=True)
    panel.to_csv(args.output_csv, index=False)

    symbol_counts = pd.DataFrame(per_symbol_rows, columns=["symbol", "rows"]) if per_symbol_rows else pd.DataFrame()
    meta = {
        "rows": int(len(panel)),
        "symbols": int(panel["Symbol"].nunique()),
        "date_min": panel["Date"].min().strftime("%Y-%m-%d"),
        "date_max": panel["Date"].max().strftime("%Y-%m-%d"),
        "feature_counts": {
            "g1": int(sum(c.startswith("g1_") for c in panel.columns)),
            "g2": int(sum(c.startswith("g2_") for c in panel.columns)),
            "g3": int(sum(c.startswith("g3_") for c in panel.columns)),
        },
        "targets": ["target_h1", "target_h5"],
        "preprocessing": {
            "denoising": [
                "symbol-level winsorization on returns and targets",
                "soft clipping of extreme range/volume-derived indicators",
            ],
            "normalization": [
                "cross-sectional z-score features per day (g3_cs_*)",
                "train-only scaling in experiment pipelines (RobustScaler/StandardScaler)",
            ],
        },
        "output_csv": str(args.output_csv),
    }

    if not symbol_counts.empty:
        meta["median_rows_per_symbol"] = float(symbol_counts["rows"].median())
        meta["min_rows_per_symbol"] = int(symbol_counts["rows"].min())
        meta["max_rows_per_symbol"] = int(symbol_counts["rows"].max())

    args.metadata_json.parent.mkdir(parents=True, exist_ok=True)
    args.metadata_json.write_text(json.dumps(meta, indent=2), encoding="utf-8")

    print("Panel feature build complete.")
    print(f"Rows: {meta['rows']} | Symbols: {meta['symbols']}")
    print(f"Range: {meta['date_min']} -> {meta['date_max']}")
    print(f"Output: {args.output_csv}")


if __name__ == "__main__":
    main()
