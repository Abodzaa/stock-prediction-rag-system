#!/usr/bin/env python3
"""Shared utilities for panel preprocessing variants."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd

ID_COLS = {"Date", "Symbol"}
TARGET_COLS = {"target_h1", "target_h5"}


def load_panel(input_csv: Path) -> pd.DataFrame:
    if not input_csv.exists():
        raise FileNotFoundError(f"Input CSV not found: {input_csv}")

    df = pd.read_csv(input_csv)
    required = {"Date", "Symbol", "target_h1", "target_h5"}
    missing = sorted(required.difference(df.columns))
    if missing:
        raise ValueError(f"Missing required columns: {missing}")

    df["Date"] = pd.to_datetime(df["Date"], errors="coerce")
    df = df.dropna(subset=["Date", "Symbol"]).sort_values(["Date", "Symbol"]).reset_index(drop=True)
    return df


def feature_columns(df: pd.DataFrame) -> list[str]:
    cols: list[str] = []
    for col in df.columns:
        if col in ID_COLS or col in TARGET_COLS:
            continue
        if col.startswith(("g1_", "g2_", "g3_", "g5_sent_")):
            cols.append(col)
    return cols


def _grouped(df: pd.DataFrame, col: str) -> pd.core.groupby.SeriesGroupBy:
    return df.groupby("Symbol", sort=False)[col]


def winsorize_by_symbol(df: pd.DataFrame, cols: list[str], low_q: float, high_q: float) -> pd.DataFrame:
    out = df.copy()
    for col in cols:
        g = _grouped(out, col)
        lo = g.transform(lambda s: s.quantile(low_q))
        hi = g.transform(lambda s: s.quantile(high_q))
        out[col] = out[col].clip(lower=lo, upper=hi)
    return out


def robust_zscore_by_symbol(df: pd.DataFrame, cols: list[str], eps: float = 1e-9) -> pd.DataFrame:
    out = df.copy()
    for col in cols:
        g = _grouped(out, col)
        median = g.transform("median")
        q75 = g.transform(lambda s: s.quantile(0.75))
        q25 = g.transform(lambda s: s.quantile(0.25))
        iqr = (q75 - q25).abs()
        out[col] = (out[col] - median) / (iqr + eps)
    return out


def ema_smooth_by_symbol(df: pd.DataFrame, cols: list[str], span: int) -> pd.DataFrame:
    out = df.copy()
    for col in cols:
        out[col] = _grouped(out, col).transform(lambda s: s.ewm(span=span, adjust=False).mean())
    return out


def standard_zscore_by_symbol(df: pd.DataFrame, cols: list[str], eps: float = 1e-9) -> pd.DataFrame:
    out = df.copy()
    for col in cols:
        g = _grouped(out, col)
        mean = g.transform("mean")
        std = g.transform("std").abs()
        out[col] = (out[col] - mean) / (std + eps)
    return out


def rolling_median_by_symbol(df: pd.DataFrame, cols: list[str], window: int) -> pd.DataFrame:
    out = df.copy()
    for col in cols:
        out[col] = _grouped(out, col).transform(lambda s: s.rolling(window=window, min_periods=1).median())
    return out


def minmax_by_symbol(df: pd.DataFrame, cols: list[str], eps: float = 1e-9) -> pd.DataFrame:
    out = df.copy()
    for col in cols:
        g = _grouped(out, col)
        low = g.transform("min")
        high = g.transform("max")
        out[col] = 2.0 * ((out[col] - low) / ((high - low) + eps)) - 1.0
    return out


def hampel_filter_by_symbol(df: pd.DataFrame, cols: list[str], window: int, n_sigma: float) -> pd.DataFrame:
    out = df.copy()
    scale = 1.4826
    for col in cols:
        def _hampel(s: pd.Series) -> pd.Series:
            med = s.rolling(window=window, center=True, min_periods=1).median()
            mad = (s - med).abs().rolling(window=window, center=True, min_periods=1).median()
            thresh = n_sigma * scale * mad
            return s.where((s - med).abs() <= thresh, med)

        out[col] = _grouped(out, col).transform(_hampel)
    return out


def cross_sectional_rank_norm(df: pd.DataFrame, cols: list[str]) -> pd.DataFrame:
    out = df.copy()
    for col in cols:
        ranks = out.groupby("Date", sort=False)[col].rank(method="average", pct=True)
        out[col] = (2.0 * ranks) - 1.0
    return out


def iqr_clip_by_symbol(df: pd.DataFrame, cols: list[str], k: float = 2.5) -> pd.DataFrame:
    out = df.copy()
    for col in cols:
        g = _grouped(out, col)
        q1 = g.transform(lambda s: s.quantile(0.25))
        q3 = g.transform(lambda s: s.quantile(0.75))
        iqr = (q3 - q1).abs()
        lo = q1 - k * iqr
        hi = q3 + k * iqr
        out[col] = out[col].clip(lower=lo, upper=hi)
    return out


def signed_log1p(df: pd.DataFrame, cols: list[str]) -> pd.DataFrame:
    out = df.copy()
    for col in cols:
        x = out[col]
        out[col] = np.sign(x) * np.log1p(np.abs(x))
    return out


def cross_sectional_zscore(df: pd.DataFrame, cols: list[str], eps: float = 1e-9) -> pd.DataFrame:
    out = df.copy()
    for col in cols:
        g = out.groupby("Date", sort=False)[col]
        mean = g.transform("mean")
        std = g.transform("std").abs()
        out[col] = (out[col] - mean) / (std + eps)
    return out


def fill_numeric_na(df: pd.DataFrame, cols: list[str], value: float = 0.0) -> pd.DataFrame:
    out = df.copy()
    out[cols] = out[cols].replace([np.inf, -np.inf], np.nan).fillna(value)
    return out


def write_outputs(
    df: pd.DataFrame,
    output_csv: Path,
    metadata_json: Path,
    method_name: str,
    denoising_steps: list[str],
    normalization_steps: list[str],
) -> None:
    output_csv.parent.mkdir(parents=True, exist_ok=True)
    metadata_json.parent.mkdir(parents=True, exist_ok=True)

    df_to_save = df.copy()
    df_to_save["Date"] = pd.to_datetime(df_to_save["Date"], errors="coerce").dt.strftime("%Y-%m-%d")
    df_to_save.to_csv(output_csv, index=False)

    cols = feature_columns(df_to_save)
    meta = {
        "method": method_name,
        "rows": int(len(df_to_save)),
        "symbols": int(df_to_save["Symbol"].nunique()),
        "date_min": str(df_to_save["Date"].min()),
        "date_max": str(df_to_save["Date"].max()),
        "feature_columns": int(len(cols)),
        "preprocessing": {
            "denoising": denoising_steps,
            "normalization": normalization_steps,
        },
        "output_csv": str(output_csv),
    }
    metadata_json.write_text(json.dumps(meta, indent=2), encoding="utf-8")
