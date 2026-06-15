#!/usr/bin/env python3
"""Utilities for loading news corpora and aggregating daily sentiment features."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd


TIMESTAMP_CANDIDATES = [
    "published_utc",
    "published_at",
    "publish_time",
    "datetime",
    "timestamp",
    "time",
    "Date",
]

HEADLINE_CANDIDATES = ["headline", "title", "news_title"]
BODY_CANDIDATES = ["summary", "content", "body", "text", "description"]
SOURCE_CANDIDATES = ["source", "provider", "publisher"]
URL_CANDIDATES = ["url", "link", "canonical_url"]
ID_CANDIDATES = ["news_id", "uuid", "id"]


def _pick_column(df: pd.DataFrame, candidates: list[str]) -> str | None:
    for col in candidates:
        if col in df.columns:
            return col
    return None


def load_news_corpus(
    input_csv: Path,
    max_rows: int | None = None,
    start_date: str | None = None,
    end_date: str | None = None,
    max_text_chars: int = 1200,
) -> pd.DataFrame:
    if not input_csv.exists():
        raise FileNotFoundError(f"News CSV not found: {input_csv}")

    df = pd.read_csv(input_csv)
    if df.empty:
        raise ValueError(f"News CSV is empty: {input_csv}")

    ts_col = _pick_column(df, TIMESTAMP_CANDIDATES)
    if ts_col is None:
        raise ValueError(f"News CSV must contain one of timestamp columns: {TIMESTAMP_CANDIDATES}")

    headline_col = _pick_column(df, HEADLINE_CANDIDATES)
    body_col = _pick_column(df, BODY_CANDIDATES)
    source_col = _pick_column(df, SOURCE_CANDIDATES)
    url_col = _pick_column(df, URL_CANDIDATES)
    id_col = _pick_column(df, ID_CANDIDATES)

    work = pd.DataFrame()
    work["published_utc"] = pd.to_datetime(df[ts_col], errors="coerce", utc=True)
    work = work.dropna(subset=["published_utc"])

    if headline_col is not None:
        work["headline"] = df.loc[work.index, headline_col].astype(str).fillna("")
    else:
        work["headline"] = ""

    if body_col is not None:
        work["body"] = df.loc[work.index, body_col].astype(str).fillna("")
    else:
        work["body"] = ""

    work["headline"] = work["headline"].str.strip()
    work["body"] = work["body"].str.strip()

    text = (work["headline"] + "\n" + work["body"]).str.strip()
    text = text.str.replace(r"\s+", " ", regex=True).str.slice(0, max_text_chars)
    work["text"] = text

    if source_col is not None:
        work["source"] = df.loc[work.index, source_col].astype(str).fillna("unknown")
    else:
        work["source"] = "unknown"

    if url_col is not None:
        work["url"] = df.loc[work.index, url_col].astype(str).fillna("")
    else:
        work["url"] = ""

    if id_col is not None:
        work["news_id"] = df.loc[work.index, id_col].astype(str).fillna("")
    else:
        work["news_id"] = ""

    work = work[work["text"].str.len() > 0].copy()

    # Stable fallback identifier for dedupe when raw IDs are missing.
    fallback_id = (
        work["published_utc"].dt.strftime("%Y%m%d%H%M%S")
        + "|"
        + work["headline"].str.slice(0, 120)
        + "|"
        + work["url"].str.slice(0, 200)
    )
    work["news_id"] = work["news_id"].mask(work["news_id"].str.len() == 0, fallback_id)

    work = work.drop_duplicates(subset=["news_id"], keep="first")

    if start_date:
        work = work[work["published_utc"] >= pd.Timestamp(start_date, tz="UTC")]
    if end_date:
        work = work[work["published_utc"] < pd.Timestamp(end_date, tz="UTC")]

    work = work.sort_values("published_utc").reset_index(drop=True)

    if max_rows is not None and max_rows > 0:
        work = work.tail(max_rows).reset_index(drop=True)

    work["Date"] = work["published_utc"].dt.tz_convert("UTC").dt.floor("D").dt.tz_localize(None)
    return work


def entropy_from_probs(pos: float, neg: float, neu: float) -> float:
    probs = np.clip(np.array([pos, neg, neu], dtype=float), 1e-8, 1.0)
    probs = probs / probs.sum()
    return float(-(probs * np.log(probs)).sum())


def aggregate_daily_sentiment(
    article_df: pd.DataFrame,
    prefix: str,
    extra_numeric_cols: list[str] | None = None,
) -> pd.DataFrame:
    required = {"Date", "sent_pos", "sent_neg", "sent_neu", "sent_score", "sent_entropy", "source"}
    missing = sorted(required.difference(article_df.columns))
    if missing:
        raise ValueError(f"Missing required sentiment columns: {missing}")

    extras = extra_numeric_cols or []

    grouped = article_df.groupby("Date", as_index=False).agg(
        article_count=("sent_score", "size"),
        source_count=("source", "nunique"),
        sent_pos_mean=("sent_pos", "mean"),
        sent_neg_mean=("sent_neg", "mean"),
        sent_neu_mean=("sent_neu", "mean"),
        sent_score_mean=("sent_score", "mean"),
        sent_score_std=("sent_score", "std"),
        sent_entropy_mean=("sent_entropy", "mean"),
    )

    for col in extras:
        if col in article_df.columns:
            stats = article_df.groupby("Date")[col].agg(["mean", "std"]).reset_index()
            stats = stats.rename(
                columns={
                    "mean": f"{col}_mean",
                    "std": f"{col}_std",
                }
            )
            grouped = grouped.merge(stats, on="Date", how="left")

    grouped = grouped.sort_values("Date").reset_index(drop=True)

    grouped["sent_score_roll3"] = grouped["sent_score_mean"].rolling(3, min_periods=1).mean()
    grouped["sent_score_roll5"] = grouped["sent_score_mean"].rolling(5, min_periods=1).mean()
    grouped["article_count_roll3"] = grouped["article_count"].rolling(3, min_periods=1).mean()
    grouped["article_count_roll5"] = grouped["article_count"].rolling(5, min_periods=1).mean()

    rename_map = {col: f"{prefix}_{col}" for col in grouped.columns if col != "Date"}
    out = grouped.rename(columns=rename_map)
    return out
