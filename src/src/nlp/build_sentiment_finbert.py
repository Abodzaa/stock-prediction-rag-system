#!/usr/bin/env python3
"""Build FinBERT-based daily sentiment features from timestamped news."""

from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd
import torch
from transformers import pipeline

from news_dataset_utils import aggregate_daily_sentiment, entropy_from_probs, load_news_corpus


DEFAULT_INPUT_CSV = Path("data/raw_news/yfinance_sp500_news.csv")
DEFAULT_ARTICLE_OUTPUT_CSV = Path("data/processed/features/news_sentiment_finbert_articles.csv")
DEFAULT_DAILY_OUTPUT_CSV = Path("data/processed/features/news_sentiment_finbert_daily.csv")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Build FinBERT sentiment features from a news CSV.")
    parser.add_argument("--input-csv", type=Path, default=DEFAULT_INPUT_CSV)
    parser.add_argument("--article-output-csv", type=Path, default=DEFAULT_ARTICLE_OUTPUT_CSV)
    parser.add_argument("--daily-output-csv", type=Path, default=DEFAULT_DAILY_OUTPUT_CSV)
    parser.add_argument("--model-name", default="ProsusAI/finbert")
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--max-length", type=int, default=256)
    parser.add_argument("--max-rows", type=int, default=0)
    parser.add_argument("--start-date", default="")
    parser.add_argument("--end-date", default="")
    parser.add_argument("--max-text-chars", type=int, default=1200)
    parser.add_argument("--device", choices=["auto", "cpu", "cuda"], default="auto")
    parser.add_argument("--require-cuda", action="store_true")
    return parser.parse_args()


def resolve_device(raw: str, require_cuda: bool) -> tuple[str, int]:
    cuda_ok = torch.cuda.is_available()
    if raw == "cuda":
        if not cuda_ok:
            raise RuntimeError("CUDA requested but torch.cuda.is_available() is False.")
        return "cuda", 0
    if raw == "cpu":
        if require_cuda:
            raise RuntimeError("--require-cuda set but --device cpu was requested.")
        return "cpu", -1

    if cuda_ok:
        return "cuda", 0

    if require_cuda:
        raise RuntimeError("--require-cuda set but CUDA is unavailable.")
    return "cpu", -1


def label_map(scores: list[dict]) -> tuple[float, float, float]:
    pos = 0.0
    neg = 0.0
    neu = 0.0

    for s in scores:
        label = str(s.get("label", "")).lower()
        score = float(s.get("score", 0.0))

        if "pos" in label:
            pos = score
        elif "neg" in label:
            neg = score
        elif "neu" in label:
            neu = score

    total = pos + neg + neu
    if total <= 0:
        return (0.0, 0.0, 1.0)
    return (pos / total, neg / total, neu / total)


def main() -> None:
    args = parse_args()
    device_name, pipeline_device = resolve_device(args.device, args.require_cuda)

    news = load_news_corpus(
        input_csv=args.input_csv,
        max_rows=args.max_rows if args.max_rows > 0 else None,
        start_date=args.start_date or None,
        end_date=args.end_date or None,
        max_text_chars=args.max_text_chars,
    )

    if news.empty:
        raise RuntimeError("No news rows available after loading and filtering.")

    clf = pipeline(
        task="text-classification",
        model=args.model_name,
        tokenizer=args.model_name,
        top_k=None,
        device=pipeline_device,
    )

    texts = news["text"].tolist()
    all_scores = clf(
        texts,
        batch_size=args.batch_size,
        truncation=True,
        max_length=args.max_length,
    )

    sent_pos = []
    sent_neg = []
    sent_neu = []
    sent_score = []
    sent_entropy = []

    for scores in all_scores:
        pos, neg, neu = label_map(scores)
        sent_pos.append(pos)
        sent_neg.append(neg)
        sent_neu.append(neu)
        sent_score.append(pos - neg)
        sent_entropy.append(entropy_from_probs(pos, neg, neu))

    article_df = news.copy()
    article_df["sent_pos"] = sent_pos
    article_df["sent_neg"] = sent_neg
    article_df["sent_neu"] = sent_neu
    article_df["sent_score"] = sent_score
    article_df["sent_entropy"] = sent_entropy

    daily_df = aggregate_daily_sentiment(article_df=article_df, prefix="g5_sent_finbert")

    args.article_output_csv.parent.mkdir(parents=True, exist_ok=True)
    args.daily_output_csv.parent.mkdir(parents=True, exist_ok=True)

    article_df.to_csv(args.article_output_csv, index=False)
    daily_df.to_csv(args.daily_output_csv, index=False)

    print(f"Input news rows: {len(news)}")
    print(f"Device: {device_name}")
    print(f"Article sentiment rows: {len(article_df)}")
    print(f"Daily rows: {len(daily_df)}")
    print(f"Daily feature columns: {len([c for c in daily_df.columns if c.startswith('g5_sent_finbert_')])}")
    print(f"Outputs: {args.article_output_csv} | {args.daily_output_csv}")


if __name__ == "__main__":
    main()
