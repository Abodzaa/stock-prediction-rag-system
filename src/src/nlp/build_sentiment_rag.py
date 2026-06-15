#!/usr/bin/env python3
"""Build retrieval-augmented sentiment features from timestamped news."""

from __future__ import annotations

import argparse
import json
import os
import re
import time
from pathlib import Path

import numpy as np
import pandas as pd
from sentence_transformers import SentenceTransformer
import torch
from sklearn.neighbors import NearestNeighbors
from sklearn.metrics.pairwise import cosine_similarity

from news_dataset_utils import aggregate_daily_sentiment, entropy_from_probs, load_news_corpus

try:
    from google import genai
except Exception:  # pragma: no cover - optional dependency at runtime
    genai = None


DEFAULT_INPUT_CSV = Path("data/raw_news/yfinance_sp500_news.csv")
DEFAULT_ARTICLE_OUTPUT_CSV = Path("data/processed/features/news_sentiment_rag_articles.csv")
DEFAULT_DAILY_OUTPUT_CSV = Path("data/processed/features/news_sentiment_rag_daily.csv")
DEFAULT_GEMINI_CACHE_CSV = Path("data/processed/features/news_sentiment_rag_gemini_cache.csv")

POSITIVE_LEXICON = {
    "beat",
    "beats",
    "upgrade",
    "upgraded",
    "growth",
    "strong",
    "bullish",
    "surge",
    "rally",
    "record",
    "profit",
    "profits",
    "outperform",
    "outperformance",
    "gain",
    "gains",
    "rebound",
    "optimistic",
}

NEGATIVE_LEXICON = {
    "miss",
    "misses",
    "downgrade",
    "downgraded",
    "weak",
    "bearish",
    "drop",
    "drops",
    "plunge",
    "selloff",
    "loss",
    "losses",
    "underperform",
    "recession",
    "bankruptcy",
    "cut",
    "cuts",
    "warn",
    "warning",
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Build RAG-style sentiment features from a news CSV.")
    parser.add_argument("--input-csv", type=Path, default=DEFAULT_INPUT_CSV)
    parser.add_argument("--article-output-csv", type=Path, default=DEFAULT_ARTICLE_OUTPUT_CSV)
    parser.add_argument("--daily-output-csv", type=Path, default=DEFAULT_DAILY_OUTPUT_CSV)
    parser.add_argument("--backend", choices=["sbert", "gemini"], default="sbert")
    parser.add_argument("--embedding-model", default="sentence-transformers/all-MiniLM-L6-v2")
    parser.add_argument("--k", type=int, default=8)
    parser.add_argument("--gemini-model", default="gemini-2.5-flash")
    parser.add_argument("--gemini-api-key", default="")
    parser.add_argument("--gemini-temperature", type=float, default=0.0)
    parser.add_argument("--gemini-max-output-tokens", type=int, default=128)
    parser.add_argument("--gemini-sleep-seconds", type=float, default=0.0)
    parser.add_argument("--cache-csv", type=Path, default=DEFAULT_GEMINI_CACHE_CSV)
    parser.add_argument("--device", choices=["auto", "cpu", "cuda"], default="auto")
    parser.add_argument("--require-cuda", action="store_true")
    parser.add_argument("--max-rows", type=int, default=0)
    parser.add_argument("--start-date", default="")
    parser.add_argument("--end-date", default="")
    parser.add_argument("--max-text-chars", type=int, default=1200)
    return parser.parse_args()


def tokenize(text: str) -> list[str]:
    return re.findall(r"[A-Za-z]+", text.lower())


def lexicon_sentiment_score(text: str) -> float:
    tokens = tokenize(text)
    if not tokens:
        return 0.0

    pos_hits = sum(1 for t in tokens if t in POSITIVE_LEXICON)
    neg_hits = sum(1 for t in tokens if t in NEGATIVE_LEXICON)
    raw = pos_hits - neg_hits
    # Saturating transform to keep score in [-1, 1].
    return float(np.tanh(raw / 3.0))


def score_to_probs(score: float) -> tuple[float, float, float]:
    # Convert scalar sentiment score to pseudo-class probabilities.
    pos = max(score, 0.0)
    neg = max(-score, 0.0)
    neu = max(1.0 - abs(score), 0.0)
    total = pos + neg + neu
    if total <= 0:
        return (0.0, 0.0, 1.0)
    return (pos / total, neg / total, neu / total)


def normalize_probs(pos: float, neg: float, neu: float) -> tuple[float, float, float]:
    pos = max(float(pos), 0.0)
    neg = max(float(neg), 0.0)
    neu = max(float(neu), 0.0)
    total = pos + neg + neu
    if total <= 0:
        return (0.0, 0.0, 1.0)
    return (pos / total, neg / total, neu / total)


def resolve_device(raw: str, require_cuda: bool) -> str:
    cuda_ok = torch.cuda.is_available()
    if raw == "cuda":
        if not cuda_ok:
            raise RuntimeError("CUDA requested but torch.cuda.is_available() is False.")
        return "cuda"
    if raw == "cpu":
        if require_cuda:
            raise RuntimeError("--require-cuda set but --device cpu was requested.")
        return "cpu"

    if cuda_ok:
        return "cuda"

    if require_cuda:
        raise RuntimeError("--require-cuda set but CUDA is unavailable.")
    return "cpu"


def load_cache(cache_csv: Path) -> dict[str, dict[str, float | str]]:
    if not cache_csv.exists():
        return {}

    df = pd.read_csv(cache_csv)
    required = {"news_id", "sent_pos", "sent_neg", "sent_neu", "sent_score", "rag_support", "mode", "error"}
    if not required.issubset(df.columns):
        return {}

    out: dict[str, dict[str, float | str]] = {}
    for _, row in df.iterrows():
        news_id = str(row["news_id"])
        if not news_id:
            continue
        out[news_id] = {
            "sent_pos": float(row["sent_pos"]),
            "sent_neg": float(row["sent_neg"]),
            "sent_neu": float(row["sent_neu"]),
            "sent_score": float(row["sent_score"]),
            "rag_support": float(row["rag_support"]),
            "mode": str(row["mode"]),
            "error": str(row["error"]),
        }
    return out


def write_cache(cache_csv: Path, cache: dict[str, dict[str, float | str]]) -> None:
    rows = []
    for news_id, payload in cache.items():
        rows.append({"news_id": news_id, **payload})

    cache_csv.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(rows).to_csv(cache_csv, index=False)


def build_neighbor_index(embeddings: np.ndarray, k: int) -> tuple[np.ndarray, np.ndarray]:
    if len(embeddings) == 0:
        return np.empty((0, 0), dtype=int), np.empty((0, 0), dtype=float)

    n_neighbors = min(max(2, k + 1), len(embeddings))
    knn = NearestNeighbors(n_neighbors=n_neighbors, metric="cosine")
    knn.fit(embeddings)
    distances, indices = knn.kneighbors(embeddings, return_distance=True)

    sim = 1.0 - distances
    sim = np.clip(sim, 0.0, 1.0)
    return indices, sim


def parse_gemini_probabilities(raw_text: str) -> tuple[float, float, float]:
    text = (raw_text or "").strip()
    if text.startswith("```"):
        text = re.sub(r"^```(?:json)?\s*", "", text)
        text = re.sub(r"\s*```$", "", text)

    try:
        payload = json.loads(text)
        if isinstance(payload, dict):
            pos = payload.get("positive", payload.get("pos", 0.0))
            neg = payload.get("negative", payload.get("neg", 0.0))
            neu = payload.get("neutral", payload.get("neu", 0.0))
            return normalize_probs(float(pos), float(neg), float(neu))
    except Exception:
        pass

    patterns = {
        "pos": r"(?:positive|pos)\s*[:=]\s*(-?\d+(?:\.\d+)?)",
        "neg": r"(?:negative|neg)\s*[:=]\s*(-?\d+(?:\.\d+)?)",
        "neu": r"(?:neutral|neu)\s*[:=]\s*(-?\d+(?:\.\d+)?)",
    }
    vals = {}
    for key, pattern in patterns.items():
        m = re.search(pattern, text, flags=re.IGNORECASE)
        if m:
            vals[key] = float(m.group(1))

    if {"pos", "neg", "neu"}.issubset(vals.keys()):
        return normalize_probs(vals["pos"], vals["neg"], vals["neu"])

    raise ValueError("Could not parse positive/negative/neutral probabilities from Gemini response.")


def gemini_prompt(article_text: str, context_texts: list[str]) -> str:
    context_block = "\n\n".join([f"Context {i + 1}: {t}" for i, t in enumerate(context_texts)])
    return (
        "You are a financial sentiment model. "
        "Given the target market-news article and retrieval context, estimate probabilities for "
        "positive, negative, and neutral sentiment of the target article for near-term equity impact.\n\n"
        "Return ONLY a strict JSON object with these keys exactly: "
        "positive, negative, neutral.\n"
        "Each value must be a float in [0,1], and values must sum to 1.\n\n"
        f"Target article:\n{article_text}\n\n"
        f"Retrieved context:\n{context_block}\n"
    )


def run_sbert_pipeline(news: pd.DataFrame, args: argparse.Namespace, device_name: str) -> tuple[list[float], list[float], list[float], list[float], list[float], list[float], list[str], list[str]]:
    embedder = SentenceTransformer(args.embedding_model, device=device_name)
    doc_embeddings = embedder.encode(news["text"].tolist(), show_progress_bar=False)

    query_bank = {
        "positive": [
            "strong earnings beat and upward guidance",
            "bullish outlook and margin expansion",
            "positive macro surprise supporting equities",
            "analyst upgrades and strong demand",
        ],
        "negative": [
            "earnings miss and lowered guidance",
            "bearish outlook and margin compression",
            "negative macro shock and risk-off move",
            "analyst downgrades and demand weakness",
        ],
        "neutral": [
            "mixed outlook with balanced risks",
            "market update without major directional signal",
            "flat expectations and stable guidance",
        ],
    }

    query_texts = []
    query_labels = []
    for label, items in query_bank.items():
        query_texts.extend(items)
        query_labels.extend([label] * len(items))

    query_embeddings = embedder.encode(query_texts, show_progress_bar=False)
    sim = cosine_similarity(doc_embeddings, query_embeddings)

    sent_pos = []
    sent_neg = []
    sent_neu = []
    sent_score = []
    sent_entropy = []
    rag_support = []
    rag_mode = []
    rag_error = []

    k = max(1, int(args.k))

    for i, row in news.iterrows():
        sims = sim[i]
        top_idx = np.argsort(sims)[::-1][:k]
        label_weights = {"positive": 0.0, "negative": 0.0, "neutral": 0.0}

        for idx in top_idx:
            w = float(max(sims[idx], 0.0))
            label = query_labels[idx]
            label_weights[label] += w

        weight_sum = sum(label_weights.values())
        if weight_sum <= 0:
            rag_pos, rag_neg, rag_neu = (0.0, 0.0, 1.0)
        else:
            rag_pos = label_weights["positive"] / weight_sum
            rag_neg = label_weights["negative"] / weight_sum
            rag_neu = label_weights["neutral"] / weight_sum

        lex_score = lexicon_sentiment_score(row["text"])
        lex_pos, lex_neg, lex_neu = score_to_probs(lex_score)

        blend = 0.7
        pos = blend * rag_pos + (1.0 - blend) * lex_pos
        neg = blend * rag_neg + (1.0 - blend) * lex_neg
        neu = blend * rag_neu + (1.0 - blend) * lex_neu
        pos, neg, neu = normalize_probs(pos, neg, neu)

        score = float(pos - neg)
        sent_pos.append(pos)
        sent_neg.append(neg)
        sent_neu.append(neu)
        sent_score.append(score)
        sent_entropy.append(entropy_from_probs(pos, neg, neu))
        rag_support.append(float(np.mean(sims[top_idx])) if len(top_idx) else 0.0)
        rag_mode.append("sbert")
        rag_error.append("")

    return sent_pos, sent_neg, sent_neu, sent_score, sent_entropy, rag_support, rag_mode, rag_error


def run_gemini_pipeline(news: pd.DataFrame, args: argparse.Namespace, device_name: str) -> tuple[list[float], list[float], list[float], list[float], list[float], list[float], list[str], list[str]]:
    if genai is None:
        raise ImportError("google-genai package is required for backend=gemini.")

    api_key = args.gemini_api_key or os.getenv("GEMINI_API_KEY") or os.getenv("GOOGLE_API_KEY")
    if not api_key:
        raise ValueError("Gemini API key not found. Set GEMINI_API_KEY or GOOGLE_API_KEY, or pass --gemini-api-key.")

    client = genai.Client(api_key=api_key)
    embedder = SentenceTransformer(args.embedding_model, device=device_name)
    doc_embeddings = embedder.encode(news["text"].tolist(), show_progress_bar=False)

    k = max(1, int(args.k))
    neighbor_idx, neighbor_sim = build_neighbor_index(doc_embeddings, k=k)

    cache = load_cache(args.cache_csv)

    sent_pos = []
    sent_neg = []
    sent_neu = []
    sent_score = []
    sent_entropy = []
    rag_support = []
    rag_mode = []
    rag_error = []

    for i, row in news.iterrows():
        news_id = str(row["news_id"])
        cached = cache.get(news_id)
        if cached is not None:
            pos = float(cached["sent_pos"])
            neg = float(cached["sent_neg"])
            neu = float(cached["sent_neu"])
            score = float(cached["sent_score"])
            support = float(cached["rag_support"])
            mode = str(cached.get("mode", "gemini"))
            err = str(cached.get("error", ""))
        else:
            nbr_ids = []
            nbr_sims = []
            if i < len(neighbor_idx):
                for idx, sim in zip(neighbor_idx[i], neighbor_sim[i]):
                    if int(idx) == i:
                        continue
                    nbr_ids.append(int(idx))
                    nbr_sims.append(float(sim))
                    if len(nbr_ids) >= k:
                        break

            context_texts = []
            for idx in nbr_ids:
                context_texts.append(str(news.iloc[idx]["text"])[:320])

            prompt = gemini_prompt(article_text=str(row["text"]), context_texts=context_texts)
            support = float(np.mean(nbr_sims)) if nbr_sims else 0.0

            try:
                response = client.models.generate_content(
                    model=args.gemini_model,
                    contents=prompt,
                    config={
                        "temperature": float(args.gemini_temperature),
                        "max_output_tokens": int(args.gemini_max_output_tokens),
                    },
                )
                response_text = str(getattr(response, "text", "") or "")
                pos, neg, neu = parse_gemini_probabilities(response_text)
                score = float(pos - neg)
                mode = "gemini"
                err = ""
            except Exception as exc:
                # Keep pipeline resilient if API fails on some rows.
                lex_score = lexicon_sentiment_score(str(row["text"]))
                pos, neg, neu = score_to_probs(lex_score)
                score = float(pos - neg)
                mode = "lexicon_fallback"
                err = str(exc)

            cache[news_id] = {
                "sent_pos": float(pos),
                "sent_neg": float(neg),
                "sent_neu": float(neu),
                "sent_score": float(score),
                "rag_support": float(support),
                "mode": mode,
                "error": err,
            }

            if args.gemini_sleep_seconds > 0:
                time.sleep(args.gemini_sleep_seconds)

        sent_pos.append(pos)
        sent_neg.append(neg)
        sent_neu.append(neu)
        sent_score.append(score)
        sent_entropy.append(entropy_from_probs(pos, neg, neu))
        rag_support.append(support)
        rag_mode.append(mode)
        rag_error.append(err)

        if (i + 1) % 100 == 0 or i + 1 == len(news):
            write_cache(args.cache_csv, cache)
            print(f"Gemini sentiment progress: {i + 1}/{len(news)}")

    return sent_pos, sent_neg, sent_neu, sent_score, sent_entropy, rag_support, rag_mode, rag_error


def main() -> None:
    args = parse_args()
    device_name = resolve_device(args.device, args.require_cuda)

    news = load_news_corpus(
        input_csv=args.input_csv,
        max_rows=args.max_rows if args.max_rows > 0 else None,
        start_date=args.start_date or None,
        end_date=args.end_date or None,
        max_text_chars=args.max_text_chars,
    )

    if news.empty:
        raise RuntimeError("No news rows available after loading and filtering.")

    if args.backend == "gemini":
        (
            sent_pos,
            sent_neg,
            sent_neu,
            sent_score,
            sent_entropy,
            rag_support,
            rag_mode,
            rag_error,
        ) = run_gemini_pipeline(news=news, args=args, device_name=device_name)
    else:
        (
            sent_pos,
            sent_neg,
            sent_neu,
            sent_score,
            sent_entropy,
            rag_support,
            rag_mode,
            rag_error,
        ) = run_sbert_pipeline(news=news, args=args, device_name=device_name)

    article_df = news.copy()
    article_df["sent_pos"] = sent_pos
    article_df["sent_neg"] = sent_neg
    article_df["sent_neu"] = sent_neu
    article_df["sent_score"] = sent_score
    article_df["sent_entropy"] = sent_entropy
    article_df["rag_support"] = rag_support
    article_df["rag_mode"] = rag_mode
    article_df["rag_error"] = rag_error

    daily_df = aggregate_daily_sentiment(
        article_df=article_df,
        prefix="g5_sent_rag",
        extra_numeric_cols=["rag_support"],
    )

    args.article_output_csv.parent.mkdir(parents=True, exist_ok=True)
    args.daily_output_csv.parent.mkdir(parents=True, exist_ok=True)
    article_df.to_csv(args.article_output_csv, index=False)
    daily_df.to_csv(args.daily_output_csv, index=False)

    print(f"Input news rows: {len(news)}")
    print(f"Backend: {args.backend}")
    print(f"Device: {device_name}")
    print(f"Article sentiment rows: {len(article_df)}")
    print(f"Daily rows: {len(daily_df)}")
    print(f"Daily feature columns: {len([c for c in daily_df.columns if c.startswith('g5_sent_rag_')])}")
    print(f"Outputs: {args.article_output_csv} | {args.daily_output_csv}")


if __name__ == "__main__":
    main()
