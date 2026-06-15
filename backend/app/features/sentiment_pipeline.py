"""Daily news-sentiment features (G4/G5), reproducing the training schema.

Reuses aggregate_daily_sentiment / entropy_from_probs from the research code so
the produced columns match g5_sent_finbert_* (12) and g5_sent_rag_* (14) exactly.

Two scorers:
  * finbert  -> ProsusAI/finbert per-article pos/neg/neu
  * rag      -> SBERT query-bank similarity blended with a finance lexicon
                (mirrors src/nlp/build_sentiment_rag.py run_sbert_pipeline)
Also exposes article-level sentiment for the explanation UI.
"""
from __future__ import annotations

import re

import numpy as np
import pandas as pd

from app.config import settings
from app.repositories.news_repository import NewsRepository, get_news_repository
from app.research_bridge import load_research_callables

_R = load_research_callables()

_POS_LEX = {"beat", "beats", "upgrade", "upgraded", "growth", "strong", "bullish", "surge",
            "rally", "record", "profit", "profits", "outperform", "outperformance", "gain",
            "gains", "rebound", "optimistic"}
_NEG_LEX = {"miss", "misses", "downgrade", "downgraded", "weak", "bearish", "drop", "drops",
            "plunge", "selloff", "loss", "losses", "underperform", "recession", "bankruptcy",
            "cut", "cuts", "warn", "warning"}

_QUERY_BANK = {
    "positive": ["strong earnings beat and upward guidance", "bullish outlook and margin expansion",
                 "positive macro surprise supporting equities", "analyst upgrades and strong demand"],
    "negative": ["earnings miss and lowered guidance", "bearish outlook and margin compression",
                 "negative macro shock and risk-off move", "analyst downgrades and demand weakness"],
    "neutral": ["mixed outlook with balanced risks", "market update without major directional signal",
                "flat expectations and stable guidance"],
}


def _norm3(p, n, u):
    p, n, u = max(p, 0.0), max(n, 0.0), max(u, 0.0)
    t = p + n + u
    return (0.0, 0.0, 1.0) if t <= 0 else (p / t, n / t, u / t)


class SentimentPipeline:
    def __init__(self, news_repo: NewsRepository | None = None) -> None:
        self.news = news_repo or get_news_repository()
        self._finbert = None
        self._sbert = None

    # ---------------------------------------------------------------- public
    def article_sentiment(self, symbol: str, family: str) -> pd.DataFrame:
        articles = self._gather(symbol)
        if not articles:
            return pd.DataFrame(columns=["Date", "sent_pos", "sent_neg", "sent_neu",
                                         "sent_score", "sent_entropy", "source", "headline"])
        df = pd.DataFrame(articles)
        df["text"] = (df["headline"].fillna("") + "\n" + df["summary"].fillna("")).str.strip()
        df["Date"] = pd.to_datetime(df["published_utc"], utc=True, errors="coerce").dt.tz_convert(
            "UTC").dt.floor("D").dt.tz_localize(None)
        df = df.dropna(subset=["Date"])
        use_rag = "rag" in family.lower()
        scored = self._score_rag(df) if use_rag else self._score_finbert(df)
        return scored

    def daily_features(self, symbol: str, family: str, columns: list[str]) -> pd.DataFrame:
        scored = self.article_sentiment(symbol, family)
        if scored.empty:
            return pd.DataFrame(columns=["Date", *columns])
        use_rag = "rag" in family.lower()
        prefix = "g5_sent_rag" if use_rag else "g5_sent_finbert"
        extras = ["rag_support"] if use_rag else None
        daily = _R["aggregate_daily_sentiment"](article_df=scored, prefix=prefix, extra_numeric_cols=extras)
        keep = ["Date"] + [c for c in columns if c in daily.columns]
        return daily[keep]

    # --------------------------------------------------------------- scorers
    def _score_finbert(self, df: pd.DataFrame) -> pd.DataFrame:
        clf = self._finbert_pipe()
        scores = clf(df["text"].str.slice(0, 1000).tolist(), batch_size=16, truncation=True, max_length=256)
        pos, neg, neu, sc, ent = [], [], [], [], []
        for s in scores:
            p, n, u = self._labels(s)
            pos.append(p); neg.append(n); neu.append(u)
            sc.append(p - n); ent.append(_R["entropy_from_probs"](p, n, u))
        out = df.copy()
        out["sent_pos"], out["sent_neg"], out["sent_neu"] = pos, neg, neu
        out["sent_score"], out["sent_entropy"] = sc, ent
        return out

    def _score_rag(self, df: pd.DataFrame) -> pd.DataFrame:
        from sklearn.metrics.pairwise import cosine_similarity

        embedder = self._sbert_model()
        doc = embedder.encode(df["text"].tolist(), show_progress_bar=False)
        q_texts, q_labels = [], []
        for label, items in _QUERY_BANK.items():
            q_texts += items; q_labels += [label] * len(items)
        sim = cosine_similarity(doc, embedder.encode(q_texts, show_progress_bar=False))
        pos, neg, neu, sc, ent, support = [], [], [], [], [], []
        k = 8
        for i in range(len(df)):
            sims = sim[i]
            top = np.argsort(sims)[::-1][:k]
            w = {"positive": 0.0, "negative": 0.0, "neutral": 0.0}
            for j in top:
                w[q_labels[j]] += float(max(sims[j], 0.0))
            tot = sum(w.values())
            rp, rn, ru = (0.0, 0.0, 1.0) if tot <= 0 else (w["positive"] / tot, w["negative"] / tot, w["neutral"] / tot)
            lp, ln, lu = self._lexicon(df["text"].iloc[i])
            p, n, u = _norm3(0.7 * rp + 0.3 * lp, 0.7 * rn + 0.3 * ln, 0.7 * ru + 0.3 * lu)
            pos.append(p); neg.append(n); neu.append(u); sc.append(p - n)
            ent.append(_R["entropy_from_probs"](p, n, u))
            support.append(float(np.mean(sims[top])) if len(top) else 0.0)
        out = df.copy()
        out["sent_pos"], out["sent_neg"], out["sent_neu"] = pos, neg, neu
        out["sent_score"], out["sent_entropy"], out["rag_support"] = sc, ent, support
        return out

    # ------------------------------------------------------------- internals
    def _gather(self, symbol: str) -> list[dict]:
        if symbol.startswith("^") or symbol == settings.index_symbol:
            from app.repositories.price_repository import get_price_repository

            rows: list[dict] = []
            for s in get_price_repository().get_universe()[:12]:
                rows += self.news.fetch(s, limit=10)
            return rows
        return self.news.fetch(symbol, limit=40)

    def _finbert_pipe(self):
        if self._finbert is None:
            from transformers import pipeline

            self._finbert = pipeline("text-classification", model=settings.finbert_model,
                                     tokenizer=settings.finbert_model, top_k=None, device=-1)
        return self._finbert

    def _sbert_model(self):
        if self._sbert is None:
            from sentence_transformers import SentenceTransformer

            self._sbert = SentenceTransformer(settings.sbert_model)
        return self._sbert

    @staticmethod
    def _labels(scores):
        p = n = u = 0.0
        for s in scores:
            lab = str(s.get("label", "")).lower(); val = float(s.get("score", 0.0))
            if "pos" in lab: p = val
            elif "neg" in lab: n = val
            elif "neu" in lab: u = val
        return _norm3(p, n, u)

    @staticmethod
    def _lexicon(text: str):
        toks = re.findall(r"[A-Za-z]+", str(text).lower())
        if not toks:
            return (0.0, 0.0, 1.0)
        raw = sum(t in _POS_LEX for t in toks) - sum(t in _NEG_LEX for t in toks)
        score = float(np.tanh(raw / 3.0))
        return _norm3(max(score, 0.0), max(-score, 0.0), max(1.0 - abs(score), 0.0))


_singleton: SentimentPipeline | None = None


def get_sentiment_pipeline() -> SentimentPipeline:
    global _singleton
    if _singleton is None:
        _singleton = SentimentPipeline()
    return _singleton
