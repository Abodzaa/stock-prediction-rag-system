"""Retrieval strategies (Strategy pattern), selected at runtime from RETRIEVAL_MODE.

The service depends on the RetrievalStrategy interface, not a concrete mode.
"""
from __future__ import annotations

from typing import Protocol

from app.config import settings
from app.repositories.embedding_repository import EmbeddingRepository
from app.repositories.qdrant_repository import QdrantRepository


class RetrievalStrategy(Protocol):
    def search(self, query: str, top_k: int, symbol: str | None = None) -> list[dict]: ...


def _to_hits(points) -> list[dict]:
    out = []
    for p in points:
        payload = dict(p.payload or {})
        payload["_score"] = float(getattr(p, "score", 0.0) or 0.0)
        payload["_id"] = str(getattr(p, "id", ""))
        out.append(payload)
    return out


class DenseStrategy:
    def __init__(self, emb: EmbeddingRepository, store: QdrantRepository) -> None:
        self.emb, self.store = emb, store

    def search(self, query: str, top_k: int, symbol: str | None = None) -> list[dict]:
        vec = self.emb.embed_query(query)
        return _to_hits(self.store.search_dense(vec, top_k, symbol))


class SparseStrategy:
    def __init__(self, store: QdrantRepository) -> None:
        self.store = store

    def search(self, query: str, top_k: int, symbol: str | None = None) -> list[dict]:
        return _to_hits(self.store.search_sparse(query, top_k, symbol))


class HybridStrategy:
    """Reciprocal Rank Fusion over dense + sparse result lists."""

    def __init__(self, emb: EmbeddingRepository, store: QdrantRepository, rrf_k: int = 60) -> None:
        self.dense = DenseStrategy(emb, store)
        self.sparse = SparseStrategy(store)
        self.rrf_k = rrf_k

    def search(self, query: str, top_k: int, symbol: str | None = None) -> list[dict]:
        dense = self.dense.search(query, top_k, symbol)
        sparse = self.sparse.search(query, top_k, symbol)
        scores: dict[str, float] = {}
        by_id: dict[str, dict] = {}
        for ranked in (dense, sparse):
            for rank, hit in enumerate(ranked):
                key = hit.get("_id") or hit.get("news_id") or hit.get("url")
                by_id[key] = hit
                scores[key] = scores.get(key, 0.0) + 1.0 / (self.rrf_k + rank + 1)
        ordered = sorted(scores, key=scores.get, reverse=True)[:top_k]
        out = []
        for key in ordered:
            hit = by_id[key]
            hit["_rrf"] = scores[key]
            out.append(hit)
        return out


def make_strategy(emb: EmbeddingRepository, store: QdrantRepository) -> RetrievalStrategy:
    mode = settings.retrieval_mode.lower()
    if mode == "dense":
        return DenseStrategy(emb, store)
    if mode == "sparse":
        return SparseStrategy(store)
    return HybridStrategy(emb, store)
