"""RetrievalService: thin facade over the active retrieval strategy."""
from __future__ import annotations

from app.config import settings
from app.patterns.retrieval_strategy import make_strategy
from app.repositories.embedding_repository import get_embedding_repository
from app.repositories.qdrant_repository import get_qdrant_repository


class RetrievalService:
    def __init__(self) -> None:
        self.strategy = make_strategy(get_embedding_repository(), get_qdrant_repository())

    def retrieve(self, query: str, symbol: str | None = None, top_k: int | None = None) -> list[dict]:
        k = top_k or settings.top_k
        hits = self.strategy.search(query, k, symbol)
        return hits[: settings.final_k]


_singleton: RetrievalService | None = None


def get_retrieval_service() -> RetrievalService:
    global _singleton
    if _singleton is None:
        _singleton = RetrievalService()
    return _singleton
