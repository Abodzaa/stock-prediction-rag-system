"""IngestService: news -> embed -> Qdrant upsert (idempotent).

Command-style: fetch for a symbol, embed passages with e5, upsert with
deterministic IDs so re-ingest updates instead of duplicating.
"""
from __future__ import annotations

from app.models.schemas import IngestResponse
from app.repositories.embedding_repository import EmbeddingRepository, get_embedding_repository
from app.repositories.news_repository import NewsRepository, get_news_repository
from app.repositories.qdrant_repository import QdrantRepository, get_qdrant_repository


class IngestService:
    def __init__(
        self,
        news: NewsRepository | None = None,
        emb: EmbeddingRepository | None = None,
        store: QdrantRepository | None = None,
    ) -> None:
        self.news = news or get_news_repository()
        self.emb = emb or get_embedding_repository()
        self.store = store or get_qdrant_repository()

    def ingest_symbol(self, symbol: str, lookback_days: int | None = None) -> IngestResponse:
        articles = self.news.fetch(symbol, limit=40)
        upserted = 0
        if articles:
            texts = [f"{a.get('headline','')} {a.get('summary','')}".strip() for a in articles]
            dense = self.emb.embed_passages(texts)
            upserted = self.store.upsert(articles, dense)
        from app.config import settings

        return IngestResponse(
            symbol=symbol.upper(), fetched=len(articles), upserted=upserted,
            collection=settings.qdrant_collection,
        )


_singleton: IngestService | None = None


def get_ingest_service() -> IngestService:
    global _singleton
    if _singleton is None:
        _singleton = IngestService()
    return _singleton
