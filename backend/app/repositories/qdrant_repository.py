"""Qdrant vector store adapter with hybrid (dense + sparse) named vectors.

Dense  = e5 (768, cosine). Sparse = hashed term-frequency bag-of-words.
Point IDs are deterministic (uuid5 of news_id) so re-ingest is idempotent.
"""
from __future__ import annotations

import re
import uuid
from collections import Counter

import numpy as np

from app.config import settings

_SPARSE_DIM = 2**18
_TOKEN_RE = re.compile(r"[A-Za-z][A-Za-z0-9]+")


def sparse_encode(text: str) -> tuple[list[int], list[float]]:
    toks = [t.lower() for t in _TOKEN_RE.findall(text)]
    if not toks:
        return [], []
    counts = Counter(hash(t) % _SPARSE_DIM for t in toks)
    n = float(len(toks))
    idx = sorted(counts)
    vals = [counts[i] / n for i in idx]
    return idx, vals


def point_id(news_id: str) -> str:
    return str(uuid.uuid5(uuid.NAMESPACE_URL, news_id))


class QdrantRepository:
    def __init__(self) -> None:
        self._client = None
        self._ready = False

    def _lazy(self):
        if self._client is None:
            from qdrant_client import QdrantClient

            self._client = QdrantClient(url=settings.qdrant_url, api_key=settings.qdrant_api_key)
        return self._client

    def ensure_collection(self) -> None:
        if self._ready:
            return
        from qdrant_client import models

        client = self._lazy()
        existing = {c.name for c in client.get_collections().collections}
        if settings.qdrant_collection not in existing:
            client.create_collection(
                collection_name=settings.qdrant_collection,
                vectors_config={"dense": models.VectorParams(size=settings.embedding_dim,
                                                             distance=models.Distance.COSINE)},
                sparse_vectors_config={"sparse": models.SparseVectorParams()},
            )
            client.create_payload_index(
                settings.qdrant_collection, field_name="symbol",
                field_schema=models.PayloadSchemaType.KEYWORD,
            )
        self._ready = True

    def upsert(self, records: list[dict], dense: np.ndarray) -> int:
        from qdrant_client import models

        self.ensure_collection()
        client = self._lazy()
        points = []
        for i, rec in enumerate(records):
            idx, vals = sparse_encode(f"{rec.get('headline','')} {rec.get('summary','')}")
            points.append(models.PointStruct(
                id=point_id(rec["news_id"]),
                vector={
                    "dense": dense[i].tolist(),
                    "sparse": models.SparseVector(indices=idx, values=vals),
                },
                payload=rec,
            ))
        if points:
            client.upsert(collection_name=settings.qdrant_collection, points=points)
        return len(points)

    def search_dense(self, query_vec: np.ndarray, top_k: int, symbol: str | None = None):
        from qdrant_client import models

        self.ensure_collection()
        client = self._lazy()
        flt = self._symbol_filter(symbol, models)
        res = client.query_points(
            collection_name=settings.qdrant_collection,
            query=query_vec.tolist(), using="dense", limit=top_k, query_filter=flt,
            with_payload=True,
        )
        return res.points

    def search_sparse(self, text: str, top_k: int, symbol: str | None = None):
        from qdrant_client import models

        self.ensure_collection()
        client = self._lazy()
        idx, vals = sparse_encode(text)
        if not idx:
            return []
        flt = self._symbol_filter(symbol, models)
        res = client.query_points(
            collection_name=settings.qdrant_collection,
            query=models.SparseVector(indices=idx, values=vals), using="sparse",
            limit=top_k, query_filter=flt, with_payload=True,
        )
        return res.points

    @staticmethod
    def _symbol_filter(symbol: str | None, models):
        if not symbol:
            return None
        return models.Filter(must=[models.FieldCondition(
            key="symbol", match=models.MatchValue(value=symbol.upper()))])


_singleton: QdrantRepository | None = None


def get_qdrant_repository() -> QdrantRepository:
    global _singleton
    if _singleton is None:
        _singleton = QdrantRepository()
    return _singleton
