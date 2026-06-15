"""e5 embedding adapter.

multilingual-e5-base was trained with mandatory prefixes:
  documents -> "passage: " + text     queries -> "query: " + text
Vectors are L2-normalized for cosine.
"""
from __future__ import annotations

import numpy as np

from app.config import settings


class EmbeddingRepository:
    def __init__(self) -> None:
        self._model = None

    def _lazy(self):
        if self._model is None:
            from sentence_transformers import SentenceTransformer

            self._model = SentenceTransformer(settings.embedding_model)
        return self._model

    def embed_passages(self, texts: list[str]) -> np.ndarray:
        m = self._lazy()
        return np.asarray(
            m.encode([f"passage: {t}" for t in texts], normalize_embeddings=True, show_progress_bar=False)
        )

    def embed_query(self, text: str) -> np.ndarray:
        m = self._lazy()
        return np.asarray(
            m.encode(f"query: {text}", normalize_embeddings=True, show_progress_bar=False)
        )

    @property
    def dim(self) -> int:
        return settings.embedding_dim


_singleton: EmbeddingRepository | None = None


def get_embedding_repository() -> EmbeddingRepository:
    global _singleton
    if _singleton is None:
        _singleton = EmbeddingRepository()
    return _singleton
