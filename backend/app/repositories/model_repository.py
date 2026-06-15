"""Loads and caches models. Adapter over two backends (sklearn joblib + torch).

Deep models are rebuilt with the *original* architecture factory
(src.models.model_factory.build_torch_model) and their saved state_dict loaded
with strict=True, guaranteeing the reconstruction matches training exactly.
"""
from __future__ import annotations

import warnings
from collections import OrderedDict
from pathlib import Path

import numpy as np

from app.config import settings
from app.models.registry import ModelEntry
from app.research_bridge import load_research_callables

warnings.filterwarnings("ignore")
_R = load_research_callables()


class LoadedModel:
    """Wraps a concrete predictor with a uniform predict() returning a scalar log-return."""

    def __init__(self, entry: ModelEntry, obj, kind: str) -> None:
        self.entry = entry
        self.obj = obj
        self.kind = kind

    def predict_classical(self, X) -> float:
        return float(np.asarray(self.obj.predict(X)).ravel()[0])

    def predict_deep(self, window: np.ndarray) -> float:
        import torch

        with torch.no_grad():
            t = torch.from_numpy(window.astype(np.float32))
            out = self.obj(t)
            return float(np.asarray(out.detach().cpu().numpy()).ravel()[0])


class ModelRepository:
    def __init__(self, models_dir: Path | None = None, cache_size: int | None = None) -> None:
        self.models_dir = Path(models_dir or settings.models_dir)
        self._cap = cache_size or settings.model_cache_size
        self._cache: "OrderedDict[str, LoadedModel]" = OrderedDict()

    def _abs(self, entry: ModelEntry) -> Path:
        return self.models_dir / entry.path

    def load(self, entry: ModelEntry) -> LoadedModel:
        if entry.model_id in self._cache:
            self._cache.move_to_end(entry.model_id)
            return self._cache[entry.model_id]

        lm = self._load_classical(entry) if entry.kind == "classical" else self._load_deep(entry)
        self._cache[entry.model_id] = lm
        self._cache.move_to_end(entry.model_id)
        while len(self._cache) > self._cap:
            self._cache.popitem(last=False)
        return lm

    def _load_classical(self, entry: ModelEntry) -> LoadedModel:
        import joblib

        path = self._abs(entry)
        if not path.exists():
            raise FileNotFoundError(f"Checkpoint not found: {path}")
        pipe = joblib.load(path)
        return LoadedModel(entry, pipe, "classical")

    def _load_deep(self, entry: ModelEntry) -> LoadedModel:
        import torch

        path = self._abs(entry)
        if not path.exists():
            raise FileNotFoundError(f"Checkpoint not found: {path}")
        ck = torch.load(path, map_location="cpu", weights_only=False)
        cfg = ck.get("config", {}) or entry.config
        hidden = int(cfg.get("hidden_size", 32))
        seq_len = int(ck.get("seq_len", entry.seq_len or cfg.get("lag_window", 64)))
        n_features = int(ck.get("n_features", entry.n_features))
        model = _R["build_torch_model"](
            model_name=ck.get("model_name", entry.model_name),
            input_size=n_features,
            seq_len=seq_len,
            hidden_size=hidden,
        )
        state = ck["state_dict"]
        # NHITS uses LazyLinear; run one dummy forward to materialize shapes before load.
        if any("LazyLinear" in type(m).__name__ for m in model.modules()) or entry.model_name == "NHITS":
            with torch.no_grad():
                model(torch.zeros(1, seq_len, n_features))
        model.load_state_dict(state, strict=True)
        model.eval()
        return LoadedModel(entry, model, "deep")


_singleton: ModelRepository | None = None


def get_model_repository() -> ModelRepository:
    global _singleton
    if _singleton is None:
        _singleton = ModelRepository()
    return _singleton
