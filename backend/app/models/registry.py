"""In-memory model registry loaded from model_registry.json (Singleton)."""
from __future__ import annotations

import json
from dataclasses import dataclass, field
from functools import lru_cache
from typing import Any

from app.config import settings


@dataclass(frozen=True)
class ModelEntry:
    model_id: str
    kind: str            # classical | deep
    family: str
    horizon: str         # h1 | h5
    group: str           # G1..G5
    algo: str
    model_name: str
    basis: str           # panel | index
    feature_names: list[str]
    n_features: int
    seq_len: int
    path: str
    symbol: str | None = None
    variant: str | None = None
    metrics: dict[str, Any] = field(default_factory=dict)
    config: dict[str, Any] = field(default_factory=dict)

    @property
    def needs_symbol(self) -> bool:
        """Symbol-basis models predict a user-chosen stock; index-basis predict the market."""
        return self.basis == "panel"

    @property
    def needs_sentiment(self) -> bool:
        return self.group in {"G4", "G5"}


class Registry:
    def __init__(self, path: str | None = None) -> None:
        p = path or str(settings.registry_path)
        with open(p, "r", encoding="utf-8") as fh:
            payload = json.load(fh)
        self.models_dir = payload.get("models_dir", str(settings.models_dir))
        self._by_id: dict[str, ModelEntry] = {}
        for mid, m in payload["models"].items():
            self._by_id[mid] = ModelEntry(
                model_id=m["model_id"],
                kind=m["kind"],
                family=m["family"],
                horizon=m["horizon"],
                group=m["group"],
                algo=m["algo"],
                model_name=m["model_name"],
                basis=m["basis"],
                feature_names=m["feature_names"],
                n_features=m["n_features"],
                seq_len=m["seq_len"],
                path=m["path"],
                symbol=m.get("symbol"),
                variant=m.get("variant"),
                metrics=m.get("metrics", {}) or {},
                config=m.get("config", {}) or {},
            )

    def get(self, model_id: str) -> ModelEntry:
        if model_id not in self._by_id:
            raise KeyError(f"Unknown model_id: {model_id}")
        return self._by_id[model_id]

    def all(self) -> list[ModelEntry]:
        return list(self._by_id.values())

    def filter(
        self,
        *,
        horizon: str | None = None,
        kind: str | None = None,
        family: str | None = None,
        group: str | None = None,
        basis: str | None = None,
        symbol: str | None = None,
    ) -> list[ModelEntry]:
        out = self.all()
        if horizon:
            out = [m for m in out if m.horizon == horizon]
        if kind:
            out = [m for m in out if m.kind == kind]
        if family:
            out = [m for m in out if m.family == family]
        if group:
            out = [m for m in out if m.group == group.upper()]
        if basis:
            out = [m for m in out if m.basis == basis]
        if symbol:
            out = [m for m in out if (m.symbol is None or m.symbol == symbol.upper())]
        return out

    def families(self) -> list[str]:
        return sorted({m.family for m in self.all()})

    def symbols(self) -> list[str]:
        return sorted({m.symbol for m in self.all() if m.symbol})


@lru_cache(maxsize=1)
def get_registry() -> Registry:
    return Registry()
