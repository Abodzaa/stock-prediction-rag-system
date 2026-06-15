"""Models catalog controller (REST). Thin: shape registry -> DTOs."""
from __future__ import annotations

from fastapi import APIRouter, Query

from app.models.registry import get_registry
from app.models.schemas import ModelSummary

router = APIRouter(prefix="/api/models", tags=["models"])


def _to_summary(m) -> ModelSummary:
    return ModelSummary(
        model_id=m.model_id, kind=m.kind, family=m.family, horizon=m.horizon,
        group=m.group, model_name=m.model_name, basis=m.basis,
        needs_symbol=m.needs_symbol, needs_sentiment=m.needs_sentiment,
        n_features=m.n_features, seq_len=m.seq_len, symbol=m.symbol,
        variant=m.variant, metrics=m.metrics,
    )


@router.get("")
def list_models(
    horizon: str | None = Query(None, pattern="^h[15]$"),
    kind: str | None = None,
    family: str | None = None,
    group: str | None = None,
    basis: str | None = None,
    symbol: str | None = None,
    limit: int = 500,
    offset: int = 0,
):
    reg = get_registry()
    items = reg.filter(horizon=horizon, kind=kind, family=family, group=group, basis=basis, symbol=symbol)
    total = len(items)
    items = items[offset: offset + limit]
    return {"total": total, "count": len(items), "models": [_to_summary(m).model_dump() for m in items]}


@router.get("/facets")
def facets():
    reg = get_registry()
    ms = reg.all()
    return {
        "total": len(ms),
        "families": reg.families(),
        "horizons": sorted({m.horizon for m in ms}),
        "kinds": sorted({m.kind for m in ms}),
        "groups": sorted({m.group for m in ms}),
        "bases": sorted({m.basis for m in ms}),
        "symbols": reg.symbols(),
    }


@router.get("/{model_id}")
def get_model(model_id: str):
    m = get_registry().get(model_id)
    out = _to_summary(m).model_dump()
    out["feature_names"] = m.feature_names
    out["path"] = m.path
    out["config"] = m.config
    return out
