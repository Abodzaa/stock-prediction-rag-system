"""Signal context controller."""
from __future__ import annotations

from fastapi import APIRouter, HTTPException, Query

from app.models.schemas import SignalContextResponse
from app.services.signal_context_service import get_signal_context_service

router = APIRouter(prefix="/api", tags=["context"])


@router.get("/signal-context", response_model=SignalContextResponse)
def signal_context(
    symbol: str = Query(..., min_length=1),
    horizon: str = Query(..., pattern="^h[15]$"),
    predicted_pct: float = Query(...),
    benchmark_symbol: str | None = Query(None),
):
    try:
        return get_signal_context_service().get_context(
            symbol=symbol,
            horizon=horizon,
            predicted_pct=predicted_pct,
            benchmark_symbol=benchmark_symbol,
        )
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc))
    except Exception as exc:
        raise HTTPException(status_code=500, detail=f"{type(exc).__name__}: {exc}")
