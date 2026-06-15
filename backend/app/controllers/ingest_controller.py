"""Ingestion controller (REST enqueue) + news passthrough for the UI."""
from __future__ import annotations

from fastapi import APIRouter, HTTPException, Query

from app.models.schemas import IngestRequest, IngestResponse
from app.repositories.news_repository import get_news_repository
from app.services.ingest_service import get_ingest_service

router = APIRouter(prefix="/api", tags=["news"])


@router.post("/ingest", response_model=IngestResponse)
def ingest(req: IngestRequest):
    try:
        return get_ingest_service().ingest_symbol(req.symbol, req.lookback_days)
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"{type(e).__name__}: {e}")


@router.get("/news")
def news(symbol: str = Query(...), limit: int = 20):
    try:
        return {"symbol": symbol.upper(), "articles": get_news_repository().fetch(symbol, limit=limit)}
    except Exception as e:
        raise HTTPException(status_code=502, detail=f"{type(e).__name__}: {e}")
