"""Explanation controller. SSE streaming for the RAG narrative + a JSON variant."""
from __future__ import annotations

import json

from fastapi import APIRouter, HTTPException, Query
from fastapi.responses import StreamingResponse

from app.models.schemas import ExplainRequest
from app.services.explanation_service import get_explanation_service

router = APIRouter(prefix="/api", tags=["explain"])


def _sse(event: dict) -> str:
    return f"data: {json.dumps(event)}\n\n"


@router.get("/explain/stream")
def explain_stream(
    model_id: str = Query(...),
    symbol: str | None = Query(None),
    as_of: str | None = Query(None),
    question: str | None = Query(None),
):
    svc = get_explanation_service()

    def gen():
        try:
            for event in svc.explain_stream(model_id, symbol, as_of, question):
                yield _sse(event)
        except KeyError as e:
            yield _sse({"type": "error", "data": f"unknown model: {e}"})
        except ValueError as e:
            yield _sse({"type": "error", "data": str(e)})
        except Exception as e:
            yield _sse({"type": "error", "data": f"{type(e).__name__}: {e}"})

    return StreamingResponse(gen(), media_type="text/event-stream",
                             headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no"})


@router.post("/explain")
def explain(req: ExplainRequest):
    try:
        return get_explanation_service().explain(req.model_id, req.symbol, req.as_of, req.question)
    except KeyError as e:
        raise HTTPException(status_code=404, detail=str(e))
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"{type(e).__name__}: {e}")
