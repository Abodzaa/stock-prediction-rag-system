"""Prediction controller (REST, request->response)."""
from __future__ import annotations

from fastapi import APIRouter, HTTPException

from app.models.schemas import PredictRequest, PredictResponse
from app.services.inference_service import get_inference_service

router = APIRouter(prefix="/api", tags=["predict"])


@router.post("/predict", response_model=PredictResponse)
def predict(req: PredictRequest):
    try:
        return get_inference_service().predict(req.model_id, req.symbol, req.as_of)
    except KeyError as e:
        raise HTTPException(status_code=404, detail=str(e))
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"{type(e).__name__}: {e}")
