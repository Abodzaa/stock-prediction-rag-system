"""Pydantic request/response DTOs."""
from __future__ import annotations

from typing import Any

from pydantic import BaseModel, Field


class ModelSummary(BaseModel):
    model_id: str
    kind: str
    family: str
    horizon: str
    group: str
    model_name: str
    basis: str
    needs_symbol: bool
    needs_sentiment: bool
    n_features: int
    seq_len: int
    symbol: str | None = None
    variant: str | None = None
    metrics: dict[str, Any] = {}


class FeatureContribution(BaseModel):
    feature: str
    value: float | None = None
    contribution: float | None = None
    importance: float | None = None


class PredictRequest(BaseModel):
    model_id: str
    symbol: str | None = Field(default=None, description="Required for panel/symbol-basis models")
    as_of: str | None = Field(default=None, description="ISO date; defaults to latest available")


class PredictResponse(BaseModel):
    model_id: str
    symbol: str
    horizon: str
    as_of: str
    predicted_log_return: float
    predicted_pct: float
    direction: str           # up | down | flat
    confidence: float        # 0..1 heuristic
    drivers: list[FeatureContribution] = []
    feature_snapshot: dict[str, float] = {}
    metrics: dict[str, Any] = {}
    warnings: list[str] = []


class ExplainRequest(BaseModel):
    model_id: str
    symbol: str | None = None
    as_of: str | None = None
    question: str | None = None


class NewsCitation(BaseModel):
    doc_id: str
    headline: str
    source: str
    url: str
    published: str
    score: float | None = None


class IngestRequest(BaseModel):
    symbol: str
    lookback_days: int | None = None


class IngestResponse(BaseModel):
    symbol: str
    fetched: int
    upserted: int
    collection: str
