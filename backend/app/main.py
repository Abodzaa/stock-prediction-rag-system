"""FastAPI application wiring (MVC). Controllers only; logic lives in services."""
from __future__ import annotations

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from app.config import settings
from app.controllers import (
    context_controller,
    explain_controller,
    ingest_controller,
    models_controller,
    predict_controller,
)
from app.models.registry import get_registry

app = FastAPI(
    title="Equity Prediction + RAG Explanation API",
    version="1.0.0",
    description="Inference over a deduplicated model zoo (classical + deep, H1/H5) with "
                "retrieval-augmented news explanations of each prediction.",
)

origins = ["*"] if settings.cors_origins.strip() == "*" else [
    o.strip() for o in settings.cors_origins.split(",") if o.strip()
]
app.add_middleware(
    CORSMiddleware, allow_origins=origins, allow_credentials=True,
    allow_methods=["*"], allow_headers=["*"],
)

app.include_router(models_controller.router)
app.include_router(predict_controller.router)
app.include_router(context_controller.router)
app.include_router(explain_controller.router)
app.include_router(ingest_controller.router)


@app.get("/api/health")
def health():
    reg = get_registry()
    return {
        "status": "ok",
        "models": len(reg.all()),
        "models_dir": reg.models_dir,
        "groq_configured": bool(settings.groq_api_key),
        "news_provider": settings.news_provider,
        "retrieval_mode": settings.retrieval_mode,
    }


@app.get("/")
def root():
    return {"service": "equity-rag-inference", "docs": "/docs", "health": "/api/health"}
