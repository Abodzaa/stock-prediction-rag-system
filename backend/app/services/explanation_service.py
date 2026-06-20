"""ExplanationService (RAG): prediction + retrieved news -> grounded narrative.

Pipeline: predict -> ingest fresh news -> retrieve (hybrid) -> Groq generation.
Streams structured SSE events (prediction, citations, tokens, done) and is
resilient: if Qdrant/Groq are unavailable it falls back to direct news + an
offline summary so the endpoint still returns something useful.
"""
from __future__ import annotations

from collections.abc import Iterator

from app.config import settings
from app.models.schemas import NewsCitation, PredictResponse
from app.repositories.groq_repository import get_llm_client
from app.repositories.news_repository import get_news_repository
from app.services.inference_service import InferenceService, get_inference_service
from app.services.ingest_service import get_ingest_service
from app.services.retrieval_service import get_retrieval_service

_SYSTEM = (
    "You are a financial research assistant. Using ONLY the model prediction and the "
    "numbered news context provided, explain in 3-5 sentences why {symbol} may move "
    "{direction} over the {horizon} horizon. Reference supporting headlines inline as [n]. "
    "Tie the explanation to the quantitative drivers when given. If the news context is "
    "insufficient to explain the move, say so explicitly. Be concise and neutral; do not "
    "give investment advice."
)


class ExplanationService:
    def __init__(self, inference: InferenceService | None = None) -> None:
        self.inference = inference or get_inference_service()
        self.news = get_news_repository()

    # ------------------------------------------------------------- retrieval
    def _gather_citations(self, symbol: str, direction: str) -> list[NewsCitation]:
        move = "rally gains surge upgrade beat" if direction == "up" else (
            "decline drop selloff downgrade miss" if direction == "down" else "outlook guidance update")
        query = f"{symbol} stock {move} earnings catalyst news"
        hits: list[dict] = []
        try:
            get_ingest_service().ingest_symbol(symbol)
            hits = get_retrieval_service().retrieve(query, symbol=symbol)
        except Exception:
            hits = []
        if not hits:  # fallback: newest articles directly from the provider
            hits = self.news.fetch(symbol, limit=settings.final_k)
        cites = []
        for h in hits[: settings.final_k]:
            cites.append(NewsCitation(
                doc_id=str(h.get("_id") or h.get("news_id") or h.get("url") or ""),
                headline=str(h.get("headline", ""))[:240],
                summary=str(h.get("summary", ""))[:420],
                source=str(h.get("source", "")),
                url=str(h.get("url", "")),
                published=str(h.get("published_utc", "")),
                score=h.get("_rrf") or h.get("_score"),
            ))
        return cites

    @staticmethod
    def _build_user_prompt(pred: PredictResponse, cites: list[NewsCitation], question: str | None) -> str:
        lines = [
            f"Symbol: {pred.symbol}",
            f"Model: {pred.model_id}",
            f"Horizon: {pred.horizon} ({'1 day' if pred.horizon == 'h1' else '5 days'})",
            f"Predicted move: {pred.direction.upper()} {pred.predicted_pct:+.2f}% "
            f"(log-return {pred.predicted_log_return:+.5f})",
            f"Model confidence: {pred.confidence:.2f}",
        ]
        if pred.drivers:
            drv = ", ".join(
                f"{d.feature}"
                + (f" (contrib {d.contribution:+.4f})" if d.contribution is not None
                   else f" (importance {d.importance:.3f})" if d.importance is not None else "")
                for d in pred.drivers[:6]
            )
            lines.append(f"Top quantitative drivers: {drv}")
        lines.append("")
        lines.append("News context:")
        if cites:
            for i, c in enumerate(cites, 1):
                lines.append(f"[{i}] ({c.source}, {c.published[:10]}) {c.headline}")
        else:
            lines.append("(no recent news retrieved)")
        if question:
            lines.append("")
            lines.append(f"User question: {question}")
        return "\n".join(lines)

    # --------------------------------------------------------------- public
    def explain_stream(
        self, model_id: str, symbol: str | None, as_of: str | None, question: str | None
    ) -> Iterator[dict]:
        pred = self.inference.predict(model_id, symbol, as_of)
        yield {"type": "prediction", "data": pred.model_dump()}

        cites = self._gather_citations(pred.symbol, pred.direction)
        yield {"type": "citations", "data": [c.model_dump() for c in cites]}

        system = _SYSTEM.format(symbol=pred.symbol, direction=pred.direction,
                                horizon="1-day" if pred.horizon == "h1" else "5-day")
        user = self._build_user_prompt(pred, cites, question)
        llm = get_llm_client()
        try:
            for token in llm.generate_stream(system, user):
                yield {"type": "token", "data": token}
        except Exception as exc:
            yield {"type": "token", "data": f"\n[generation error: {exc}]"}
        yield {"type": "done", "data": {}}

    def explain(self, model_id: str, symbol: str | None, as_of: str | None, question: str | None) -> dict:
        pred = self.inference.predict(model_id, symbol, as_of)
        cites = self._gather_citations(pred.symbol, pred.direction)
        system = _SYSTEM.format(symbol=pred.symbol, direction=pred.direction,
                                horizon="1-day" if pred.horizon == "h1" else "5-day")
        user = self._build_user_prompt(pred, cites, question)
        try:
            text = get_llm_client().generate(system, user)
        except Exception as exc:
            text = f"[generation unavailable: {exc}]\n\n{user}"
        return {
            "prediction": pred.model_dump(),
            "citations": [c.model_dump() for c in cites],
            "explanation": text,
        }


_singleton: ExplanationService | None = None


def get_explanation_service() -> ExplanationService:
    global _singleton
    if _singleton is None:
        _singleton = ExplanationService()
    return _singleton
