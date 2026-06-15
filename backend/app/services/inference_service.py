"""InferenceService: features -> model -> prediction + feature drivers.

Business logic only; no HTTP awareness (MVC Model layer).
"""
from __future__ import annotations

import math

import numpy as np

from app.config import settings
from app.features.feature_pipeline import FeaturePipeline
from app.models.registry import ModelEntry, Registry, get_registry
from app.models.schemas import FeatureContribution, PredictResponse
from app.repositories.model_repository import ModelRepository, get_model_repository


class InferenceService:
    def __init__(
        self,
        registry: Registry | None = None,
        models: ModelRepository | None = None,
        features: FeaturePipeline | None = None,
    ) -> None:
        self.registry = registry or get_registry()
        self.models = models or get_model_repository()
        self.features = features or FeaturePipeline()

    def resolve_symbol(self, entry: ModelEntry, symbol: str | None) -> str:
        if entry.needs_symbol:
            if not symbol:
                raise ValueError(f"Model {entry.model_id} requires a 'symbol'.")
            return symbol.upper()
        # index/market-basis models predict the market index
        return symbol.upper() if symbol else settings.index_symbol

    def predict(self, model_id: str, symbol: str | None, as_of: str | None) -> PredictResponse:
        entry = self.registry.get(model_id)
        sym = self.resolve_symbol(entry, symbol)
        loaded = self.models.load(entry)
        warnings: list[str] = []

        if entry.kind == "classical":
            X, values, used_date, w = self.features.vector_for_classical(entry, sym, as_of)
            warnings += w
            pred = loaded.predict_classical(X)
            drivers = self._classical_drivers(loaded.obj, entry, X)
            snapshot = {k: _f(v) for k, v in values.items()}
        else:
            window, snapshot, used_date, w = self.features.window_for_deep(entry, sym, as_of)
            warnings += w
            pred = loaded.predict_deep(window)
            drivers = []  # deep drivers are not linearly decomposable; explanation uses news

        horizon_days = 5 if entry.horizon == "h5" else 1
        pct = (math.exp(pred) - 1.0) * 100.0
        direction = "up" if pred > 1e-5 else ("down" if pred < -1e-5 else "flat")
        confidence = self._confidence(pred, entry, horizon_days)

        return PredictResponse(
            model_id=entry.model_id,
            symbol=sym,
            horizon=entry.horizon,
            as_of=used_date,
            predicted_log_return=_f(pred),
            predicted_pct=_f(pct),
            direction=direction,
            confidence=_f(confidence),
            drivers=drivers,
            feature_snapshot={k: v for k, v in snapshot.items() if v is not None},
            metrics=entry.metrics,
            warnings=warnings,
        )

    # ----------------------------------------------------------- drivers
    def _classical_drivers(self, pipe, entry: ModelEntry, X) -> list[FeatureContribution]:
        model = pipe.named_steps.get("model", pipe.steps[-1][1])
        names = entry.feature_names
        # Linear models: contribution = coef * scaled feature value.
        if hasattr(model, "coef_"):
            try:
                pre = pipe[:-1].transform(X)
                scaled = np.asarray(pre).ravel()
                coef = np.asarray(model.coef_).ravel()
                contrib = coef * scaled
                idx = np.argsort(np.abs(contrib))[::-1][:8]
                return [
                    FeatureContribution(
                        feature=names[i],
                        value=_f(np.asarray(X)[0][i]),
                        contribution=_f(contrib[i]),
                    )
                    for i in idx
                ]
            except Exception:
                pass
        # Tree models: global feature importances.
        if hasattr(model, "feature_importances_"):
            imp = np.asarray(model.feature_importances_).ravel()
            idx = np.argsort(imp)[::-1][:8]
            return [
                FeatureContribution(
                    feature=names[i],
                    value=_f(np.asarray(X)[0][i]),
                    importance=_f(imp[i]),
                )
                for i in idx
            ]
        return []

    def _confidence(self, pred: float, entry: ModelEntry, horizon_days: int) -> float:
        # Magnitude signal scaled by a per-horizon typical daily move (~1.5%/day).
        typical = 0.015 * math.sqrt(horizon_days)
        mag = math.tanh(abs(pred) / max(typical, 1e-6))
        # Blend with the model's historical directional accuracy when available.
        da = entry.metrics.get("directional_accuracy")
        if isinstance(da, (int, float)) and da == da:
            return float(0.5 * mag + 0.5 * max(0.0, min(1.0, da)))
        return float(mag)


def _f(v) -> float:
    try:
        f = float(v)
        return 0.0 if (math.isnan(f) or math.isinf(f)) else f
    except (TypeError, ValueError):
        return 0.0


_singleton: InferenceService | None = None


def get_inference_service() -> InferenceService:
    global _singleton
    if _singleton is None:
        _singleton = InferenceService()
    return _singleton
