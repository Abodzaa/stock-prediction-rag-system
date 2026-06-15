"""Bridge to the original research codebase.

We *reuse* the exact training-time implementations (model architectures, feature
formulas, sequence building, sentiment aggregation) instead of reimplementing
them, so inference matches training. Nothing is retrained.

This module wires sys.path so the ``src`` package under the research root is
importable, then re-exports the functions the backend needs.
"""
from __future__ import annotations

import sys
from pathlib import Path

from app.config import settings

_research_root = str(settings.research_root)
_nlp_dir = str(settings.research_root / "src" / "nlp")  # nlp scripts use bare imports

for p in (_research_root, _nlp_dir):
    if p not in sys.path:
        sys.path.insert(0, p)


def load_research_callables():
    """Import and return the reused research functions (lazy, so import errors surface clearly)."""
    from src.models.model_factory import build_torch_model
    from src.features.build_equity_panel_features import (
        compute_symbol_features,
        add_cross_sectional_features,
        winsorize_series,
    )
    from src.features.build_research_features import (
        compute_g1_features,
        compute_g2_technical,
    )
    from src.experiments.run_deep_torch_experiments import (
        build_sequences,
        standardize_by_train,
    )
    from src.nlp.news_dataset_utils import (
        aggregate_daily_sentiment,
        entropy_from_probs,
    )

    return {
        "build_torch_model": build_torch_model,
        "compute_symbol_features": compute_symbol_features,
        "add_cross_sectional_features": add_cross_sectional_features,
        "winsorize_series": winsorize_series,
        "compute_g1_features": compute_g1_features,
        "compute_g2_technical": compute_g2_technical,
        "build_sequences": build_sequences,
        "standardize_by_train": standardize_by_train,
        "aggregate_daily_sentiment": aggregate_daily_sentiment,
        "entropy_from_probs": entropy_from_probs,
    }
