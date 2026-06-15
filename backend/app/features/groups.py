"""Canonical feature-group definitions, mirrored from the original research code.

These lists define the *exact ordered* feature columns each model group expects.
They are the single source of truth shared by the registry builder, the feature
pipeline, and the inference services.

Two bases exist (see DECISIONS.md):

* ``panel``  - per-symbol features; G3 = cross-sectional z-scores + breadth
               (from src/features/build_equity_panel_features.py)
* ``index``  - market/index-level features; G3 = breadth aggregates
               (from src/features/build_research_features.py)

G1/G2 column *names* are identical across bases; only the underlying series
(symbol vs index) and the G3 columns differ.
"""

from __future__ import annotations

# --- shared price / technical blocks (identical names in both bases) ---------
G1_COLS: list[str] = [
    "g1_ret_1d",
    "g1_ret_5d",
    "g1_ret_10d",
    "g1_intraday_log_ret",
    "g1_overnight_log_ret",
    "g1_hl_range",
    "g1_volume_log",
    "g1_volume_chg_1d",
]

G2_COLS: list[str] = [
    "g2_realized_vol_10",
    "g2_realized_vol_20",
    "g2_atr_14",
    "g2_rsi_14",
    "g2_macd",
    "g2_macd_signal",
    "g2_macd_hist",
    "g2_bollinger_z20",
    "g2_volume_z20",
]

# --- G3 differs by basis -----------------------------------------------------
# Panel basis (per-symbol cross-sectional + breadth) -> build_equity_panel_features
G3_PANEL_COLS: list[str] = [
    "g3_cs_g1_ret_1d_z",
    "g3_cs_g2_realized_vol_20_z",
    "g3_cs_g1_volume_chg_1d_z",
    "g3_adv_dec_spread",
    "g3_adv_dec_ratio",
]

# Index basis (market breadth) -> build_research_features.compute_g3_breadth
G3_INDEX_COLS: list[str] = [
    "g3_active_constituents",
    "g3_pct_above_50dma",
    "g3_pct_above_200dma",
    "g3_adv_dec_spread",
    "g3_adv_dec_ratio",
    "g3_ad_line",
    "g3_equal_weight_ret_1d",
]

# --- sentiment blocks (market-level daily aggregates) ------------------------
# Produced by aggregate_daily_sentiment(prefix=...) in src/nlp/news_dataset_utils.py
def _sentiment_cols(prefix: str, with_rag_support: bool) -> list[str]:
    base = [
        "article_count",
        "source_count",
        "sent_pos_mean",
        "sent_neg_mean",
        "sent_neu_mean",
        "sent_score_mean",
        "sent_score_std",
        "sent_entropy_mean",
    ]
    if with_rag_support:
        base += ["rag_support_mean", "rag_support_std"]
    base += [
        "sent_score_roll3",
        "sent_score_roll5",
        "article_count_roll3",
        "article_count_roll5",
    ]
    return [f"{prefix}_{c}" for c in base]


G5_FINBERT_COLS: list[str] = _sentiment_cols("g5_sent_finbert", with_rag_support=False)  # 12
G5_RAG_COLS: list[str] = _sentiment_cols("g5_sent_rag", with_rag_support=True)  # 14


def sentiment_cols_for(family: str) -> list[str]:
    """Return the sentiment column block used by a given experiment family."""
    if "rag" in family.lower():
        return G5_RAG_COLS
    return G5_FINBERT_COLS


def basis_for(family: str) -> str:
    """Map an experiment family (top-level models/ dir) to its feature basis."""
    fam = family.lower()
    if fam.startswith("panel_full") or fam.startswith("preprocessing_variants"):
        return "panel"
    if fam.startswith("nasdaq100_best_deep_per_symbol"):
        return "panel"
    # sentiment_eval_*, deep_preprocessed_sentiment* -> index/market basis
    return "index"


def expected_columns(family: str, group: str) -> list[str]:
    """Ordered feature columns expected by (family, group).

    Used for deep checkpoints (which carry no feature_names_in_) and as a
    fallback for classical models whose joblib fails to unpickle.
    """
    basis = basis_for(family)
    g3 = G3_PANEL_COLS if basis == "panel" else G3_INDEX_COLS
    sent = sentiment_cols_for(family)

    g = group.upper()
    if g.startswith("G1"):
        return list(G1_COLS)
    if g.startswith("G2"):
        return G1_COLS + G2_COLS
    if g.startswith("G3"):
        return G1_COLS + G2_COLS + g3
    if g.startswith("G4"):
        return G1_COLS + G2_COLS + g3 + sent
    if g.startswith("G5"):
        return list(sent)
    raise ValueError(f"Unknown feature group: {group}")
