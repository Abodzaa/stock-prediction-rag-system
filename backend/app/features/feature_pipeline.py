"""Feature engineering for live inference.

Reproduces the *exact* training-time features by reusing the research code
(compute_symbol_features, add_cross_sectional_features, compute_g1/g2, ...).
Two bases are supported (see DECISIONS.md):

* panel  -> per-symbol g1/g2 (+ winsor/clip) + cross-sectional g3 from the universe
* index  -> index g1/g2 + market breadth g3 + daily news sentiment g5

Deep models additionally need a (seq_len x n_features) window, standardized
feature-wise. Training standardized per fold on the train slice; that slice mean/
std is not persisted, so at inference we standardize using the available trailing
history (the inference-time analogue). This is documented as an approximation.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from app.config import settings
from app.models.registry import ModelEntry
from app.repositories.price_repository import PriceRepository, get_price_repository
from app.research_bridge import load_research_callables

_R = load_research_callables()


class FeatureBuildError(RuntimeError):
    pass


class FeaturePipeline:
    def __init__(self, price_repo: PriceRepository | None = None, sentiment_pipeline=None) -> None:
        self.price = price_repo or get_price_repository()
        self.sentiment = sentiment_pipeline  # lazily provided; only needed for G4/G5

    # ------------------------------------------------------------------ panel
    def _panel_symbol_block(self, symbol: str) -> pd.DataFrame:
        df = self.price.get_ohlcv(symbol)
        feats = _R["compute_symbol_features"](df.set_index("Date"))
        feats = feats.reset_index().rename(columns={"index": "Date"})
        if "Date" not in feats.columns:
            feats = feats.rename(columns={feats.columns[0]: "Date"})
        for col in ["g1_ret_1d", "g1_ret_5d", "g1_ret_10d", "target_h1", "target_h5"]:
            feats[col] = _R["winsorize_series"](feats[col], 0.005, 0.995)
        for col in ["g1_hl_range", "g2_volume_z20", "g2_bollinger_z20"]:
            feats[col] = feats[col].clip(lower=-10.0, upper=10.0)
        return feats

    def _panel_cross_sectional(self, target_symbol: str) -> pd.DataFrame:
        universe = self.price.get_universe()
        if target_symbol.upper() not in {s.upper() for s in universe}:
            universe = [target_symbol] + universe
        ohlcv = self.price.get_universe_ohlcv(universe)
        if target_symbol not in ohlcv:
            ohlcv[target_symbol] = self.price.get_ohlcv(target_symbol)

        needed = ["g1_ret_1d", "g2_realized_vol_20", "g1_volume_chg_1d"]
        frames = []
        for sym, df in ohlcv.items():
            f = _R["compute_symbol_features"](df.set_index("Date"))
            f = f.reset_index().rename(columns={"index": "Date"})
            if "Date" not in f.columns:
                f = f.rename(columns={f.columns[0]: "Date"})
            f = f[["Date", *needed]].copy()
            f["Symbol"] = sym
            frames.append(f)
        panel = pd.concat(frames, ignore_index=True).sort_values(["Date", "Symbol"])
        panel = _R["add_cross_sectional_features"](panel)
        g3_cols = [
            "g3_cs_g1_ret_1d_z",
            "g3_cs_g2_realized_vol_20_z",
            "g3_cs_g1_volume_chg_1d_z",
            "g3_adv_dec_spread",
            "g3_adv_dec_ratio",
        ]
        sub = panel[panel["Symbol"] == target_symbol][["Date", *g3_cols]].reset_index(drop=True)
        return sub

    # ------------------------------------------------------------------ index
    def _index_block(self) -> pd.DataFrame:
        idx = self.price.get_ohlcv(settings.index_symbol)
        g1 = _R["compute_g1_features"](idx)
        g2 = _R["compute_g2_technical"](idx)
        return g1.merge(g2, on="Date", how="left")

    def _index_breadth(self) -> pd.DataFrame:
        """Live re-implementation of compute_g3_breadth from fetched universe closes."""
        universe = self.price.get_universe()
        ohlcv = self.price.get_universe_ohlcv(universe)
        if not ohlcv:
            raise FeatureBuildError("Could not fetch universe data for breadth features.")
        closes = {}
        for sym, df in ohlcv.items():
            s = df.set_index("Date")["Close"].astype(float)
            closes[sym] = s
        close_df = pd.DataFrame(closes).sort_index()
        ret_df = np.log(close_df).diff(1)
        above50 = close_df > close_df.rolling(50, min_periods=50).mean()
        above200 = close_df > close_df.rolling(200, min_periods=200).mean()
        active = close_df.notna()
        active_count = active.sum(axis=1).replace(0, np.nan)
        adv = (ret_df > 0).where(active).sum(axis=1)
        dec = (ret_df < 0).where(active).sum(axis=1)
        spread = adv - dec
        out = pd.DataFrame({
            "Date": close_df.index,
            "g3_active_constituents": active_count.values,
            "g3_pct_above_50dma": (above50.where(active).sum(axis=1) / active_count).values,
            "g3_pct_above_200dma": (above200.where(active).sum(axis=1) / active_count).values,
            "g3_adv_dec_spread": spread.values,
            "g3_adv_dec_ratio": (adv / dec.replace(0, np.nan)).values,
            "g3_ad_line": spread.cumsum().values,
            "g3_equal_weight_ret_1d": ret_df.where(active).mean(axis=1).values,
        })
        return out

    # --------------------------------------------------------------- assembly
    def build_frame(self, entry: ModelEntry, symbol: str) -> pd.DataFrame:
        """Return a Date-indexed frame containing every column in entry.feature_names."""
        needed = set(entry.feature_names)
        if entry.basis == "panel":
            frame = self._panel_symbol_block(symbol)
            if any(c.startswith("g3_") for c in needed):
                g3 = self._panel_cross_sectional(symbol)
                frame = frame.merge(g3, on="Date", how="left")
        else:  # index basis
            frame = self._index_block()
            if any(c.startswith("g3_") for c in needed):
                frame = frame.merge(self._index_breadth(), on="Date", how="left")
            if any(c.startswith("g5_sent_") for c in needed):
                frame = self._attach_sentiment(frame, entry, symbol, needed)

        missing = [c for c in entry.feature_names if c not in frame.columns]
        for c in missing:
            frame[c] = np.nan  # tolerated; imputer (classical) / zero (deep) handles it
        frame = frame.sort_values("Date").reset_index(drop=True)
        return frame

    def _attach_sentiment(self, frame, entry, symbol, needed) -> pd.DataFrame:
        sent_cols = [c for c in needed if c.startswith("g5_sent_")]
        if not settings.enable_sentiment:
            for c in sent_cols:
                frame[c] = 0.0
            return frame
        if self.sentiment is None:
            from app.features.sentiment_pipeline import get_sentiment_pipeline

            self.sentiment = get_sentiment_pipeline()
        try:
            daily = self.sentiment.daily_features(
                symbol=symbol, family=entry.family, columns=sent_cols
            )
            frame = frame.merge(daily, on="Date", how="left")
            for c in sent_cols:
                if c not in frame.columns:
                    frame[c] = 0.0
                frame[c] = frame[c].ffill()
        except Exception:
            for c in sent_cols:
                frame[c] = 0.0
        return frame

    @staticmethod
    def _row_at(frame: pd.DataFrame, feature_names: list[str], as_of: str | None):
        valid = frame.dropna(subset=[c for c in feature_names if c in frame.columns], how="all")
        if as_of:
            cutoff = pd.to_datetime(as_of)
            valid = valid[valid["Date"] <= cutoff]
        if valid.empty:
            raise FeatureBuildError("No feature rows available for the requested date.")
        return valid.iloc[-1]

    # --------------------------------------------------------------- classical
    def vector_for_classical(self, entry: ModelEntry, symbol: str, as_of: str | None):
        frame = self.build_frame(entry, symbol)
        row = self._row_at(frame, entry.feature_names, as_of)
        values = {c: _to_float(row.get(c)) for c in entry.feature_names}
        warnings: list[str] = []
        n_nan = sum(1 for v in values.values() if v is None or (isinstance(v, float) and np.isnan(v)))
        if n_nan:
            warnings.append(f"{n_nan}/{len(values)} features missing; pipeline imputer will fill them.")
        used_date = pd.to_datetime(row["Date"]).strftime("%Y-%m-%d")
        ordered = pd.DataFrame([[values[c] for c in entry.feature_names]], columns=entry.feature_names)
        return ordered, values, used_date, warnings

    # -------------------------------------------------------------------- deep
    def window_for_deep(self, entry: ModelEntry, symbol: str, as_of: str | None):
        frame = self.build_frame(entry, symbol)
        cols = entry.feature_names
        sub = frame[["Date", *cols]].copy()
        if as_of:
            sub = sub[sub["Date"] <= pd.to_datetime(as_of)]
        sub = sub.dropna(subset=cols, how="all").reset_index(drop=True)
        seq_len = entry.seq_len or 64
        if len(sub) < seq_len:
            raise FeatureBuildError(
                f"Need {seq_len} rows for {entry.model_name}; only {len(sub)} available "
                f"(increase PRICE_LOOKBACK_DAYS)."
            )
        hist = sub[cols].to_numpy(dtype=float)
        # Standardize feature-wise on trailing history (inference analogue of train stats).
        mean = np.nanmean(hist, axis=0)
        std = np.nanstd(hist, axis=0)
        std = np.where(std < 1e-8, 1.0, std)
        window = hist[-seq_len:]
        z = (window - mean[None, :]) / std[None, :]
        z = np.nan_to_num(z, nan=0.0, posinf=0.0, neginf=0.0).astype(np.float32)
        used_date = pd.to_datetime(sub["Date"].iloc[-1]).strftime("%Y-%m-%d")
        snapshot = {c: _to_float(sub[c].iloc[-1]) for c in cols}
        warnings: list[str] = [
            "Deep input standardized on trailing history (train-fold stats not persisted)."
        ]
        return z[None, :, :], snapshot, used_date, warnings


def _to_float(v) -> float | None:
    try:
        f = float(v)
        return f
    except (TypeError, ValueError):
        return None
