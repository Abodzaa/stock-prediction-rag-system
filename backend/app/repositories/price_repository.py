"""Market price data access (yfinance) + index universe.

Adapter behind a stable interface so the data source can be swapped without
touching services.
"""
from __future__ import annotations

import time
from pathlib import Path

import pandas as pd

from app.config import settings

_OHLCV_COLS = ["Open", "High", "Low", "Close", "Volume"]


class PriceRepository:
    def __init__(self, ttl_seconds: int = 900) -> None:
        self._ttl = ttl_seconds
        self._cache: dict[str, tuple[float, pd.DataFrame]] = {}

    # -- single symbol -------------------------------------------------------
    def get_ohlcv(self, symbol: str, lookback_days: int | None = None) -> pd.DataFrame:
        lookback = lookback_days or settings.price_lookback_days
        key = f"{symbol}:{lookback}"
        now = time.time()
        hit = self._cache.get(key)
        if hit and now - hit[0] < self._ttl:
            return hit[1].copy()

        df = self._download(symbol, lookback)
        self._cache[key] = (now, df)
        return df.copy()

    def _download(self, symbol: str, lookback_days: int) -> pd.DataFrame:
        import yfinance as yf

        period = f"{max(lookback_days + 40, 60)}d"
        raw = yf.download(
            symbol, period=period, interval="1d",
            auto_adjust=False, progress=False, threads=False,
        )
        if raw is None or raw.empty:
            raise ValueError(f"No price data returned for {symbol}")
        if isinstance(raw.columns, pd.MultiIndex):
            raw.columns = raw.columns.get_level_values(0)
        raw = raw.reset_index().rename(columns={"index": "Date", "Datetime": "Date"})
        raw["Date"] = pd.to_datetime(raw["Date"]).dt.tz_localize(None)
        for c in _OHLCV_COLS:
            if c not in raw.columns:
                raise ValueError(f"{symbol}: missing column {c}")
        out = raw[["Date", *_OHLCV_COLS]].dropna(subset=["Close"]).sort_values("Date")
        return out.reset_index(drop=True)

    # -- universe ------------------------------------------------------------
    def get_universe(self) -> list[str]:
        name = "nasdaq100" if settings.universe == "nasdaq100" else "sp500"
        snap = settings.metadata_dir / f"{name}_constituents_snapshot.csv"
        if snap.exists():
            df = pd.read_csv(snap)
            col = "YahooSymbol" if "YahooSymbol" in df.columns else "Symbol"
            syms = df[col].astype(str).str.strip().tolist()
        else:
            syms = list(_NASDAQ100_FALLBACK)
        return syms[: settings.universe_max]

    def get_universe_ohlcv(
        self, symbols: list[str], lookback_days: int | None = None
    ) -> dict[str, pd.DataFrame]:
        """Batch-download OHLCV for the universe (used for cross-sectional G3 features)."""
        import yfinance as yf

        lookback = lookback_days or settings.price_lookback_days
        period = f"{max(lookback + 40, 60)}d"
        data = yf.download(
            symbols, period=period, interval="1d",
            auto_adjust=False, progress=False, threads=True, group_by="ticker",
        )
        out: dict[str, pd.DataFrame] = {}
        for sym in symbols:
            try:
                sub = data[sym] if isinstance(data.columns, pd.MultiIndex) else data
            except KeyError:
                continue
            sub = sub.reset_index().rename(columns={"index": "Date", "Datetime": "Date"})
            if "Date" not in sub.columns or "Close" not in sub.columns:
                continue
            sub["Date"] = pd.to_datetime(sub["Date"]).dt.tz_localize(None)
            keep = [c for c in ["Date", *_OHLCV_COLS] if c in sub.columns]
            sub = sub[keep].dropna(subset=["Close"]).sort_values("Date")
            if len(sub) > 30:
                out[sym] = sub.reset_index(drop=True)
        return out


# Minimal offline fallback if the snapshot CSV is unavailable.
_NASDAQ100_FALLBACK = [
    "AAPL", "MSFT", "AMZN", "NVDA", "GOOGL", "GOOG", "META", "TSLA", "AVGO", "COST",
    "PEP", "ADBE", "NFLX", "AMD", "CSCO", "TMUS", "INTC", "CMCSA", "QCOM", "INTU",
    "AMGN", "TXN", "HON", "AMAT", "BKNG", "ISRG", "ADP", "VRTX", "REGN", "MU",
]


_singleton: PriceRepository | None = None


def get_price_repository() -> PriceRepository:
    global _singleton
    if _singleton is None:
        _singleton = PriceRepository()
    return _singleton
