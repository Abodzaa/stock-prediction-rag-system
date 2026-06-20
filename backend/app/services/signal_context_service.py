"""Signal context service: price history, volatility, and scheduled event context."""
from __future__ import annotations

import math
from datetime import datetime, timedelta, timezone

import pandas as pd

from app.models.schemas import PricePoint, SignalContextResponse, UpcomingEvent
from app.repositories.price_repository import PriceRepository, get_price_repository


class SignalContextService:
    def __init__(self, price_repo: PriceRepository | None = None) -> None:
        self.price = price_repo or get_price_repository()

    def get_context(
        self,
        symbol: str,
        horizon: str,
        predicted_pct: float,
        benchmark_symbol: str | None = None,
    ) -> SignalContextResponse:
        history = self.price.get_ohlcv(symbol, lookback_days=160).tail(90).reset_index(drop=True)
        if history.empty:
            raise ValueError(f"No price history available for {symbol}")

        current_price = float(history["Close"].iloc[-1])
        price_as_of = pd.to_datetime(history["Date"].iloc[-1]).date().isoformat()
        predicted_price = current_price * (1.0 + (predicted_pct / 100.0))

        horizon_days = 5 if horizon == "h5" else 1
        recent_returns = history["Close"].pct_change().dropna().tail(20)
        daily_vol = float(recent_returns.std()) if not recent_returns.empty else 0.0
        annualized_pct = daily_vol * math.sqrt(252) * 100.0 if daily_vol else 0.0
        interval_move = max(abs(predicted_pct) / 100.0 * 0.65, daily_vol * math.sqrt(horizon_days) * 1.28)
        range_low = current_price * (1.0 + (predicted_pct / 100.0) - interval_move)
        range_high = current_price * (1.0 + (predicted_pct / 100.0) + interval_move)
        volatility_label, volatility_detail = self._volatility_badge(annualized_pct)

        benchmark_history: list[PricePoint] = []
        if benchmark_symbol:
            try:
                benchmark_df = self.price.get_ohlcv(benchmark_symbol, lookback_days=160).tail(90)
                benchmark_history = self._to_points(benchmark_df)
            except Exception:
                benchmark_history = []

        return SignalContextResponse(
            symbol=symbol.upper(),
            benchmark_symbol=benchmark_symbol,
            current_price=_f(current_price),
            price_as_of=price_as_of,
            predicted_price=_f(predicted_price),
            range_low=_f(range_low),
            range_high=_f(range_high),
            volatility_label=volatility_label,
            volatility_annualized_pct=_f(annualized_pct),
            volatility_detail=volatility_detail,
            price_history=self._to_points(history),
            benchmark_history=benchmark_history,
            upcoming_events=self._upcoming_events(symbol, horizon_days),
            track_record=None,
        )

    def _to_points(self, frame: pd.DataFrame) -> list[PricePoint]:
        points: list[PricePoint] = []
        for row in frame.itertuples(index=False):
            points.append(
                PricePoint(
                    date=pd.to_datetime(row.Date).date().isoformat(),
                    close=_f(row.Close),
                )
            )
        return points

    def _volatility_badge(self, annualized_pct: float) -> tuple[str, str]:
        if annualized_pct >= 42:
            return "High volatility", "Recent price swings have been unusually wide."
        if annualized_pct >= 24:
            return "Medium volatility", "Recent price swings are meaningful but not extreme."
        return "Low volatility", "Recent price swings have been relatively contained."

    def _upcoming_events(self, symbol: str, horizon_days: int) -> list[UpcomingEvent]:
        try:
            import yfinance as yf

            ticker = yf.Ticker(symbol)
            calendar = ticker.calendar
        except Exception:
            return []

        now = datetime.now(timezone.utc)
        horizon_limit = now + timedelta(days=horizon_days + 1)
        events: list[UpcomingEvent] = []

        if isinstance(calendar, pd.DataFrame):
            if not calendar.empty:
                data = calendar.iloc[:, 0].to_dict()
            else:
                data = {}
        elif isinstance(calendar, dict):
            data = calendar
        else:
            data = {}

        for key, kind in (
            ("Earnings Date", "earnings"),
            ("Ex-Dividend Date", "dividend"),
            ("Dividend Date", "dividend"),
        ):
            value = data.get(key)
            dates = _flatten_event_dates(value)
            for event_date in dates:
                within_horizon = now.date() <= event_date.date() <= horizon_limit.date()
                if not within_horizon:
                    continue
                label = (
                    "Upcoming earnings"
                    if kind == "earnings"
                    else "Upcoming dividend-related date"
                )
                events.append(
                    UpcomingEvent(
                        kind=kind,
                        label=label,
                        date=event_date.date().isoformat(),
                        within_horizon=True,
                    )
                )

        deduped: dict[tuple[str, str], UpcomingEvent] = {}
        for event in events:
            deduped[(event.kind, event.date)] = event
        return list(deduped.values())


def _flatten_event_dates(value) -> list[datetime]:
    if value is None:
        return []

    if isinstance(value, pd.Series):
        values = value.tolist()
    elif isinstance(value, (list, tuple, set)):
        values = list(value)
    else:
        values = [value]

    out: list[datetime] = []
    for item in values:
        try:
            ts = pd.to_datetime(item, utc=True)
        except Exception:
            continue
        if ts is None or pd.isna(ts):
            continue
        out.append(ts.to_pydatetime())
    return out


def _f(value: float | None) -> float:
    try:
        numeric = float(value)
    except (TypeError, ValueError):
        return 0.0
    if math.isnan(numeric) or math.isinf(numeric):
        return 0.0
    return round(numeric, 4)


_singleton: SignalContextService | None = None


def get_signal_context_service() -> SignalContextService:
    global _singleton
    if _singleton is None:
        _singleton = SignalContextService()
    return _singleton
