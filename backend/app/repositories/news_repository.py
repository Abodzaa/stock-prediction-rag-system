"""News access. Adapter + Factory over multiple providers (yfinance default).

Returns a normalized article schema:
  {news_id, symbol, published_utc(iso), headline, summary, source, url}
"""
from __future__ import annotations

import time
from datetime import datetime, timezone

import pandas as pd

from app.config import settings


def _norm_ts(value) -> str:
    try:
        ts = pd.to_datetime(value, utc=True)
        if ts is None or pd.isna(ts):
            raise ValueError
        return ts.isoformat()
    except Exception:
        return datetime.now(timezone.utc).isoformat()


class NewsProvider:
    def fetch(self, symbol: str, limit: int = 30) -> list[dict]:  # pragma: no cover - interface
        raise NotImplementedError


class YFinanceNews(NewsProvider):
    def fetch(self, symbol: str, limit: int = 30) -> list[dict]:
        import yfinance as yf

        try:
            items = yf.Ticker(symbol).news or []
        except Exception:
            items = []
        rows = []
        for it in items[:limit]:
            content = it.get("content") if isinstance(it.get("content"), dict) else {}
            title = content.get("title") or it.get("title") or ""
            summary = content.get("summary") or content.get("description") or it.get("summary") or ""
            prov = content.get("provider") if isinstance(content.get("provider"), dict) else {}
            source = prov.get("displayName") or it.get("publisher") or "unknown"
            canon = content.get("canonicalUrl") if isinstance(content.get("canonicalUrl"), dict) else {}
            click = content.get("clickThroughUrl") if isinstance(content.get("clickThroughUrl"), dict) else {}
            url = canon.get("url") or click.get("url") or it.get("link") or ""
            ts = it.get("providerPublishTime") or content.get("pubDate") or content.get("displayTime")
            if isinstance(ts, (int, float)):
                published = pd.to_datetime(ts, unit="s", utc=True).isoformat()
            else:
                published = _norm_ts(ts)
            nid = str(content.get("id") or it.get("uuid") or it.get("id") or f"{symbol}-{published}-{title[:40]}")
            if not title and not summary:
                continue
            rows.append({
                "news_id": nid, "symbol": symbol.upper(), "published_utc": published,
                "headline": str(title), "summary": str(summary),
                "source": str(source), "url": str(url),
            })
        return rows


class FinnhubNews(NewsProvider):
    def fetch(self, symbol: str, limit: int = 50) -> list[dict]:
        import requests

        end = datetime.now(timezone.utc).date()
        start = end - pd.Timedelta(days=settings.news_lookback_days)
        r = requests.get(
            "https://finnhub.io/api/v1/company-news",
            params={"symbol": symbol.upper(), "from": str(start), "to": str(end),
                    "token": settings.news_api_key},
            timeout=20,
        )
        r.raise_for_status()
        rows = []
        for it in r.json()[:limit]:
            published = pd.to_datetime(it.get("datetime"), unit="s", utc=True).isoformat()
            rows.append({
                "news_id": str(it.get("id") or f"{symbol}-{published}"),
                "symbol": symbol.upper(), "published_utc": published,
                "headline": str(it.get("headline", "")), "summary": str(it.get("summary", "")),
                "source": str(it.get("source", "finnhub")), "url": str(it.get("url", "")),
            })
        return rows


class NewsApiOrg(NewsProvider):
    def fetch(self, symbol: str, limit: int = 50) -> list[dict]:
        import requests

        r = requests.get(
            "https://newsapi.org/v2/everything",
            params={"q": symbol, "language": "en", "sortBy": "publishedAt",
                    "pageSize": min(limit, 100), "apiKey": settings.news_api_key},
            timeout=20,
        )
        r.raise_for_status()
        rows = []
        for it in r.json().get("articles", [])[:limit]:
            published = _norm_ts(it.get("publishedAt"))
            src = (it.get("source") or {}).get("name", "newsapi")
            rows.append({
                "news_id": str(it.get("url") or f"{symbol}-{published}"),
                "symbol": symbol.upper(), "published_utc": published,
                "headline": str(it.get("title", "")), "summary": str(it.get("description", "")),
                "source": str(src), "url": str(it.get("url", "")),
            })
        return rows


def _make_provider() -> NewsProvider:
    p = settings.news_provider.lower()
    if p == "finnhub" and settings.news_api_key:
        return FinnhubNews()
    if p in {"newsapi", "newsapi.org"} and settings.news_api_key:
        return NewsApiOrg()
    return YFinanceNews()


class NewsRepository:
    def __init__(self) -> None:
        self._provider = _make_provider()
        self._cache: dict[str, tuple[float, list[dict]]] = {}
        self._ttl = 900

    def fetch(self, symbol: str, limit: int = 40) -> list[dict]:
        now = time.time()
        hit = self._cache.get(symbol)
        if hit and now - hit[0] < self._ttl:
            return hit[1]
        rows = self._provider.fetch(symbol, limit=limit)
        # newest first
        rows.sort(key=lambda r: r["published_utc"], reverse=True)
        self._cache[symbol] = (now, rows)
        return rows


_singleton: NewsRepository | None = None


def get_news_repository() -> NewsRepository:
    global _singleton
    if _singleton is None:
        _singleton = NewsRepository()
    return _singleton
