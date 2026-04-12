"""
NewsAPI integration — fetches the top 1 news article per stock ticker.

Free tier: 100 requests/day, articles from past month.
Get a free key at: https://newsapi.org/register

Add to .env:
  NEWSAPI_KEY=your_key_here
"""

import logging
import os
from datetime import datetime, timedelta
from typing import Dict, List, Optional

import requests

logger = logging.getLogger(__name__)

NEWSAPI_URL = "https://newsapi.org/v2/everything"

TICKER_SEARCH_TERMS = {
    "SPY": "S&P 500 stock market",
    "QQQ": "Nasdaq QQQ ETF",
    "AAPL": "Apple AAPL stock",
    "MSFT": "Microsoft MSFT stock",
    "AMZN": "Amazon AMZN stock",
    "GOOGL": "Google Alphabet stock",
    "NVDA": "Nvidia NVDA stock",
    "META": "Meta Facebook stock",
    "TSLA": "Tesla TSLA stock",
    "AMD": "AMD semiconductor stock",
}


def _time_ago(published_at: str) -> str:
    """Convert ISO timestamp to human-readable time ago string."""
    try:
        dt = datetime.strptime(published_at[:19], "%Y-%m-%dT%H:%M:%S")
        delta = datetime.utcnow() - dt
        hours = int(delta.total_seconds() / 3600)
        if hours < 1:
            mins = int(delta.total_seconds() / 60)
            return f"{mins}m ago"
        if hours < 24:
            return f"{hours}h ago"
        return f"{delta.days}d ago"
    except Exception:
        return ""


def fetch_top_article(symbol: str, api_key: str) -> Optional[dict]:
    """
    Fetch the single most recent news article for a given ticker.
    Returns dict with: title, source, url, published_at, time_ago
    """
    query = TICKER_SEARCH_TERMS.get(symbol, f"{symbol} stock")

    params = {
        "q": query,
        "sortBy": "publishedAt",
        "language": "en",
        "pageSize": 1,
        "from": (datetime.utcnow() - timedelta(days=3)).strftime("%Y-%m-%d"),
        "apiKey": api_key,
    }

    try:
        resp = requests.get(NEWSAPI_URL, params=params, timeout=8)
        if resp.status_code == 200:
            data = resp.json()
            articles = data.get("articles", [])
            if articles:
                a = articles[0]
                return {
                    "symbol": symbol,
                    "title": a.get("title", "")[:120],
                    "source": a.get("source", {}).get("name", ""),
                    "url": a.get("url", ""),
                    "published_at": a.get("publishedAt", ""),
                    "time_ago": _time_ago(a.get("publishedAt", "")),
                }
        elif resp.status_code == 401:
            logger.warning("NewsAPI: Invalid API key")
        elif resp.status_code == 429:
            logger.warning("NewsAPI: Rate limit hit")
        else:
            logger.warning(f"NewsAPI {symbol}: HTTP {resp.status_code}")
    except requests.Timeout:
        logger.warning(f"NewsAPI timeout for {symbol}")
    except Exception as e:
        logger.warning(f"NewsAPI error for {symbol}: {e}")

    return None


def fetch_news_for_symbols(symbols: List[str]) -> Dict[str, Optional[dict]]:
    """
    Fetch top article for each symbol. Returns dict of symbol → article (or None).
    Gracefully handles missing API key or failures — news is optional.
    """
    api_key = os.getenv("NEWSAPI_KEY", "")
    if not api_key:
        logger.info("NEWSAPI_KEY not set — skipping news fetch")
        return {sym: None for sym in symbols}

    results = {}
    for sym in symbols:
        article = fetch_top_article(sym, api_key)
        results[sym] = article
        if article:
            logger.info(f"News [{sym}]: {article['title'][:60]}... ({article['time_ago']})")

    fetched = sum(1 for v in results.values() if v)
    logger.info(f"News fetched: {fetched}/{len(symbols)} symbols")
    return results
