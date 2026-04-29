"""OHLCV historical data retrieval with business-day gap repair."""

import logging
from datetime import datetime, timedelta
from typing import Optional

import pandas as pd

from core.timeutil import utc_now

logger = logging.getLogger(__name__)


class HistoricalFetcher:
    """Fetch and normalize OHLCV bars from an Alpaca data client."""

    def __init__(self, data_client) -> None:
        self._client = data_client

    def get_bars(
        self,
        symbol: str,
        timeframe: str = "1Day",
        start: Optional[datetime] = None,
        end: Optional[datetime] = None,
        limit: int = 2000,
    ) -> pd.DataFrame:
        from alpaca.data.requests import StockBarsRequest
        from alpaca.data.timeframe import TimeFrame
        from alpaca.data.enums import DataFeed

        if start is None:
            start = utc_now() - timedelta(days=limit * 1.5)
        if end is None:
            end = utc_now()

        tf_map = {"1Day": TimeFrame.Day, "1Hour": TimeFrame.Hour, "5Min": TimeFrame.Minute, "1Min": TimeFrame.Minute}
        req = StockBarsRequest(
            symbol_or_symbols=symbol,
            timeframe=tf_map.get(timeframe, TimeFrame.Day),
            start=start, end=end, limit=limit, feed=DataFeed.IEX,
        )
        try:
            bars = self._client.get_stock_bars(req)
            df = bars.df
            if isinstance(df.index, pd.MultiIndex):
                df = df.xs(symbol, level=0)
            df = df[["open", "high", "low", "close", "volume"]].sort_index()
            return self._fill_gaps(df)
        except Exception as e:
            logger.error("get_bars(%s) failed: %s", symbol, e)
            return pd.DataFrame()

    def get_latest_bar(self, symbol: str) -> Optional[pd.Series]:
        from alpaca.data.requests import StockLatestBarRequest
        from alpaca.data.enums import DataFeed
        try:
            req = StockLatestBarRequest(symbol_or_symbols=symbol, feed=DataFeed.IEX)
            bar = self._client.get_stock_latest_bar(req)[symbol]
            return pd.Series(
                {"open": bar.open, "high": bar.high, "low": bar.low, "close": bar.close, "volume": bar.volume},
                name=bar.timestamp,
            )
        except Exception as e:
            logger.error("get_latest_bar(%s) failed: %s", symbol, e)
            return None

    def get_latest_quote(self, symbol: str) -> Optional[dict]:
        from alpaca.data.requests import StockLatestQuoteRequest
        from alpaca.data.enums import DataFeed
        try:
            req = StockLatestQuoteRequest(symbol_or_symbols=symbol, feed=DataFeed.IEX)
            q = self._client.get_stock_latest_quote(req)[symbol]
            return {"bid": q.bid_price, "ask": q.ask_price, "spread_pct": (q.ask_price - q.bid_price) / q.ask_price}
        except Exception as e:
            logger.error("get_latest_quote(%s) failed: %s", symbol, e)
            return None

    def get_snapshot(self, symbol: str) -> Optional[dict]:
        from alpaca.data.requests import StockSnapshotRequest
        from alpaca.data.enums import DataFeed
        try:
            req = StockSnapshotRequest(symbol_or_symbols=symbol, feed=DataFeed.IEX)
            snap = self._client.get_stock_snapshot(req)[symbol]
            return {"symbol": symbol, "latest_trade_price": snap.latest_trade.price, "daily_bar": snap.daily_bar}
        except Exception as e:
            logger.error("get_snapshot(%s) failed: %s", symbol, e)
            return None

    @staticmethod
    def _fill_gaps(df: pd.DataFrame) -> pd.DataFrame:
        if df.empty:
            return df
        if hasattr(df.index, "tz") and df.index.tz is not None:
            df.index = df.index.tz_convert("UTC").tz_localize(None)
        df.index = pd.to_datetime(df.index).normalize()
        full_idx = pd.bdate_range(df.index[0], df.index[-1])
        df = df[~df.index.duplicated(keep="last")].reindex(full_idx)
        for col in ("close", "open", "high", "low"):
            df[col] = df[col].ffill()
        df["volume"] = df["volume"].fillna(0)
        return df.dropna(subset=["close"])
