"""MarketData: thin facade over HistoricalFetcher and StreamManager."""

from datetime import datetime
from typing import Callable, Dict, List, Optional

import pandas as pd

from data.historical_fetcher import HistoricalFetcher
from data.stream_manager import StreamManager


class MarketData:
    """OHLCV fetch and streaming facade over Alpaca."""

    def __init__(self, alpaca_client) -> None:
        self._fetcher = HistoricalFetcher(alpaca_client.data_client)
        self._stream = StreamManager()
        self._cache: Dict[str, pd.DataFrame] = {}

    def get_historical_bars(
        self,
        symbol: str,
        timeframe: str = "1Day",
        start: Optional[datetime] = None,
        end: Optional[datetime] = None,
        limit: int = 2000,
    ) -> pd.DataFrame:
        df = self._fetcher.get_bars(symbol, timeframe, start, end, limit)
        if not df.empty:
            self._cache[symbol] = df
        return df

    def get_latest_bar(self, symbol: str) -> Optional[pd.Series]:
        return self._fetcher.get_latest_bar(symbol)

    def get_latest_quote(self, symbol: str) -> Optional[dict]:
        return self._fetcher.get_latest_quote(symbol)

    def get_snapshot(self, symbol: str) -> Optional[dict]:
        return self._fetcher.get_snapshot(symbol)

    def get_cached_bars(self, symbol: str) -> Optional[pd.DataFrame]:
        return self._cache.get(symbol)

    def subscribe_bars(self, symbols: List[str], timeframe: str, callback: Callable) -> None:
        self._stream.subscribe_bars(symbols, callback)

    def stop_stream(self) -> None:
        self._stream.stop()
