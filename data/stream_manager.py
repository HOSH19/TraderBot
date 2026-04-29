"""WebSocket bar stream management for live trading."""

import logging
import os
import threading
from typing import Callable, List

logger = logging.getLogger(__name__)


class StreamManager:
    """Spawn and manage an Alpaca StockDataStream background thread."""

    def __init__(self) -> None:
        self._bar_callbacks: List[Callable] = []
        self._stream = None
        self._stream_thread = None

    def subscribe_bars(self, symbols: List[str], callback: Callable) -> None:
        self._bar_callbacks.append(callback)
        if self._stream is None:
            self._start(symbols)

    def stop(self) -> None:
        if self._stream:
            try:
                self._stream.stop()
            except Exception as e:
                logger.warning("Error stopping stream: %s", e)

    def _start(self, symbols: List[str]) -> None:
        from alpaca.data.live import StockDataStream

        api_key = (os.getenv("ALPACA_API_KEY") or "").strip()
        secret_key = (os.getenv("ALPACA_SECRET_KEY") or "").strip()
        stream = StockDataStream(api_key, secret_key)

        async def _on_bar(bar):
            for cb in self._bar_callbacks:
                try:
                    cb(bar)
                except Exception as e:
                    logger.error("Bar callback error: %s", e)

        stream.subscribe_bars(_on_bar, *symbols)
        self._stream = stream
        self._stream_thread = threading.Thread(target=stream.run, daemon=True)
        self._stream_thread.start()
