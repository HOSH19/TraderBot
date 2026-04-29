"""Alpaca-backed position sync and optional streaming fill handling.

Mutates a shared :class:`~core.risk.portfolio_state.PortfolioState` on startup and fills.
"""

import logging
import threading
from typing import Dict, Optional

from core.risk import PortfolioState, Position
from core.timeutil import utc_now

logger = logging.getLogger(__name__)


class PositionTracker:
    """Reconcile broker positions into an in-memory ``PortfolioState``."""

    def __init__(self, alpaca_client, portfolio_state: PortfolioState) -> None:
        """Share one ``PortfolioState`` instance with the rest of the app.

        Args:
            alpaca_client: Connected ``AlpacaClient``.
            portfolio_state: Object updated by :meth:`sync_from_alpaca` and stream handlers.
        """
        self.client = alpaca_client
        self.portfolio = portfolio_state
        self._lock = threading.Lock()
        self._ws_thread: Optional[threading.Thread] = None

    def sync_from_alpaca(self) -> None:
        """Pull account and open lots from Alpaca into ``portfolio``."""
        try:
            account = self.client.get_account()
            self.portfolio.equity = float(account.equity)
            self.portfolio.cash = float(account.cash)
            self.portfolio.buying_power = float(account.buying_power)

            if self.portfolio.peak_equity == 0:
                self.portfolio.peak_equity = self.portfolio.equity
            if self.portfolio.daily_start_equity == 0:
                self.portfolio.daily_start_equity = self.portfolio.equity
            if self.portfolio.weekly_start_equity == 0:
                self.portfolio.weekly_start_equity = self.portfolio.equity

            alpaca_positions = self.client.get_positions()
            with self._lock:
                self.portfolio.positions = {}
                for pos in alpaca_positions:
                    self.portfolio.positions[pos.symbol] = Position(
                        symbol=pos.symbol,
                        shares=float(pos.qty),
                        entry_price=float(pos.avg_entry_price),
                        entry_time=getattr(pos, "created_at", None) or utc_now(),
                        current_price=float(pos.current_price),
                        stop_loss=0.0,
                        regime_at_entry="UNKNOWN",
                    )
        except Exception as e:
            logger.error(f"sync_from_alpaca failed: {e}")

    def update_position_price(self, symbol: str, price: float):
        """Update the current price for a tracked position and record the timestamp."""
        with self._lock:
            if symbol in self.portfolio.positions:
                self.portfolio.positions[symbol].current_price = price
                self.portfolio.last_updated = utc_now()

    def update_stop(self, symbol: str, new_stop: float):
        """Update the stop-loss price for the given symbol's tracked position."""
        with self._lock:
            if symbol in self.portfolio.positions:
                self.portfolio.positions[symbol].stop_loss = new_stop

    def _refresh_equity(self):
        """Pull the latest account values from Alpaca and update P&L fields on the portfolio."""
        try:
            account = self.client.get_account()
            with self._lock:
                self.portfolio.equity = float(account.equity)
                self.portfolio.cash = float(account.cash)
                self.portfolio.buying_power = float(account.buying_power)
                self.portfolio.daily_pnl = self.portfolio.equity - self.portfolio.daily_start_equity
                self.portfolio.weekly_pnl = self.portfolio.equity - self.portfolio.weekly_start_equity
                if self.portfolio.equity > self.portfolio.peak_equity:
                    self.portfolio.peak_equity = self.portfolio.equity
        except Exception as e:
            logger.warning(f"_refresh_equity failed: {e}")

    def get_position(self, symbol: str) -> Optional[Position]:
        """Return the Position for the given symbol, or None if not currently held."""
        return self.portfolio.positions.get(symbol)

    def get_all_positions(self) -> Dict[str, Position]:
        """Return a thread-safe snapshot of all open positions keyed by symbol."""
        with self._lock:
            return dict(self.portfolio.positions)

    def reset_daily(self):
        """Reset the daily equity baseline and P&L counter (call once each trading day start)."""
        self.portfolio.daily_start_equity = self.portfolio.equity
        self.portfolio.daily_pnl = 0.0

    def reset_weekly(self):
        """Reset the weekly equity baseline and P&L counter (call once each week start)."""
        self.portfolio.weekly_start_equity = self.portfolio.equity
        self.portfolio.weekly_pnl = 0.0
