"""ATR-based trailing stop computation and GTC stop order management.

Each open position gets a hard stop placed as a live GTC StopOrder on Alpaca
so the stop survives process restarts. On each bar, the stop tightens if price
moved favorably (trailing) but never widens.
"""

import logging
from typing import Dict, Optional

import pandas as pd

from core.signals.indicators import atr

logger = logging.getLogger(__name__)

_ATR_MULTIPLIER_BY_TIER = {
    "low": 1.5,    # LowVolBull — tighter stops in calm regimes
    "mid": 2.0,    # MidVolCautious
    "high": 3.0,   # HighVolDefensive — wide stops in volatile regimes
}


def _tier(strategy_type: str) -> str:
    t = strategy_type.lower()
    if "low" in t:
        return "low"
    if "high" in t or "defensive" in t:
        return "high"
    return "mid"


class StopManager:
    """Compute ATR-based stop levels and keep live GTC orders in sync."""

    def __init__(self, order_executor, atr_period: int = 14) -> None:
        self._executor = order_executor
        self._atr_period = atr_period
        self._stop_order_ids: Dict[str, str] = {}   # symbol → Alpaca order id
        self._stop_prices: Dict[str, float] = {}    # symbol → current stop price

    def update_stops(
        self,
        bars_by_symbol: Dict[str, pd.DataFrame],
        positions: dict,
        regime_info,
    ) -> None:
        """Recompute stops for all open positions and tighten if price has moved up."""
        tier = _tier(regime_info.recommended_strategy_type if regime_info else "mid")
        atr_mult = _ATR_MULTIPLIER_BY_TIER[tier]

        for symbol, pos in positions.items():
            bars = bars_by_symbol.get(symbol)
            if bars is None or len(bars) < self._atr_period + 1:
                continue
            new_stop = self._compute_stop(bars, pos.current_price, atr_mult)
            self._sync_stop(symbol, pos, new_stop)

    def register_new_position(self, symbol: str, bars: pd.DataFrame, entry_price: float, tier: str = "mid") -> float:
        """Compute and place an initial GTC stop for a new entry. Returns the stop price."""
        atr_mult = _ATR_MULTIPLIER_BY_TIER[_tier(tier)]
        stop_price = self._compute_stop(bars, entry_price, atr_mult)
        self._place_stop_order(symbol, stop_price)
        return stop_price

    def remove_position_stop(self, symbol: str) -> None:
        order_id = self._stop_order_ids.pop(symbol, None)
        self._stop_prices.pop(symbol, None)
        if order_id:
            self._executor.cancel_order(order_id)

    # ------------------------------------------------------------------ #
    # Private                                                              #
    # ------------------------------------------------------------------ #

    def _compute_stop(self, bars: pd.DataFrame, current_price: float, atr_mult: float) -> float:
        atr_val = float(atr(bars, self._atr_period).iloc[-1])
        return round(current_price - atr_mult * atr_val, 2)

    def _sync_stop(self, symbol: str, pos, new_stop: float) -> None:
        current_stop = self._stop_prices.get(symbol, 0.0)

        # Never widen — only tighten
        if new_stop <= current_stop:
            return

        logger.info("Trailing stop tightened: %s $%.2f → $%.2f", symbol, current_stop, new_stop)
        existing_order = self._stop_order_ids.get(symbol)

        if existing_order:
            success = self._executor.modify_stop(symbol, existing_order, new_stop, current_stop)
            if success:
                self._stop_prices[symbol] = new_stop
        else:
            self._place_stop_order(symbol, new_stop)

    def _place_stop_order(self, symbol: str, stop_price: float) -> None:
        try:
            from alpaca.trading.requests import StopOrderRequest
            from alpaca.trading.enums import OrderSide, TimeInForce
            req = StopOrderRequest(
                symbol=symbol,
                qty=1,  # Alpaca will close the full position
                side=OrderSide.SELL,
                time_in_force=TimeInForce.GTC,
                stop_price=round(stop_price, 2),
            )
            order = self._executor.client.trading_client.submit_order(req)
            self._stop_order_ids[symbol] = order.id
            self._stop_prices[symbol] = stop_price
            logger.info("GTC stop placed: %s @ $%.2f (order %s)", symbol, stop_price, order.id)
        except Exception as e:
            logger.error("Failed to place GTC stop for %s: %s", symbol, e)
