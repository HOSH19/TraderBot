"""
Order placement, modification, and cancellation.
LIMIT orders by default (±0.1% of price), cancel after 30s, optionally retry at market.
Unique trade_id links signal → risk_decision → order → fill.
"""

import logging
import time
import uuid
from typing import Optional

from core.regime_strategies import Signal
from core.risk_manager import RiskDecision

logger = logging.getLogger(__name__)

LIMIT_OFFSET_PCT = 0.001
ORDER_TIMEOUT_SECONDS = 30


class OrderExecutor:
    """Handles order submission, modification, and cancellation via Alpaca."""

    def __init__(self, alpaca_client, dry_run: bool = False):
        """
        Initialize OrderExecutor.

        Args:
            alpaca_client: Connected AlpacaClient instance.
            dry_run: If True, log intended actions without submitting real orders.
        """
        self.client = alpaca_client
        self.dry_run = dry_run
        self._open_orders: dict = {}

    def _gen_trade_id(self, symbol: str) -> str:
        """Generate a short unique trade identifier linking signal to fill."""
        return f"{symbol}-{uuid.uuid4().hex[:8]}"

    def submit_order(
        self,
        signal: Signal,
        risk_decision: RiskDecision,
        retry_at_market: bool = True,
    ) -> Optional[str]:
        """
        Submit a LIMIT order for the given signal.
        Returns order_id or None if dry_run / rejected.
        """
        from alpaca.trading.requests import LimitOrderRequest, MarketOrderRequest
        from alpaca.trading.enums import OrderSide, TimeInForce

        if not risk_decision.approved or risk_decision.modified_signal is None:
            return None

        sig = risk_decision.modified_signal
        trade_id = self._gen_trade_id(sig.symbol)

        account = self.client.get_account()
        equity = float(account.equity)
        qty = int(equity * sig.position_size_pct * sig.leverage / sig.entry_price)

        if qty <= 0:
            logger.warning(f"Computed qty=0 for {sig.symbol}, skipping.")
            return None

        limit_price = round(sig.entry_price * (1 + LIMIT_OFFSET_PCT), 2)

        if self.dry_run:
            return trade_id

        try:
            req = LimitOrderRequest(
                symbol=sig.symbol,
                qty=qty,
                side=OrderSide.BUY if sig.direction == "LONG" else OrderSide.SELL,
                time_in_force=TimeInForce.DAY,
                limit_price=limit_price,
                client_order_id=trade_id,
            )
            order = self.client.trading_client.submit_order(req)
            self._open_orders[trade_id] = order.id

            time.sleep(ORDER_TIMEOUT_SECONDS)
            try:
                filled = self.client.trading_client.get_order_by_id(order.id)
                if filled.status not in ("filled", "partially_filled"):
                    self.client.trading_client.cancel_order_by_id(order.id)
                    logger.warning(f"LIMIT order {trade_id} unfilled after {ORDER_TIMEOUT_SECONDS}s — cancelled")
                    if retry_at_market:
                        mkt_req = MarketOrderRequest(
                            symbol=sig.symbol,
                            qty=qty,
                            side=OrderSide.BUY if sig.direction == "LONG" else OrderSide.SELL,
                            time_in_force=TimeInForce.DAY,
                            client_order_id=trade_id + "-mkt",
                        )
                        mkt_order = self.client.trading_client.submit_order(mkt_req)
                        return mkt_order.id
            except Exception as e:
                logger.error(f"Failed to check/cancel order {trade_id}: {e}")

            return order.id

        except Exception as e:
            logger.error(f"Order submission failed for {sig.symbol}: {e}")
            return None

    def submit_bracket_order(
        self,
        signal: Signal,
        risk_decision: RiskDecision,
    ) -> Optional[str]:
        """Submit entry + stop + take_profit via Alpaca OCO bracket order."""
        from alpaca.trading.requests import MarketOrderRequest, TakeProfitRequest, StopLossRequest
        from alpaca.trading.enums import OrderSide, TimeInForce

        if not risk_decision.approved or risk_decision.modified_signal is None:
            return None

        sig = risk_decision.modified_signal
        trade_id = self._gen_trade_id(sig.symbol)

        account = self.client.get_account()
        equity = float(account.equity)
        qty = int(equity * sig.position_size_pct * sig.leverage / sig.entry_price)

        if qty <= 0:
            return None

        if self.dry_run:
            return trade_id

        try:
            req = MarketOrderRequest(
                symbol=sig.symbol,
                qty=qty,
                side=OrderSide.BUY,
                time_in_force=TimeInForce.DAY,
                client_order_id=trade_id,
                order_class="bracket",
                stop_loss=StopLossRequest(stop_price=round(sig.stop_loss, 2)),
                take_profit=TakeProfitRequest(limit_price=round(sig.take_profit, 2)) if sig.take_profit else None,
            )
            order = self.client.trading_client.submit_order(req)
            return order.id
        except Exception as e:
            logger.error(f"Bracket order failed for {sig.symbol}: {e}")
            return None

    def modify_stop(self, symbol: str, order_id: str, new_stop: float, current_stop: float) -> bool:
        """Only tighten stops, never widen."""
        if new_stop <= current_stop:
            logger.warning(f"modify_stop: new_stop ${new_stop:.2f} would widen stop — rejected")
            return False
        try:
            self.client.trading_client.replace_order_by_id(
                order_id,
                stop_price=round(new_stop, 2),
            )
            return True
        except Exception as e:
            logger.error(f"modify_stop failed for {symbol}: {e}")
            return False

    def cancel_order(self, order_id: str) -> bool:
        """
        Cancel an open order by its Alpaca order ID.

        Returns True on success, False on error.
        """
        if self.dry_run:
            return True
        try:
            self.client.trading_client.cancel_order_by_id(order_id)
            return True
        except Exception as e:
            logger.error(f"Cancel order {order_id} failed: {e}")
            return False

    def close_position(self, symbol: str) -> bool:
        """
        Close the entire open position for the given symbol.

        Returns True on success, False on error.
        """
        if self.dry_run:
            return True
        try:
            self.client.trading_client.close_position(symbol)
            return True
        except Exception as e:
            logger.error(f"close_position {symbol} failed: {e}")
            return False

    def close_all_positions(self) -> bool:
        """
        Close all open positions and cancel all pending orders.

        Returns True on success, False on error.
        """
        if self.dry_run:
            return True
        try:
            self.client.trading_client.close_all_positions(cancel_orders=True)
            return True
        except Exception as e:
            logger.error(f"close_all_positions failed: {e}")
            return False
