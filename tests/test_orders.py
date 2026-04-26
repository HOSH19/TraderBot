"""
Tests for order executor in dry-run mode (no real API calls).
"""

import os
import sys
import pytest
from datetime import datetime
from unittest.mock import MagicMock, patch

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from core.timeutil import utc_now


def _make_signal(symbol="SPY", entry=400.0, stop=390.0):
    """Build a LONG Signal fixture with 20% allocation, 1.0× leverage, and the given entry/stop prices."""
    from core.regime_strategies import Signal
    return Signal(
        symbol=symbol, direction="LONG", confidence=0.75,
        entry_price=entry, stop_loss=stop, take_profit=None,
        position_size_pct=0.20, leverage=1.0,
        regime_id=0, regime_name="BULL", regime_probability=0.75,
        timestamp=utc_now(), reasoning="test", strategy_name="Test",
    )


def _make_risk_decision(signal):
    """Wrap a signal in an approved RiskDecision fixture."""
    from core.risk_manager import RiskDecision
    return RiskDecision(approved=True, modified_signal=signal, rejection_reason="")


def _mock_alpaca():
    """Return a MagicMock Alpaca client whose get_account().equity is $100,000."""
    client = MagicMock()
    account = MagicMock()
    account.equity = "100000"
    client.get_account.return_value = account
    return client


class TestOrderExecutorDryRun:
    """Tests for OrderExecutor operating in dry-run mode; no real API calls are made."""

    def test_dry_run_submit_returns_trade_id(self):
        """submit_order in dry-run mode must return a non-None trade ID that includes the ticker symbol."""
        from broker.order_executor import OrderExecutor
        client = _mock_alpaca()
        executor = OrderExecutor(client, dry_run=True)
        signal = _make_signal()
        decision = _make_risk_decision(signal)
        order_id = executor.submit_order(signal, decision)
        assert order_id is not None
        assert "SPY" in order_id

    def test_rejected_signal_not_submitted(self):
        """submit_order must return None and skip execution when the RiskDecision is not approved."""
        from broker.order_executor import OrderExecutor
        from core.risk_manager import RiskDecision
        client = _mock_alpaca()
        executor = OrderExecutor(client, dry_run=True)
        signal = _make_signal()
        rejected = RiskDecision(approved=False, modified_signal=None, rejection_reason="test rejection")
        order_id = executor.submit_order(signal, rejected)
        assert order_id is None

    def test_modify_stop_only_tightens(self):
        """modify_stop must reject a new stop that is below (looser than) the current stop."""
        from broker.order_executor import OrderExecutor
        client = _mock_alpaca()
        executor = OrderExecutor(client, dry_run=True)
        result = executor.modify_stop("SPY", "order123", new_stop=385.0, current_stop=390.0)
        assert result is False

    def test_modify_stop_tighter_accepted(self):
        """modify_stop must accept a new stop that is above (tighter than) the current stop."""
        from broker.order_executor import OrderExecutor
        client = _mock_alpaca()
        executor = OrderExecutor(client, dry_run=True)
        result = executor.modify_stop("SPY", "order123", new_stop=395.0, current_stop=390.0)
        assert result is True

    def test_close_all_dry_run(self):
        """close_all_positions in dry-run mode must return True without calling the real API."""
        from broker.order_executor import OrderExecutor
        client = _mock_alpaca()
        executor = OrderExecutor(client, dry_run=True)
        result = executor.close_all_positions()
        assert result is True
        client.trading_client.close_all_positions.assert_not_called()

    def test_trade_id_is_unique(self):
        """Each successive submit_order call must produce a distinct trade ID."""
        from broker.order_executor import OrderExecutor
        client = _mock_alpaca()
        executor = OrderExecutor(client, dry_run=True)
        ids = set()
        for _ in range(10):
            signal = _make_signal()
            decision = _make_risk_decision(signal)
            order_id = executor.submit_order(signal, decision)
            ids.add(order_id)
        assert len(ids) == 10, "trade_ids should be unique"

    def test_bracket_order_dry_run(self):
        """submit_bracket_order in dry-run mode must return a non-None order ID for a signal with a take-profit."""
        from broker.order_executor import OrderExecutor
        from core.regime_strategies import Signal
        client = _mock_alpaca()
        executor = OrderExecutor(client, dry_run=True)
        signal = Signal(
            symbol="SPY", direction="LONG", confidence=0.75,
            entry_price=400.0, stop_loss=390.0, take_profit=420.0,
            position_size_pct=0.20, leverage=1.0,
            regime_id=0, regime_name="BULL", regime_probability=0.75,
            timestamp=utc_now(), reasoning="test", strategy_name="Test",
        )
        decision = _make_risk_decision(signal)
        order_id = executor.submit_bracket_order(signal, decision)
        assert order_id is not None
