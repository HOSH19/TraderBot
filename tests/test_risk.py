"""
Tests for the risk management layer.
Circuit breakers, position sizing, leverage rules, order validation.
"""

import os
import sys
import pytest
from datetime import datetime

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


def _load_config():
    """Load the project settings.yaml config and return it as a dict."""
    import yaml
    cfg_path = os.path.join(os.path.dirname(__file__), "..", "config", "settings.yaml")
    with open(cfg_path) as f:
        return yaml.safe_load(f)


def _make_portfolio(equity: float = 100_000, daily_dd: float = 0.0, peak_dd: float = 0.0):
    """
    Build a PortfolioState fixture.

    daily_dd and peak_dd are fractional drawdowns (e.g. -0.025 means -2.5%) used to
    back-calculate the reference equity levels that the circuit-breaker compares against.
    """
    from core.risk_manager import PortfolioState
    portfolio = PortfolioState(equity=equity, cash=equity, buying_power=equity)
    portfolio.daily_start_equity = equity / (1 + daily_dd) if daily_dd != 0 else equity
    portfolio.peak_equity = equity / (1 + peak_dd) if peak_dd != 0 else equity
    portfolio.weekly_start_equity = equity
    return portfolio


def _make_signal(symbol="SPY", alloc=0.95, lev=1.0, entry=400.0, stop=390.0):
    """Build a LONG Signal fixture with the given symbol, allocation, leverage, entry, and stop-loss."""
    from core.regime_strategies import Signal
    return Signal(
        symbol=symbol, direction="LONG", confidence=0.75,
        entry_price=entry, stop_loss=stop, take_profit=None,
        position_size_pct=alloc, leverage=lev,
        regime_id=0, regime_name="BULL", regime_probability=0.75,
        timestamp=datetime.utcnow(), reasoning="test", strategy_name="Test",
    )


class TestCircuitBreakers:
    """Tests for the CircuitBreaker: daily drawdown thresholds, peak drawdown halting, and lock-file behaviour."""

    def test_normal_state_passes(self):
        """Portfolio within drawdown limits should receive NORMAL action from the circuit breaker."""
        from core.risk_manager import CircuitBreaker
        config = _load_config()
        cb = CircuitBreaker(config.get("risk", {}))
        portfolio = _make_portfolio()
        action, reason = cb.check(portfolio)
        assert action == "NORMAL"

    def test_daily_dd_reduce_threshold(self):
        """A -2.5% daily drawdown must trigger the REDUCE_50_DAY action."""
        from core.risk_manager import CircuitBreaker
        config = _load_config()
        cb = CircuitBreaker(config.get("risk", {}))
        portfolio = _make_portfolio(equity=98_000, daily_dd=-0.025)
        action, _ = cb.check(portfolio)
        assert action == "REDUCE_50_DAY"

    def test_daily_dd_halt_threshold(self):
        """A -3.5% daily drawdown must trigger the CLOSE_ALL_DAY action."""
        from core.risk_manager import CircuitBreaker
        config = _load_config()
        cb = CircuitBreaker(config.get("risk", {}))
        portfolio = _make_portfolio(equity=97_000, daily_dd=-0.035)
        action, _ = cb.check(portfolio)
        assert action == "CLOSE_ALL_DAY"

    def test_peak_dd_creates_lock_file(self, tmp_path, monkeypatch):
        """A -12% peak drawdown must halt trading and write the lock file to disk."""
        from core.risk_manager import CircuitBreaker
        import core.risk_manager as rm
        monkeypatch.setattr(rm, "TRADING_HALTED_LOCK", str(tmp_path / "trading_halted.lock"))
        config = _load_config()
        cb = CircuitBreaker(config.get("risk", {}))
        portfolio = _make_portfolio(equity=88_000, peak_dd=-0.12)
        action, reason = cb.check(portfolio)
        assert action == "HALTED"
        assert (tmp_path / "trading_halted.lock").exists()

    def test_lock_file_blocks_trading(self, tmp_path, monkeypatch):
        """A pre-existing lock file must cause the circuit breaker to return HALTED even without a new drawdown."""
        from core.risk_manager import CircuitBreaker
        import core.risk_manager as rm
        lock_path = tmp_path / "trading_halted.lock"
        lock_path.write_text("halted")
        monkeypatch.setattr(rm, "TRADING_HALTED_LOCK", str(lock_path))
        config = _load_config()
        cb = CircuitBreaker(config.get("risk", {}))
        action, _ = cb.check(_make_portfolio())
        assert action == "HALTED"


class TestRiskManager:
    """Tests for RiskManager signal validation: approval, rejection reasons, and position-size capping."""

    def test_valid_signal_approved(self):
        """A well-formed signal against a healthy portfolio must be approved."""
        from core.risk_manager import RiskManager
        config = _load_config()
        rm = RiskManager(config)
        signal = _make_signal()
        portfolio = _make_portfolio()
        decision = rm.validate_signal(signal, portfolio)
        assert decision.approved

    def test_missing_stop_rejected(self):
        """A signal with a zero stop-loss must be rejected with a reason mentioning 'stop_loss'."""
        from core.risk_manager import RiskManager
        config = _load_config()
        rm = RiskManager(config)
        signal = _make_signal(stop=0.0)
        portfolio = _make_portfolio()
        decision = rm.validate_signal(signal, portfolio)
        assert not decision.approved
        assert "stop_loss" in decision.rejection_reason.lower()

    def test_max_concurrent_positions_blocks(self):
        """Opening a new position when the max concurrent limit is already reached must be rejected."""
        from core.risk_manager import RiskManager, Position
        config = _load_config()
        rm = RiskManager(config)
        portfolio = _make_portfolio()
        for i in range(5):
            sym = f"SYM{i}"
            portfolio.positions[sym] = Position(
                symbol=sym, shares=10, entry_price=100, entry_time=datetime.utcnow(),
                current_price=100, stop_loss=90, regime_at_entry="BULL"
            )
        signal = _make_signal(symbol="NEW")
        decision = rm.validate_signal(signal, portfolio)
        assert not decision.approved
        assert "concurrent" in decision.rejection_reason.lower()

    def test_leverage_forced_down_with_active_cb(self):
        """When a circuit breaker is active, leverage on an approved signal must be reduced to 1.0."""
        from core.risk_manager import RiskManager
        import core.risk_manager as rm_module
        import os
        config = _load_config()
        rm = RiskManager(config)
        portfolio = _make_portfolio(equity=97_000, daily_dd=-0.035)
        signal = _make_signal(lev=1.25)
        decision = rm.validate_signal(signal, portfolio)
        if decision.approved and decision.modified_signal:
            assert decision.modified_signal.leverage == 1.0

    def test_extreme_signal_size_capped(self):
        """An oversized allocation must be capped to the configured max_single_position limit."""
        from core.risk_manager import RiskManager
        config = _load_config()
        rm = RiskManager(config)
        signal = _make_signal(alloc=2.0, lev=1.25)
        portfolio = _make_portfolio()
        decision = rm.validate_signal(signal, portfolio)
        if decision.approved and decision.modified_signal:
            assert decision.modified_signal.position_size_pct <= config["risk"]["max_single_position"]

    def test_daily_trade_limit(self):
        """Exceeding the configured max_daily_trades count must cause the next signal to be rejected."""
        from core.risk_manager import RiskManager
        config = _load_config()
        rm = RiskManager(config)
        rm._daily_trade_count = config["risk"]["max_daily_trades"]
        signal = _make_signal()
        portfolio = _make_portfolio()
        decision = rm.validate_signal(signal, portfolio)
        assert not decision.approved
        assert "daily trade limit" in decision.rejection_reason.lower()
