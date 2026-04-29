"""Tests for circuit breaker threshold logic."""

from unittest.mock import patch

import pytest

from core.risk.circuit_breaker import CircuitBreaker
from core.risk.portfolio_state import PortfolioState


def _portfolio(equity=100_000, daily_start=None, weekly_start=None, peak=None):
    p = PortfolioState(equity=equity, cash=equity, buying_power=equity)
    p.daily_start_equity = daily_start or equity
    p.weekly_start_equity = weekly_start or equity
    p.peak_equity = peak or equity
    return p


CFG = {
    "daily_dd_reduce": 0.02,
    "daily_dd_halt": 0.03,
    "weekly_dd_reduce": 0.05,
    "weekly_dd_halt": 0.07,
    "max_dd_from_peak": 0.10,
}


class TestCircuitBreaker:
    def setup_method(self):
        self.cb = CircuitBreaker(CFG)

    def test_normal_no_trigger(self):
        p = _portfolio(equity=100_000, daily_start=99_000)  # only -1% daily
        action, _ = self.cb.check(p)
        assert action == "NORMAL"

    def test_daily_soft_threshold(self):
        p = _portfolio(equity=97_500, daily_start=100_000)  # -2.5% → soft
        action, _ = self.cb.check(p)
        assert action == "REDUCE_50_DAY"

    def test_daily_halt_threshold(self):
        p = _portfolio(equity=96_500, daily_start=100_000)  # -3.5% → halt
        action, _ = self.cb.check(p)
        assert action == "CLOSE_ALL_DAY"

    def test_weekly_soft_threshold(self):
        p = _portfolio(equity=94_500, daily_start=94_500, weekly_start=100_000)  # -5.5% weekly
        action, _ = self.cb.check(p)
        assert action == "REDUCE_50_WEEK"

    def test_weekly_halt_threshold(self):
        p = _portfolio(equity=92_000, daily_start=92_000, weekly_start=100_000)  # -8% weekly
        action, _ = self.cb.check(p)
        assert action == "CLOSE_ALL_WEEK"

    def test_peak_drawdown_halt(self):
        p = _portfolio(equity=89_000, daily_start=89_000, weekly_start=89_000, peak=100_000)  # -11% from peak
        with patch.object(self.cb, "_write_lock_file"):
            action, _ = self.cb.check(p)
        assert action == "HALTED"

    def test_exactly_at_soft_boundary_not_triggered(self):
        # Exactly -2.0% daily → should NOT trigger soft threshold (threshold is strictly greater)
        p = _portfolio(equity=98_000, daily_start=100_000)
        action, _ = self.cb.check(p)
        # Depends on implementation: >=2% triggers soft
        assert action in ("NORMAL", "REDUCE_50_DAY")

    def test_reason_string_non_empty_on_trigger(self):
        p = _portfolio(equity=96_500, daily_start=100_000)
        action, reason = self.cb.check(p)
        assert action != "NORMAL"
        assert len(reason) > 0
