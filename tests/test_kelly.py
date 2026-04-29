"""Accuracy tests for Kelly Criterion math and correlation-aware sizing."""

import numpy as np
import pandas as pd
import pytest

from core.risk.kelly_sizer import KellySizer, kelly_fraction


class TestKellyFormula:
    def test_known_values(self):
        # f* = (p*b - q) / b = (0.6*2 - 0.4) / 2 = 0.8/2 = 0.4 (full Kelly)
        f = kelly_fraction(win_rate=0.6, payoff_ratio=2.0)
        assert abs(f - 0.4) < 1e-6

    def test_breakeven_edge_zero(self):
        # p=0.5, b=1 → f* = (0.5*1 - 0.5)/1 = 0
        f = kelly_fraction(win_rate=0.5, payoff_ratio=1.0)
        assert f == 0.0

    def test_negative_edge_clamped_to_zero(self):
        # p=0.3, b=1 → f* negative → clamp to 0
        f = kelly_fraction(win_rate=0.3, payoff_ratio=1.0)
        assert f == 0.0

    def test_full_kelly_fraction_clamped_to_1(self):
        f = kelly_fraction(win_rate=0.99, payoff_ratio=10.0)
        assert f <= 1.0

    def test_default_priors_give_positive_size(self):
        # Default win_rate=0.52, payoff=1.5: (0.52*1.5-0.48)/1.5 = 0.2 (full Kelly)
        f = kelly_fraction(win_rate=0.52, payoff_ratio=1.5)
        assert abs(f - 0.2) < 1e-6


def _make_bars(n=60, trend=0.001):
    closes = np.cumprod(1 + np.random.normal(trend, 0.01, n))
    return pd.DataFrame({
        "open": closes * 0.999,
        "high": closes * 1.005,
        "low": closes * 0.995,
        "close": closes,
        "volume": np.ones(n) * 1e6,
    })


class TestKellySizer:
    def setup_method(self):
        self.config = {
            "risk": {
                "max_single_position": 0.15,
                "correlation_reduce_threshold": 0.70,
                "correlation_reject_threshold": 0.85,
            }
        }
        self.sizer = KellySizer(self.config)

    def test_size_positive_no_existing(self):
        np.random.seed(10)
        bars = _make_bars()
        size, reason = self.sizer.size("SPY", None, None, bars, {})
        assert size > 0
        assert size <= 0.15

    def test_correlated_position_reduces_size(self):
        np.random.seed(42)
        base = np.cumprod(1 + np.random.normal(0.001, 0.01, 60))
        bars_a = pd.DataFrame({"close": base, "open": base, "high": base * 1.01, "low": base * 0.99, "volume": 1e6})
        # Highly correlated (almost identical)
        bars_b = pd.DataFrame({"close": base * 1.001, "open": base, "high": base * 1.01, "low": base * 0.99, "volume": 1e6})

        size_no_existing, _ = self.sizer.size("B", None, None, bars_b, {})
        size_with_existing, _ = self.sizer.size("B", None, None, bars_b, {"A": bars_a})

        # Size should be reduced (or rejected) when correlated position exists
        assert size_with_existing <= size_no_existing

    def test_highly_correlated_position_rejected(self):
        np.random.seed(7)
        base = np.cumprod(1 + np.random.normal(0, 0.01, 60))
        bars = pd.DataFrame({"close": base, "open": base, "high": base * 1.001, "low": base * 0.999, "volume": 1e6})
        # Identical returns → correlation = 1.0 → should be rejected
        size, reason = self.sizer.size("B", None, None, bars, {"A": bars})
        assert size == 0.0
