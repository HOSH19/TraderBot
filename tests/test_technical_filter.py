"""Tests for TechnicalSignalFilter gating logic."""

import numpy as np
import pandas as pd
import pytest

from core.signals.technical_filter import TechnicalSignalFilter


def _make_bars(closes, vol_mult=1.0):
    closes = np.array(closes, dtype=float)
    n = len(closes)
    return pd.DataFrame({
        "open": closes * 0.999,
        "high": closes * (1 + 0.01 * vol_mult),
        "low": closes * (1 - 0.01 * vol_mult),
        "close": closes,
        "volume": np.ones(n) * 1e6,
    })


CONFIG = {
    "technical": {
        "rsi_bull_min": 50,
        "rsi_bull_max": 75,
        "bb_period": 20,
        "bb_std": 2.0,
    }
}


class TestMomentumGate:
    def setup_method(self):
        self.filt = TechnicalSignalFilter(CONFIG)

    def test_strong_uptrend_passes(self):
        # Mix some small losses into an uptrend so RSI doesn't go NaN (all-gains → avg_loss=0→NaN)
        np.random.seed(99)
        closes = [100 + i * 0.5 + np.random.normal(0, 0.1) for i in range(80)]
        bars = _make_bars(closes)
        result = self.filt.evaluate(bars, "LowVolBull")
        assert result.confirmed
        assert result.signal_type == "momentum"
        assert result.strength > 0

    def test_strong_downtrend_blocked(self):
        closes = [200 - i * 1.5 for i in range(60)]
        bars = _make_bars(closes)
        result = self.filt.evaluate(bars, "LowVolBull")
        assert not result.confirmed

    def test_strength_in_0_to_1(self):
        closes = [100 + i * 0.5 for i in range(60)]
        bars = _make_bars(closes)
        result = self.filt.evaluate(bars, "LowVolBull")
        assert 0.0 <= result.strength <= 1.0


class TestMeanReversionGate:
    def setup_method(self):
        self.filt = TechnicalSignalFilter(CONFIG)

    def test_price_at_lower_band_passes(self):
        base = [100.0] * 40
        # Add a sharp drop at the end to push price below lower band
        closes = base + [100 - i * 3 for i in range(1, 21)]
        bars = _make_bars(closes)
        result = self.filt.evaluate(bars, "MidVolCautious")
        assert result.confirmed
        assert result.signal_type == "mean_reversion"

    def test_price_above_sma_blocked(self):
        # Price consistently above its own SMA (strong uptrend)
        closes = [80 + i * 1.5 for i in range(60)]
        bars = _make_bars(closes)
        result = self.filt.evaluate(bars, "MidVolCautious")
        assert not result.confirmed


class TestHighVolNoGate:
    def test_high_vol_always_passes(self):
        filt = TechnicalSignalFilter(CONFIG)
        # Downtrending — would fail momentum or mean-reversion gates
        closes = [200 - i for i in range(60)]
        bars = _make_bars(closes, vol_mult=3.0)
        result = filt.evaluate(bars, "HighVolDefensive")
        assert result.confirmed
        assert result.strength == 1.0
