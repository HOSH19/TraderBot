"""Accuracy tests for indicator math in core/signals/indicators.py."""

import numpy as np
import pandas as pd
import pytest

from core.signals.indicators import atr, bollinger, macd, rsi


def _close(values):
    return pd.Series(values, dtype=float)


def _ohlcv(highs, lows, closes):
    return pd.DataFrame({"high": highs, "low": lows, "close": closes}, dtype=float)


class TestRSI:
    def test_mostly_gains_high_rsi(self):
        # Some noise ensures avg_loss > 0 so RSI stays defined; strong uptrend → high RSI
        np.random.seed(0)
        gains = [100.0 + i * 0.3 + np.random.normal(0, 0.5) for i in range(60)]
        result = rsi(_close(gains), period=14).dropna()
        assert len(result) > 0
        assert result.iloc[-1] > 60

    def test_mostly_losses_low_rsi(self):
        np.random.seed(1)
        losses = [100.0 - i * 0.3 + np.random.normal(0, 0.5) for i in range(60)]
        result = rsi(_close(losses), period=14).dropna()
        assert len(result) > 0
        assert result.iloc[-1] < 40

    def test_flat_prices_returns_50(self):
        close = _close([100.0] * 30)
        result = rsi(close, period=14)
        # All changes are 0; RSI is undefined (0/0) — result should be NaN or 50
        last = result.dropna()
        if len(last) > 0:
            assert abs(last.iloc[-1] - 50) < 1 or np.isnan(last.iloc[-1])

    def test_returns_series_same_length(self):
        close = _close(range(50))
        result = rsi(close, period=14)
        assert len(result) == 50

    def test_bounded_0_to_100(self):
        np.random.seed(42)
        close = _close(np.cumprod(1 + np.random.normal(0, 0.02, 100)))
        result = rsi(close, period=14).dropna()
        assert (result >= 0).all() and (result <= 100).all()


class TestMACD:
    def test_returns_expected_columns(self):
        close = _close(np.cumprod(1 + np.random.normal(0, 0.01, 100)))
        result = macd(close)
        assert set(result.columns) >= {"macd", "signal", "hist"}

    def test_histogram_equals_macd_minus_signal(self):
        np.random.seed(1)
        close = _close(np.cumprod(1 + np.random.normal(0, 0.01, 100)))
        result = macd(close).dropna()
        diff = (result["macd"] - result["signal"] - result["hist"]).abs()
        assert diff.max() < 1e-10

    def test_trending_up_produces_positive_macd(self):
        close = _close([100.0 + i * 0.5 for i in range(100)])
        result = macd(close).dropna()
        assert result["macd"].iloc[-1] > 0

    def test_trending_down_produces_negative_macd(self):
        close = _close([200.0 - i * 0.5 for i in range(100)])
        result = macd(close).dropna()
        assert result["macd"].iloc[-1] < 0


class TestBollinger:
    def test_returns_expected_columns(self):
        close = _close(np.cumprod(1 + np.random.normal(0, 0.01, 50)))
        result = bollinger(close)
        assert set(result.columns) >= {"upper", "mid", "lower"}

    def test_mid_is_rolling_mean(self):
        np.random.seed(2)
        close = _close(np.cumprod(1 + np.random.normal(0, 0.01, 60)))
        result = bollinger(close, period=20)
        expected_mid = close.rolling(20).mean()
        pd.testing.assert_series_equal(result["mid"], expected_mid, check_names=False)

    def test_upper_above_lower(self):
        np.random.seed(3)
        close = _close(np.cumprod(1 + np.random.normal(0, 0.01, 50)))
        result = bollinger(close).dropna()
        assert (result["upper"] > result["lower"]).all()

    def test_band_width_scales_with_std(self):
        close_volatile = _close(np.cumprod(1 + np.random.normal(0, 0.05, 50)))
        close_calm = _close(np.cumprod(1 + np.random.normal(0, 0.001, 50)))
        volatile_width = (bollinger(close_volatile)["upper"] - bollinger(close_volatile)["lower"]).dropna().mean()
        calm_width = (bollinger(close_calm)["upper"] - bollinger(close_calm)["lower"]).dropna().mean()
        assert volatile_width > calm_width


class TestATR:
    def test_atr_positive(self):
        df = _ohlcv(
            [110 + i for i in range(30)],
            [90 + i for i in range(30)],
            [100 + i for i in range(30)],
        )
        result = atr(df, period=14).dropna()
        assert (result > 0).all()

    def test_zero_range_returns_near_zero(self):
        prices = [100.0] * 30
        df = _ohlcv(prices, prices, prices)
        result = atr(df, period=14).dropna()
        assert result.max() < 1e-10

    def test_wider_range_gives_larger_atr(self):
        n = 30
        narrow = _ohlcv([101.0] * n, [99.0] * n, [100.0] * n)
        wide = _ohlcv([110.0] * n, [90.0] * n, [100.0] * n)
        assert atr(wide, period=14).dropna().mean() > atr(narrow, period=14).dropna().mean()
