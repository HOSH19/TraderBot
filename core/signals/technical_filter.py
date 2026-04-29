"""TechnicalSignalFilter — momentum/mean-reversion confirmation gated by HMM regime.

The HMM regime determines which signal type is appropriate:
  - Low-vol / bull regimes  → momentum signals (RSI trend, MACD crossover)
  - Mid-vol / neutral       → mean-reversion signals (Bollinger band bounce)
  - High-vol / bear         → no technical confirmation required (stay defensive)

Returns a TechnicalConfirmation that the orchestrator uses to scale position size.
"""

from dataclasses import dataclass
from typing import Optional

import pandas as pd

from core.signals.indicators import bollinger, macd, rsi


@dataclass
class TechnicalConfirmation:
    confirmed: bool
    signal_type: str       # "momentum" | "mean_reversion" | "none"
    strength: float        # [0, 1] — scales position sizing
    reason: str


class TechnicalSignalFilter:
    """Check RSI, MACD, and Bollinger conditions and return a size-scaling confirmation."""

    def __init__(self, config: dict) -> None:
        tc = config.get("technical", {})
        self._rsi_period = tc.get("rsi_period", 14)
        self._rsi_bull_min = tc.get("rsi_bull_min", 50)
        self._rsi_bull_max = tc.get("rsi_bull_max", 75)
        self._rsi_bear_max = tc.get("rsi_bear_max", 50)
        self._macd_fast = tc.get("macd_fast", 12)
        self._macd_slow = tc.get("macd_slow", 26)
        self._macd_signal = tc.get("macd_signal_period", 9)
        self._bb_period = tc.get("bb_period", 20)
        self._bb_std = tc.get("bb_std", 2.0)
        self._min_bars = tc.get("min_bars", 60)

    def evaluate(self, bars: pd.DataFrame, regime_vol_tier: str) -> TechnicalConfirmation:
        """Return a confirmation based on regime tier and indicator state.

        Args:
            bars: OHLCV DataFrame for one symbol.
            regime_vol_tier: "low" | "mid" | "high" — from RegimeInfo.recommended_strategy_type.
        """
        if len(bars) < self._min_bars:
            return TechnicalConfirmation(confirmed=True, signal_type="none", strength=1.0, reason="insufficient bars")

        tier = regime_vol_tier.lower()
        if "high" in tier or "defensive" in tier:
            return TechnicalConfirmation(confirmed=True, signal_type="none", strength=1.0, reason="defensive regime")

        close = bars["close"]
        if "low" in tier or "bull" in tier:
            return self._momentum_check(close)
        return self._mean_reversion_check(close, bars)

    def _momentum_check(self, close: pd.Series) -> TechnicalConfirmation:
        rsi_val = rsi(close, self._rsi_period).iloc[-1]
        macd_df = macd(close, self._macd_fast, self._macd_slow, self._macd_signal)
        hist = macd_df["hist"].iloc[-1]
        prev_hist = macd_df["hist"].iloc[-2] if len(macd_df) > 1 else hist

        rsi_ok = self._rsi_bull_min <= rsi_val <= self._rsi_bull_max
        macd_ok = hist > 0 and hist > prev_hist  # positive and growing

        if rsi_ok and macd_ok:
            strength = min(1.0, (rsi_val - self._rsi_bull_min) / (self._rsi_bull_max - self._rsi_bull_min))
            return TechnicalConfirmation(confirmed=True, signal_type="momentum", strength=0.7 + 0.3 * strength,
                                         reason=f"RSI={rsi_val:.0f} MACD↑")
        if rsi_ok or macd_ok:
            return TechnicalConfirmation(confirmed=True, signal_type="momentum", strength=0.6,
                                         reason=f"partial momentum RSI={rsi_val:.0f}")
        return TechnicalConfirmation(confirmed=False, signal_type="momentum", strength=0.0,
                                     reason=f"momentum not confirmed RSI={rsi_val:.0f}")

    def _mean_reversion_check(self, close: pd.Series, bars: pd.DataFrame) -> TechnicalConfirmation:
        bb = bollinger(close, self._bb_period, self._bb_std)
        price = close.iloc[-1]
        lower = bb["lower"].iloc[-1]
        mid = bb["mid"].iloc[-1]
        upper = bb["upper"].iloc[-1]

        if price <= lower:
            strength = min(1.0, (lower - price) / (upper - lower + 1e-10) * 2 + 0.8)
            return TechnicalConfirmation(confirmed=True, signal_type="mean_reversion", strength=min(strength, 1.0),
                                         reason=f"price at/below lower BB ({price:.2f} ≤ {lower:.2f})")
        if price <= mid:
            return TechnicalConfirmation(confirmed=True, signal_type="mean_reversion", strength=0.6,
                                         reason=f"price below BB midline ({price:.2f} < {mid:.2f})")
        return TechnicalConfirmation(confirmed=False, signal_type="mean_reversion", strength=0.0,
                                     reason=f"price above BB midline ({price:.2f} > {mid:.2f})")
