"""
Volatility-based allocation strategies.

Three strategy classes sorted by VOLATILITY RANK — independent of return-based regime labels.
Always LONG, never SHORT. High vol = reduce allocation, not reverse direction.
"""

import logging
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from datetime import datetime

from core.timeutil import utc_now
from typing import List, Optional, Dict, Any

import numpy as np
import pandas as pd

from core.hmm_engine import RegimeInfo, RegimeState
from data.feature_engineering import compute_atr

logger = logging.getLogger(__name__)


@dataclass
class Signal:
    """Trade signal produced by a strategy, carrying all order and sizing parameters."""

    symbol: str
    direction: str
    confidence: float
    entry_price: float
    stop_loss: float
    take_profit: Optional[float]
    position_size_pct: float
    leverage: float
    regime_id: int
    regime_name: str
    regime_probability: float
    timestamp: datetime
    reasoning: str
    strategy_name: str
    metadata: Dict[str, Any] = field(default_factory=dict)


def _ema(series: pd.Series, span: int) -> pd.Series:
    """Return the exponential moving average of series with the given span."""
    return series.ewm(span=span, adjust=False).mean()


def _compute_stop_and_params(
    bars: pd.DataFrame,
    strategy_type: str,
) -> tuple:
    """Return (atr, ema50, current_price)."""
    close = bars["close"] if "close" in bars.columns else bars["Close"]
    high = bars["high"] if "high" in bars.columns else bars["High"]
    low = bars["low"] if "low" in bars.columns else bars["Low"]

    atr_series = compute_atr(high, low, close, 14)
    ema50 = _ema(close, 50)

    current_price = float(close.iloc[-1])
    atr = float(atr_series.iloc[-1])
    ema50_val = float(ema50.iloc[-1])

    return current_price, atr, ema50_val


def _cap_long_stop_below_entry(entry: float, stop: float, atr: float) -> float:
    """Clamp raw stop so a LONG bracket is strictly below entry (EMA-based rules can sit above spot)."""
    cushion = max(0.01 * atr, entry * 1e-6, 1e-4)
    return min(stop, entry - cushion)


class BaseStrategy(ABC):
    """Abstract base class for all regime-driven allocation strategies."""

    name: str = "BaseStrategy"

    def __init__(self, config: dict):
        """Initialize the strategy with configuration parameters."""
        self.config = config

    @abstractmethod
    def generate_signal(
        self,
        symbol: str,
        bars: pd.DataFrame,
        regime_state: RegimeState,
    ) -> Optional[Signal]:
        """Generate a trade signal for symbol given OHLCV bars and the current regime state."""
        pass


class LowVolBullStrategy(BaseStrategy):
    """
    Lowest third of regimes by expected_volatility.
    Direction: LONG, Allocation: 95%, Leverage: 1.25x
    Stop: max(price - 3*ATR, 50EMA - 0.5*ATR)
    """
    name = "LowVolBullStrategy"

    def generate_signal(self, symbol, bars, regime_state) -> Optional[Signal]:
        """Generate a fully allocated long signal with 1.25x leverage in calm low-volatility conditions."""
        price, atr, ema50 = _compute_stop_and_params(bars, self.name)
        if atr == 0 or price == 0:
            return None

        stop = max(price - 3 * atr, ema50 - 0.5 * atr)
        stop = _cap_long_stop_below_entry(price, stop, atr)
        alloc = self.config.get("low_vol_allocation", 0.95)
        leverage = self.config.get("low_vol_leverage", 1.25)

        return Signal(
            symbol=symbol,
            direction="LONG",
            confidence=regime_state.probability,
            entry_price=price,
            stop_loss=stop,
            take_profit=None,
            position_size_pct=alloc,
            leverage=leverage,
            regime_id=regime_state.state_id,
            regime_name=regime_state.label,
            regime_probability=regime_state.probability,
            timestamp=utc_now(),
            reasoning=f"Low-vol regime ({regime_state.label}, p={regime_state.probability:.2f}). "
                      f"Calm market — full allocation with modest leverage.",
            strategy_name=self.name,
            metadata={"atr": atr, "ema50": ema50},
        )


class MidVolCautiousStrategy(BaseStrategy):
    """
    Middle third of regimes by expected_volatility.
    If price > 50EMA: 95% / 1.0x (trend intact)
    If price < 50EMA: 60% / 1.0x (trend broken, reduce)
    Stop: 50EMA - 0.5*ATR
    """
    name = "MidVolCautiousStrategy"

    def generate_signal(self, symbol, bars, regime_state) -> Optional[Signal]:
        """Generate a long signal sized by whether price is above or below the 50 EMA."""
        price, atr, ema50 = _compute_stop_and_params(bars, self.name)
        if atr == 0 or price == 0:
            return None

        stop = _cap_long_stop_below_entry(price, ema50 - 0.5 * atr, atr)
        trend_intact = price > ema50

        if trend_intact:
            alloc = self.config.get("mid_vol_allocation_trend", 0.95)
            leverage = 1.0
            reason_suffix = "Trend intact (price > 50EMA). Stay invested."
        else:
            alloc = self.config.get("mid_vol_allocation_no_trend", 0.60)
            leverage = 1.0
            reason_suffix = "Trend broken (price < 50EMA). Reduce allocation."

        return Signal(
            symbol=symbol,
            direction="LONG",
            confidence=regime_state.probability,
            entry_price=price,
            stop_loss=stop,
            take_profit=None,
            position_size_pct=alloc,
            leverage=leverage,
            regime_id=regime_state.state_id,
            regime_name=regime_state.label,
            regime_probability=regime_state.probability,
            timestamp=utc_now(),
            reasoning=f"Mid-vol regime ({regime_state.label}, p={regime_state.probability:.2f}). {reason_suffix}",
            strategy_name=self.name,
            metadata={"atr": atr, "ema50": ema50, "trend_intact": trend_intact},
        )


class HighVolDefensiveStrategy(BaseStrategy):
    """
    Top third of regimes by expected_volatility.
    Direction: LONG (NOT short — catches V-shaped rebounds)
    Allocation: 60%, Leverage: 1.0x
    Stop: 50EMA - 1.0*ATR (wider for volatile conditions)
    """
    name = "HighVolDefensiveStrategy"

    def generate_signal(self, symbol, bars, regime_state) -> Optional[Signal]:
        """Generate a reduced-allocation long signal to stay positioned for V-shaped rebounds."""
        price, atr, ema50 = _compute_stop_and_params(bars, self.name)
        if atr == 0 or price == 0:
            return None

        stop = _cap_long_stop_below_entry(price, ema50 - 1.0 * atr, atr)
        alloc = self.config.get("high_vol_allocation", 0.60)

        return Signal(
            symbol=symbol,
            direction="LONG",
            confidence=regime_state.probability,
            entry_price=price,
            stop_loss=stop,
            take_profit=None,
            position_size_pct=alloc,
            leverage=1.0,
            regime_id=regime_state.state_id,
            regime_name=regime_state.label,
            regime_probability=regime_state.probability,
            timestamp=utc_now(),
            reasoning=f"High-vol regime ({regime_state.label}, p={regime_state.probability:.2f}). "
                      f"Reduced allocation — staying 60% long to catch rebounds.",
            strategy_name=self.name,
            metadata={"atr": atr, "ema50": ema50},
        )


CrashDefensiveStrategy = HighVolDefensiveStrategy
BearTrendStrategy = HighVolDefensiveStrategy
MeanReversionStrategy = MidVolCautiousStrategy
BullTrendStrategy = LowVolBullStrategy
EuphoriaCautiousStrategy = LowVolBullStrategy

LABEL_TO_STRATEGY: Dict[str, type] = {
    "CRASH": HighVolDefensiveStrategy,
    "STRONG_BEAR": HighVolDefensiveStrategy,
    "WEAK_BEAR": MidVolCautiousStrategy,
    "BEAR": MidVolCautiousStrategy,
    "NEUTRAL": MidVolCautiousStrategy,
    "WEAK_BULL": LowVolBullStrategy,
    "BULL": LowVolBullStrategy,
    "STRONG_BULL": LowVolBullStrategy,
    "EUPHORIA": LowVolBullStrategy,
}


class StrategyOrchestrator:
    """
    Maps regime_id → vol_rank → strategy class.
    Vol rank is computed independently of return-based labels.
    """

    def __init__(self, config: dict, regime_infos: List[RegimeInfo]):
        """Initialize the orchestrator and build the initial regime-to-strategy mapping."""
        self.config = config
        self._strategy_map: Dict[int, BaseStrategy] = {}
        self._update_mapping(regime_infos)

    def _update_mapping(self, regime_infos: List[RegimeInfo]):
        """Build the regime_id-to-strategy map by ranking regimes by expected volatility."""
        self._strategy_map = {}
        n = len(regime_infos)
        if n == 0:
            return

        sorted_by_vol = sorted(regime_infos, key=lambda r: r.expected_volatility)
        for rank, info in enumerate(sorted_by_vol):
            position = rank / max(n - 1, 1)
            if position <= 0.33:
                strategy_cls = LowVolBullStrategy
            elif position >= 0.67:
                strategy_cls = HighVolDefensiveStrategy
            else:
                strategy_cls = MidVolCautiousStrategy
            self._strategy_map[info.regime_id] = strategy_cls(self.config)

    def update_regime_infos(self, regime_infos: List[RegimeInfo]):
        """Rebuild strategy mapping after HMM retrain."""
        self._update_mapping(regime_infos)

    def generate_signals(
        self,
        symbols: List[str],
        bars_by_symbol: Dict[str, pd.DataFrame],
        regime_state: RegimeState,
        is_flickering: bool,
        current_allocations: Optional[Dict[str, float]] = None,
    ) -> List[Signal]:
        """
        Generate signals for all symbols using the strategy mapped to the current regime.

        Applies uncertainty scaling when confidence is low or the regime is flickering, and
        skips signals that fall within rebalance_threshold of the current allocation.
        """
        if regime_state.state_id not in self._strategy_map:
            logger.warning(f"No strategy for regime_id={regime_state.state_id}, returning empty signals.")
            return []

        strategy = self._strategy_map[regime_state.state_id]
        rebalance_threshold = self.config.get("rebalance_threshold", 0.10)
        uncertainty_mult = self.config.get("uncertainty_size_mult", 0.50)
        min_confidence = self.config.get("min_confidence", 0.55)

        signals = []
        for symbol in symbols:
            bars = bars_by_symbol.get(symbol)
            if bars is None or len(bars) < 60:
                continue

            raw_signal = strategy.generate_signal(symbol, bars, regime_state)
            if raw_signal is None:
                continue

            if regime_state.probability < min_confidence or is_flickering:
                raw_signal.position_size_pct *= uncertainty_mult
                raw_signal.leverage = 1.0
                raw_signal.reasoning += " [UNCERTAINTY — size halved]"

            if current_allocations:
                current = current_allocations.get(symbol, 0.0)
                if abs(raw_signal.position_size_pct - current) < rebalance_threshold:
                    continue

            signals.append(raw_signal)

        return signals
