"""Instantiate vol-tier strategies per regime_id and batch generate_signal calls.

Technical signal confirmation (RSI, MACD, Bollinger) scales position size per symbol
when the regime allows it.
"""

import logging
from typing import Dict, List, Optional

import pandas as pd

from core.hmm.regime_info import RegimeInfo
from core.hmm.regime_state import RegimeState
from core.signals.technical_filter import TechnicalSignalFilter
from core.strategies.base_strategy import BaseStrategy
from core.strategies.signal import Signal
from core.strategies.vol_tier import _strategy_class_for_vol_rank_fraction

logger = logging.getLogger(__name__)


class StrategyOrchestrator:
    """Route each regime_id to a vol-tier strategy and apply technical confirmation."""

    def __init__(self, config: dict, regime_infos: List[RegimeInfo]) -> None:
        self.config = config
        self._strategy_map: Dict[int, BaseStrategy] = {}
        self._regime_info_map: Dict[int, RegimeInfo] = {}
        self._tech_filter = TechnicalSignalFilter(config)
        self._update_mapping(regime_infos)

    def update_regime_infos(self, regime_infos: List[RegimeInfo]) -> None:
        self._update_mapping(regime_infos)

    def generate_signals(
        self,
        symbols: List[str],
        bars_by_symbol: Dict[str, pd.DataFrame],
        regime_state: RegimeState,
        is_flickering: bool,
        current_allocations: Optional[Dict[str, float]] = None,
    ) -> List[Signal]:
        if regime_state.state_id not in self._strategy_map:
            logger.warning("No strategy for regime_id=%s", regime_state.state_id)
            return []

        strategy = self._strategy_map[regime_state.state_id]
        regime_info = self._regime_info_map.get(regime_state.state_id)
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

            raw_signal = self._apply_uncertainty_scaling(raw_signal, regime_state, is_flickering, min_confidence, uncertainty_mult)
            raw_signal = self._apply_technical_confirmation(raw_signal, bars, regime_info)
            if raw_signal.position_size_pct == 0.0:
                continue

            if current_allocations:
                current = current_allocations.get(symbol, 0.0)
                if abs(raw_signal.position_size_pct - current) < rebalance_threshold:
                    continue

            signals.append(raw_signal)

        return signals

    # ------------------------------------------------------------------ #
    # Private                                                              #
    # ------------------------------------------------------------------ #

    def _update_mapping(self, regime_infos: List[RegimeInfo]) -> None:
        n = len(regime_infos)
        if n == 0:
            return
        denom = max(n - 1, 1)
        sorted_by_vol = sorted(regime_infos, key=lambda r: r.expected_volatility)
        self._strategy_map = {}
        self._regime_info_map = {}
        for rank, info in enumerate(sorted_by_vol):
            cls = _strategy_class_for_vol_rank_fraction(rank / denom)
            self._strategy_map[info.regime_id] = cls(self.config)
            self._regime_info_map[info.regime_id] = info

    @staticmethod
    def _apply_uncertainty_scaling(signal: Signal, regime_state: RegimeState, is_flickering: bool,
                                    min_confidence: float, uncertainty_mult: float) -> Signal:
        if regime_state.probability < min_confidence or is_flickering:
            return Signal(**{**signal.__dict__,
                             "position_size_pct": signal.position_size_pct * uncertainty_mult,
                             "leverage": 1.0,
                             "reasoning": signal.reasoning + " [UNCERTAINTY — size halved]"})
        return signal

    def _apply_technical_confirmation(self, signal: Signal, bars: pd.DataFrame, regime_info: Optional[RegimeInfo]) -> Signal:
        if regime_info is None:
            return signal
        tier = regime_info.recommended_strategy_type
        confirmation = self._tech_filter.evaluate(bars, tier)
        if not confirmation.confirmed:
            return Signal(**{**signal.__dict__,
                             "position_size_pct": 0.0,
                             "reasoning": signal.reasoning + f" [TECH BLOCKED: {confirmation.reason}]"})
        if confirmation.strength < 1.0:
            new_size = signal.position_size_pct * confirmation.strength
            return Signal(**{**signal.__dict__,
                             "position_size_pct": new_size,
                             "reasoning": signal.reasoning + f" [TECH: {confirmation.reason}]"})
        return signal
