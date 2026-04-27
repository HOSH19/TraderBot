"""Instantiate vol-tier strategies per ``regime_id`` and batch ``generate_signal`` calls."""

import logging
from typing import Dict, List, Optional

import pandas as pd  # noqa: F401 — used in type annotation for bars_by_symbol

from core.hmm.regime_info import RegimeInfo
from core.hmm.regime_state import RegimeState
from core.strategies.base_strategy import BaseStrategy
from core.strategies.signal import Signal
from core.strategies.vol_tier import _strategy_class_for_vol_rank_fraction

logger = logging.getLogger(__name__)


class StrategyOrchestrator:
    """Route each ``regime_id`` to a low / mid / high vol template by ranked ``expected_volatility``."""

    def __init__(self, config: dict, regime_infos: List[RegimeInfo]) -> None:
        """Build the regime→strategy map from training-time ``RegimeInfo`` rows.

        Args:
            config: Strategy settings (deadband, uncertainty multiplier, etc.).
            regime_infos: One info row per HMM state after ``HMMEngine.train``.
        """
        self.config = config
        self._strategy_map: Dict[int, BaseStrategy] = {}
        self._update_mapping(regime_infos)

    def _update_mapping(self, regime_infos: List[RegimeInfo]) -> None:
        """Sort states by ``expected_volatility`` and assign tier strategies."""
        self._strategy_map = {}
        n = len(regime_infos)
        if n == 0:
            return

        denom = max(n - 1, 1)
        sorted_by_vol = sorted(regime_infos, key=lambda r: r.expected_volatility)
        for rank, info in enumerate(sorted_by_vol):
            strategy_cls = _strategy_class_for_vol_rank_fraction(rank / denom)
            self._strategy_map[info.regime_id] = strategy_cls(self.config)

    def update_regime_infos(self, regime_infos: List[RegimeInfo]) -> None:
        """Refresh mappings after a new ``train()`` (new ``n_regimes`` or vol ordering)."""
        self._update_mapping(regime_infos)

    def generate_signals(
        self,
        symbols: List[str],
        bars_by_symbol: Dict[str, pd.DataFrame],
        regime_state: RegimeState,
        is_flickering: bool,
        current_allocations: Optional[Dict[str, float]] = None,
    ) -> List[Signal]:
        """Emit targets per symbol for the strategy tied to ``regime_state.state_id``.

        Args:
            symbols: Tickers to evaluate.
            bars_by_symbol: Causal OHLCV history per symbol.
            regime_state: Filtered HMM state for the primary series.
            is_flickering: If true, shrink size and pin leverage to 1.0.
            current_allocations: Optional live weights; suppresses noise inside the deadband.

        Returns:
            List of approved ``Signal`` objects (may be empty).
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
