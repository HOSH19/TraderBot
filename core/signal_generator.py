"""
Combines HMM regime state + strategy orchestrator into the unified signal pipeline.
"""

import logging
from typing import Dict, List, Optional, Tuple

import pandas as pd

from core.hmm_engine import HMMEngine, RegimeState
from core.regime_strategies import Signal, StrategyOrchestrator

logger = logging.getLogger(__name__)


class SignalGenerator:
    """Combines the HMM regime detector and strategy orchestrator into a single signal pipeline."""

    def __init__(self, hmm_engine: HMMEngine, orchestrator: StrategyOrchestrator, config: dict):
        """Initialize with a trained HMMEngine, a StrategyOrchestrator, and config dict."""
        self.hmm = hmm_engine
        self.orchestrator = orchestrator
        self.cfg = config

    def generate(
        self,
        symbols: List[str],
        bars_by_symbol: Dict[str, pd.DataFrame],
        current_allocations: Optional[Dict[str, float]] = None,
    ) -> Tuple[List[Signal], Optional[RegimeState]]:
        """Produce strategy signals and regime state using the first symbol as the HMM series."""
        primary = symbols[0]
        primary_bars = bars_by_symbol.get(primary)

        if primary_bars is None or len(primary_bars) < self.cfg.get("hmm", {}).get("min_train_bars", 504):
            logger.warning("Insufficient bars for regime detection")
            return [], None

        try:
            regime_state = self.hmm.predict_regime_filtered(primary_bars)
        except Exception as e:
            logger.error(f"HMM prediction failed: {e}. Holding current regime.")
            return [], None

        is_flickering = self.hmm.is_flickering()

        signals = self.orchestrator.generate_signals(
            symbols=symbols,
            bars_by_symbol=bars_by_symbol,
            regime_state=regime_state,
            is_flickering=is_flickering,
            current_allocations=current_allocations,
        )

        return signals, regime_state
