"""
Tests for volatility-based allocation strategies and StrategyOrchestrator.
"""

import os
import sys
import numpy as np
import pandas as pd
import pytest
from datetime import datetime

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


def _make_bars(n: int = 300) -> pd.DataFrame:
    """Generate n bars of synthetic OHLCV price data starting at $400."""
    rng = np.random.default_rng(99)
    prices = [400.0]
    for _ in range(n - 1):
        prices.append(prices[-1] * np.exp(rng.normal(0.0003, 0.01)))
    prices = np.array(prices)
    idx = pd.bdate_range("2022-01-01", periods=n)
    return pd.DataFrame({
        "open": prices, "high": prices * 1.005,
        "low": prices * 0.995, "close": prices,
        "volume": rng.integers(1_000_000, 5_000_000, n).astype(float),
    }, index=idx)


def _load_config():
    """Load the project settings.yaml config and return it as a dict."""
    import yaml
    cfg_path = os.path.join(os.path.dirname(__file__), "..", "config", "settings.yaml")
    with open(cfg_path) as f:
        return yaml.safe_load(f)


def _make_regime_state(state_id: int = 0, label: str = "BULL", prob: float = 0.75):
    """Build a confirmed RegimeState fixture with the given id, label, and probability."""
    from core.hmm_engine import RegimeState
    return RegimeState(
        label=label,
        state_id=state_id,
        probability=prob,
        state_probabilities=np.array([prob, 1 - prob]),
        timestamp=datetime.utcnow(),
        is_confirmed=True,
        consecutive_bars=5,
    )


def _make_regime_info(regime_id: int, vol: float, strategy_type: str):
    """Build a RegimeInfo fixture with the given id, expected volatility, and strategy type."""
    from core.hmm_engine import RegimeInfo
    return RegimeInfo(
        regime_id=regime_id,
        regime_name="TEST",
        expected_return=0.001,
        expected_volatility=vol,
        recommended_strategy_type=strategy_type,
        max_leverage_allowed=1.25 if strategy_type == "LowVolBull" else 1.0,
        max_position_size_pct=0.95,
        min_confidence_to_act=0.55,
    )


class TestStrategies:
    """Unit tests for individual regime strategies and the StrategyOrchestrator."""

    def test_low_vol_bull_direction_long(self):
        """LowVolBullStrategy must generate a LONG signal in a bull regime."""
        from core.regime_strategies import LowVolBullStrategy
        config = _load_config()
        strat = LowVolBullStrategy(config.get("strategy", {}))
        signal = strat.generate_signal("SPY", _make_bars(), _make_regime_state())
        assert signal is not None
        assert signal.direction == "LONG"

    def test_low_vol_bull_leverage(self):
        """LowVolBullStrategy must apply 1.25× leverage in a confirmed bull regime."""
        from core.regime_strategies import LowVolBullStrategy
        config = _load_config()
        strat = LowVolBullStrategy(config.get("strategy", {}))
        signal = strat.generate_signal("SPY", _make_bars(), _make_regime_state())
        assert signal.leverage == 1.25

    def test_high_vol_allocation_reduced(self):
        """HighVolDefensiveStrategy must cap position size at 60% with 1.0× leverage in a crash regime."""
        from core.regime_strategies import HighVolDefensiveStrategy
        config = _load_config()
        strat = HighVolDefensiveStrategy(config.get("strategy", {}))
        signal = strat.generate_signal("SPY", _make_bars(), _make_regime_state(label="CRASH"))
        assert signal.position_size_pct == pytest.approx(0.60)
        assert signal.leverage == 1.0

    def test_mid_vol_trend_intact(self):
        """MidVolCautiousStrategy must still generate a signal when trend is intact in a neutral regime."""
        from core.regime_strategies import MidVolCautiousStrategy
        config = _load_config()
        strat = MidVolCautiousStrategy(config.get("strategy", {}))
        bars = _make_bars(300)
        bars["close"] = bars["close"] * 1.1
        signal = strat.generate_signal("SPY", bars, _make_regime_state(label="NEUTRAL"))
        assert signal is not None

    def test_stop_loss_below_entry(self):
        """Every strategy's stop_loss must be strictly below its entry_price."""
        from core.regime_strategies import LowVolBullStrategy, MidVolCautiousStrategy, HighVolDefensiveStrategy
        config = _load_config()
        bars = _make_bars(300)
        regime = _make_regime_state()
        for StratCls in [LowVolBullStrategy, MidVolCautiousStrategy, HighVolDefensiveStrategy]:
            sig = StratCls(config.get("strategy", {})).generate_signal("SPY", bars, regime)
            if sig:
                assert sig.stop_loss < sig.entry_price, f"{StratCls.name} stop_loss >= entry_price"

    def test_uncertainty_mode_halves_size(self):
        """When regime probability is below the confidence threshold the orchestrator must halve position sizes."""
        from core.regime_strategies import StrategyOrchestrator
        config = _load_config()
        infos = [
            _make_regime_info(0, 0.005, "LowVolBull"),
            _make_regime_info(1, 0.015, "MidVolCautious"),
            _make_regime_info(2, 0.025, "HighVolDefensive"),
        ]
        orch = StrategyOrchestrator(config.get("strategy", {}), infos)
        low_prob_regime = _make_regime_state(state_id=0, prob=0.45)
        signals = orch.generate_signals(
            ["SPY"], {"SPY": _make_bars(300)}, low_prob_regime, is_flickering=False
        )
        if signals:
            assert signals[0].position_size_pct <= 0.95 * 0.5 + 0.01

    def test_no_short_signals(self):
        """Strategies must never emit a SHORT direction signal under any regime label."""
        from core.regime_strategies import LowVolBullStrategy, MidVolCautiousStrategy, HighVolDefensiveStrategy
        config = _load_config()
        bars = _make_bars(300)
        for StratCls in [LowVolBullStrategy, MidVolCautiousStrategy, HighVolDefensiveStrategy]:
            for label in ["CRASH", "BEAR", "NEUTRAL", "BULL", "EUPHORIA"]:
                regime = _make_regime_state(label=label)
                sig = StratCls(config.get("strategy", {})).generate_signal("SPY", bars, regime)
                if sig:
                    assert sig.direction in ("LONG", "FLAT"), f"Got SHORT signal from {StratCls.name}"

    def test_rebalance_threshold_prevents_churn(self):
        """Orchestrator must suppress signals when current allocation is already near the target."""
        from core.regime_strategies import StrategyOrchestrator
        config = _load_config()
        infos = [_make_regime_info(0, 0.005, "LowVolBull")]
        orch = StrategyOrchestrator(config.get("strategy", {}), infos)
        regime = _make_regime_state(state_id=0, prob=0.80)
        signals = orch.generate_signals(
            ["SPY"], {"SPY": _make_bars(300)}, regime,
            is_flickering=False,
            current_allocations={"SPY": 0.90},
        )
        assert len(signals) == 0, "Should not rebalance when already near target allocation"
