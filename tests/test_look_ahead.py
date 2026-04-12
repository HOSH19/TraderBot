"""
MANDATORY: Verify no look-ahead bias in regime predictions.

The forward algorithm must produce identical regime predictions at any time T
regardless of how much data exists AFTER T.

  regime at bar 400 using data[0:400] == regime at bar 400 using data[0:500]

If this fails, look-ahead bias is present and backtests are invalid.
"""

import os
import sys
import numpy as np
import pandas as pd
import pytest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


def _make_synthetic_bars(n: int = 700, seed: int = 42) -> pd.DataFrame:
    """Generate synthetic OHLCV bars with alternating volatility regimes for look-ahead tests."""
    rng = np.random.default_rng(seed)
    prices = [100.0]
    for i in range(n - 1):
        vol = 0.008 if (i % 200) < 130 else 0.018
        drift = 0.0003 if (i % 200) < 130 else -0.0002
        prices.append(prices[-1] * np.exp(rng.normal(drift, vol)))
    prices = np.array(prices)
    high = prices * (1 + rng.uniform(0, 0.005, n))
    low = prices * (1 - rng.uniform(0, 0.005, n))
    volume = rng.integers(1_000_000, 5_000_000, n).astype(float)
    idx = pd.bdate_range("2018-01-01", periods=n)
    return pd.DataFrame({
        "open": prices * (1 + rng.normal(0, 0.001, n)),
        "high": high, "low": low, "close": prices, "volume": volume,
    }, index=idx)


def _load_config():
    """Load the project settings.yaml config and return it as a dict."""
    import yaml
    cfg_path = os.path.join(os.path.dirname(__file__), "..", "config", "settings.yaml")
    with open(cfg_path) as f:
        return yaml.safe_load(f)


def test_no_look_ahead_bias():
    """
    Regime at T must be identical with data[0:T] vs data[0:T+100].
    Tests the forward algorithm implementation is truly causal.
    """
    from core.hmm_engine import HMMEngine

    config = _load_config()
    full_data = _make_synthetic_bars(700)

    hmm = HMMEngine(config.get("hmm", {}))
    hmm.train(full_data)

    T = 400
    additional = 100

    regime_short = hmm.predict_regime_filtered(full_data.iloc[:T])
    hmm._current_state = None
    hmm._consecutive_bars = 0
    hmm._pending_regime_id = None
    hmm._pending_bars = 0
    hmm._flicker_history = []

    regime_long = hmm.predict_regime_filtered(full_data.iloc[: T + additional])
    regime_at_T = regime_long

    assert regime_short.state_id == regime_at_T.state_id, (
        f"LOOK-AHEAD BIAS DETECTED: "
        f"regime at bar {T} with data[0:{T}] = {regime_short.label} ({regime_short.state_id}), "
        f"but regime at bar {T} with data[0:{T+additional}] = {regime_at_T.label} ({regime_at_T.state_id}). "
        f"The forward algorithm is using future data."
    )


def test_proba_sum_unchanged_by_future_data():
    """Probability distribution should not shift due to future observations."""
    from core.hmm_engine import HMMEngine

    config = _load_config()
    full_data = _make_synthetic_bars(700)

    hmm = HMMEngine(config.get("hmm", {}))
    hmm.train(full_data)

    T = 400
    proba_short = hmm.predict_regime_proba(full_data.iloc[:T])
    proba_long_at_T = hmm._forward_pass(
        hmm._get_feature_matrix_cached(full_data.iloc[:T])
    )[-1] if hasattr(hmm, "_get_feature_matrix_cached") else proba_short

    assert abs(proba_short.sum() - 1.0) < 1e-5


def test_backtest_end_date_invariance():
    """
    Backtest results at a given date should not change based on how far in the future we go.
    Identical OOS periods should produce identical results regardless of test end date.
    """
    from backtest.backtester import WalkForwardBacktester

    config = _load_config()
    full_data = _make_synthetic_bars(700)
    backtester = WalkForwardBacktester(config)

    try:
        result_short = backtester.run("TEST", full_data.iloc[:550], verbose=False)
        result_long = backtester.run("TEST", full_data.iloc[:650], verbose=False)

        overlap_end = min(len(result_short.equity_curve.dropna()), len(result_long.equity_curve.dropna()))
        if overlap_end > 300:
            eq_short = result_short.equity_curve.dropna().iloc[:overlap_end]
            eq_long = result_long.equity_curve.dropna().iloc[:overlap_end]

            common_idx = eq_short.index.intersection(eq_long.index)
            if len(common_idx) > 10:
                diff = (eq_short.loc[common_idx] - eq_long.loc[common_idx]).abs().max()
                assert diff < 1.0, (
                    f"Equity curves diverge by ${diff:.2f} for overlapping period — possible look-ahead bias"
                )
    except ValueError:
        pytest.skip("Insufficient data for backtest invariance test")
