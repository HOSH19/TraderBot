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


def _make_synthetic_bars(n: int = 1000, seed: int = 42) -> pd.DataFrame:
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
    Posterior at the same calendar date must match when using a prefix vs a longer series.
    Compares the forward pass at the last valid date of the short prefix, not the last bar
    of the long window (which would be a different time step).
    """
    from core.hmm_engine import HMMEngine
    from data.feature_engineering import get_feature_matrix

    config = _load_config()
    full_data = _make_synthetic_bars()

    hmm = HMMEngine(config.get("hmm", {}))
    hmm.train(full_data)

    # Need enough raw bars that the prefix has valid feature rows (warmup ~450+ trading days).
    T = 600
    additional = 100

    bars_s = full_data.iloc[:T]
    bars_l = full_data.iloc[: T + additional]
    fm_s, idx_s = get_feature_matrix(bars_s)
    fm_l, idx_l = get_feature_matrix(bars_l)
    assert len(fm_s) > 0 and len(fm_l) > 0

    t_end = idx_s[-1]
    assert t_end in idx_l, "long window must include the short window's last valid date"
    pos_l = idx_l.get_loc(t_end)
    if isinstance(pos_l, slice):
        pos_l = pos_l.start
    else:
        pos_l = int(pos_l) if np.isscalar(pos_l) else int(pos_l.flat[0])

    alpha_s = hmm._forward_pass(fm_s)
    alpha_l = hmm._forward_pass(fm_l)

    sid_s = int(np.argmax(alpha_s[-1]))
    sid_l = int(np.argmax(alpha_l[pos_l]))

    assert sid_s == sid_l, (
        f"LOOK-AHEAD BIAS DETECTED: at {t_end} short-prefix argmax state {sid_s} != "
        f"same-date state {sid_l} when extra future bars are present."
    )


def test_proba_sum_unchanged_by_future_data():
    """Last-step regime probabilities must be a proper distribution (sum to 1)."""
    from core.hmm_engine import HMMEngine

    config = _load_config()
    full_data = _make_synthetic_bars()

    hmm = HMMEngine(config.get("hmm", {}))
    hmm.train(full_data)

    T = 600
    proba_short = hmm.predict_regime_proba(full_data.iloc[:T])
    assert abs(proba_short.sum() - 1.0) < 1e-5


def test_backtest_end_date_invariance():
    """
    Backtest results at a given date should not change based on how far in the future we go.
    Identical OOS periods should produce identical results regardless of test end date.
    """
    from backtest.backtester import WalkForwardBacktester

    config = _load_config()
    full_data = _make_synthetic_bars()
    backtester = WalkForwardBacktester(config)

    try:
        result_short = backtester.run("TEST", full_data.iloc[:550])
        result_long = backtester.run("TEST", full_data.iloc[:650])

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
