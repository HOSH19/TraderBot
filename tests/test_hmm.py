"""
Tests for HMM engine: training, BIC selection, regime labeling, forward algorithm.
"""

import os
import sys
import numpy as np
import pandas as pd
import pytest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


def _make_synthetic_bars(n: int = 700, seed: int = 42) -> pd.DataFrame:
    """Generate synthetic OHLCV data with regime-like behavior."""
    rng = np.random.default_rng(seed)
    prices = [100.0]
    for i in range(n - 1):
        regime = "bull" if (i % 200) < 130 else "bear"
        vol = 0.008 if regime == "bull" else 0.018
        drift = 0.0003 if regime == "bull" else -0.0002
        prices.append(prices[-1] * np.exp(rng.normal(drift, vol)))

    prices = np.array(prices)
    high = prices * (1 + rng.uniform(0, 0.005, n))
    low = prices * (1 - rng.uniform(0, 0.005, n))
    volume = rng.integers(1_000_000, 5_000_000, n).astype(float)

    idx = pd.bdate_range("2018-01-01", periods=n)
    return pd.DataFrame({
        "open": prices * (1 + rng.normal(0, 0.001, n)),
        "high": high,
        "low": low,
        "close": prices,
        "volume": volume,
    }, index=idx)


def _load_config():
    """Load the project settings.yaml config and return it as a dict."""
    import yaml
    cfg_path = os.path.join(os.path.dirname(__file__), "..", "config", "settings.yaml")
    with open(cfg_path) as f:
        return yaml.safe_load(f)


class TestHMMEngine:
    """Integration tests for HMMEngine: training, BIC selection, regime labeling, and persistence."""

    def test_train_selects_best_n(self):
        """Verify that BIC selection chooses a number of regimes within the allowed range."""
        from core.hmm_engine import HMMEngine
        config = _load_config()
        hmm = HMMEngine(config.get("hmm", {}))
        bars = _make_synthetic_bars(700)
        hmm.train(bars)
        assert hmm.n_regimes in [3, 4, 5, 6, 7]
        assert hmm.bic_score < float("inf")

    def test_regime_labels_assigned(self):
        """Ensure each trained regime receives a non-empty string label."""
        from core.hmm_engine import HMMEngine
        config = _load_config()
        hmm = HMMEngine(config.get("hmm", {}))
        bars = _make_synthetic_bars(700)
        hmm.train(bars)
        assert len(hmm.labels) == hmm.n_regimes
        for label in hmm.labels:
            assert isinstance(label, str) and len(label) > 0

    def test_regime_infos_built(self):
        """Confirm RegimeInfo objects are built for each regime with valid leverage and position-size constraints."""
        from core.hmm_engine import HMMEngine
        config = _load_config()
        hmm = HMMEngine(config.get("hmm", {}))
        bars = _make_synthetic_bars(700)
        hmm.train(bars)
        assert len(hmm.regime_infos) == hmm.n_regimes
        for info in hmm.regime_infos:
            assert info.max_leverage_allowed in [1.0, 1.25]
            assert 0 < info.max_position_size_pct <= 1.0

    def test_predict_returns_regime_state(self):
        """Verify predict_regime_filtered returns a valid RegimeState with probability in [0, 1]."""
        from core.hmm_engine import HMMEngine
        config = _load_config()
        hmm = HMMEngine(config.get("hmm", {}))
        bars = _make_synthetic_bars(700)
        hmm.train(bars)
        state = hmm.predict_regime_filtered(bars)
        assert state is not None
        assert 0.0 <= state.probability <= 1.0
        assert state.label in hmm.labels

    def test_forward_probs_sum_to_one(self):
        """Assert that the forward-algorithm probability vector sums to 1.0 within floating-point tolerance."""
        from core.hmm_engine import HMMEngine
        config = _load_config()
        hmm = HMMEngine(config.get("hmm", {}))
        bars = _make_synthetic_bars(700)
        hmm.train(bars)
        proba = hmm.predict_regime_proba(bars)
        assert abs(proba.sum() - 1.0) < 1e-6

    def test_save_and_load(self, tmp_path):
        """Confirm that a trained model can be serialized and deserialized with identical regime count and labels."""
        from core.hmm_engine import HMMEngine
        config = _load_config()
        hmm = HMMEngine(config.get("hmm", {}))
        bars = _make_synthetic_bars(700)
        hmm.train(bars)
        path = str(tmp_path / "model.pkl")
        hmm.save(path)

        hmm2 = HMMEngine(config.get("hmm", {}))
        hmm2.load(path)
        assert hmm2.n_regimes == hmm.n_regimes
        assert hmm2.labels == hmm.labels

    def test_stale_detection(self):
        """Verify is_stale returns False for a freshly trained model and True after simulating an old training date."""
        from core.hmm_engine import HMMEngine
        from datetime import datetime, timedelta
        config = _load_config()
        hmm = HMMEngine(config.get("hmm", {}))
        bars = _make_synthetic_bars(700)
        hmm.train(bars)
        assert not hmm.is_stale(max_days=7)
        hmm.training_date = datetime.utcnow() - timedelta(days=10)
        assert hmm.is_stale(max_days=7)
