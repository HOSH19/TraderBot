"""Gaussian HMM regime detector used as a volatility / environment classifier.

Live inference uses the forward algorithm only (never Viterbi) to avoid look-ahead bias.
Training selects the state count via BIC over ``n_components`` in ``[3, 4, 5, 6, 7]``.
"""

import pickle
import logging
import warnings
from datetime import datetime
from typing import List, Optional, Dict, Tuple

import numpy as np
from hmmlearn import hmm

from core.hmm.labels import REGIME_LABELS
from core.hmm.regime_info import RegimeInfo
from core.hmm.regime_state import RegimeState
from core.timeutil import ensure_utc, utc_now
from data.feature_engineering import get_feature_matrix

logger = logging.getLogger(__name__)


class HMMEngine:
    """Market regime detector: Gaussian HMM, BIC model selection, and stability filtering."""

    def __init__(self, config: dict) -> None:
        """Create an engine from settings.

        Args:
            config: HMM hyperparameters and thresholds (e.g. ``min_train_bars``, ``n_candidates``).
        """
        self.config = config
        self.model: Optional[hmm.GaussianHMM] = None
        self.n_regimes: int = 0
        self.regime_infos: List[RegimeInfo] = []
        self.training_date: Optional[datetime] = None
        self.bic_score: float = float("inf")
        self.labels: List[str] = []

        self._current_state: Optional[RegimeState] = None
        self._consecutive_bars: int = 0
        self._pending_regime_id: Optional[int] = None
        self._pending_bars: int = 0
        self._flicker_history: List[int] = []

    def train(self, bars) -> "HMMEngine":
        """Fit the HMM with BIC-selected state count and build regime metadata.

        Args:
            bars: OHLCV bars (DataFrame) used for feature extraction and training.

        Returns:
            ``self`` for chaining.

        Raises:
            ValueError: If fewer than ``min_train_bars`` rows remain after feature computation.
        """
        feature_matrix, valid_idx = get_feature_matrix(bars)

        if len(feature_matrix) < self.config.get("min_train_bars", 504):
            raise ValueError(
                f"Need at least {self.config.get('min_train_bars', 504)} bars after feature "
                f"computation, got {len(feature_matrix)}."
            )

        candidates = self.config.get("n_candidates", [3, 4, 5, 6, 7])
        n_init = self.config.get("n_init", 10)
        cov_type = self.config.get("covariance_type", "full")

        best_bic, best_model, best_n, bic_scores = self._select_by_bic(
            feature_matrix, candidates, n_init, cov_type
        )
        logger.info("HMM trained: n_regimes=%s BIC=%.2f", best_n, best_bic)

        self.model = best_model
        self.n_regimes = best_n
        self.bic_score = best_bic
        self.training_date = utc_now()
        self._cached_alpha = None

        self._build_regime_infos(feature_matrix)
        return self

    def _select_by_bic(
        self,
        feature_matrix: np.ndarray,
        candidates: List[int],
        n_init: int,
        cov_type: str,
    ) -> Tuple[float, hmm.GaussianHMM, int, Dict[int, float]]:
        """Fit each candidate state count and pick the lowest-BIC model.

        Returns:
            Tuple of best BIC score, fitted model, winning ``n_components``, and all BIC scores.
        """
        best_bic = float("inf")
        best_model = None
        best_n: Optional[int] = None
        bic_scores: Dict[int, float] = {}

        for n in candidates:
            bic, model = self._fit_with_bic(feature_matrix, n, n_init, cov_type)
            bic_scores[n] = bic
            if bic < best_bic:
                best_bic = bic
                best_model = model
                best_n = n

        assert best_model is not None and best_n is not None
        return best_bic, best_model, best_n, bic_scores

    def _fit_with_bic(
        self, X: np.ndarray, n: int, n_init: int, cov_type: str
    ) -> Tuple[float, hmm.GaussianHMM]:
        """Fit one candidate ``n_components`` with multiple random restarts.

        Returns:
            BIC score and the best-scoring fitted ``GaussianHMM``.

        Raises:
            RuntimeError: If every fit attempt fails.
        """
        best_score = float("-inf")
        best_model = None

        for seed in range(n_init):
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                m = hmm.GaussianHMM(
                    n_components=n,
                    covariance_type=cov_type,
                    n_iter=200,
                    random_state=seed,
                    tol=1e-4,
                )
                try:
                    m.fit(X)
                    score = m.score(X)
                    if score > best_score:
                        best_score = score
                        best_model = m
                except Exception:
                    continue

        if best_model is None:
            raise RuntimeError(f"All HMM fits failed for n_components={n}")

        n_params = n * n + n * X.shape[1] + n * X.shape[1] * X.shape[1]
        bic = -2 * best_score + n_params * np.log(len(X))
        return bic, best_model

    def _build_regime_infos(self, feature_matrix: np.ndarray) -> None:
        """Assign return-ordered labels and build ``RegimeInfo`` rows from training features."""
        hidden_seq = self.model.predict(feature_matrix)
        mean_returns, mean_vols = self._state_mean_returns_vols(feature_matrix, hidden_seq)
        self._assign_return_ordered_labels(mean_returns)
        vol_rank_frac = self._vol_rank_fractions(mean_vols)
        min_conf = self.config.get("min_confidence", 0.55)

        self.regime_infos = []
        for i in range(self.n_regimes):
            stype, max_lev, max_pos = self._strategy_params_for_vol_rank(float(vol_rank_frac[i]))
            self.regime_infos.append(RegimeInfo(
                regime_id=i,
                regime_name=self.labels[i],
                expected_return=mean_returns[i],
                expected_volatility=mean_vols[i],
                recommended_strategy_type=stype,
                max_leverage_allowed=max_lev,
                max_position_size_pct=max_pos,
                min_confidence_to_act=min_conf,
            ))

    def _state_mean_returns_vols(
        self, feature_matrix: np.ndarray, hidden_seq: np.ndarray
    ) -> Tuple[List[float], List[float]]:
        """Mean return (feature col 0) and mean |vol| (col 3) per state from Viterbi paths."""
        mean_returns: List[float] = []
        mean_vols: List[float] = []
        ret_col, vol_col = 0, 3

        for i in range(self.n_regimes):
            mask = hidden_seq == i
            if mask.sum() == 0:
                mean_returns.append(0.0)
                mean_vols.append(1.0)
                continue
            mean_returns.append(float(feature_matrix[mask, ret_col].mean()))
            mean_vols.append(float(np.abs(feature_matrix[mask, vol_col]).mean()))

        return mean_returns, mean_vols

    def _assign_return_ordered_labels(self, mean_returns: List[float]) -> None:
        """Map mixture components to human labels by ascending expected return."""
        sorted_by_return = np.argsort(mean_returns)
        labels_for_n = REGIME_LABELS[self.n_regimes]
        self.labels = [""] * self.n_regimes
        for rank, regime_id in enumerate(sorted_by_return):
            self.labels[int(regime_id)] = labels_for_n[rank]

    def _vol_rank_fractions(self, mean_vols: List[float]) -> np.ndarray:
        """Volatility rank of each state in ``[0, 1]`` (0 = calmest among states)."""
        sorted_by_vol = np.argsort(mean_vols)
        vol_ranks = np.empty(self.n_regimes)
        denom = max(self.n_regimes - 1, 1)
        for rank, regime_id in enumerate(sorted_by_vol):
            vol_ranks[int(regime_id)] = rank / denom
        return vol_ranks

    @staticmethod
    def _strategy_params_for_vol_rank(vol_rank_frac: float) -> Tuple[str, float, float]:
        """Map normalized vol rank to strategy name, max leverage, and max position fraction."""
        if vol_rank_frac <= 0.33:
            return "LowVolBull", 1.25, 0.95
        if vol_rank_frac >= 0.67:
            return "HighVolDefensive", 1.0, 0.60
        return "MidVolCautious", 1.0, 0.95

    def predict_regime_filtered(self, bars) -> RegimeState:
        """Filtered state probabilities at the last bar using the forward algorithm only.

        Does not use Viterbi / ``predict()`` on the live path. Applies the stability filter
        and updates flicker bookkeeping. Caches the final alpha vector for incremental use.

        Args:
            bars: OHLCV history (DataFrame) ending at the bar to score.

        Returns:
            ``RegimeState`` for the last row after stability filtering.

        Raises:
            RuntimeError: If the model is not trained.
            ValueError: If no valid feature rows remain.
        """
        if self.model is None:
            raise RuntimeError("Model not trained. Call train() first.")

        feature_matrix, valid_idx = get_feature_matrix(bars)
        if len(feature_matrix) == 0:
            raise ValueError("No valid feature rows after NaN removal.")

        alpha = self._forward_pass(feature_matrix)
        state_probs = alpha[-1]
        state_id = int(np.argmax(state_probs))
        probability = float(state_probs[state_id])

        regime_state = self._apply_stability_filter(state_id, probability, state_probs)
        return regime_state

    def predict_regime_proba(self, bars) -> np.ndarray:
        """Return the regime probability vector for the last bar.

        Args:
            bars: OHLCV history (DataFrame).

        Returns:
            Length-``n_regimes`` probability vector (forward filter at ``t = T-1``).
        """
        feature_matrix, _ = get_feature_matrix(bars)
        alpha = self._forward_pass(feature_matrix)
        return alpha[-1]

    def _forward_pass(self, X: np.ndarray) -> np.ndarray:
        """Run the HMM forward recursion with per-step normalization.

        Each row ``alpha[t]`` is ``P(o_1..t, s_t | λ)`` normalized to sum to 1.

        Args:
            X: Feature matrix ``(n_obs, n_features)``.

        Returns:
            Array of shape ``(n_obs, n_states)`` of filtered state probabilities.
        """
        n_states = self.model.n_components
        n_obs = len(X)
        alpha = np.zeros((n_obs, n_states))

        log_emission_0 = self._log_emission(X[0])
        log_alpha_0 = np.log(self.model.startprob_ + 1e-300) + log_emission_0
        alpha[0] = self._normalize_log(log_alpha_0)

        log_transmat = np.log(self.model.transmat_ + 1e-300)

        for t in range(1, n_obs):
            log_alpha_prev = np.log(alpha[t - 1] + 1e-300)
            log_alpha_t = np.logaddexp.reduce(
                log_alpha_prev[:, None] + log_transmat, axis=0
            ) + self._log_emission(X[t])
            alpha[t] = self._normalize_log(log_alpha_t)

        return alpha

    def _log_emission(self, obs: np.ndarray) -> np.ndarray:
        """Log emission density for Gaussian emissions per state for one observation."""
        log_probs = np.zeros(self.model.n_components)
        for i in range(self.model.n_components):
            diff = obs - self.model.means_[i]
            cov = self.model.covars_[i]
            try:
                inv_cov = np.linalg.inv(cov)
                sign, log_det = np.linalg.slogdet(cov)
                if sign <= 0:
                    log_probs[i] = -1e10
                    continue
                log_probs[i] = -0.5 * (diff @ inv_cov @ diff + log_det + len(obs) * np.log(2 * np.pi))
            except np.linalg.LinAlgError:
                log_probs[i] = -1e10
        return log_probs

    @staticmethod
    def _normalize_log(log_alpha: np.ndarray) -> np.ndarray:
        """Stabilize log alphas and convert them to a normalized probability vector."""
        max_val = np.max(log_alpha)
        shifted = log_alpha - max_val
        alpha = np.exp(shifted)
        total = alpha.sum()
        return alpha / (total + 1e-300) if total > 0 else np.ones_like(alpha) / len(alpha)

    def _regime_state(
        self,
        state_id: int,
        probability: float,
        state_probs: np.ndarray,
        *,
        is_confirmed: bool,
        consecutive_bars: int,
    ) -> RegimeState:
        """Construct a ``RegimeState`` snapshot for the given argmax state and filter flags."""
        return RegimeState(
            label=self.labels[state_id],
            state_id=state_id,
            probability=probability,
            state_probabilities=state_probs,
            timestamp=utc_now(),
            is_confirmed=is_confirmed,
            consecutive_bars=consecutive_bars,
        )

    def _apply_stability_filter(
        self, raw_state_id: int, probability: float, state_probs: np.ndarray
    ) -> RegimeState:
        """Apply ``stability_bars`` debouncing before confirming a regime switch."""
        stability_bars = self.config.get("stability_bars", 3)
        if self._current_state is None:
            return self._stability_bootstrap(raw_state_id, probability, state_probs)

        if raw_state_id == self._current_state.state_id:
            return self._stability_hold_current(probability, state_probs)

        switched = self._stability_try_confirm_switch(
            raw_state_id, probability, state_probs, stability_bars
        )
        if switched is not None:
            return switched

        self._flicker_history.append(0)
        self._trim_flicker_window()
        return self._regime_state(
            self._current_state.state_id,
            probability,
            state_probs,
            is_confirmed=False,
            consecutive_bars=self._consecutive_bars,
        )

    def _stability_bootstrap(
        self, raw_state_id: int, probability: float, state_probs: np.ndarray
    ) -> RegimeState:
        """Initialize filter state on the first bar."""
        self._current_state = self._regime_state(
            raw_state_id, probability, state_probs,
            is_confirmed=True, consecutive_bars=1,
        )
        self._consecutive_bars = 1
        return self._current_state

    def _stability_hold_current(self, probability: float, state_probs: np.ndarray) -> RegimeState:
        """Extend the streak when the raw argmax matches the confirmed regime."""
        self._pending_regime_id = None
        self._pending_bars = 0
        self._consecutive_bars += 1
        self._trim_flicker_window()
        return self._regime_state(
            self._current_state.state_id,
            probability,
            state_probs,
            is_confirmed=True,
            consecutive_bars=self._consecutive_bars,
        )

    def _stability_try_confirm_switch(
        self,
        raw_state_id: int,
        probability: float,
        state_probs: np.ndarray,
        stability_bars: int,
    ) -> Optional[RegimeState]:
        """Count toward a switch; confirm when the pending state persists long enough."""
        if self._pending_regime_id == raw_state_id:
            self._pending_bars += 1
        else:
            self._pending_regime_id = raw_state_id
            self._pending_bars = 1

        if self._pending_bars < stability_bars:
            return None

        self._flicker_history.append(1)
        self._current_state = self._regime_state(
            raw_state_id, probability, state_probs,
            is_confirmed=True, consecutive_bars=self._pending_bars,
        )
        self._consecutive_bars = self._pending_bars
        self._pending_regime_id = None
        self._pending_bars = 0
        logger.warning(
            "Regime confirmed: %s (p=%.3f)", self._current_state.label, probability
        )
        return self._current_state

    def _trim_flicker_window(self) -> None:
        """Drop flicker history older than ``flicker_window``."""
        window = self.config.get("flicker_window", 20)
        self._flicker_history = self._flicker_history[-window:]

    def get_regime_stability(self) -> int:
        """Number of consecutive bars in the current confirmed regime."""
        return self._consecutive_bars

    def get_transition_matrix(self) -> np.ndarray:
        """Return the trained transition matrix ``A`` where ``A[i,j] = P(s_t=j | s_{t-1}=i)``.

        Raises:
            RuntimeError: If the model is not trained.
        """
        if self.model is None:
            raise RuntimeError("Model not trained.")
        return self.model.transmat_

    def detect_regime_change(self) -> bool:
        """Whether a regime change was confirmed on the current update."""
        return self._pending_bars == 0 and self._consecutive_bars == self.config.get("stability_bars", 3)

    def get_regime_flicker_rate(self) -> int:
        """Count of confirmed switches recorded in the flicker window."""
        return sum(self._flicker_history)

    def is_flickering(self) -> bool:
        """Whether recent confirmed switches exceed ``flicker_threshold``."""
        return self.get_regime_flicker_rate() > self.config.get("flicker_threshold", 4)

    def get_current_regime_info(self) -> Optional[RegimeInfo]:
        """Metadata for the active regime, or ``None`` if state is not initialized."""
        if self._current_state is None or not self.regime_infos:
            return None
        return self.regime_infos[self._current_state.state_id]

    def save(self, path: str) -> None:
        """Pickle the fitted model and regime metadata to ``path``."""
        payload = {
            "model": self.model,
            "n_regimes": self.n_regimes,
            "bic_score": self.bic_score,
            "training_date": self.training_date,
            "labels": self.labels,
            "regime_infos": self.regime_infos,
        }
        with open(path, "wb") as f:
            pickle.dump(payload, f)

    def load(self, path: str) -> "HMMEngine":
        """Restore model and metadata from ``path``.

        Returns:
            ``self`` for chaining.
        """
        with open(path, "rb") as f:
            payload = pickle.load(f)
        self.model = payload["model"]
        self.n_regimes = payload["n_regimes"]
        self.bic_score = payload["bic_score"]
        self.training_date = ensure_utc(payload["training_date"])
        self.labels = payload["labels"]
        self.regime_infos = payload["regime_infos"]
        self._cached_alpha = None
        return self

    def is_stale(self, max_days: int = 3) -> bool:
        """Whether the model is missing a training date or older than ``max_days``."""
        if self.training_date is None:
            return True
        age = (utc_now() - ensure_utc(self.training_date)).days
        return age > max_days
