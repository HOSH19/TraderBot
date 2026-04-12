"""
HMM Regime Detection Engine.

Design: volatility classifier — detects calm vs turbulent environments.
Uses forward algorithm only (never Viterbi) to prevent look-ahead bias.
BIC model selection across n_components [3,4,5,6,7].
"""

import pickle
import logging
import warnings
from dataclasses import dataclass, field
from datetime import datetime
from typing import List, Optional, Dict, Tuple

import numpy as np
from hmmlearn import hmm

from data.feature_engineering import compute_features, get_feature_matrix

logger = logging.getLogger(__name__)

REGIME_LABELS = {
    3: ["BEAR", "NEUTRAL", "BULL"],
    4: ["CRASH", "BEAR", "BULL", "EUPHORIA"],
    5: ["CRASH", "BEAR", "NEUTRAL", "BULL", "EUPHORIA"],
    6: ["CRASH", "STRONG_BEAR", "WEAK_BEAR", "WEAK_BULL", "STRONG_BULL", "EUPHORIA"],
    7: ["CRASH", "STRONG_BEAR", "WEAK_BEAR", "NEUTRAL", "WEAK_BULL", "STRONG_BULL", "EUPHORIA"],
}


@dataclass
class RegimeInfo:
    """Metadata describing a detected HMM regime, including allocation and risk parameters."""

    regime_id: int
    regime_name: str
    expected_return: float
    expected_volatility: float
    recommended_strategy_type: str
    max_leverage_allowed: float
    max_position_size_pct: float
    min_confidence_to_act: float


@dataclass
class RegimeState:
    """Snapshot of the current regime produced by the stability filter."""

    label: str
    state_id: int
    probability: float
    state_probabilities: np.ndarray
    timestamp: datetime
    is_confirmed: bool
    consecutive_bars: int


class HMMEngine:
    """Gaussian HMM-based market regime detector with BIC model selection and stability filtering."""

    def __init__(self, config: dict):
        """Initialize the HMM engine with configuration settings."""
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
        self._cached_alpha: Optional[np.ndarray] = None

    def train(self, bars) -> "HMMEngine":
        """Train with BIC model selection. bars is a DataFrame with OHLCV columns."""
        feature_matrix, valid_idx = get_feature_matrix(bars)

        if len(feature_matrix) < self.config.get("min_train_bars", 504):
            raise ValueError(
                f"Need at least {self.config.get('min_train_bars', 504)} bars after feature "
                f"computation, got {len(feature_matrix)}."
            )

        candidates = self.config.get("n_candidates", [3, 4, 5, 6, 7])
        n_init = self.config.get("n_init", 10)
        cov_type = self.config.get("covariance_type", "full")

        best_bic = float("inf")
        best_model = None
        best_n = None
        bic_scores: Dict[int, float] = {}

        for n in candidates:
            bic, model = self._fit_with_bic(feature_matrix, n, n_init, cov_type)
            bic_scores[n] = bic
            logger.info(f"HMM n_components={n}  BIC={bic:.2f}")
            if bic < best_bic:
                best_bic = bic
                best_model = model
                best_n = n

        logger.info(f"Selected n_components={best_n} (BIC={best_bic:.2f})  Scores: {bic_scores}")

        self.model = best_model
        self.n_regimes = best_n
        self.bic_score = best_bic
        self.training_date = datetime.utcnow()
        self._cached_alpha = None

        self._build_regime_infos(feature_matrix)
        return self

    def _fit_with_bic(
        self, X: np.ndarray, n: int, n_init: int, cov_type: str
    ) -> Tuple[float, hmm.GaussianHMM]:
        """Fit a GaussianHMM with n_init random seeds and return the (BIC, best_model) pair."""
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

    def _build_regime_infos(self, feature_matrix: np.ndarray):
        """Label regimes by mean return (ascending) and compute vol-based metadata."""
        hidden_seq = self.model.predict(feature_matrix)
        mean_returns = []
        mean_vols = []

        for i in range(self.n_regimes):
            mask = hidden_seq == i
            if mask.sum() == 0:
                mean_returns.append(0.0)
                mean_vols.append(1.0)
                continue
            ret_col = 0
            vol_col = 3
            mean_returns.append(feature_matrix[mask, ret_col].mean())
            mean_vols.append(np.abs(feature_matrix[mask, vol_col]).mean())

        sorted_by_return = np.argsort(mean_returns)
        labels_for_n = REGIME_LABELS[self.n_regimes]
        self.labels = [""] * self.n_regimes

        for rank, regime_id in enumerate(sorted_by_return):
            self.labels[regime_id] = labels_for_n[rank]

        sorted_by_vol = np.argsort(mean_vols)
        vol_ranks = np.empty(self.n_regimes)
        for rank, regime_id in enumerate(sorted_by_vol):
            vol_ranks[regime_id] = rank / max(self.n_regimes - 1, 1)

        self.regime_infos = []
        for i in range(self.n_regimes):
            vr = vol_ranks[i]
            if vr <= 0.33:
                strategy_type = "LowVolBull"
                max_lev = 1.25
                max_pos = 0.95
            elif vr >= 0.67:
                strategy_type = "HighVolDefensive"
                max_lev = 1.0
                max_pos = 0.60
            else:
                strategy_type = "MidVolCautious"
                max_lev = 1.0
                max_pos = 0.95

            self.regime_infos.append(RegimeInfo(
                regime_id=i,
                regime_name=self.labels[i],
                expected_return=mean_returns[i],
                expected_volatility=mean_vols[i],
                recommended_strategy_type=strategy_type,
                max_leverage_allowed=max_lev,
                max_position_size_pct=max_pos,
                min_confidence_to_act=self.config.get("min_confidence", 0.55),
            ))

        logger.info(f"Regime infos built: {[(r.regime_name, r.recommended_strategy_type) for r in self.regime_infos]}")

    def predict_regime_filtered(self, bars) -> RegimeState:
        """
        Compute P(state_t | observations_1:t) using the forward algorithm.
        Uses ONLY past and present data. Never uses model.predict() (Viterbi).
        Caches alpha for incremental live updates.
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

        self._cached_alpha = alpha[-1].copy()

        regime_state = self._apply_stability_filter(state_id, probability, state_probs)
        return regime_state

    def predict_regime_proba(self, bars) -> np.ndarray:
        """Return full probability distribution over regimes for the last bar."""
        feature_matrix, _ = get_feature_matrix(bars)
        alpha = self._forward_pass(feature_matrix)
        return alpha[-1]

    def _forward_pass(self, X: np.ndarray) -> np.ndarray:
        """
        Manual forward algorithm.
        alpha[t] = P(o_1,...,o_t, s_t) normalized.
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
        """Log emission probability for each state given observation."""
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
        """Convert log-space alpha values to a normalized probability distribution."""
        max_val = np.max(log_alpha)
        shifted = log_alpha - max_val
        alpha = np.exp(shifted)
        total = alpha.sum()
        return alpha / (total + 1e-300) if total > 0 else np.ones_like(alpha) / len(alpha)

    def _apply_stability_filter(
        self, raw_state_id: int, probability: float, state_probs: np.ndarray
    ) -> RegimeState:
        """Require stability_bars consecutive signals before confirming a regime change."""
        stability_bars = self.config.get("stability_bars", 3)
        confirmed = False

        if self._current_state is None:
            self._current_state = RegimeState(
                label=self.labels[raw_state_id],
                state_id=raw_state_id,
                probability=probability,
                state_probabilities=state_probs,
                timestamp=datetime.utcnow(),
                is_confirmed=True,
                consecutive_bars=1,
            )
            self._consecutive_bars = 1
            return self._current_state

        current_id = self._current_state.state_id

        if raw_state_id == current_id:
            self._pending_regime_id = None
            self._pending_bars = 0
            self._consecutive_bars += 1
            confirmed = True
        else:
            if self._pending_regime_id == raw_state_id:
                self._pending_bars += 1
            else:
                self._pending_regime_id = raw_state_id
                self._pending_bars = 1

            if self._pending_bars >= stability_bars:
                self._flicker_history.append(1)
                self._current_state = RegimeState(
                    label=self.labels[raw_state_id],
                    state_id=raw_state_id,
                    probability=probability,
                    state_probabilities=state_probs,
                    timestamp=datetime.utcnow(),
                    is_confirmed=True,
                    consecutive_bars=self._pending_bars,
                )
                self._consecutive_bars = self._pending_bars
                self._pending_regime_id = None
                self._pending_bars = 0
                logger.warning(
                    f"Regime confirmed: {self._current_state.label} (p={probability:.3f})"
                )
                return self._current_state
            else:
                self._flicker_history.append(0)

        window = self.config.get("flicker_window", 20)
        self._flicker_history = self._flicker_history[-window:]

        return RegimeState(
            label=self.labels[self._current_state.state_id],
            state_id=self._current_state.state_id,
            probability=probability,
            state_probabilities=state_probs,
            timestamp=datetime.utcnow(),
            is_confirmed=confirmed,
            consecutive_bars=self._consecutive_bars,
        )

    def get_regime_stability(self) -> int:
        """Consecutive bars in current confirmed regime."""
        return self._consecutive_bars

    def get_transition_matrix(self) -> np.ndarray:
        """Return the trained HMM transition probability matrix."""
        if self.model is None:
            raise RuntimeError("Model not trained.")
        return self.model.transmat_

    def detect_regime_change(self) -> bool:
        """True only if a regime change was confirmed this bar."""
        return self._pending_bars == 0 and self._consecutive_bars == self.config.get("stability_bars", 3)

    def get_regime_flicker_rate(self) -> int:
        """Number of regime changes in the current flicker window."""
        return sum(self._flicker_history)

    def is_flickering(self) -> bool:
        """Return True if the number of recent regime changes exceeds the configured threshold."""
        return self.get_regime_flicker_rate() > self.config.get("flicker_threshold", 4)

    def get_current_regime_info(self) -> Optional[RegimeInfo]:
        """Return the RegimeInfo for the currently active regime, or None if not yet set."""
        if self._current_state is None or not self.regime_infos:
            return None
        return self.regime_infos[self._current_state.state_id]

    def save(self, path: str):
        """Serialize the trained model and metadata to a pickle file at path."""
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
        logger.info(f"HMM model saved to {path}")

    def load(self, path: str) -> "HMMEngine":
        """Load a previously saved HMM model from path and return self."""
        with open(path, "rb") as f:
            payload = pickle.load(f)
        self.model = payload["model"]
        self.n_regimes = payload["n_regimes"]
        self.bic_score = payload["bic_score"]
        self.training_date = payload["training_date"]
        self.labels = payload["labels"]
        self.regime_infos = payload["regime_infos"]
        self._cached_alpha = None
        logger.info(
            f"HMM model loaded from {path} "
            f"(n={self.n_regimes}, trained={self.training_date}, BIC={self.bic_score:.2f})"
        )
        return self

    def is_stale(self, max_days: int = 7) -> bool:
        """Return True if the model is untrained or was trained more than max_days ago."""
        if self.training_date is None:
            return True
        age = (datetime.utcnow() - self.training_date).days
        return age > max_days
