"""HMM regime detector: BIC model selection, stability filtering, and live forward inference.

Supports Gaussian and Student-t emission models via ``emission_type`` config key.
Live inference uses the forward algorithm only (never Viterbi) to avoid look-ahead bias.
Macro features (VIX, yield curve, credit proxy) are used when stored via ``set_macro_df``.
"""

import pickle
import logging
from datetime import datetime
from typing import Dict, List, Optional, Tuple

import numpy as np

from core.hmm.base_model import BaseHMMModel
from core.hmm.gaussian_model import GaussianHMMModel
from core.hmm.labels import REGIME_LABELS
from core.hmm.regime_info import RegimeInfo
from core.hmm.regime_state import RegimeState
from core.hmm.student_t_model import StudentTHMMModel
from core.timeutil import ensure_utc, utc_now
from data.feature_engineering import get_feature_matrix

logger = logging.getLogger(__name__)


class HMMEngine:
    """Market regime detector: pluggable emission model, BIC selection, and stability filter."""

    def __init__(self, config: dict) -> None:
        """Create an engine from settings.

        Args:
            config: HMM hyperparameters. Key options:
                    ``emission_type`` (``"gaussian"`` or ``"student_t"``),
                    ``student_t_dof``, ``n_candidates``, ``n_init``,
                    ``min_train_bars``, ``stability_bars``, ``flicker_threshold``.
        """
        self.config = config
        self._model: Optional[BaseHMMModel] = None
        self._macro_df = None

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

    # ------------------------------------------------------------------ #
    # Public API                                                           #
    # ------------------------------------------------------------------ #

    def set_macro_df(self, macro_df) -> None:
        """Store macro features used by ``predict_regime_filtered`` and ``train``."""
        self._macro_df = macro_df

    def train(self, bars) -> "HMMEngine":
        """Fit an emission model with BIC-selected state count and build regime metadata.

        Args:
            bars: OHLCV DataFrame. Macro features from ``set_macro_df`` are merged automatically.

        Returns:
            ``self`` for chaining.

        Raises:
            ValueError: If fewer than ``min_train_bars`` valid rows remain after feature extraction.
        """
        feature_matrix, _ = get_feature_matrix(bars, macro_df=self._macro_df)

        min_bars = self.config.get("min_train_bars", 504)
        if len(feature_matrix) < min_bars:
            raise ValueError(
                f"Need at least {min_bars} bars after feature computation, got {len(feature_matrix)}."
            )

        candidates = self.config.get("n_candidates", [3, 4, 5, 6, 7])
        n_init = self.config.get("n_init", 10)

        best_bic, self._model, best_n = self._select_by_bic(feature_matrix, candidates, n_init)
        logger.info("HMM trained: emission=%s n_regimes=%s BIC=%.2f",
                    self.config.get("emission_type", "gaussian"), best_n, best_bic)

        self.n_regimes = best_n
        self.bic_score = best_bic
        self.training_date = utc_now()
        self._build_regime_infos(feature_matrix)
        return self

    def predict_regime_filtered(self, bars) -> RegimeState:
        """Run the forward algorithm on ``bars`` and apply the stability filter.

        Uses macro features stored via ``set_macro_df`` if available.

        Args:
            bars: OHLCV history ending at the bar to score.

        Returns:
            Filtered ``RegimeState`` for the final bar.

        Raises:
            RuntimeError: If the model has not been trained.
            ValueError: If no valid feature rows remain.
        """
        if self._model is None:
            raise RuntimeError("Model not trained. Call train() first.")

        feature_matrix, _ = get_feature_matrix(bars, macro_df=self._macro_df)
        if len(feature_matrix) == 0:
            raise ValueError("No valid feature rows after NaN removal.")

        log_emit = self._model.log_emission_matrix(feature_matrix)
        alpha = self._forward_pass(log_emit)
        state_probs = self._normalize_log(np.log(alpha[-1] + 1e-300))
        state_id = int(np.argmax(state_probs))
        probability = float(state_probs[state_id])

        return self._apply_stability_filter(state_id, probability, state_probs)

    def predict_regime_proba(self, bars) -> np.ndarray:
        """Return state probabilities at the final bar without the stability filter.

        Returns:
            Array of shape ``(K,)`` summing to 1.
        """
        feature_matrix, _ = get_feature_matrix(bars, macro_df=self._macro_df)
        log_emit = self._model.log_emission_matrix(feature_matrix)
        alpha = self._forward_pass(log_emit)
        return alpha[-1]

    # ------------------------------------------------------------------ #
    # Model selection                                                      #
    # ------------------------------------------------------------------ #

    def _select_by_bic(
        self, X: np.ndarray, candidates: List[int], n_init: int
    ) -> Tuple[float, BaseHMMModel, int]:
        """Fit each candidate state count and return the lowest-BIC model."""
        best_bic = float("inf")
        best_model: Optional[BaseHMMModel] = None
        best_n: Optional[int] = None

        for n in candidates:
            bic, model = self._fit_with_bic(X, n, n_init)
            if bic < best_bic:
                best_bic, best_model, best_n = bic, model, n

        assert best_model is not None and best_n is not None
        return best_bic, best_model, best_n

    def _fit_with_bic(self, X: np.ndarray, n: int, n_init: int) -> Tuple[float, BaseHMMModel]:
        """Fit one candidate with multiple random restarts; return BIC and best model.

        Raises:
            RuntimeError: If every restart fails.
        """
        best_score = float("-inf")
        best_model: Optional[BaseHMMModel] = None

        emission_type = self.config.get("emission_type", "gaussian")
        # Fewer restarts for Student-t since k-means init is stable
        effective_inits = n_init if emission_type == "gaussian" else max(3, n_init // 3)

        for seed in range(effective_inits):
            model = self._build_model(n, seed)
            try:
                model.fit(X)
                score = model.score(X)
                if score > best_score:
                    best_score = score
                    best_model = model
            except Exception:
                continue

        if best_model is None:
            raise RuntimeError(f"All HMM fits failed for n_components={n}")

        n_params = best_model.n_free_params(X.shape[1])
        bic = -2 * best_score + n_params * np.log(len(X))
        return bic, best_model

    def _build_model(self, n_components: int, seed: int) -> BaseHMMModel:
        """Instantiate the configured emission model for one BIC candidate."""
        emission_type = self.config.get("emission_type", "gaussian")
        cov_type = self.config.get("covariance_type", "full")
        dof = float(self.config.get("student_t_dof", 4.0))

        if emission_type == "student_t":
            return StudentTHMMModel(n_components=n_components, dof=dof)
        return GaussianHMMModel(
            n_components=n_components,
            covariance_type=cov_type,
            n_iter=200,
            random_state=seed,
        )

    # ------------------------------------------------------------------ #
    # Regime metadata                                                      #
    # ------------------------------------------------------------------ #

    def _build_regime_infos(self, feature_matrix: np.ndarray) -> None:
        """Assign return-ordered labels and build ``RegimeInfo`` rows from training features."""
        hidden_seq = self._model.predict(feature_matrix)
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
        """Mean return (col 0) and mean |vol| (col 3) per state from Viterbi paths."""
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
        """Volatility rank of each state in [0, 1] (0 = calmest)."""
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

    # ------------------------------------------------------------------ #
    # Forward algorithm                                                    #
    # ------------------------------------------------------------------ #

    def _forward_pass(self, log_emit: np.ndarray) -> np.ndarray:
        """Normalized forward recursion on a precomputed (T, K) log-emission matrix.

        Per-step normalization prevents underflow without requiring full log-space arithmetic.

        Args:
            log_emit: Log-emission probabilities of shape (T, K).

        Returns:
            Normalized alpha matrix of shape (T, K); each row sums to 1.
        """
        n_obs, n_states = log_emit.shape
        alpha = np.zeros((n_obs, n_states))

        log_alpha_0 = np.log(self._model.startprob_ + 1e-300) + log_emit[0]
        alpha[0] = self._normalize_log(log_alpha_0)

        log_transmat = np.log(self._model.transmat_ + 1e-300)
        for t in range(1, n_obs):
            log_alpha_t = (
                np.logaddexp.reduce(np.log(alpha[t - 1] + 1e-300)[:, None] + log_transmat, axis=0)
                + log_emit[t]
            )
            alpha[t] = self._normalize_log(log_alpha_t)

        return alpha

    @staticmethod
    def _normalize_log(log_alpha: np.ndarray) -> np.ndarray:
        """Convert log-probabilities to a normalized probability vector."""
        shifted = log_alpha - np.max(log_alpha)
        alpha = np.exp(shifted)
        total = alpha.sum()
        return alpha / (total + 1e-300) if total > 0 else np.ones_like(alpha) / len(alpha)

    # ------------------------------------------------------------------ #
    # Stability filter                                                     #
    # ------------------------------------------------------------------ #

    def _regime_state(
        self,
        state_id: int,
        probability: float,
        state_probs: np.ndarray,
        *,
        is_confirmed: bool,
        consecutive_bars: int,
    ) -> RegimeState:
        """Construct a ``RegimeState`` snapshot for the given state and filter flags."""
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
        logger.warning("Regime confirmed: %s (p=%.3f)", self._current_state.label, probability)
        return self._current_state

    def _trim_flicker_window(self) -> None:
        """Drop flicker history older than ``flicker_window``."""
        window = self.config.get("flicker_window", 20)
        self._flicker_history = self._flicker_history[-window:]

    # ------------------------------------------------------------------ #
    # Diagnostic accessors                                                 #
    # ------------------------------------------------------------------ #

    def get_regime_stability(self) -> int:
        """Number of consecutive bars in the current confirmed regime."""
        return self._consecutive_bars

    def get_transition_matrix(self) -> np.ndarray:
        """Trained transition matrix A where A[i,j] = P(s_t=j | s_{t-1}=i).

        Raises:
            RuntimeError: If the model is not trained.
        """
        if self._model is None:
            raise RuntimeError("Model not trained.")
        return self._model.transmat_

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

    # ------------------------------------------------------------------ #
    # Persistence                                                          #
    # ------------------------------------------------------------------ #

    def save(self, path: str) -> None:
        """Pickle the fitted model and regime metadata to ``path``."""
        payload = {
            "model": self._model,
            "macro_df": self._macro_df,
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

        Automatically migrates old pickle format (bare hmmlearn model) to the new wrapper.

        Returns:
            ``self`` for chaining.
        """
        with open(path, "rb") as f:
            payload = pickle.load(f)

        raw_model = payload["model"]
        if isinstance(raw_model, BaseHMMModel):
            self._model = raw_model
        else:
            # Migrate: old format stored a bare hmmlearn GaussianHMM
            self._model = GaussianHMMModel.from_fitted(raw_model)

        self._macro_df = payload.get("macro_df")
        self.n_regimes = payload["n_regimes"]
        self.bic_score = payload["bic_score"]
        self.training_date = ensure_utc(payload["training_date"])
        self.labels = payload["labels"]
        self.regime_infos = payload["regime_infos"]
        return self

    def is_stale(self, max_days: int = 3) -> bool:
        """Whether the model is missing a training date or older than ``max_days``."""
        if self.training_date is None:
            return True
        age = (utc_now() - ensure_utc(self.training_date)).days
        return age > max_days
