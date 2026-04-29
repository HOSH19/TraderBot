"""HMMEngine: thin coordinator over ModelSelector, StabilityFilter, RegimeMetadata, and Persistence."""

import logging
from datetime import datetime
from typing import List, Optional

import numpy as np

from core.hmm.base_model import BaseHMMModel
from core.hmm.forward_algorithm import forward_pass, normalize_log
from core.hmm.gaussian_model import GaussianHMMModel
from core.hmm.model_selector import ModelSelector
from core.hmm.persistence import load, save
from core.hmm.regime_info import RegimeInfo
from core.hmm.regime_metadata import build_regime_infos
from core.hmm.regime_state import RegimeState
from core.hmm.stability_filter import StabilityFilter
from core.timeutil import ensure_utc, utc_now
from data.feature_engineering import get_feature_matrix

logger = logging.getLogger(__name__)


class HMMEngine:
    """Market regime detector: BIC model selection, stability filtering, forward inference."""

    def __init__(self, config: dict) -> None:
        self.config = config
        self._model: Optional[BaseHMMModel] = None
        self._macro_df = None
        self._selector = ModelSelector(config)
        self._stability = StabilityFilter(config)

        self.n_regimes: int = 0
        self.regime_infos: List[RegimeInfo] = []
        self.training_date: Optional[datetime] = None
        self.bic_score: float = float("inf")
        self.labels: List[str] = []

    # ------------------------------------------------------------------ #
    # Public API                                                           #
    # ------------------------------------------------------------------ #

    def set_macro_df(self, macro_df) -> None:
        self._macro_df = macro_df

    def train(self, bars) -> "HMMEngine":
        feature_matrix, _ = get_feature_matrix(bars, macro_df=self._macro_df)
        min_bars = self.config.get("min_train_bars", 504)
        if len(feature_matrix) < min_bars:
            raise ValueError(f"Need at least {min_bars} bars, got {len(feature_matrix)}.")

        self.bic_score, self._model, self.n_regimes = self._selector.select(feature_matrix)
        logger.info("HMM trained: emission=%s n=%s BIC=%.2f",
                    self.config.get("emission_type", "gaussian"), self.n_regimes, self.bic_score)

        self.training_date = utc_now()
        min_conf = self.config.get("min_confidence", 0.55)
        self.labels, self.regime_infos = build_regime_infos(self._model, feature_matrix, self.n_regimes, min_conf)
        return self

    def predict_regime_filtered(self, bars) -> RegimeState:
        if self._model is None:
            raise RuntimeError("Model not trained. Call train() first.")
        feature_matrix, _ = get_feature_matrix(bars, macro_df=self._macro_df)
        if len(feature_matrix) == 0:
            raise ValueError("No valid feature rows after NaN removal.")

        state_probs = self._infer_state_probs(feature_matrix)
        state_id = int(np.argmax(state_probs))
        probability = float(state_probs[state_id])
        return self._stability.update(state_id, probability, state_probs, self.labels)

    def predict_regime_proba(self, bars) -> np.ndarray:
        feature_matrix, _ = get_feature_matrix(bars, macro_df=self._macro_df)
        return self._infer_state_probs(feature_matrix)

    # ------------------------------------------------------------------ #
    # Diagnostics                                                          #
    # ------------------------------------------------------------------ #

    def get_regime_stability(self) -> int:
        return self._stability.get_stability()

    def get_regime_flicker_rate(self) -> int:
        return self._stability.get_flicker_rate()

    def is_flickering(self) -> bool:
        return self._stability.is_flickering()

    def get_transition_matrix(self) -> np.ndarray:
        if self._model is None:
            raise RuntimeError("Model not trained.")
        return self._model.transmat_

    def get_current_regime_info(self) -> Optional[RegimeInfo]:
        state = self._stability.current_state
        if state is None or not self.regime_infos:
            return None
        return self.regime_infos[state.state_id]

    def detect_regime_change(self) -> bool:
        stability_bars = self.config.get("stability_bars", 3)
        return self._stability.consecutive_bars == stability_bars

    # ------------------------------------------------------------------ #
    # Persistence                                                          #
    # ------------------------------------------------------------------ #

    def save(self, path: str) -> None:
        save(path, {
            "model": self._model,
            "macro_df": self._macro_df,
            "n_regimes": self.n_regimes,
            "bic_score": self.bic_score,
            "training_date": self.training_date,
            "labels": self.labels,
            "regime_infos": self.regime_infos,
        })

    def load(self, path: str) -> "HMMEngine":
        payload = load(path)
        raw_model = payload["model"]
        self._model = raw_model if isinstance(raw_model, BaseHMMModel) else GaussianHMMModel.from_fitted(raw_model)
        self._macro_df = payload.get("macro_df")
        self.n_regimes = payload["n_regimes"]
        self.bic_score = payload["bic_score"]
        self.training_date = ensure_utc(payload["training_date"])
        self.labels = payload["labels"]
        self.regime_infos = payload["regime_infos"]
        return self

    def is_stale(self, max_days: int = 3) -> bool:
        if self.training_date is None:
            return True
        return (utc_now() - ensure_utc(self.training_date)).days > max_days

    # ------------------------------------------------------------------ #
    # Private                                                              #
    # ------------------------------------------------------------------ #

    def _infer_state_probs(self, feature_matrix: np.ndarray) -> np.ndarray:
        log_emit = self._model.log_emission_matrix(feature_matrix)
        alpha = forward_pass(log_emit, self._model.startprob_, self._model.transmat_)
        return normalize_log(np.log(alpha[-1] + 1e-300))
