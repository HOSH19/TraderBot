"""BIC-based model selection — fit candidate state counts and return the best."""

import logging
from typing import List, Optional, Tuple

import numpy as np

from core.hmm.base_model import BaseHMMModel

logger = logging.getLogger(__name__)


class ModelSelector:
    """Fit each candidate n_components and return the lowest-BIC model."""

    def __init__(self, config: dict) -> None:
        self._cfg = config

    def select(self, X: np.ndarray) -> Tuple[float, BaseHMMModel, int]:
        """Return (best_bic, best_model, best_n)."""
        candidates = self._cfg.get("n_candidates", [3, 4, 5, 6, 7])
        n_init = self._cfg.get("n_init", 10)

        best_bic = float("inf")
        best_model: Optional[BaseHMMModel] = None
        best_n: Optional[int] = None

        for n in candidates:
            bic, model = self._fit_candidate(X, n, n_init)
            if bic < best_bic:
                best_bic, best_model, best_n = bic, model, n

        assert best_model is not None and best_n is not None
        return best_bic, best_model, best_n

    def _fit_candidate(self, X: np.ndarray, n: int, n_init: int) -> Tuple[float, BaseHMMModel]:
        """Multi-restart fit; return (BIC, best_model). Raises if all restarts fail."""
        emission_type = self._cfg.get("emission_type", "gaussian")
        effective_inits = n_init if emission_type == "gaussian" else max(3, n_init // 3)

        best_score = float("-inf")
        best_model: Optional[BaseHMMModel] = None

        for seed in range(effective_inits):
            model = self._build_model(n, seed)
            try:
                model.fit(X)
                score = model.score(X)
                if score > best_score:
                    best_score, best_model = score, model
            except Exception:
                continue

        if best_model is None:
            raise RuntimeError(f"All HMM fits failed for n_components={n}")

        n_params = best_model.n_free_params(X.shape[1])
        bic = -2 * best_score + n_params * np.log(len(X))
        return bic, best_model

    def _build_model(self, n_components: int, seed: int) -> BaseHMMModel:
        from core.hmm.gaussian_model import GaussianHMMModel
        from core.hmm.student_t_model import StudentTHMMModel

        emission_type = self._cfg.get("emission_type", "gaussian")
        if emission_type == "student_t":
            return StudentTHMMModel(n_components=n_components, dof=float(self._cfg.get("student_t_dof", 4.0)))
        return GaussianHMMModel(
            n_components=n_components,
            covariance_type=self._cfg.get("covariance_type", "full"),
            n_iter=200,
            random_state=seed,
        )
