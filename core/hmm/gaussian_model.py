"""Gaussian HMM backed by hmmlearn, implementing BaseHMMModel."""

import warnings

import numpy as np
from hmmlearn import hmm
from scipy.stats import multivariate_normal

from core.hmm.base_model import BaseHMMModel


class GaussianHMMModel(BaseHMMModel):
    """Multivariate Gaussian HMM wrapper around hmmlearn.GaussianHMM."""

    def __init__(
        self,
        n_components: int,
        covariance_type: str = "full",
        n_iter: int = 200,
        random_state: int = 0,
        tol: float = 1e-4,
    ) -> None:
        self._n = n_components
        self._inner: hmm.GaussianHMM = hmm.GaussianHMM(
            n_components=n_components,
            covariance_type=covariance_type,
            n_iter=n_iter,
            random_state=random_state,
            tol=tol,
        )

    @classmethod
    def from_fitted(cls, fitted: hmm.GaussianHMM) -> "GaussianHMMModel":
        """Wrap an already-fitted hmmlearn model (migration from old pickle format)."""
        obj = cls.__new__(cls)
        obj._n = fitted.n_components
        obj._inner = fitted
        return obj

    def fit(self, X: np.ndarray) -> None:
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            self._inner.fit(X)

    def log_emission_matrix(self, X: np.ndarray) -> np.ndarray:
        """Vectorized Gaussian log-pdf for all observations; shape (T, K)."""
        log_probs = np.full((len(X), self._n), -1e10)
        for k in range(self._n):
            try:
                log_probs[:, k] = multivariate_normal.logpdf(
                    X, mean=self._inner.means_[k], cov=self._inner.covars_[k]
                )
            except Exception:
                pass
        return log_probs

    def predict(self, X: np.ndarray) -> np.ndarray:
        return self._inner.predict(X)

    def score(self, X: np.ndarray) -> float:
        return float(self._inner.score(X))

    def n_free_params(self, n_features: int) -> int:
        K, d = self._n, n_features
        return K * K + K * d + K * d * d

    @property
    def n_components(self) -> int:
        return self._n

    @property
    def transmat_(self) -> np.ndarray:
        return self._inner.transmat_

    @property
    def startprob_(self) -> np.ndarray:
        return self._inner.startprob_

    @property
    def means_(self) -> np.ndarray:
        return self._inner.means_

    @property
    def covars_(self) -> np.ndarray:
        return self._inner.covars_
