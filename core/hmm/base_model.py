"""Abstract interface shared by Gaussian and Student-t HMM emission models."""

from abc import ABC, abstractmethod

import numpy as np


class BaseHMMModel(ABC):
    """Common contract for HMM emission implementations used by HMMEngine."""

    @abstractmethod
    def fit(self, X: np.ndarray) -> None:
        """Fit the model to feature matrix X of shape (T, d)."""

    @abstractmethod
    def log_emission_matrix(self, X: np.ndarray) -> np.ndarray:
        """Return (T, K) log-emission probabilities for all observations and states."""

    @abstractmethod
    def predict(self, X: np.ndarray) -> np.ndarray:
        """Viterbi decoding; return (T,) integer state sequence."""

    @abstractmethod
    def score(self, X: np.ndarray) -> float:
        """Total log-likelihood of X under the fitted model."""

    @abstractmethod
    def n_free_params(self, n_features: int) -> int:
        """Number of free parameters, used for BIC computation."""

    @property
    @abstractmethod
    def n_components(self) -> int:
        """Number of hidden states K."""

    @property
    @abstractmethod
    def transmat_(self) -> np.ndarray:
        """Transition matrix of shape (K, K)."""

    @property
    @abstractmethod
    def startprob_(self) -> np.ndarray:
        """Initial state distribution of shape (K,)."""

    @property
    @abstractmethod
    def means_(self) -> np.ndarray:
        """Emission means of shape (K, d)."""

    @property
    @abstractmethod
    def covars_(self) -> np.ndarray:
        """Emission covariances of shape (K, d, d)."""
