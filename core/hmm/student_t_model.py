"""Multivariate Student-t HMM with custom EM via the Gaussian scale-mixture representation.

Student-t(ν) = integral over Gaussian(x | μ, Σ/τ) * Gamma(τ | ν/2, ν/2) dτ.
This gives heavier tails than Gaussian, which better models financial return regimes.

The EM adds an auxiliary weight E[τ_{t,k}] per observation per state that up-weights
outlier-like observations less in the M-step covariance update.
"""

import numpy as np
from scipy.special import gammaln, logsumexp
from sklearn.cluster import KMeans

from core.hmm.base_model import BaseHMMModel


class StudentTHMMModel(BaseHMMModel):
    """HMM with multivariate Student-t emissions and fixed degrees of freedom."""

    def __init__(
        self,
        n_components: int,
        dof: float = 4.0,
        n_iter: int = 200,
        tol: float = 1e-4,
    ) -> None:
        """
        Args:
            n_components: Number of hidden states.
            dof: Degrees of freedom ν; lower = heavier tails. ν=4 matches empirical equity data.
            n_iter: Maximum EM iterations.
            tol: Convergence threshold on log-likelihood change.
        """
        self._n = n_components
        self._dof = float(dof)
        self._n_iter = n_iter
        self._tol = tol

        self._means: np.ndarray = np.array([])
        self._covars: np.ndarray = np.array([])
        self._transmat: np.ndarray = np.array([])
        self._startprob: np.ndarray = np.array([])

    # ------------------------------------------------------------------ #
    # BaseHMMModel interface                                               #
    # ------------------------------------------------------------------ #

    def fit(self, X: np.ndarray) -> None:
        """Fit via Student-t EM: forward-backward E-step + scale-mixture M-step."""
        self._initialize(X)
        prev_ll = float("-inf")
        for _ in range(self._n_iter):
            log_emit = self.log_emission_matrix(X)
            gamma, xi, ll = self._e_step(log_emit)
            u = self._aux_weights(X, gamma)
            self._m_step(X, gamma, xi, u)
            if abs(ll - prev_ll) < self._tol:
                break
            prev_ll = ll

    def log_emission_matrix(self, X: np.ndarray) -> np.ndarray:
        """Student-t log-pdf for every (observation, state) pair; shape (T, K)."""
        T, d = X.shape
        nu = self._dof
        log_norm = (
            gammaln((nu + d) / 2)
            - gammaln(nu / 2)
            - d / 2 * np.log(nu * np.pi)
        )
        log_probs = np.full((T, self._n), -1e10)
        for k in range(self._n):
            diff = X - self._means[k]
            try:
                cov_inv = np.linalg.inv(self._covars[k])
                sign, log_det = np.linalg.slogdet(self._covars[k])
                if sign <= 0:
                    continue
                delta = np.einsum("td,dd,td->t", diff, cov_inv, diff)
                log_probs[:, k] = (
                    log_norm - 0.5 * log_det - (nu + d) / 2 * np.log1p(delta / nu)
                )
            except np.linalg.LinAlgError:
                pass
        return log_probs

    def predict(self, X: np.ndarray) -> np.ndarray:
        """Viterbi decoding via log-space DP; returns (T,) state sequence."""
        log_emit = self.log_emission_matrix(X)
        T, K = log_emit.shape
        log_trans = np.log(self._transmat + 1e-300)

        dp = np.full((T, K), -np.inf)
        back = np.zeros((T, K), dtype=int)
        dp[0] = np.log(self._startprob + 1e-300) + log_emit[0]

        for t in range(1, T):
            candidates = dp[t - 1, :, None] + log_trans
            back[t] = candidates.argmax(axis=0)
            dp[t] = candidates.max(axis=0) + log_emit[t]

        states = np.zeros(T, dtype=int)
        states[-1] = dp[-1].argmax()
        for t in range(T - 2, -1, -1):
            states[t] = back[t + 1, states[t + 1]]
        return states

    def score(self, X: np.ndarray) -> float:
        """Total log-likelihood via the forward pass."""
        log_alpha = self._forward(self.log_emission_matrix(X))
        return float(logsumexp(log_alpha[-1]))

    def n_free_params(self, n_features: int) -> int:
        K, d = self._n, n_features
        return K * K + K * d + K * d * d

    # ------------------------------------------------------------------ #
    # Properties                                                           #
    # ------------------------------------------------------------------ #

    @property
    def n_components(self) -> int:
        return self._n

    @property
    def transmat_(self) -> np.ndarray:
        return self._transmat

    @property
    def startprob_(self) -> np.ndarray:
        return self._startprob

    @property
    def means_(self) -> np.ndarray:
        return self._means

    @property
    def covars_(self) -> np.ndarray:
        return self._covars

    # ------------------------------------------------------------------ #
    # EM internals                                                         #
    # ------------------------------------------------------------------ #

    def _initialize(self, X: np.ndarray) -> None:
        """Seed parameters from k-means centroids with diagonal covariances."""
        km = KMeans(n_clusters=self._n, n_init=10, random_state=0)
        km.fit(X)
        labels = km.labels_
        d = X.shape[1]

        self._means = km.cluster_centers_.copy()
        self._covars = np.array([
            np.cov(X[labels == k].T) + 1e-6 * np.eye(d)
            if (labels == k).sum() > 1 else np.eye(d)
            for k in range(self._n)
        ])
        # High self-transition prior reduces flicker
        self._transmat = np.full((self._n, self._n), 0.1 / max(self._n - 1, 1))
        np.fill_diagonal(self._transmat, 0.9)
        self._startprob = np.ones(self._n) / self._n

    def _e_step(self, log_emit: np.ndarray) -> tuple:
        """Forward-backward to get responsibilities γ and transition counts ξ."""
        log_alpha = self._forward(log_emit)
        log_beta = self._backward(log_emit)

        log_gamma = log_alpha + log_beta
        log_gamma -= logsumexp(log_gamma, axis=1, keepdims=True)
        gamma = np.exp(log_gamma)

        T, K = log_emit.shape
        log_trans = np.log(self._transmat + 1e-300)
        xi = np.zeros((T - 1, K, K))
        for t in range(T - 1):
            lxi = (
                log_alpha[t, :, None]
                + log_trans
                + log_emit[t + 1]
                + log_beta[t + 1]
            )
            xi[t] = np.exp(lxi - logsumexp(lxi.ravel()))

        ll = float(logsumexp(log_alpha[-1]))
        return gamma, xi, ll

    def _aux_weights(self, X: np.ndarray, gamma: np.ndarray) -> np.ndarray:
        """E[τ_{t,k}] = (ν + d) / (ν + δ_{t,k}); shape (T, K).

        τ is the Gaussian scale mixture weight; outliers get lower τ, shrinking
        their influence on the covariance M-step.
        """
        nu, d = self._dof, X.shape[1]
        u = np.zeros_like(gamma)
        for k in range(self._n):
            diff = X - self._means[k]
            try:
                cov_inv = np.linalg.inv(self._covars[k])
                delta = np.einsum("td,dd,td->t", diff, cov_inv, diff)
            except np.linalg.LinAlgError:
                delta = np.full(len(X), nu)
            u[:, k] = (nu + d) / (nu + delta)
        return u

    def _m_step(self, X: np.ndarray, gamma: np.ndarray, xi: np.ndarray, u: np.ndarray) -> None:
        """Update π, A, μ, Σ using responsibility-weighted Student-t updates."""
        self._startprob = gamma[0] / (gamma[0].sum() + 1e-300)

        self._transmat = xi.sum(axis=0)
        self._transmat /= self._transmat.sum(axis=1, keepdims=True) + 1e-300

        d = X.shape[1]
        for k in range(self._n):
            w = gamma[:, k] * u[:, k]
            w_sum = w.sum() + 1e-300
            self._means[k] = (w[:, None] * X).sum(axis=0) / w_sum
            diff = X - self._means[k]
            # Covariance denominator uses γ (not γu) to keep scale correct
            gamma_sum = gamma[:, k].sum() + 1e-300
            self._covars[k] = (
                np.einsum("t,td,te->de", w, diff, diff) / gamma_sum
                + 1e-6 * np.eye(d)
            )

    def _forward(self, log_emit: np.ndarray) -> np.ndarray:
        """Log-space forward recursion; returns log_alpha (T, K)."""
        T, K = log_emit.shape
        log_trans = np.log(self._transmat + 1e-300)
        log_alpha = np.zeros((T, K))
        log_alpha[0] = np.log(self._startprob + 1e-300) + log_emit[0]
        for t in range(1, T):
            log_alpha[t] = (
                logsumexp(log_alpha[t - 1, :, None] + log_trans, axis=0)
                + log_emit[t]
            )
        return log_alpha

    def _backward(self, log_emit: np.ndarray) -> np.ndarray:
        """Log-space backward recursion; returns log_beta (T, K)."""
        T, K = log_emit.shape
        log_trans = np.log(self._transmat + 1e-300)
        log_beta = np.zeros((T, K))
        for t in range(T - 2, -1, -1):
            log_beta[t] = logsumexp(
                log_trans + log_emit[t + 1] + log_beta[t + 1], axis=1
            )
        return log_beta
