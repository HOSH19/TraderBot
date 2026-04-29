"""Normalized forward recursion — no look-ahead bias, no Viterbi."""

import numpy as np


def forward_pass(log_emit: np.ndarray, startprob: np.ndarray, transmat: np.ndarray) -> np.ndarray:
    """Normalized forward recursion on precomputed (T, K) log-emission matrix.

    Per-step normalization prevents underflow without full log-space arithmetic.
    Returns normalized alpha matrix of shape (T, K); each row sums to 1.
    """
    n_obs, n_states = log_emit.shape
    alpha = np.zeros((n_obs, n_states))

    log_alpha_0 = np.log(startprob + 1e-300) + log_emit[0]
    alpha[0] = normalize_log(log_alpha_0)

    log_transmat = np.log(transmat + 1e-300)
    for t in range(1, n_obs):
        log_alpha_t = (
            np.logaddexp.reduce(np.log(alpha[t - 1] + 1e-300)[:, None] + log_transmat, axis=0)
            + log_emit[t]
        )
        alpha[t] = normalize_log(log_alpha_t)

    return alpha


def normalize_log(log_alpha: np.ndarray) -> np.ndarray:
    """Convert log-probabilities to a normalized probability vector."""
    shifted = log_alpha - np.max(log_alpha)
    alpha = np.exp(shifted)
    total = alpha.sum()
    return alpha / (total + 1e-300) if total > 0 else np.ones_like(alpha) / len(alpha)
