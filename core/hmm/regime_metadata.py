"""Build RegimeInfo rows from a fitted HMM model and training feature matrix."""

from typing import List, Tuple

import numpy as np

from core.hmm.labels import REGIME_LABELS
from core.hmm.regime_info import RegimeInfo


def build_regime_infos(model, feature_matrix: np.ndarray, n_regimes: int, min_confidence: float) -> Tuple[List[str], List[RegimeInfo]]:
    """Derive labels and RegimeInfo rows from training-time Viterbi paths."""
    hidden_seq = model.predict(feature_matrix)
    mean_returns, mean_vols = _state_mean_returns_vols(feature_matrix, hidden_seq, n_regimes)
    labels = _assign_return_ordered_labels(mean_returns, n_regimes)
    vol_rank_frac = _vol_rank_fractions(mean_vols, n_regimes)

    regime_infos = []
    for i in range(n_regimes):
        stype, max_lev, max_pos = _strategy_params_for_vol_rank(float(vol_rank_frac[i]))
        regime_infos.append(RegimeInfo(
            regime_id=i,
            regime_name=labels[i],
            expected_return=mean_returns[i],
            expected_volatility=mean_vols[i],
            recommended_strategy_type=stype,
            max_leverage_allowed=max_lev,
            max_position_size_pct=max_pos,
            min_confidence_to_act=min_confidence,
        ))
    return labels, regime_infos


def _state_mean_returns_vols(
    feature_matrix: np.ndarray, hidden_seq: np.ndarray, n_regimes: int
) -> Tuple[List[float], List[float]]:
    mean_returns, mean_vols = [], []
    for i in range(n_regimes):
        mask = hidden_seq == i
        if mask.sum() == 0:
            mean_returns.append(0.0)
            mean_vols.append(1.0)
        else:
            mean_returns.append(float(feature_matrix[mask, 0].mean()))
            mean_vols.append(float(np.abs(feature_matrix[mask, 3]).mean()))
    return mean_returns, mean_vols


def _assign_return_ordered_labels(mean_returns: List[float], n_regimes: int) -> List[str]:
    sorted_by_return = np.argsort(mean_returns)
    labels_for_n = REGIME_LABELS[n_regimes]
    labels = [""] * n_regimes
    for rank, regime_id in enumerate(sorted_by_return):
        labels[int(regime_id)] = labels_for_n[rank]
    return labels


def _vol_rank_fractions(mean_vols: List[float], n_regimes: int) -> np.ndarray:
    sorted_by_vol = np.argsort(mean_vols)
    vol_ranks = np.empty(n_regimes)
    denom = max(n_regimes - 1, 1)
    for rank, regime_id in enumerate(sorted_by_vol):
        vol_ranks[int(regime_id)] = rank / denom
    return vol_ranks


def _strategy_params_for_vol_rank(vol_rank_frac: float) -> Tuple[str, float, float]:
    if vol_rank_frac <= 0.33:
        return "LowVolBull", 1.25, 0.95
    if vol_rank_frac >= 0.67:
        return "HighVolDefensive", 1.0, 0.60
    return "MidVolCautious", 1.0, 0.95
