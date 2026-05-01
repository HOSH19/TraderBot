"""Causal OHLCV features and rolling z-scores for :class:`~core.hmm.engine.HMMEngine`.

Every series uses information available at or before each bar; default z-score window is 252 sessions.
Optional macro features (VIX, yield curve, credit proxy) are appended when a macro DataFrame is supplied.
"""

from typing import Optional, Tuple

import numpy as np
import pandas as pd


def compute_log_returns(close: pd.Series, period: int) -> pd.Series:
    """Return log returns of close price over the given lookback period."""
    return np.log(close / close.shift(period))


def compute_realized_volatility(close: pd.Series, window: int = 20) -> pd.Series:
    """Return rolling realized volatility as the standard deviation of daily log returns."""
    log_ret = compute_log_returns(close, 1)
    return log_ret.rolling(window).std()


def compute_vol_ratio(close: pd.Series, short: int = 5, long: int = 20) -> pd.Series:
    """Return the ratio of short-term to long-term realized volatility, indicating vol regime changes."""
    short_vol = compute_log_returns(close, 1).rolling(short).std()
    long_vol = compute_log_returns(close, 1).rolling(long).std()
    return short_vol / (long_vol + 1e-10)


def compute_normalized_volume(volume: pd.Series, window: int = 50) -> pd.Series:
    """Return volume z-scored against its own rolling mean and standard deviation."""
    mean = volume.rolling(window).mean()
    std = volume.rolling(window).std()
    return (volume - mean) / (std + 1e-10)


def compute_volume_trend(volume: pd.Series, sma_window: int = 10) -> pd.Series:
    """Return the first difference of the rolling SMA of volume, capturing directional trend."""
    sma = volume.rolling(sma_window).mean()
    return sma.diff()


def compute_adx(high: pd.Series, low: pd.Series, close: pd.Series, period: int = 14) -> pd.Series:
    """
    Compute the Average Directional Index (ADX) to measure trend strength.

    Uses exponential smoothing on the true range and directional movement components.
    Returns values between 0 and 100; higher values indicate a stronger trend.
    """
    tr = pd.concat([
        high - low,
        (high - close.shift(1)).abs(),
        (low - close.shift(1)).abs()
    ], axis=1).max(axis=1)

    atr = tr.ewm(span=period, adjust=False).mean()

    up_move = high - high.shift(1)
    down_move = low.shift(1) - low

    plus_dm = np.where((up_move > down_move) & (up_move > 0), up_move, 0.0)
    minus_dm = np.where((down_move > up_move) & (down_move > 0), down_move, 0.0)

    plus_di = 100 * pd.Series(plus_dm, index=close.index).ewm(span=period, adjust=False).mean() / (atr + 1e-10)
    minus_di = 100 * pd.Series(minus_dm, index=close.index).ewm(span=period, adjust=False).mean() / (atr + 1e-10)

    dx = (100 * (plus_di - minus_di).abs() / (plus_di + minus_di + 1e-10))
    adx = dx.ewm(span=period, adjust=False).mean()
    return adx


def compute_sma_slope(close: pd.Series, window: int = 50) -> pd.Series:
    """Return the one-period change in the rolling SMA of close price, approximating its slope."""
    sma = close.rolling(window).mean()
    return sma.diff()


def compute_rsi(close: pd.Series, period: int = 14) -> pd.Series:
    """Compute the Relative Strength Index (RSI), returning values in the range [0, 100]."""
    delta = close.diff()
    gain = delta.clip(lower=0).ewm(span=period, adjust=False).mean()
    loss = (-delta.clip(upper=0)).ewm(span=period, adjust=False).mean()
    rs = gain / (loss + 1e-10)
    return 100 - (100 / (1 + rs))


def compute_rsi_zscore(close: pd.Series, rsi_period: int = 14, zscore_window: int = 252) -> pd.Series:
    """Return a rolling z-score of RSI, normalising its level relative to its historical distribution."""
    rsi = compute_rsi(close, rsi_period)
    mean = rsi.rolling(zscore_window).mean()
    std = rsi.rolling(zscore_window).std()
    return (rsi - mean) / (std + 1e-10)


def compute_distance_from_sma(close: pd.Series, window: int = 200) -> pd.Series:
    """Return the fractional deviation of close price from its rolling SMA (positive = above SMA)."""
    sma = close.rolling(window).mean()
    return (close - sma) / (sma + 1e-10)


def compute_roc(close: pd.Series, period: int = 10) -> pd.Series:
    """Return the Rate of Change (ROC) of close price over the given lookback period."""
    return (close - close.shift(period)) / (close.shift(period) + 1e-10)


def compute_atr(high: pd.Series, low: pd.Series, close: pd.Series, period: int = 14) -> pd.Series:
    """Compute the Average True Range (ATR) using exponential smoothing over the true range."""
    tr = pd.concat([
        high - low,
        (high - close.shift(1)).abs(),
        (low - close.shift(1)).abs()
    ], axis=1).max(axis=1)
    return tr.ewm(span=period, adjust=False).mean()


def compute_normalized_atr(high: pd.Series, low: pd.Series, close: pd.Series, period: int = 14) -> pd.Series:
    """Return ATR divided by close price, expressing volatility as a fraction of price level."""
    atr = compute_atr(high, low, close, period)
    return atr / (close + 1e-10)


def rolling_zscore(series: pd.Series, window: int = 252) -> pd.Series:
    """Standardise a series using a rolling mean and standard deviation over the given window."""
    mean = series.rolling(window).mean()
    std = series.rolling(window).std()
    return (series - mean) / (std + 1e-10)


def compute_features(
    bars: pd.DataFrame,
    macro_df: Optional[pd.DataFrame] = None,
    zscore_window: int = 252,
) -> pd.DataFrame:
    """Build the full feature matrix (z-scored columns) from OHLCV ``bars``.

    Args:
        bars: Must expose ``open``, ``high``, ``low``, ``close``, ``volume`` (any case).
        macro_df: Optional DataFrame with columns [vix, yield_spread, credit_proxy].
                  When supplied, three macro features are appended after price features.
        zscore_window: Rolling window for standardization (default 252).

    Returns:
        Feature frame aligned to ``bars.index``; warm-up rows remain ``NaN`` until populated.
    """
    bars = bars.copy()
    bars.columns = [c.lower() for c in bars.columns]

    close = bars["close"]
    high = bars["high"]
    low = bars["low"]
    volume = bars["volume"]

    features = pd.DataFrame(index=bars.index)

    features["ret_1"] = rolling_zscore(compute_log_returns(close, 1), zscore_window)
    features["ret_5"] = rolling_zscore(compute_log_returns(close, 5), zscore_window)
    features["ret_20"] = rolling_zscore(compute_log_returns(close, 20), zscore_window)

    features["realized_vol"] = rolling_zscore(compute_realized_volatility(close, 20), zscore_window)
    features["vol_ratio"] = rolling_zscore(compute_vol_ratio(close, 5, 20), zscore_window)

    features["vol_norm"] = rolling_zscore(compute_normalized_volume(volume, 50), zscore_window)
    features["vol_trend"] = rolling_zscore(compute_volume_trend(volume, 10), zscore_window)

    features["adx"] = rolling_zscore(compute_adx(high, low, close, 14), zscore_window)
    features["sma50_slope"] = rolling_zscore(compute_sma_slope(close, 50), zscore_window)

    features["rsi_zscore"] = compute_rsi_zscore(close, 14, zscore_window)
    features["dist_sma200"] = rolling_zscore(compute_distance_from_sma(close, 200), zscore_window)

    features["roc_10"] = rolling_zscore(compute_roc(close, 10), zscore_window)
    features["roc_20"] = rolling_zscore(compute_roc(close, 20), zscore_window)

    features["norm_atr"] = rolling_zscore(compute_normalized_atr(high, low, close, 14), zscore_window)

    if macro_df is not None:
        features = _append_macro_features(features, macro_df, zscore_window)

    features = features.replace([np.inf, -np.inf], np.nan)
    return features


def _append_macro_features(
    features: pd.DataFrame, macro_df: pd.DataFrame, zscore_window: int
) -> pd.DataFrame:
    """Forward-fill macro_df to match ``features`` index and append z-scored columns."""
    aligned = macro_df.reindex(features.index, method="ffill")
    for col in ("vix", "yield_spread", "credit_proxy"):
        if col in aligned.columns:
            features[f"macro_{col}"] = rolling_zscore(aligned[col], zscore_window)
    return features


def get_multi_symbol_feature_matrix(
    bars_by_symbol: dict,
    macro_df: Optional[pd.DataFrame] = None,
    zscore_window: int = 252,
) -> Tuple[np.ndarray, pd.Index]:
    """Average features across multiple symbols on their shared date index.

    Each symbol's feature matrix is computed independently and z-scored, then
    the per-date mean is taken. Only dates present in all symbols are kept.
    """
    frames = []
    for bars in bars_by_symbol.values():
        feat = compute_features(bars, macro_df=macro_df, zscore_window=zscore_window)
        frames.append(feat)

    # align on shared index, average, drop any remaining NaNs
    combined = pd.concat(frames, keys=range(len(frames))).groupby(level=1).mean()
    valid = combined.dropna()
    return valid.values, valid.index


def get_feature_matrix(
    bars: pd.DataFrame,
    macro_df: Optional[pd.DataFrame] = None,
    zscore_window: int = 252,
) -> Tuple[np.ndarray, pd.Index]:
    """Return ``compute_features`` rows with complete cases only.

    Args:
        bars: OHLCV history.
        macro_df: Optional macro features to append (see ``compute_features``).
        zscore_window: Rolling window for z-scoring.

    Returns:
        ``(values, index)`` where ``values`` is a ``float64`` ndarray for model fitting.
    """
    features = compute_features(bars, macro_df=macro_df, zscore_window=zscore_window)
    valid = features.dropna()
    return valid.values, valid.index
