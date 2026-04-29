"""Fractional Kelly position sizing with correlation-aware capping.

Full Kelly = (edge / odds). We use half-Kelly (0.5f) capped at max_fraction
to avoid overbetting and limit correlation-driven concentration.
"""

from typing import Dict, Optional, Tuple

import numpy as np
import pandas as pd


_KELLY_FRACTION = 0.5        # half-Kelly is standard for live trading
_DEFAULT_WIN_RATE = 0.52     # conservative prior when history is thin
_DEFAULT_PAYOFF = 1.5        # avg win / avg loss


def kelly_fraction(win_rate: float, payoff_ratio: float) -> float:
    """Classic Kelly formula: f = (p * b - q) / b, clamped to [0, 1]."""
    q = 1.0 - win_rate
    f = (win_rate * payoff_ratio - q) / payoff_ratio
    return max(0.0, min(1.0, f))


class KellySizer:
    """Compute Kelly-scaled position sizes with correlation and max-fraction caps."""

    def __init__(self, config: dict) -> None:
        rc = config.get("risk", {})
        self._max_fraction = rc.get("max_single_position", 0.15)
        self._corr_reduce_threshold = rc.get("correlation_reduce_threshold", 0.70)
        self._corr_reject_threshold = rc.get("correlation_reject_threshold", 0.85)

    def size(
        self,
        symbol: str,
        win_rate: Optional[float],
        payoff_ratio: Optional[float],
        bars: pd.DataFrame,
        existing_bars: Dict[str, pd.DataFrame],
    ) -> Tuple[float, str]:
        """Return (size_pct, reason) for the proposed position.

        Args:
            symbol: Ticker being sized.
            win_rate: Historical win rate for this strategy on this symbol.
            payoff_ratio: avg_win / avg_loss from backtest.
            bars: OHLCV for the symbol being added.
            existing_bars: OHLCV for symbols already in the portfolio.

        Returns:
            Tuple of (position_size_fraction, human_readable_reason).
        """
        wr = win_rate if win_rate is not None else _DEFAULT_WIN_RATE
        pr = payoff_ratio if payoff_ratio is not None else _DEFAULT_PAYOFF

        raw_kelly = kelly_fraction(wr, pr)
        sized = raw_kelly * _KELLY_FRACTION
        sized = min(sized, self._max_fraction)

        sized, reason = self._apply_correlation_cap(symbol, sized, bars, existing_bars)
        return sized, reason

    def _apply_correlation_cap(
        self,
        symbol: str,
        size: float,
        bars: pd.DataFrame,
        existing_bars: Dict[str, pd.DataFrame],
    ) -> Tuple[float, str]:
        if not existing_bars:
            return size, f"Kelly={size*100:.1f}% (no existing positions)"

        returns = bars["close"].pct_change().dropna()
        max_corr = 0.0
        most_correlated = ""

        for sym, ex_bars in existing_bars.items():
            if sym == symbol:
                continue
            ex_ret = ex_bars["close"].pct_change().dropna()
            aligned = pd.concat([returns, ex_ret], axis=1).dropna()
            if len(aligned) < 20:
                continue
            corr = float(aligned.iloc[:, 0].corr(aligned.iloc[:, 1]))
            if corr > max_corr:
                max_corr = corr
                most_correlated = sym

        if max_corr >= self._corr_reject_threshold:
            return 0.0, f"REJECTED: correlation {max_corr:.2f} with {most_correlated} ≥ {self._corr_reject_threshold}"

        if max_corr >= self._corr_reduce_threshold:
            reduced = size * (1.0 - max_corr)
            return reduced, f"Kelly reduced to {reduced*100:.1f}% (corr={max_corr:.2f} with {most_correlated})"

        return size, f"Kelly={size*100:.1f}% (max_corr={max_corr:.2f})"
