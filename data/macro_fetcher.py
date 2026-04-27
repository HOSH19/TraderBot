"""Fetch macro conditioning features via yfinance for HMM enrichment.

Three features are returned:
  - vix:           CBOE Volatility Index level (fear gauge)
  - yield_spread:  10-year minus 3-month Treasury yield (curve steepness)
  - credit_proxy:  Daily log-return of HYG minus LQD (removes duration, isolates credit risk)

All require yfinance to be installed (``pip install yfinance``). Failures are
non-fatal — the caller receives None and falls back to price-only features.
"""

import logging
from datetime import datetime
from typing import Optional

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)

_VIX = "^VIX"
_YIELD_10Y = "^TNX"
_YIELD_3M = "^IRX"
_HY_ETF = "HYG"
_IG_ETF = "LQD"


def fetch_macro_df(start: datetime, end: datetime) -> Optional[pd.DataFrame]:
    """Download and align macro features to the given date range.

    Args:
        start: First date to fetch (inclusive).
        end:   Last date to fetch (inclusive).

    Returns:
        DataFrame with columns [vix, yield_spread, credit_proxy] indexed by date,
        or None if yfinance is unavailable or the download fails.
    """
    try:
        import yfinance as yf
    except ImportError:
        logger.warning("yfinance not installed — macro features disabled. Run: pip install yfinance")
        return None

    try:
        tickers = [_VIX, _YIELD_10Y, _YIELD_3M, _HY_ETF, _IG_ETF]
        raw = yf.download(tickers, start=start, end=end, progress=False, auto_adjust=True)

        close = raw["Close"] if isinstance(raw.columns, pd.MultiIndex) else raw
        close = close.dropna(how="all")

        vix = close[_VIX]
        yield_spread = close[_YIELD_10Y] - close[_YIELD_3M]
        credit_proxy = np.log(close[_HY_ETF] / close[_IG_ETF]).diff()

        macro = pd.DataFrame({
            "vix": vix,
            "yield_spread": yield_spread,
            "credit_proxy": credit_proxy,
        }).dropna()

        logger.info("Macro features fetched: %d rows (%s → %s)", len(macro), macro.index[0].date(), macro.index[-1].date())
        return macro

    except Exception as exc:
        logger.warning("Macro feature fetch failed: %s", exc)
        return None
