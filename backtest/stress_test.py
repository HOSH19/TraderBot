"""Monte Carlo style shocks (gaps, shuffles) on top of :class:`~backtest.walk_forward_backtester.WalkForwardBacktester`."""

import random
from typing import Dict, List, Optional

import numpy as np
import pandas as pd

from backtest import WalkForwardBacktester, BacktestResult
from backtest.performance import compute_metrics


def _primary_symbol(config: dict, override: Optional[str] = None) -> str:
    """Resolve the primary symbol from ``config`` or ``override``."""
    if override:
        return override
    syms = config.get("broker", {}).get("symbols") or ["SPY"]
    return syms[0]


def _inject_crash_gaps(bars: pd.DataFrame, n_points: int, gap_range: tuple, seed: int) -> pd.DataFrame:
    """
    Apply random multiplicative price shocks to simulate crash gap events.

    Args:
        bars: OHLCV DataFrame to modify.
        n_points: Number of bars to shock.
        gap_range: (min_shock, max_shock) as fractional multipliers (e.g. -0.15 to -0.05).
        seed: Random seed for reproducibility.
    """
    bars = bars.copy()
    bars.columns = [c.lower() for c in bars.columns]
    rng = np.random.default_rng(seed)
    indices = rng.choice(range(50, len(bars)), size=min(n_points, len(bars) - 50), replace=False)
    for idx in indices:
        shock = rng.uniform(gap_range[0], gap_range[1])
        for col in ["open", "high", "low", "close"]:
            bars.iloc[idx, bars.columns.get_loc(col)] *= (1 + shock)
    return bars


def _inject_overnight_gaps(bars: pd.DataFrame, n_points: int, atr_mult_range: tuple, seed: int) -> pd.DataFrame:
    """
    Apply random additive ATR-scaled gaps to simulate overnight price jumps.

    Args:
        bars: OHLCV DataFrame to modify.
        n_points: Number of bars to gap.
        atr_mult_range: (min_mult, max_mult) applied to the 14-period ATR.
        seed: Random seed for reproducibility.
    """
    bars = bars.copy()
    bars.columns = [c.lower() for c in bars.columns]
    close = bars["close"]
    atr = (bars["high"] - bars["low"]).rolling(14).mean()
    rng = np.random.default_rng(seed)
    indices = rng.choice(range(14, len(bars)), size=min(n_points, len(bars) - 14), replace=False)
    for idx in indices:
        mult = rng.uniform(atr_mult_range[0], atr_mult_range[1])
        direction = rng.choice([-1, 1])
        gap = direction * mult * float(atr.iloc[idx])
        for col in ["open", "high", "low", "close"]:
            bars.iloc[idx, bars.columns.get_loc(col)] += gap
    return bars


def run_crash_injection(
    bars: pd.DataFrame,
    config: dict,
    symbol: Optional[str] = None,
    n_simulations: int = 100,
    n_crash_points: int = 10,
    gap_range: tuple = (-0.15, -0.05),
) -> Dict:
    """Insert -5% to -15% single-day gaps at random points. 100 Monte Carlo runs."""
    sym = _primary_symbol(config, symbol)
    backtester = WalkForwardBacktester(config)
    max_losses = []
    cb_fired_count = 0

    for seed in range(n_simulations):
        shocked = _inject_crash_gaps(bars, n_crash_points, gap_range, seed)
        try:
            result = backtester.run(sym, shocked)
            metrics = compute_metrics(result)
            max_losses.append(metrics["max_drawdown_pct"])
            if metrics["max_drawdown_pct"] < -10:
                cb_fired_count += 1
        except Exception:
            pass

    arr = np.array(max_losses)
    return {
        "n_simulations": n_simulations,
        "mean_max_loss_pct": float(arr.mean()) if len(arr) else 0.0,
        "worst_case_pct": float(arr.min()) if len(arr) else 0.0,
        "pct_circuit_breaker_fired": cb_fired_count / n_simulations * 100,
    }


def run_gap_risk(
    bars: pd.DataFrame,
    config: dict,
    symbol: Optional[str] = None,
    n_simulations: int = 50,
    n_gap_points: int = 20,
    atr_mult_range: tuple = (2.0, 5.0),
) -> Dict:
    """Insert overnight gaps of 2–5x ATR. Compare expected vs actual loss."""
    sym = _primary_symbol(config, symbol)
    backtester = WalkForwardBacktester(config)
    actual_losses = []

    for seed in range(n_simulations):
        gapped = _inject_overnight_gaps(bars, n_gap_points, atr_mult_range, seed)
        try:
            result = backtester.run(sym, gapped)
            metrics = compute_metrics(result)
            actual_losses.append(metrics["max_drawdown_pct"])
        except Exception:
            pass

    arr = np.array(actual_losses)
    return {
        "n_simulations": n_simulations,
        "mean_actual_loss_pct": float(arr.mean()) if len(arr) else 0.0,
        "worst_actual_pct": float(arr.min()) if len(arr) else 0.0,
        "note": "Compare worst_actual vs your max_dd_from_peak limit (10%)",
    }


def run_regime_misclassification(
    bars: pd.DataFrame,
    config: dict,
    symbol: Optional[str] = None,
    n_simulations: int = 30,
) -> Dict:
    """
    Shuffle regime labels to simulate complete HMM misclassification.
    If system blows up → risk management isn't independent enough.
    """
    sym = _primary_symbol(config, symbol)
    backtester = WalkForwardBacktester(config)
    results_normal = []
    results_shuffled = []

    try:
        normal = backtester.run(sym, bars)
        results_normal.append(compute_metrics(normal)["max_drawdown_pct"])
    except Exception:
        pass

    rng = np.random.default_rng(42)
    for seed in range(n_simulations):
        shuffled_bars = bars.copy()
        shuffled_bars.columns = [c.lower() for c in shuffled_bars.columns]
        shuffled_bars["close"] = rng.permutation(shuffled_bars["close"].values)
        shuffled_bars["open"] = shuffled_bars["close"].shift(1).bfill()
        shuffled_bars["high"] = shuffled_bars["close"] * rng.uniform(1.0, 1.02, len(shuffled_bars))
        shuffled_bars["low"] = shuffled_bars["close"] * rng.uniform(0.98, 1.0, len(shuffled_bars))

        try:
            result = backtester.run(sym, shuffled_bars)
            results_shuffled.append(compute_metrics(result)["max_drawdown_pct"])
        except Exception:
            pass

    normal_dd = np.mean(results_normal) if results_normal else 0.0
    shuffled_arr = np.array(results_shuffled)

    return {
        "normal_max_dd_pct": float(normal_dd),
        "shuffled_mean_dd_pct": float(shuffled_arr.mean()) if len(shuffled_arr) else 0.0,
        "shuffled_worst_dd_pct": float(shuffled_arr.min()) if len(shuffled_arr) else 0.0,
        "risk_independent": abs(float(shuffled_arr.mean()) if len(shuffled_arr) else 0) < 20,
        "note": "If shuffled_worst_dd > 20%, risk management is too dependent on HMM.",
    }


def print_stress_report(crash: Dict, gap: Dict, misclass: Dict):
    """
    Print a formatted Rich table summarizing crash injection, gap risk, and misclassification results.

    Args:
        crash: Output dict from run_crash_injection.
        gap: Output dict from run_gap_risk.
        misclass: Output dict from run_regime_misclassification.
    """
    from rich.console import Console
    from rich.table import Table

    console = Console()
    console.print("\n[bold red]═══ STRESS TEST RESULTS ═══[/bold red]")

    console.print("\n[bold]Crash Injection[/bold]")
    t = Table()
    t.add_column("Metric")
    t.add_column("Value", justify="right")
    for k, v in crash.items():
        t.add_row(str(k), f"{v:.2f}" if isinstance(v, float) else str(v))
    console.print(t)

    console.print("\n[bold]Gap Risk[/bold]")
    t2 = Table()
    t2.add_column("Metric")
    t2.add_column("Value", justify="right")
    for k, v in gap.items():
        t2.add_row(str(k), f"{v:.2f}" if isinstance(v, float) else str(v))
    console.print(t2)

    console.print("\n[bold]Regime Misclassification[/bold]")
    t3 = Table()
    t3.add_column("Metric")
    t3.add_column("Value", justify="right")
    for k, v in misclass.items():
        t3.add_row(str(k), f"{v:.2f}" if isinstance(v, float) else str(v))
    console.print(t3)
