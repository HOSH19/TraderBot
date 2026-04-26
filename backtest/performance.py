"""
Performance metrics and benchmark comparisons for backtest results.
"""

import logging
from typing import Dict, List, Optional

import numpy as np
import pandas as pd

from backtest.backtester import BacktestResult, Trade

logger = logging.getLogger(__name__)


def _annualized_return(equity: pd.Series) -> float:
    """Return compound annual growth rate assuming 252 trading days per year."""
    if len(equity) < 2:
        return 0.0
    years = len(equity) / 252
    return (equity.iloc[-1] / equity.iloc[0]) ** (1 / years) - 1 if years > 0 else 0.0


def _max_drawdown(equity: pd.Series) -> tuple:
    """Returns (max_dd_pct, max_dd_duration_days)."""
    peak = equity.expanding().max()
    dd = (equity - peak) / peak
    max_dd = float(dd.min())

    underwater = dd < 0
    duration = 0
    current = 0
    for u in underwater:
        if u:
            current += 1
            duration = max(duration, current)
        else:
            current = 0

    return max_dd, duration


def _sharpe(returns: pd.Series, risk_free_rate: float = 0.045) -> float:
    """Return annualized Sharpe ratio using excess returns over the risk-free rate."""
    daily_rf = risk_free_rate / 252
    excess = returns - daily_rf
    if excess.std() == 0:
        return 0.0
    return float(excess.mean() / excess.std() * np.sqrt(252))


def _sortino(returns: pd.Series, risk_free_rate: float = 0.045) -> float:
    """Return annualized Sortino ratio, penalizing only downside volatility."""
    daily_rf = risk_free_rate / 252
    excess = returns - daily_rf
    downside = excess[excess < 0].std()
    if downside == 0:
        return 0.0
    return float(excess.mean() / downside * np.sqrt(252))


def _trade_stats_from_log(result: BacktestResult) -> Dict[str, float]:
    """Win rate, averages, and profit factor between successive rebalance marks on the equity curve."""
    trades = result.trade_log
    n_trades = len(trades)
    if n_trades <= 1:
        return {
            "win_rate": 0.0,
            "avg_win_pct": 0.0,
            "avg_loss_pct": 0.0,
            "profit_factor": 0.0,
        }

    trade_returns = []
    for i in range(1, len(trades)):
        a = trades[i - 1]
        b = trades[i]
        idx_a = result.equity_curve.index.get_loc(a.timestamp) if a.timestamp in result.equity_curve.index else None
        idx_b = result.equity_curve.index.get_loc(b.timestamp) if b.timestamp in result.equity_curve.index else None
        if idx_a is not None and idx_b is not None:
            eq_a = result.equity_curve.iloc[idx_a]
            eq_b = result.equity_curve.iloc[idx_b]
            if eq_a > 0:
                trade_returns.append(eq_b / eq_a - 1)

    trade_returns = np.array(trade_returns)
    wins = trade_returns[trade_returns > 0]
    losses = trade_returns[trade_returns < 0]
    win_rate = len(wins) / len(trade_returns) if len(trade_returns) > 0 else 0.0
    avg_win = float(wins.mean()) if len(wins) > 0 else 0.0
    avg_loss = float(losses.mean()) if len(losses) > 0 else 0.0
    profit_factor = abs(wins.sum() / losses.sum()) if losses.sum() != 0 else float("inf")

    return {
        "win_rate": win_rate,
        "avg_win_pct": avg_win * 100,
        "avg_loss_pct": avg_loss * 100,
        "profit_factor": profit_factor,
    }


def compute_metrics(
    result: BacktestResult,
    risk_free_rate: float = 0.045,
) -> Dict:
    """
    Compute a full suite of performance metrics from a backtest result.

    Returns a dict with keys including total_return_pct, cagr_pct, sharpe,
    sortino, calmar, max_drawdown_pct, win_rate, profit_factor, and worst-period stats.
    """
    equity = result.equity_curve.dropna()
    returns = equity.pct_change().dropna()

    total_return = float(equity.iloc[-1] / equity.iloc[0] - 1)
    cagr = _annualized_return(equity)
    sharpe = _sharpe(returns, risk_free_rate)
    sortino = _sortino(returns, risk_free_rate)
    max_dd, max_dd_dur = _max_drawdown(equity)
    calmar = cagr / abs(max_dd) if max_dd != 0 else 0.0

    ts = _trade_stats_from_log(result)
    n_trades = len(result.trade_log)

    worst_day = float(returns.min())
    worst_week = float(returns.rolling(5).sum().min()) if len(returns) >= 5 else worst_day
    worst_month = float(returns.rolling(21).sum().min()) if len(returns) >= 21 else worst_day

    return {
        "total_return_pct": total_return * 100,
        "cagr_pct": cagr * 100,
        "sharpe": sharpe,
        "sortino": sortino,
        "calmar": calmar,
        "max_drawdown_pct": max_dd * 100,
        "max_drawdown_duration_days": max_dd_dur,
        "win_rate": ts["win_rate"],
        "avg_win_pct": ts["avg_win_pct"],
        "avg_loss_pct": ts["avg_loss_pct"],
        "profit_factor": ts["profit_factor"],
        "total_trades": n_trades,
        "worst_day_pct": worst_day * 100,
        "worst_week_pct": worst_week * 100,
        "worst_month_pct": worst_month * 100,
    }


def regime_breakdown(result: BacktestResult) -> pd.DataFrame:
    """
    Summarize time spent and trade count per regime label.

    Returns a DataFrame with columns Regime, % Time In, Trades, and Avg Prob,
    sorted by time descending. Returns empty DataFrame if no regime history.
    """
    if result.regime_history.empty or not result.trade_log:
        return pd.DataFrame()

    rows = []
    for regime in result.regime_history["regime"].unique():
        mask = result.regime_history["regime"] == regime
        pct_time = mask.sum() / len(result.regime_history)
        regime_trades = [t for t in result.trade_log if t.regime == regime]
        rows.append({
            "Regime": regime,
            "% Time In": f"{pct_time*100:.1f}%",
            "Trades": len(regime_trades),
            "Avg Prob": f"{result.regime_history.loc[mask, 'probability'].mean():.2f}",
        })

    return pd.DataFrame(rows).sort_values("% Time In", ascending=False)


def confidence_breakdown(result: BacktestResult) -> pd.DataFrame:
    """
    Bucket trades by regime probability confidence tier (<50%, 50-60%, 60-70%, 70%+).

    Returns a DataFrame with columns Confidence and Trades.
    """
    buckets = [(0.0, 0.5, "<50%"), (0.5, 0.6, "50-60%"), (0.6, 0.7, "60-70%"), (0.7, 1.01, "70%+")]
    rows = []
    for lo, hi, label in buckets:
        bucket_trades = [
            t for t in result.trade_log if lo <= t.regime_prob < hi
        ]
        rows.append({"Confidence": label, "Trades": len(bucket_trades)})
    return pd.DataFrame(rows)


def buy_and_hold_benchmark(
    bars: pd.DataFrame, initial_capital: float = 100_000
) -> pd.Series:
    """
    Compute equity curve for a simple buy-and-hold strategy.

    Buys at the first close price and holds to the end. Returns a Series of equity values.
    """
    close = bars["close"] if "close" in bars.columns else bars["Close"]
    shares = initial_capital / float(close.iloc[0])
    return close * shares


def sma200_benchmark(
    bars: pd.DataFrame, initial_capital: float = 100_000
) -> pd.Series:
    """
    Compute equity curve for a 200-day SMA trend-following benchmark.

    Enters long when price crosses above SMA-200, exits when price falls below.
    Returns a forward-filled Series of equity values.
    """
    close = bars["close"] if "close" in bars.columns else bars["Close"]
    sma200 = close.rolling(200).mean()
    equity = pd.Series(index=close.index, dtype=float)
    cash = initial_capital
    shares = 0.0

    for i in range(len(close)):
        price = float(close.iloc[i])
        sma = float(sma200.iloc[i]) if not np.isnan(sma200.iloc[i]) else np.nan

        if np.isnan(sma):
            equity.iloc[i] = cash + shares * price
            continue

        if price > sma and shares == 0:
            shares = cash / price
            cash = 0.0
        elif price <= sma and shares > 0:
            cash = shares * price
            shares = 0.0

        equity.iloc[i] = cash + shares * price

    return equity.ffill()


def random_allocation_benchmark(
    bars: pd.DataFrame,
    n_simulations: int = 100,
    initial_capital: float = 100_000,
    rebalance_threshold: float = 0.10,
    seed: int = 42,
) -> Dict:
    """
    Simulate random allocation changes to produce a Monte Carlo baseline.

    At each bar, randomly decides to rebalance to 60% or 95% allocation.
    Returns a dict with mean, std, min, and max final return across all simulations.
    """
    close = bars["close"] if "close" in bars.columns else bars["Close"]
    rng = np.random.default_rng(seed)
    final_returns = []

    allocations = [0.60, 0.95]

    for s in range(n_simulations):
        cash = initial_capital
        shares = 0.0
        current_alloc = 0.0

        for i in range(len(close)):
            price = float(close.iloc[i])
            equity = cash + shares * price

            if rng.random() < rebalance_threshold:
                target_alloc = float(rng.choice(allocations))
                if abs(target_alloc - current_alloc) >= rebalance_threshold:
                    target_shares = int(equity * target_alloc / price)
                    delta = target_shares - shares
                    cash -= delta * price
                    shares = target_shares
                    current_alloc = target_alloc

        final_equity = cash + shares * float(close.iloc[-1])
        final_returns.append(final_equity / initial_capital - 1)

    arr = np.array(final_returns)
    return {"mean": float(arr.mean()), "std": float(arr.std()), "min": float(arr.min()), "max": float(arr.max())}


def print_report(result: BacktestResult, bars: pd.DataFrame, risk_free_rate: float = 0.045):
    """
    Print a formatted Rich report of backtest metrics, regime breakdown, and benchmarks.

    Args:
        result: Completed BacktestResult from WalkForwardBacktester.run.
        bars: Original OHLCV DataFrame used for benchmark calculations.
        risk_free_rate: Annual risk-free rate used in ratio calculations.
    """
    from rich.console import Console
    from rich.table import Table

    console = Console()
    metrics = compute_metrics(result, risk_free_rate)

    console.print("\n[bold cyan]═══ BACKTEST RESULTS ═══[/bold cyan]")
    t = Table(show_header=True, header_style="bold magenta")
    t.add_column("Metric", style="cyan")
    t.add_column("Value", justify="right")
    for k, v in metrics.items():
        if isinstance(v, float):
            t.add_row(k, f"{v:.2f}")
        else:
            t.add_row(k, str(v))
    console.print(t)

    console.print("\n[bold cyan]═══ REGIME BREAKDOWN ═══[/bold cyan]")
    rb = regime_breakdown(result)
    if not rb.empty:
        console.print(rb.to_string(index=False))

    initial_capital = result.config.get("backtest", {}).get("initial_capital", 100_000)
    bah = buy_and_hold_benchmark(bars, initial_capital)
    bah_ret = float(bah.iloc[-1] / bah.iloc[0] - 1) * 100
    sma = sma200_benchmark(bars, initial_capital)
    sma_ret = float(sma.iloc[-1] / sma.iloc[0] - 1) * 100
    rand = random_allocation_benchmark(bars, initial_capital=initial_capital)

    console.print("\n[bold cyan]═══ BENCHMARK COMPARISON ═══[/bold cyan]")
    bt = Table(show_header=True, header_style="bold magenta")
    bt.add_column("Benchmark", style="cyan")
    bt.add_column("Total Return %", justify="right")
    bt.add_row("Buy & Hold", f"{bah_ret:.2f}%")
    bt.add_row("200 SMA Trend", f"{sma_ret:.2f}%")
    bt.add_row(f"Random Alloc (mean±std)", f"{rand['mean']*100:.2f}% ± {rand['std']*100:.2f}%")
    bt.add_row("This Strategy", f"{metrics['total_return_pct']:.2f}%")
    console.print(bt)
