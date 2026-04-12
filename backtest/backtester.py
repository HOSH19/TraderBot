"""
Walk-forward allocation backtester.

This is an ALLOCATION-BASED backtester. It does NOT track individual trade
entries and exits. It sets a target portfolio allocation each bar based on the
detected volatility regime and rebalances when the allocation changes meaningfully.

IS window:  252 trading days (1 year) for HMM training
OOS window: 126 trading days (6 months) for evaluation
Step size:  126 trading days (6 months)
"""

import logging
from dataclasses import dataclass, field
from typing import List, Dict, Optional, Tuple

import numpy as np
import pandas as pd

from core.hmm_engine import HMMEngine
from core.regime_strategies import StrategyOrchestrator
from data.feature_engineering import get_feature_matrix

logger = logging.getLogger(__name__)


@dataclass
class Trade:
    """Records a single allocation rebalance event during backtesting."""

    bar_index: int
    timestamp: pd.Timestamp
    symbol: str
    prev_allocation: float
    new_allocation: float
    price: float
    regime: str
    regime_prob: float
    slippage_cost: float


@dataclass
class BacktestResult:
    """Aggregated output from a completed walk-forward backtest run."""

    equity_curve: pd.Series
    trade_log: List[Trade]
    regime_history: pd.DataFrame
    windows: List[Dict]
    config: Dict


class WalkForwardBacktester:
    """Runs an allocation-based walk-forward backtest using HMM regime detection."""

    def __init__(self, config: dict):
        """
        Initialize the backtester with the full application config.

        Args:
            config: Dict containing 'backtest', 'hmm', and 'strategy' sub-configs.
        """
        self.cfg = config
        self.bt_cfg = config.get("backtest", {})

    def run(
        self,
        symbol: str,
        bars: pd.DataFrame,
        verbose: bool = True,
    ) -> BacktestResult:
        """
        Run walk-forward backtest on a single symbol.
        bars: DataFrame with OHLCV columns, DatetimeIndex.
        """
        bars = bars.copy()
        bars.columns = [c.lower() for c in bars.columns]

        train_window = self.bt_cfg.get("train_window", 252)
        test_window = self.bt_cfg.get("test_window", 126)
        step_size = self.bt_cfg.get("step_size", 126)
        initial_capital = self.bt_cfg.get("initial_capital", 100_000)
        slippage_pct = self.bt_cfg.get("slippage_pct", 0.0005)
        fill_delay = self.bt_cfg.get("fill_delay_bars", 1)
        rebalance_threshold = self.cfg.get("strategy", {}).get("rebalance_threshold", 0.10)

        total_bars = len(bars)
        min_start = train_window
        if total_bars < min_start + test_window:
            raise ValueError(
                f"Need at least {min_start + test_window} bars, got {total_bars}."
            )

        equity = initial_capital
        cash = initial_capital
        shares = 0.0
        current_allocation = 0.0

        all_equity = pd.Series(index=bars.index, dtype=float)
        all_equity.iloc[:min_start] = equity
        trade_log: List[Trade] = []
        regime_rows = []
        windows = []

        start = min_start
        while start < total_bars:
            is_end = start
            oos_start = start
            oos_end = min(start + test_window, total_bars)

            is_bars = bars.iloc[is_end - train_window: is_end]
            oos_bars = bars.iloc[oos_start: oos_end]

            hmm = HMMEngine(self.cfg.get("hmm", {}))
            try:
                hmm.train(is_bars)
            except Exception as e:
                logger.warning(f"HMM train failed at window starting {bars.index[oos_start]}: {e}")
                start += step_size
                continue

            orchestrator = StrategyOrchestrator(
                self.cfg.get("strategy", {}), hmm.regime_infos
            )

            window_trades = 0
            for i in range(len(oos_bars)):
                global_idx = oos_start + i
                price_row = oos_bars.iloc[i]
                current_price = float(price_row["close"])

                equity = cash + shares * current_price
                all_equity.iloc[global_idx] = equity

                history_bars = bars.iloc[: oos_start + i + 1]
                try:
                    regime_state = hmm.predict_regime_filtered(history_bars)
                except Exception:
                    regime_rows.append({
                        "timestamp": bars.index[global_idx],
                        "regime": "UNKNOWN",
                        "probability": 0.0,
                    })
                    continue

                regime_rows.append({
                    "timestamp": bars.index[global_idx],
                    "regime": regime_state.label,
                    "probability": regime_state.probability,
                    "is_confirmed": regime_state.is_confirmed,
                })

                signals = orchestrator.generate_signals(
                    symbols=[symbol],
                    bars_by_symbol={symbol: bars.iloc[: oos_start + i + 1]},
                    regime_state=regime_state,
                    is_flickering=hmm.is_flickering(),
                    current_allocations={symbol: current_allocation},
                )

                if not signals:
                    continue

                sig = signals[0]
                target_allocation = sig.position_size_pct * sig.leverage

                if abs(target_allocation - current_allocation) < rebalance_threshold:
                    continue

                fill_idx = global_idx + fill_delay
                if fill_idx >= total_bars:
                    continue

                fill_price = float(bars.iloc[fill_idx]["open"])
                slippage = fill_price * slippage_pct
                fill_price += slippage

                target_shares = int(equity * target_allocation / fill_price)
                delta = target_shares - shares
                cost = delta * fill_price
                slippage_cost = abs(delta) * slippage

                cash -= cost
                shares = target_shares

                trade_log.append(Trade(
                    bar_index=fill_idx,
                    timestamp=bars.index[fill_idx],
                    symbol=symbol,
                    prev_allocation=current_allocation,
                    new_allocation=target_allocation,
                    price=fill_price,
                    regime=regime_state.label,
                    regime_prob=regime_state.probability,
                    slippage_cost=slippage_cost,
                ))
                current_allocation = target_allocation
                window_trades += 1

            if verbose:
                oos_eq = all_equity.iloc[oos_start: oos_end].dropna()
                if len(oos_eq) > 1:
                    oos_ret = (oos_eq.iloc[-1] / oos_eq.iloc[0] - 1) * 100
                    logger.info(
                        f"Window {bars.index[oos_start].date()} → {bars.index[oos_end-1].date()}: "
                        f"OOS return={oos_ret:.2f}%  trades={window_trades}"
                    )

            windows.append({
                "is_start": bars.index[is_end - train_window],
                "is_end": bars.index[is_end - 1],
                "oos_start": bars.index[oos_start],
                "oos_end": bars.index[oos_end - 1],
                "n_regimes": hmm.n_regimes,
                "bic": hmm.bic_score,
                "trades": window_trades,
            })

            start += step_size

        all_equity = all_equity.ffill()
        regime_df = pd.DataFrame(regime_rows).set_index("timestamp") if regime_rows else pd.DataFrame()

        return BacktestResult(
            equity_curve=all_equity,
            trade_log=trade_log,
            regime_history=regime_df,
            windows=windows,
            config=self.cfg,
        )
