"""Walk-forward engine: retrain HMM in-sample, simulate target weights out-of-sample.

Allocation-only: no per-fill inventory accounting beyond one symbol. Default windows are
252 train bars, 126 OOS bars, step 126. Rebalances when strategy targets diverge from the
deadband, with optional fill delay and slippage at the next open.
"""

import logging
from typing import List

import pandas as pd

from backtest.delayed_rebalance import delayed_rebalance_trade
from backtest.result import BacktestResult
from backtest.trade import Trade
from backtest.walk_sim_state import WalkSimState
from core.hmm.engine import HMMEngine
from core.strategies.orchestrator import StrategyOrchestrator

logger = logging.getLogger(__name__)


class WalkForwardBacktester:
    """Rolling IS train + OOS simulation for one ticker."""

    def __init__(self, config: dict) -> None:
        """Parse ``backtest`` settings from the full application config.

        Args:
            config: Must include ``backtest``, ``hmm``, and ``strategy`` sections.
        """
        self.cfg = config
        self.bt_cfg = config.get("backtest", {})

    def _simulate_oos_bars(
        self,
        symbol: str,
        bars: pd.DataFrame,
        oos_bars: pd.DataFrame,
        oos_start: int,
        hmm: HMMEngine,
        orchestrator: StrategyOrchestrator,
        *,
        rebalance_threshold: float,
        fill_delay: int,
        slippage_pct: float,
        total_bars: int,
        walk: WalkSimState,
        all_equity: pd.Series,
        trade_log: List[Trade],
        regime_rows: list,
    ) -> int:
        """Walk one OOS slice; mutate ``walk``, ``all_equity``, and logs in place.

        Returns:
            Number of rebalance trades executed in this window.
        """
        window_trades = 0
        for i in range(len(oos_bars)):
            global_idx = oos_start + i
            current_price = float(oos_bars.iloc[i]["close"])
            equity = walk.cash + walk.shares * current_price
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
                current_allocations={symbol: walk.current_allocation},
            )
            if not signals:
                continue

            target = signals[0].position_size_pct * signals[0].leverage
            if abs(target - walk.current_allocation) < rebalance_threshold:
                continue

            ncash, nshares, nalloc, trade = delayed_rebalance_trade(
                symbol=symbol,
                bars=bars,
                global_idx=global_idx,
                fill_delay=fill_delay,
                total_bars=total_bars,
                equity=equity,
                cash=walk.cash,
                shares=walk.shares,
                prev_allocation=walk.current_allocation,
                target_allocation=target,
                slippage_pct=slippage_pct,
                regime_state=regime_state,
            )
            if trade is None:
                continue
            walk.cash, walk.shares, walk.current_allocation = ncash, nshares, nalloc
            trade_log.append(trade)
            window_trades += 1
        return window_trades

    def run(
        self,
        symbol: str,
        bars: pd.DataFrame,
    ) -> BacktestResult:
        """Execute rolling windows on ``bars`` for ``symbol``.

        Args:
            symbol: Ticker to simulate.
            bars: OHLCV DataFrame with a DatetimeIndex; columns are lowercased internally.

        Returns:
            :class:`BacktestResult` with curves, trades, regime history, and window stats.

        Raises:
            ValueError: If ``len(bars)`` is shorter than train + test requirements.
        """
        bars = bars.copy()
        bars.columns = [c.lower() for c in bars.columns]

        # Fetch macro features once for the full date range; sliced per window below.
        macro_df = None
        if self.cfg.get("hmm", {}).get("use_macro_features", True):
            from data.macro_fetcher import fetch_macro_df
            macro_df = fetch_macro_df(
                bars.index[0].to_pydatetime(), bars.index[-1].to_pydatetime()
            )

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

        walk = WalkSimState(
            cash=initial_capital,
            shares=0.0,
            current_allocation=0.0,
        )

        all_equity = pd.Series(index=bars.index, dtype=float)
        all_equity.iloc[:min_start] = initial_capital
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
            if macro_df is not None:
                hmm.set_macro_df(macro_df)
            try:
                hmm.train(is_bars)
            except Exception as e:
                logger.warning(f"HMM train failed at window starting {bars.index[oos_start]}: {e}")
                start += step_size
                continue

            orchestrator = StrategyOrchestrator(
                self.cfg.get("strategy", {}), hmm.regime_infos
            )

            window_trades = self._simulate_oos_bars(
                symbol, bars, oos_bars, oos_start, hmm, orchestrator,
                rebalance_threshold=rebalance_threshold,
                fill_delay=fill_delay,
                slippage_pct=slippage_pct,
                total_bars=total_bars,
                walk=walk,
                all_equity=all_equity,
                trade_log=trade_log,
                regime_rows=regime_rows,
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
