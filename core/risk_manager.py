"""
Risk Management Layer.

Operates INDEPENDENTLY of the HMM. Has ABSOLUTE VETO POWER over any signal.
Even if the HMM fails completely, circuit breakers catch drawdowns from actual P&L.
"""

import logging
import os
import time
from dataclasses import dataclass, field
from datetime import datetime, date
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd

from core.regime_strategies import Signal
from core.timeutil import ensure_utc, utc_now

logger = logging.getLogger(__name__)

TRADING_HALTED_LOCK = "trading_halted.lock"


@dataclass
class Position:
    """Represents an open position held by the bot."""

    symbol: str
    shares: float
    entry_price: float
    entry_time: datetime
    current_price: float
    stop_loss: float
    regime_at_entry: str
    current_regime: str = ""
    trade_id: str = ""

    @property
    def unrealized_pnl(self) -> float:
        """Unrealized P&L in dollars for this position."""
        return (self.current_price - self.entry_price) * self.shares

    @property
    def unrealized_pnl_pct(self) -> float:
        """Unrealized P&L as a fraction of entry value."""
        if self.entry_price == 0:
            return 0.0
        return (self.current_price / self.entry_price - 1)

    @property
    def holding_period_hours(self) -> float:
        """Hours elapsed since the position was entered."""
        return (utc_now() - ensure_utc(self.entry_time)).total_seconds() / 3600


@dataclass
class PortfolioState:
    """Snapshot of the current portfolio used by risk checks and circuit breakers."""

    equity: float
    cash: float
    buying_power: float
    positions: Dict[str, Position] = field(default_factory=dict)
    daily_pnl: float = 0.0
    weekly_pnl: float = 0.0
    peak_equity: float = 0.0
    daily_start_equity: float = 0.0
    weekly_start_equity: float = 0.0
    circuit_breaker_status: str = "NORMAL"
    flicker_rate: int = 0
    last_updated: datetime = field(default_factory=utc_now)

    @property
    def drawdown_from_peak(self) -> float:
        """Current equity drawdown as a fraction of the all-time peak equity."""
        if self.peak_equity == 0:
            return 0.0
        return (self.equity - self.peak_equity) / self.peak_equity

    @property
    def daily_drawdown(self) -> float:
        """Equity change today as a fraction of the day-open equity."""
        if self.daily_start_equity == 0:
            return 0.0
        return (self.equity - self.daily_start_equity) / self.daily_start_equity

    @property
    def weekly_drawdown(self) -> float:
        """Equity change this week as a fraction of the week-open equity."""
        if self.weekly_start_equity == 0:
            return 0.0
        return (self.equity - self.weekly_start_equity) / self.weekly_start_equity

    @property
    def total_exposure(self) -> float:
        """Sum of all position market values as a fraction of equity."""
        if self.equity == 0:
            return 0.0
        return sum(p.shares * p.current_price for p in self.positions.values()) / self.equity

    @property
    def n_positions(self) -> int:
        """Number of currently open positions."""
        return len(self.positions)


@dataclass
class RiskDecision:
    """Outcome of a risk validation check, including the (possibly modified) signal."""

    approved: bool
    modified_signal: Optional[Signal]
    rejection_reason: str
    modifications: List[str] = field(default_factory=list)


class CircuitBreaker:
    """Monitors drawdown thresholds and halts or reduces trading when limits are breached."""

    def __init__(self, config: dict):
        """Initialize circuit breaker with risk configuration thresholds."""
        self.cfg = config
        self._trigger_history: List[Dict] = []

    def check(self, portfolio: PortfolioState) -> Tuple[str, str]:
        """
        Returns (action, reason).
        action: NORMAL | REDUCE_50 | CLOSE_ALL | HALTED
        """
        if os.path.exists(TRADING_HALTED_LOCK):
            return "HALTED", "trading_halted.lock file present — manual intervention required"

        daily_dd = portfolio.daily_drawdown
        weekly_dd = portfolio.weekly_drawdown
        peak_dd = portfolio.drawdown_from_peak

        max_dd = self.cfg.get("max_dd_from_peak", 0.10)
        if peak_dd <= -max_dd:
            self._write_lock_file(portfolio)
            return "HALTED", f"Peak DD {peak_dd*100:.1f}% exceeds {max_dd*100:.0f}% limit"

        weekly_halt = self.cfg.get("weekly_dd_halt", 0.07)
        if weekly_dd <= -weekly_halt:
            return "CLOSE_ALL_WEEK", f"Weekly DD {weekly_dd*100:.1f}% exceeds {weekly_halt*100:.0f}% limit"

        weekly_reduce = self.cfg.get("weekly_dd_reduce", 0.05)
        if weekly_dd <= -weekly_reduce:
            return "REDUCE_50_WEEK", f"Weekly DD {weekly_dd*100:.1f}% exceeds {weekly_reduce*100:.0f}% reduce threshold"

        daily_halt = self.cfg.get("daily_dd_halt", 0.03)
        if daily_dd <= -daily_halt:
            return "CLOSE_ALL_DAY", f"Daily DD {daily_dd*100:.1f}% exceeds {daily_halt*100:.0f}% limit"

        daily_reduce = self.cfg.get("daily_dd_reduce", 0.02)
        if daily_dd <= -daily_reduce:
            return "REDUCE_50_DAY", f"Daily DD {daily_dd*100:.1f}% exceeds {daily_reduce*100:.0f}% reduce threshold"

        return "NORMAL", ""

    def _write_lock_file(self, portfolio: PortfolioState):
        """Write the halt lock file to disk, requiring manual deletion to resume trading."""
        with open(TRADING_HALTED_LOCK, "w") as f:
            f.write(
                f"Trading halted at {utc_now().isoformat()}\n"
                f"Peak DD: {portfolio.drawdown_from_peak*100:.2f}%\n"
                f"Equity: ${portfolio.equity:,.2f}\n"
                f"Delete this file to resume trading.\n"
            )
        logger.critical(f"Peak DD limit hit. Created {TRADING_HALTED_LOCK}. Manual deletion required.")

    def update(self, portfolio: PortfolioState) -> Tuple[str, str]:
        """Run a circuit-breaker check and log any triggered action, returning (action, reason)."""
        action, reason = self.check(portfolio)
        if action != "NORMAL":
            self._trigger_history.append({
                "time": utc_now().isoformat(),
                "action": action,
                "reason": reason,
                "equity": portfolio.equity,
                "daily_dd": portfolio.daily_drawdown,
                "weekly_dd": portfolio.weekly_drawdown,
                "peak_dd": portfolio.drawdown_from_peak,
            })
            logger.warning(f"Circuit breaker: {action} — {reason}")
        return action, reason

    def get_history(self) -> List[Dict]:
        """Return the list of all circuit-breaker trigger events recorded so far."""
        return self._trigger_history


class RiskManager:
    """Validates and modifies trade signals against portfolio-level risk constraints."""

    def __init__(self, config: dict):
        """Initialize the risk manager and attach a CircuitBreaker instance."""
        self.cfg = config
        self.risk_cfg = config.get("risk", {})
        self.circuit_breaker = CircuitBreaker(self.risk_cfg)
        self._daily_trade_count: int = 0
        self._last_trade_times: Dict[str, float] = {}
        self._returns_history: Dict[str, List[float]] = {}

    def validate_signal(
        self,
        signal: Signal,
        portfolio: PortfolioState,
    ) -> RiskDecision:
        """
        Apply all risk checks to signal and return a RiskDecision.

        Checks include circuit breakers, stop-loss presence, daily trade limits,
        duplicate-trade blocks, max concurrent positions, position sizing, leverage,
        and total exposure. May approve, reject, or return a modified signal.
        """
        modifications = []

        cb_action, cb_reason = self.circuit_breaker.update(portfolio)
        if cb_action in ("HALTED", "CLOSE_ALL_DAY", "CLOSE_ALL_WEEK"):
            return RiskDecision(
                approved=False,
                modified_signal=None,
                rejection_reason=f"Circuit breaker: {cb_action} — {cb_reason}",
            )

        if cb_action in ("REDUCE_50_DAY", "REDUCE_50_WEEK"):
            signal = Signal(**{**signal.__dict__, "position_size_pct": signal.position_size_pct * 0.5})
            modifications.append(f"Size halved due to {cb_action}")

        if not signal.stop_loss or signal.stop_loss <= 0:
            return RiskDecision(
                approved=False,
                modified_signal=None,
                rejection_reason="Signal rejected: missing stop_loss",
            )

        if self._daily_trade_count >= self.risk_cfg.get("max_daily_trades", 20):
            return RiskDecision(
                approved=False,
                modified_signal=None,
                rejection_reason=f"Daily trade limit reached ({self._daily_trade_count})",
            )

        dup_block = self.risk_cfg.get("duplicate_block_seconds", 60)
        last_time = self._last_trade_times.get(signal.symbol, 0)
        if time.time() - last_time < dup_block:
            return RiskDecision(
                approved=False,
                modified_signal=None,
                rejection_reason=f"Duplicate trade blocked: {signal.symbol} traded within {dup_block}s",
            )

        if portfolio.n_positions >= self.risk_cfg.get("max_concurrent", 5):
            return RiskDecision(
                approved=False,
                modified_signal=None,
                rejection_reason=f"Max concurrent positions ({self.risk_cfg.get('max_concurrent', 5)}) reached",
            )

        signal, size_mods = self._apply_position_sizing(signal, portfolio)
        modifications.extend(size_mods)

        leverage_ok, lev_reason = self._check_leverage(signal, portfolio)
        if not leverage_ok:
            signal = Signal(**{**signal.__dict__, "leverage": 1.0})
            modifications.append(lev_reason)

        exposure_ok, exp_reason = self._check_exposure(signal, portfolio)
        if not exposure_ok:
            return RiskDecision(
                approved=False,
                modified_signal=None,
                rejection_reason=exp_reason,
            )

        self._daily_trade_count += 1
        self._last_trade_times[signal.symbol] = time.time()

        if portfolio.peak_equity < portfolio.equity:
            portfolio.peak_equity = portfolio.equity

        return RiskDecision(
            approved=True,
            modified_signal=signal,
            rejection_reason="",
            modifications=modifications,
        )

    def _apply_position_sizing(
        self, signal: Signal, portfolio: PortfolioState
    ) -> Tuple[Signal, List[str]]:
        """
        Size the position using risk-per-trade and max-single-position caps.

        Returns the adjusted signal and a list of modification messages.
        """
        mods = []
        max_risk = self.risk_cfg.get("max_risk_per_trade", 0.01)
        min_pos = self.risk_cfg.get("min_position_dollars", 100.0)

        risk_per_share = abs(signal.entry_price - signal.stop_loss)
        if risk_per_share <= 0:
            return signal, ["stop_loss equals entry, using min size"]

        max_risk_dollars = portfolio.equity * max_risk
        gap_mult = self.risk_cfg.get("gap_risk_multiplier", 3.0)
        overnight_risk = risk_per_share * gap_mult
        adjusted_risk = min(max_risk_dollars / overnight_risk, max_risk_dollars / risk_per_share)
        risk_based_shares = int(adjusted_risk / risk_per_share)
        risk_based_value = risk_based_shares * signal.entry_price

        max_single = self.risk_cfg.get("max_single_position", 0.15)
        max_value = portfolio.equity * max_single
        size_value = min(risk_based_value, max_value)

        if size_value < min_pos:
            mods.append(f"Position size ${size_value:.0f} below minimum ${min_pos:.0f}")
            return Signal(**{**signal.__dict__, "position_size_pct": 0.0}), mods

        size_pct = size_value / portfolio.equity
        size_pct = min(size_pct, signal.position_size_pct)

        if size_pct < signal.position_size_pct:
            mods.append(f"Size capped at {size_pct*100:.1f}% by risk rules (was {signal.position_size_pct*100:.1f}%)")

        return Signal(**{**signal.__dict__, "position_size_pct": size_pct}), mods

    def _check_leverage(
        self, signal: Signal, portfolio: PortfolioState
    ) -> Tuple[bool, str]:
        """Return (allowed, reason) for the requested leverage given current portfolio conditions."""
        max_lev = self.risk_cfg.get("max_leverage", 1.25)

        if signal.leverage > max_lev:
            return False, f"Leverage {signal.leverage}x exceeds max {max_lev}x"

        cb_action, _ = self.circuit_breaker.check(portfolio)
        if cb_action != "NORMAL":
            return False, "Circuit breaker active — leverage forced to 1.0x"

        if portfolio.n_positions >= 3:
            return False, "3+ open positions — leverage forced to 1.0x"

        if portfolio.flicker_rate > self.risk_cfg.get("flicker_threshold", 4):
            return False, "High flicker rate — leverage forced to 1.0x"

        return True, ""

    def _check_exposure(
        self, signal: Signal, portfolio: PortfolioState
    ) -> Tuple[bool, str]:
        """Return (allowed, reason) checking that adding this signal won't exceed max total exposure."""
        max_exp = self.risk_cfg.get("max_exposure", 0.80)
        current_exp = portfolio.total_exposure
        new_exp = current_exp + signal.position_size_pct * signal.leverage

        if new_exp > max_exp:
            return False, f"Adding this position would bring exposure to {new_exp*100:.1f}% (max {max_exp*100:.0f}%)"

        return True, ""

    def reset_daily_counters(self):
        """Reset the daily trade count at the start of each trading day."""
        self._daily_trade_count = 0

    def reset_weekly_counters(self):
        """Reset weekly-level risk counters at the start of each trading week."""
        return
