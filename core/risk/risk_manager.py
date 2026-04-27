"""Portfolio-level validation and resizing of trade signals before execution."""

import logging
import time
from typing import Dict, List, Optional, Tuple  # noqa: F401 — Tuple used in return annotations

from core.risk.circuit_breaker import CircuitBreaker
from core.risk.portfolio_state import PortfolioState
from core.risk.risk_decision import RiskDecision
from core.strategies.signal import Signal

logger = logging.getLogger(__name__)


class RiskManager:
    """Run circuit breakers, sizing, leverage, and exposure checks on each signal."""

    def __init__(self, config: dict) -> None:
        """Create a manager and embed a :class:`CircuitBreaker` for the ``risk`` config block.

        Args:
            config: Full application settings; uses the ``risk`` sub-dict.
        """
        self.cfg = config
        self.risk_cfg = config.get("risk", {})
        self.circuit_breaker = CircuitBreaker(self.risk_cfg)
        self._daily_trade_count: int = 0
        self._last_trade_times: Dict[str, float] = {}

    def validate_signal(
        self,
        signal: Signal,
        portfolio: PortfolioState,
    ) -> RiskDecision:
        """Apply sizing, guards, and breakers; approve, reject, or return a modified signal.

        Args:
            signal: Proposed trade from the strategy layer.
            portfolio: Live or paper portfolio snapshot.

        Returns:
            ``RiskDecision`` with approval flag, optional modified signal, and reasons.
        """
        modifications: List[str] = []
        cb_action, cb_reason = self.circuit_breaker.update(portfolio)

        decision = self._reject_circuit_hard_stop(cb_action, cb_reason)
        if decision:
            return decision

        signal, modifications = self._apply_circuit_size_reduction(cb_action, signal, modifications)

        for reject in (
            self._reject_bad_stop(signal),
            self._reject_daily_trade_cap(),
            self._reject_duplicate_symbol(signal),
            self._reject_max_positions(portfolio),
        ):
            if reject:
                return reject

        signal, modifications = self._apply_position_and_leverage(signal, portfolio, modifications)

        exposure_ok, exp_reason = self._check_exposure(signal, portfolio)
        if not exposure_ok:
            return RiskDecision(approved=False, modified_signal=None, rejection_reason=exp_reason)

        return self._finalize_approval(signal, portfolio, modifications)

    @staticmethod
    def _reject_circuit_hard_stop(cb_action: str, cb_reason: str) -> Optional[RiskDecision]:
        """Return a rejection if the circuit breaker is in a hard-stop state, else None."""
        if cb_action not in ("HALTED", "CLOSE_ALL_DAY", "CLOSE_ALL_WEEK"):
            return None
        return RiskDecision(
            approved=False,
            modified_signal=None,
            rejection_reason=f"Circuit breaker: {cb_action} — {cb_reason}",
        )

    @staticmethod
    def _apply_circuit_size_reduction(
        cb_action: str, signal: Signal, modifications: List[str]
    ) -> Tuple[Signal, List[str]]:
        """Halve position size when the circuit breaker signals a soft reduction."""
        if cb_action not in ("REDUCE_50_DAY", "REDUCE_50_WEEK"):
            return signal, modifications
        signal = Signal(**{**signal.__dict__, "position_size_pct": signal.position_size_pct * 0.5})
        return signal, [*modifications, f"Size halved due to {cb_action}"]

    @staticmethod
    def _reject_bad_stop(signal: Signal) -> Optional[RiskDecision]:
        """Reject signals that have no positive stop-loss set."""
        if signal.stop_loss and signal.stop_loss > 0:
            return None
        return RiskDecision(
            approved=False,
            modified_signal=None,
            rejection_reason="Signal rejected: missing stop_loss",
        )

    def _reject_daily_trade_cap(self) -> Optional[RiskDecision]:
        """Reject if the session trade count has hit ``max_daily_trades``."""
        cap = self.risk_cfg.get("max_daily_trades", 20)
        if self._daily_trade_count < cap:
            return None
        return RiskDecision(
            approved=False,
            modified_signal=None,
            rejection_reason=f"Daily trade limit reached ({self._daily_trade_count})",
        )

    def _reject_duplicate_symbol(self, signal: Signal) -> Optional[RiskDecision]:
        """Reject if the same symbol was traded within ``duplicate_block_seconds``."""
        dup_block = self.risk_cfg.get("duplicate_block_seconds", 60)
        elapsed = time.time() - self._last_trade_times.get(signal.symbol, 0)
        if elapsed >= dup_block:
            return None
        return RiskDecision(
            approved=False,
            modified_signal=None,
            rejection_reason=f"Duplicate trade blocked: {signal.symbol} traded within {dup_block}s",
        )

    def _reject_max_positions(self, portfolio: PortfolioState) -> Optional[RiskDecision]:
        """Reject if the portfolio already holds ``max_concurrent`` positions."""
        cap = self.risk_cfg.get("max_concurrent", 5)
        if portfolio.n_positions < cap:
            return None
        return RiskDecision(
            approved=False,
            modified_signal=None,
            rejection_reason=f"Max concurrent positions ({cap}) reached",
        )

    def _apply_position_and_leverage(
        self, signal: Signal, portfolio: PortfolioState, modifications: List[str]
    ) -> Tuple[Signal, List[str]]:
        """Apply position sizing then cap leverage to 1.0 if leverage check fails."""
        signal, size_mods = self._apply_position_sizing(signal, portfolio)
        modifications = [*modifications, *size_mods]
        leverage_ok, lev_reason = self._check_leverage(signal, portfolio)
        if leverage_ok:
            return signal, modifications
        signal = Signal(**{**signal.__dict__, "leverage": 1.0})
        return signal, [*modifications, lev_reason]

    def _finalize_approval(
        self, signal: Signal, portfolio: PortfolioState, modifications: List[str]
    ) -> RiskDecision:
        """Record the trade, update peak equity, and return an approved RiskDecision."""
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
        """Cap shares using per-trade risk budget, gap multiplier, and max single name.

        Returns:
            Adjusted ``Signal`` and human-readable modification notes.
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
        """Check requested leverage against caps and portfolio stress signals.

        Returns:
            ``(True, "")`` if leverage is allowed; otherwise ``(False, reason)``.
        """
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
        """Ensure post-trade gross exposure stays under ``max_exposure``.

        Returns:
            ``(True, "")`` if room remains; otherwise ``(False, reason)``.
        """
        max_exp = self.risk_cfg.get("max_exposure", 0.80)
        current_exp = portfolio.total_exposure
        new_exp = current_exp + signal.position_size_pct * signal.leverage

        if new_exp > max_exp:
            return False, f"Adding this position would bring exposure to {new_exp*100:.1f}% (max {max_exp*100:.0f}%)"

        return True, ""

    def reset_daily_counters(self) -> None:
        """Clear the per-session trade counter (e.g. at session open)."""
        self._daily_trade_count = 0

    def reset_weekly_counters(self) -> None:
        """Placeholder for weekly risk state resets (currently no-op)."""
        pass
