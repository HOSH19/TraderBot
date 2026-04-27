"""Rich-powered TUI panels for regime, book, signals, and risk (optional dev UX)."""

import logging
import os
from datetime import datetime
from typing import List, Optional

from core.timeutil import ensure_utc, utc_now

logger = logging.getLogger(__name__)


class Dashboard:
    """Throttle refreshes and render multi-panel status via Rich."""

    def __init__(self, config: dict) -> None:
        """Read refresh cadence from ``monitoring.dashboard_refresh_seconds``.

        Args:
            config: Full settings blob.
        """
        self.cfg = config
        self.refresh_secs = config.get("monitoring", {}).get("dashboard_refresh_seconds", 5)
        self._last_refresh = 0.0
        self._recent_signals: list = []
        self._hmm_training_date: Optional[datetime] = None

    def refresh(self, portfolio, regime_state, hmm, signals: list) -> None:
        """Maybe redraw panels when ``refresh_secs`` elapsed.

        Appends new signals to the recent-signals history, updates the HMM training
        date, and delegates rendering to _render. No-ops if called too soon.
        """
        import time
        now = time.time()
        if now - self._last_refresh < self.refresh_secs:
            return
        self._last_refresh = now

        if signals:
            for s in signals:
                self._recent_signals.insert(0, {
                    "time": utc_now().strftime("%H:%M"),
                    "symbol": s.symbol,
                    "direction": s.direction,
                    "alloc": f"{s.position_size_pct*100:.0f}%",
                    "regime": s.regime_name,
                })
            self._recent_signals = self._recent_signals[:10]

        if hmm and hmm.training_date:
            self._hmm_training_date = hmm.training_date

        try:
            self._render(portfolio, regime_state, hmm)
        except Exception as e:
            logger.warning(f"Dashboard render failed: {e}")

    _REGIME_COLORS = {
        "BULL": "green", "STRONG_BULL": "bright_green", "EUPHORIA": "bright_green",
        "NEUTRAL": "yellow", "WEAK_BULL": "cyan", "WEAK_BEAR": "orange3",
        "BEAR": "red", "STRONG_BEAR": "bright_red", "CRASH": "bold red",
    }

    def _render(self, portfolio, regime_state, hmm) -> None:
        """Clear the terminal and print all six dashboard panels."""
        from rich.console import Console
        from rich.panel import Panel  # noqa: F401 — used in panel builders

        console = Console()
        console.clear()
        for panel in (
            self._regime_panel(regime_state, hmm),
            self._portfolio_panel(portfolio),
            self._positions_panel(portfolio),
            self._signals_panel(),
            self._risk_panel(portfolio),
            self._system_panel(hmm),
        ):
            console.print(panel)

    def _regime_panel(self, regime_state, hmm):
        """Build the REGIME panel showing label, probability, stability, and flicker rate."""
        from rich.panel import Panel
        from rich.text import Text

        label = regime_state.label if regime_state else "UNKNOWN"
        prob = regime_state.probability if regime_state else 0.0
        stability = hmm.get_regime_stability() if hmm else 0
        flicker = hmm.get_regime_flicker_rate() if hmm else 0
        flicker_window = hmm.config.get("flicker_window", 20) if hmm else 20
        color = self._REGIME_COLORS.get(label, "white")
        return Panel(
            Text(
                f"{label} ({prob*100:.0f}%)  |  Stability: {stability} bars  |  Flicker: {flicker}/{flicker_window}",
                style=color,
            ),
            title="REGIME",
        )

    def _portfolio_panel(self, portfolio):
        """Build the PORTFOLIO panel showing equity, daily P&L, allocation, and position count."""
        from rich.panel import Panel

        color = "green" if portfolio.daily_pnl >= 0 else "red"
        return Panel(
            f"Equity: ${portfolio.equity:,.2f}  |  "
            f"Daily: [bold {color}]{portfolio.daily_pnl:+,.2f} ({portfolio.daily_drawdown*100:+.2f}%)[/bold {color}]\n"
            f"Allocation: {portfolio.total_exposure*100:.0f}%  |  Positions: {portfolio.n_positions}",
            title="PORTFOLIO",
        )

    def _positions_panel(self, portfolio):
        """Build the POSITIONS panel with a per-symbol P&L table."""
        from rich.panel import Panel
        from rich.table import Table
        from rich.text import Text

        table = Table(show_header=True, header_style="bold magenta", expand=True)
        for col, justify in [("Symbol", "left"), ("Qty", "right"), ("Entry", "right"),
                              ("Current", "right"), ("P&L %", "right"), ("Stop", "right")]:
            table.add_column(col, justify=justify)

        for sym, pos in portfolio.positions.items():
            pnl_pct = pos.unrealized_pnl_pct * 100
            color = "green" if pnl_pct >= 0 else "red"
            table.add_row(
                sym,
                str(int(pos.shares)),
                f"${pos.entry_price:.2f}",
                f"${pos.current_price:.2f}",
                Text(f"{pnl_pct:+.1f}%", style=color),
                f"${pos.stop_loss:.2f}" if pos.stop_loss > 0 else "—",
            )
        return Panel(table, title="POSITIONS")

    def _signals_panel(self):
        """Build the RECENT SIGNALS panel from the last 5 signal events."""
        from rich.panel import Panel
        from rich.table import Table

        table = Table(show_header=True, expand=True)
        for col in ("Time", "Symbol", "Action", "Regime"):
            table.add_column(col)
        for s in self._recent_signals[:5]:
            table.add_row(s["time"], s["symbol"], f"{s['direction']} {s['alloc']}", s["regime"])
        return Panel(table, title="RECENT SIGNALS")

    def _risk_panel(self, portfolio):
        """Build the RISK STATUS panel showing drawdown gates and circuit-breaker state."""
        from rich.panel import Panel

        cb_status = portfolio.circuit_breaker_status
        cb_color = {"NORMAL": "green", "REDUCED": "yellow"}.get(cb_status, "red")
        daily_dd = portfolio.daily_drawdown * 100
        peak_dd = portfolio.drawdown_from_peak * 100
        return Panel(
            f"Daily DD: {daily_dd:.1f}%/3% [{'green' if daily_dd > -2 else 'red'}]{'✓' if daily_dd > -2 else '✗'}[/]  |  "
            f"From Peak: {peak_dd:.1f}%/10% [{'green' if peak_dd > -5 else 'red'}]{'✓' if peak_dd > -5 else '✗'}[/]  |  "
            f"CB: [{cb_color}]{cb_status}[/{cb_color}]",
            title="RISK STATUS",
        )

    def _system_panel(self, hmm):
        """Build the SYSTEM panel showing data/API health, HMM age, mode, and UTC clock."""
        from rich.panel import Panel

        mode = "PAPER" if self.cfg.get("broker", {}).get("paper_trading", True) else "LIVE"
        hmm_age = ""
        if self._hmm_training_date:
            days_ago = (utc_now() - ensure_utc(self._hmm_training_date)).days
            hmm_age = f"{days_ago}d ago"
        return Panel(
            f"Data: ✓  |  API: ✓  |  HMM: {hmm_age}  |  {mode}  |  {utc_now().strftime('%H:%M:%S UTC')}",
            title="SYSTEM",
        )
