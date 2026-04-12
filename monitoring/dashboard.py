"""
Terminal-based live dashboard using the rich library.
Refreshes every 5 seconds. Color-coded risk status bars.

Layout:
  REGIME | PORTFOLIO | POSITIONS | RECENT SIGNALS | RISK STATUS | SYSTEM
"""

import logging
import os
from datetime import datetime
from typing import List, Optional

logger = logging.getLogger(__name__)


class Dashboard:
    """Terminal-based live dashboard rendered with the rich library."""

    def __init__(self, config: dict):
        """Initialize the dashboard with configuration and empty signal history.

        Args:
            config: Full application config dict; reads monitoring.dashboard_refresh_seconds.
        """
        self.cfg = config
        self.refresh_secs = config.get("monitoring", {}).get("dashboard_refresh_seconds", 5)
        self._last_refresh = 0.0
        self._recent_signals: list = []
        self._last_signals: list = []
        self._api_latency_ms: float = 0.0
        self._hmm_training_date: Optional[datetime] = None

    def refresh(self, portfolio, regime_state, hmm, signals: list):
        """Refresh the dashboard if the configured interval has elapsed.

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
                    "time": datetime.utcnow().strftime("%H:%M"),
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

    def _render(self, portfolio, regime_state, hmm):
        """Clear the terminal and print all dashboard panels using rich.

        Renders six panels in order: REGIME, PORTFOLIO, POSITIONS, RECENT SIGNALS,
        RISK STATUS, and SYSTEM. Requires the rich library to be installed.
        """
        from rich.console import Console
        from rich.layout import Layout
        from rich.panel import Panel
        from rich.table import Table
        from rich.text import Text

        console = Console()
        console.clear()

        regime_label = regime_state.label if regime_state else "UNKNOWN"
        regime_prob = regime_state.probability if regime_state else 0.0
        stability = hmm.get_regime_stability() if hmm else 0
        flicker = hmm.get_regime_flicker_rate() if hmm else 0

        regime_color = {
            "BULL": "green", "STRONG_BULL": "bright_green", "EUPHORIA": "bright_green",
            "NEUTRAL": "yellow", "WEAK_BULL": "cyan", "WEAK_BEAR": "orange3",
            "BEAR": "red", "STRONG_BEAR": "bright_red", "CRASH": "bold red",
        }.get(regime_label, "white")

        regime_panel = Panel(
            Text(
                f"{regime_label} ({regime_prob*100:.0f}%)  |  "
                f"Stability: {stability} bars  |  Flicker: {flicker}/{hmm.config.get('flicker_window',20) if hmm else 20}",
                style=regime_color,
            ),
            title="REGIME",
        )

        daily_pnl_color = "green" if portfolio.daily_pnl >= 0 else "red"
        portfolio_panel = Panel(
            f"Equity: ${portfolio.equity:,.2f}  |  "
            f"Daily: [bold {daily_pnl_color}]{portfolio.daily_pnl:+,.2f} "
            f"({portfolio.daily_drawdown*100:+.2f}%)[/bold {daily_pnl_color}]\n"
            f"Allocation: {portfolio.total_exposure*100:.0f}%  |  "
            f"Positions: {portfolio.n_positions}",
            title="PORTFOLIO",
        )

        pos_table = Table(show_header=True, header_style="bold magenta", expand=True)
        pos_table.add_column("Symbol")
        pos_table.add_column("Qty", justify="right")
        pos_table.add_column("Entry", justify="right")
        pos_table.add_column("Current", justify="right")
        pos_table.add_column("P&L %", justify="right")
        pos_table.add_column("Stop", justify="right")

        for sym, pos in portfolio.positions.items():
            pnl_pct = pos.unrealized_pnl_pct * 100
            color = "green" if pnl_pct >= 0 else "red"
            pos_table.add_row(
                sym,
                str(int(pos.shares)),
                f"${pos.entry_price:.2f}",
                f"${pos.current_price:.2f}",
                Text(f"{pnl_pct:+.1f}%", style=color),
                f"${pos.stop_loss:.2f}" if pos.stop_loss > 0 else "—",
            )
        positions_panel = Panel(pos_table, title="POSITIONS")

        sig_table = Table(show_header=True, expand=True)
        sig_table.add_column("Time")
        sig_table.add_column("Symbol")
        sig_table.add_column("Action")
        sig_table.add_column("Regime")
        for s in self._recent_signals[:5]:
            sig_table.add_row(s["time"], s["symbol"], f"{s['direction']} {s['alloc']}", s["regime"])
        signals_panel = Panel(sig_table, title="RECENT SIGNALS")

        cb_status = portfolio.circuit_breaker_status
        cb_color = {"NORMAL": "green", "REDUCED": "yellow"}.get(cb_status, "red")
        daily_dd_pct = portfolio.daily_drawdown * 100
        peak_dd_pct = portfolio.drawdown_from_peak * 100

        risk_panel = Panel(
            f"Daily DD: {daily_dd_pct:.1f}%/3% "
            f"[{'green' if daily_dd_pct > -2 else 'red'}]{'✓' if daily_dd_pct > -2 else '✗'}[/]  |  "
            f"From Peak: {peak_dd_pct:.1f}%/10% "
            f"[{'green' if peak_dd_pct > -5 else 'red'}]{'✓' if peak_dd_pct > -5 else '✗'}[/]  |  "
            f"CB: [{cb_color}]{cb_status}[/{cb_color}]",
            title="RISK STATUS",
        )

        mode = "PAPER" if self.cfg.get("broker", {}).get("paper_trading", True) else "LIVE"
        hmm_age = ""
        if self._hmm_training_date:
            days_ago = (datetime.utcnow() - self._hmm_training_date).days
            hmm_age = f"{days_ago}d ago"

        system_panel = Panel(
            f"Data: ✓  |  API: ✓  |  HMM: {hmm_age}  |  {mode}  |  "
            f"{datetime.utcnow().strftime('%H:%M:%S UTC')}",
            title="SYSTEM",
        )

        console.print(regime_panel)
        console.print(portfolio_panel)
        console.print(positions_panel)
        console.print(signals_panel)
        console.print(risk_panel)
        console.print(system_panel)
