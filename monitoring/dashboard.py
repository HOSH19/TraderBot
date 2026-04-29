"""Rich-powered TUI dashboard — refresh throttle and rendering coordination."""

import logging
import time
from typing import Optional
from datetime import datetime

from core.timeutil import utc_now
from monitoring import panels as panel_builders

logger = logging.getLogger(__name__)


class Dashboard:
    """Throttle refreshes and render multi-panel status via Rich."""

    def __init__(self, config: dict) -> None:
        self.cfg = config
        self.refresh_secs = config.get("monitoring", {}).get("dashboard_refresh_seconds", 5)
        self._last_refresh = 0.0
        self._recent_signals: list = []
        self._hmm_training_date: Optional[datetime] = None

    def refresh(self, portfolio, regime_state, hmm, signals: list) -> None:
        now = time.time()
        if now - self._last_refresh < self.refresh_secs:
            return
        self._last_refresh = now
        self._update_signal_history(signals)
        if hmm and hmm.training_date:
            self._hmm_training_date = hmm.training_date
        try:
            self._render(portfolio, regime_state, hmm)
        except Exception as e:
            logger.warning("Dashboard render failed: %s", e)

    def _update_signal_history(self, signals: list) -> None:
        for s in signals:
            self._recent_signals.insert(0, {
                "time": utc_now().strftime("%H:%M"),
                "symbol": s.symbol,
                "direction": s.direction,
                "alloc": f"{s.position_size_pct*100:.0f}%",
                "regime": s.regime_name,
            })
        self._recent_signals = self._recent_signals[:10]

    def _render(self, portfolio, regime_state, hmm) -> None:
        from rich.console import Console

        mode = "PAPER" if self.cfg.get("broker", {}).get("paper_trading", True) else "LIVE"
        console = Console()
        console.clear()
        for panel in (
            panel_builders.regime_panel(regime_state, hmm),
            panel_builders.portfolio_panel(portfolio),
            panel_builders.positions_panel(portfolio),
            panel_builders.signals_panel(self._recent_signals),
            panel_builders.risk_panel(portfolio),
            panel_builders.system_panel(self.cfg, self._hmm_training_date, mode),
        ):
            console.print(panel)
