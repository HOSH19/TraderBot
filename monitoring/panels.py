"""Rich panel builder functions — each returns a renderable Panel."""

from core.timeutil import ensure_utc, utc_now

_REGIME_COLORS = {
    "BULL": "green", "STRONG_BULL": "bright_green", "EUPHORIA": "bright_green",
    "NEUTRAL": "yellow", "WEAK_BULL": "cyan", "WEAK_BEAR": "orange3",
    "BEAR": "red", "STRONG_BEAR": "bright_red", "CRASH": "bold red",
}


def regime_panel(regime_state, hmm):
    from rich.panel import Panel
    from rich.text import Text

    label = regime_state.label if regime_state else "UNKNOWN"
    prob = regime_state.probability if regime_state else 0.0
    stability = hmm.get_regime_stability() if hmm else 0
    flicker = hmm.get_regime_flicker_rate() if hmm else 0
    flicker_window = hmm.config.get("flicker_window", 20) if hmm else 20
    color = _REGIME_COLORS.get(label, "white")
    return Panel(
        Text(
            f"{label} ({prob*100:.0f}%)  |  Stability: {stability} bars  |  Flicker: {flicker}/{flicker_window}",
            style=color,
        ),
        title="REGIME",
    )


def portfolio_panel(portfolio):
    from rich.panel import Panel

    color = "green" if portfolio.daily_pnl >= 0 else "red"
    return Panel(
        f"Equity: ${portfolio.equity:,.2f}  |  "
        f"Daily: [bold {color}]{portfolio.daily_pnl:+,.2f} ({portfolio.daily_drawdown*100:+.2f}%)[/bold {color}]\n"
        f"Allocation: {portfolio.total_exposure*100:.0f}%  |  Positions: {portfolio.n_positions}",
        title="PORTFOLIO",
    )


def positions_panel(portfolio):
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
            sym, str(int(pos.shares)),
            f"${pos.entry_price:.2f}", f"${pos.current_price:.2f}",
            Text(f"{pnl_pct:+.1f}%", style=color),
            f"${pos.stop_loss:.2f}" if pos.stop_loss > 0 else "—",
        )
    return Panel(table, title="POSITIONS")


def signals_panel(recent_signals: list):
    from rich.panel import Panel
    from rich.table import Table

    table = Table(show_header=True, expand=True)
    for col in ("Time", "Symbol", "Action", "Regime"):
        table.add_column(col)
    for s in recent_signals[:5]:
        table.add_row(s["time"], s["symbol"], f"{s['direction']} {s['alloc']}", s["regime"])
    return Panel(table, title="RECENT SIGNALS")


def risk_panel(portfolio):
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


def system_panel(config: dict, hmm_training_date, mode: str):
    from rich.panel import Panel

    hmm_age = ""
    if hmm_training_date:
        days_ago = (utc_now() - ensure_utc(hmm_training_date)).days
        hmm_age = f"{days_ago}d ago"
    return Panel(
        f"Data: ✓  |  API: ✓  |  HMM: {hmm_age}  |  {mode}  |  {utc_now().strftime('%H:%M:%S UTC')}",
        title="SYSTEM",
    )
