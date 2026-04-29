"""Pure message-building functions that return HTML strings for Telegram."""

from datetime import datetime
from typing import List, Optional

_REGIME_EMOJI = {
    "BULL": "🐂", "STRONG_BULL": "🚀", "EUPHORIA": "🎉",
    "NEUTRAL": "😐", "WEAK_BULL": "📈", "WEAK_BEAR": "📉",
    "BEAR": "🐻", "STRONG_BEAR": "❄️", "CRASH": "💥",
}


def _regime_header(label: str, prob: float, stability: int, is_flickering: bool, stale_tag: str = "") -> List[str]:
    emoji = _REGIME_EMOJI.get(label, "❓")
    flicker = " ⚡ FLICKERING" if is_flickering else ""
    return [
        f"<b>📊 REGIME{stale_tag}</b>",
        f"{emoji} <b>{label}</b> ({prob*100:.0f}% confidence){flicker}",
        f"Stability: {stability} bars",
    ]


def _positions_section(positions: List[dict]) -> List[str]:
    if not positions:
        return ["", "📦 No open positions"]
    lines = ["", "<b>📦 OPEN POSITIONS</b>"]
    for p in positions:
        pnl_sign = "+" if p["pnl_pct"] >= 0 else ""
        lines.append(f"• {p['symbol']}: {p['shares']} shares  {pnl_sign}{p['pnl_pct']:.1f}%")
    return lines


def daily_briefing_message(
    date: datetime,
    regime_label: str,
    regime_prob: float,
    regime_stability: int,
    is_flickering: bool,
    equity: float,
    daily_pnl: float,
    daily_pnl_pct: float,
    circuit_breaker: str,
    signals: List[dict],
    orders_placed: List[dict],
    positions: List[dict],
    paper_trading: bool,
    error: Optional[str] = None,
) -> str:
    mode_tag = "📄 PAPER" if paper_trading else "💵 LIVE"
    cb_emoji = "✅" if circuit_breaker == "NORMAL" else "⚠️"
    pnl_emoji = "📈" if daily_pnl >= 0 else "📉"

    lines = [
        "<b>🤖 HMM TRADER DAILY BRIEFING</b>",
        f"<b>📅 {date.strftime('%A, %b %d %Y')}</b>  |  {mode_tag}",
        "",
        *_regime_header(regime_label, regime_prob, regime_stability, is_flickering),
        "",
        "<b>💼 PORTFOLIO</b>",
        f"Equity: <b>${equity:,.2f}</b>",
        f"{pnl_emoji} Daily P&L: <b>{daily_pnl:+,.2f} ({daily_pnl_pct:+.2f}%)</b>",
        f"Circuit Breaker: {circuit_breaker} {cb_emoji}",
    ]

    if signals:
        lines += ["", "<b>🎯 TODAY'S SIGNALS</b>"]
        for s in signals:
            lines.append(f"• {s['symbol']}: {s['direction']} {s['alloc_pct']:.0f}% @ ${s['entry']:.2f}")
    else:
        lines += ["", "🎯 <b>No signals today</b> (no rebalance needed)"]

    if orders_placed:
        lines += ["", "<b>📋 ORDERS PLACED</b>"]
        for o in orders_placed:
            lines.append(f"• {o['symbol']}: {o['side']} {o['qty']} shares @ ${o['price']:.2f} (LIMIT)")
    else:
        lines += ["", "📋 No orders placed"]

    lines += _positions_section(positions)

    if error:
        lines += ["", f"⚠️ <b>ERROR:</b> {error}"]

    lines += ["", "<i>Next run: tomorrow after market close</i>"]
    return "\n".join(lines)


def market_summary_message(
    date: datetime,
    market_status: str,
    next_open: str,
    regime_label: str,
    regime_prob: float,
    regime_stability: int,
    is_flickering: bool,
    equity: float,
    positions: List[dict],
    stock_prices: List[dict],
    paper_trading: bool,
    hmm_age_days: int = 0,
    hmm_stale_max_days: int = 3,
) -> str:
    mode_tag = "📄 PAPER" if paper_trading else "💵 LIVE"
    stale_tag = f"  ⚠️ model {hmm_age_days}d old" if hmm_age_days > hmm_stale_max_days else ""

    market_icons = {"Weekend": "🌙 <b>Weekend</b>", "Post-close": "🌆 <b>Post-close</b>"}
    market_line = f"{market_icons.get(market_status, f'📊 <b>{market_status}</b>')}  ·  <i>Next open:</i> {next_open}"

    lines = [
        "<b>🤖 REGIME TRADER MARKET SUMMARY</b>",
        f"<b>📅 {date.strftime('%A, %b %d %Y')}</b>  |  {mode_tag}",
        market_line,
        "",
        *_regime_header(regime_label, regime_prob, regime_stability, is_flickering, stale_tag=" (last close)" + stale_tag),
        "",
        "<b>💼 PORTFOLIO</b>",
        f"Equity: <b>${equity:,.2f}</b>",
    ]

    lines += _positions_section(positions)

    if stock_prices:
        lines += ["", "<b>📉 PRICE SNAPSHOT (last close)</b>"]
        for s in stock_prices:
            wk_sign = "+" if s["week_chg_pct"] >= 0 else ""
            wk_emoji = "📈" if s["week_chg_pct"] >= 0 else "📉"
            lines.append(f"• <b>{s['symbol']}</b>: ${s['close']:.2f}  {wk_emoji} {wk_sign}{s['week_chg_pct']:.1f}% wk")

    return "\n".join(lines)
