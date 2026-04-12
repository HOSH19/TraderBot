"""
Telegram daily briefing and alert notifications.

Setup:
  1. Message @BotFather on Telegram → /newbot → copy the token
  2. Start a chat with your bot, then visit:
     https://api.telegram.org/bot<TOKEN>/getUpdates
     to find your chat_id
  3. Add to .env:
     TELEGRAM_BOT_TOKEN=your_token_here
     TELEGRAM_CHAT_ID=your_chat_id_here
"""

import logging
import os
from datetime import datetime
from typing import List, Optional

import requests

logger = logging.getLogger(__name__)

TELEGRAM_API = "https://api.telegram.org/bot{token}/sendMessage"


def _format_news_section(news: dict) -> list:
    """Format news dict {symbol: article|None} into Telegram message lines."""
    articles = [(sym, a) for sym, a in news.items() if a]
    if not articles:
        return []

    lines = ["", "<b>📰 TOP NEWS</b>"]
    for sym, a in articles:
        time_tag = f"  <i>{a['time_ago']}</i>" if a.get("time_ago") else ""
        source_tag = f" — {a['source']}" if a.get("source") else ""
        title = a.get("title", "").rstrip(" - " + a.get("source", ""))
        url = a.get("url", "")
        if url:
            lines.append(f'• <b>{sym}</b>: <a href="{url}">{title}</a>{source_tag}{time_tag}')
        else:
            lines.append(f"• <b>{sym}</b>: {title}{source_tag}{time_tag}")
    return lines


class TelegramNotifier:
    """Sends Telegram messages for daily briefings, market summaries, and trade alerts."""

    def __init__(self):
        """Initialize the notifier, reading credentials from environment variables."""
        token = os.getenv("TELEGRAM_BOT_TOKEN", "") or ""
        chat_id = os.getenv("TELEGRAM_CHAT_ID", "") or ""
        self.token = token.strip()
        self.chat_id = str(chat_id).strip()
        self.enabled = bool(self.token and self.chat_id)
        if not self.enabled:
            logger.info("Telegram not configured — set TELEGRAM_BOT_TOKEN and TELEGRAM_CHAT_ID in .env")

    def send(self, message: str) -> bool:
        """Send an HTML-formatted message to the configured Telegram chat.

        Returns True on success, False if disabled or on any delivery failure.
        """
        if not self.enabled:
            logger.info(f"[TELEGRAM DISABLED] {message}")
            return False
        try:
            resp = requests.post(
                TELEGRAM_API.format(token=self.token),
                json={
                    "chat_id": self.chat_id,
                    "text": message,
                    "parse_mode": "HTML",
                },
                timeout=10,
            )
            if resp.status_code == 200:
                return True
            logger.warning(f"Telegram send failed: {resp.status_code} {resp.text}")
            return False
        except Exception as e:
            logger.error(f"Telegram error: {e}")
            return False

    def send_daily_briefing(
        self,
        date: datetime,
        regime_label: str,
        regime_prob: float,
        regime_stability: int,
        is_flickering: bool,
        equity: float,
        daily_pnl: float,
        daily_pnl_pct: float,
        peak_dd_pct: float,
        circuit_breaker: str,
        signals: List[dict],
        orders_placed: List[dict],
        positions: List[dict],
        paper_trading: bool,
        news: Optional[dict] = None,
        error: Optional[str] = None,
    ) -> bool:
        """Compose and send the end-of-day trading briefing message.

        Includes regime state, portfolio P&L, today's signals, orders placed,
        open positions, top news headlines, and any error encountered.
        Returns True if the message was delivered successfully.
        """
        mode_tag = "📄 PAPER" if paper_trading else "💵 LIVE"
        regime_emoji = {
            "BULL": "🐂", "STRONG_BULL": "🚀", "EUPHORIA": "🎉",
            "NEUTRAL": "😐", "WEAK_BULL": "📈", "WEAK_BEAR": "📉",
            "BEAR": "🐻", "STRONG_BEAR": "❄️", "CRASH": "💥",
        }.get(regime_label, "❓")

        cb_emoji = "✅" if circuit_breaker == "NORMAL" else "⚠️"
        dd_emoji = "✅" if peak_dd_pct > -5 else ("⚠️" if peak_dd_pct > -8 else "🚨")
        pnl_emoji = "📈" if daily_pnl >= 0 else "📉"
        flicker_tag = " ⚡ FLICKERING" if is_flickering else ""

        lines = [
            f"<b>🤖 HMM TRADER DAILY BRIEFING</b>",
            f"<b>📅 {date.strftime('%A, %b %d %Y')}</b>  |  {mode_tag}",
            "",
            f"<b>📊 REGIME</b>",
            f"{regime_emoji} <b>{regime_label}</b> ({regime_prob*100:.0f}% confidence){flicker_tag}",
            f"Stability: {regime_stability} bars",
            "",
            f"<b>💼 PORTFOLIO</b>",
            f"Equity: <b>${equity:,.2f}</b>",
            f"{pnl_emoji} Daily P&L: <b>{daily_pnl:+,.2f} ({daily_pnl_pct:+.2f}%)</b>",
            f"From Peak: {peak_dd_pct:.1f}%  {dd_emoji}",
            f"Circuit Breaker: {circuit_breaker} {cb_emoji}",
        ]

        if signals:
            lines += ["", "<b>🎯 TODAY'S SIGNALS</b>"]
            for s in signals:
                lines.append(
                    f"• {s['symbol']}: {s['direction']} {s['alloc_pct']:.0f}% "
                    f"@ ${s['entry']:.2f}  stop ${s['stop']:.2f}"
                )
        else:
            lines += ["", "🎯 <b>No signals today</b> (no rebalance needed)"]

        if orders_placed:
            lines += ["", "<b>📋 ORDERS PLACED</b>"]
            for o in orders_placed:
                lines.append(f"• {o['symbol']}: {o['side']} {o['qty']} shares @ ${o['price']:.2f} (LIMIT)")
        else:
            lines += ["", "📋 No orders placed"]

        if positions:
            lines += ["", "<b>📦 OPEN POSITIONS</b>"]
            for p in positions:
                pnl_sign = "+" if p['pnl_pct'] >= 0 else ""
                lines.append(
                    f"• {p['symbol']}: {p['shares']} shares  "
                    f"{pnl_sign}{p['pnl_pct']:.1f}%  stop ${p['stop']:.2f}"
                )

        if news:
            news_lines = _format_news_section(news)
            if news_lines:
                lines += news_lines

        if error:
            lines += ["", f"⚠️ <b>ERROR:</b> {error}"]

        lines += ["", f"<i>Next run: tomorrow after market close</i>"]

        message = "\n".join(lines)
        return self.send(message)

    def send_market_summary(
        self,
        date: datetime,
        market_status: str,
        next_open: str,
        regime_label: str,
        regime_prob: float,
        regime_stability: int,
        is_flickering: bool,
        equity: float,
        peak_dd_pct: float,
        positions: List[dict],
        stock_prices: List[dict],
        paper_trading: bool,
        hmm_age_days: int = 0,
        news: Optional[dict] = None,
    ) -> bool:
        """Send a market-closed summary with regime state, portfolio snapshot, and price data.

        Used on non-trading days or when the market is closed. Returns True on success.
        """
        mode_tag = "📄 PAPER" if paper_trading else "💵 LIVE"
        regime_emoji = {
            "BULL": "🐂", "STRONG_BULL": "🚀", "EUPHORIA": "🎉",
            "NEUTRAL": "😐", "WEAK_BULL": "📈", "WEAK_BEAR": "📉",
            "BEAR": "🐻", "STRONG_BEAR": "❄️", "CRASH": "💥",
        }.get(regime_label, "❓")

        flicker_tag = " ⚡ FLICKERING" if is_flickering else ""
        stale_tag = f"  ⚠️ model {hmm_age_days}d old" if hmm_age_days > 7 else ""

        lines = [
            f"<b>🤖 REGIME TRADER MARKET SUMMARY</b>",
            f"<b>📅 {date.strftime('%A, %b %d %Y')}</b>  |  {mode_tag}",
            f"🔴 Market: <b>{market_status}</b>  |  Next open: {next_open}",
            "",
            f"<b>📊 REGIME (last close){stale_tag}</b>",
            f"{regime_emoji} <b>{regime_label}</b> ({regime_prob*100:.0f}% confidence){flicker_tag}",
            f"Stability: {regime_stability} bars",
            "",
            f"<b>💼 PORTFOLIO</b>",
            f"Equity: <b>${equity:,.2f}</b>  |  From Peak: {peak_dd_pct:.1f}%",
        ]

        if positions:
            lines.append("")
            lines.append("<b>📦 OPEN POSITIONS</b>")
            for p in positions:
                pnl_emoji = "📈" if p["pnl_pct"] >= 0 else "📉"
                pnl_sign = "+" if p["pnl_pct"] >= 0 else ""
                lines.append(
                    f"• {p['symbol']}: {p['shares']} shares  "
                    f"{pnl_emoji} {pnl_sign}{p['pnl_pct']:.1f}%  stop ${p['stop']:.2f}"
                )
        else:
            lines += ["", "📦 No open positions"]

        if stock_prices:
            lines += ["", "<b>📉 PRICE SNAPSHOT (last close)</b>"]
            for s in stock_prices:
                wk_sign = "+" if s["week_chg_pct"] >= 0 else ""
                wk_emoji = "📈" if s["week_chg_pct"] >= 0 else "📉"
                lines.append(
                    f"• <b>{s['symbol']}</b>: ${s['close']:.2f}  "
                    f"{wk_emoji} {wk_sign}{s['week_chg_pct']:.1f}% wk"
                )

        if news:
            news_lines = _format_news_section(news)
            if news_lines:
                lines += news_lines

        lines += ["", f"<i>No orders placed — market closed</i>"]
        return self.send("\n".join(lines))

    def send_alert(self, event: str, detail: str) -> bool:
        """Send a critical-event alert message with an appropriate emoji prefix.

        Args:
            event: One of 'circuit_breaker', 'regime_change', 'large_pnl', 'error', 'halted'.
            detail: Human-readable description of the event.
        """
        emoji_map = {
            "circuit_breaker": "🚨",
            "regime_change": "🔄",
            "large_pnl": "💰",
            "error": "❌",
            "halted": "🛑",
        }
        emoji = emoji_map.get(event, "⚠️")
        message = f"{emoji} <b>REGIME TRADER ALERT</b>\n\n<b>{event.upper()}</b>\n{detail}"
        return self.send(message)
