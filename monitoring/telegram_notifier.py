"""Thin Telegram Bot API sender — message building lives in monitoring.messages."""

import logging
import os
from datetime import datetime
from typing import List, Optional

import requests

from monitoring.messages import daily_briefing_message, market_summary_message

logger = logging.getLogger(__name__)

TELEGRAM_API = "https://api.telegram.org/bot{token}/sendMessage"

_ALERT_EMOJI = {
    "circuit_breaker": "🚨",
    "regime_change": "🔄",
    "large_pnl": "💰",
    "error": "❌",
    "halted": "🛑",
}


class TelegramNotifier:
    """Thin ``requests`` wrapper around Telegram Bot API ``sendMessage``."""

    def __init__(self) -> None:
        token = (os.getenv("TELEGRAM_BOT_TOKEN") or "").strip()
        chat_id = str((os.getenv("TELEGRAM_CHAT_ID") or "")).strip()
        self.token = token
        self.chat_id = chat_id
        self.enabled = bool(token and chat_id)

    def send(self, message: str) -> bool:
        if not self.enabled:
            return False
        try:
            resp = requests.post(
                TELEGRAM_API.format(token=self.token),
                json={"chat_id": self.chat_id, "text": message, "parse_mode": "HTML"},
                timeout=10,
            )
            if resp.status_code == 200:
                return True
            logger.warning("Telegram send failed: %s %s", resp.status_code, resp.text)
            return False
        except Exception as e:
            logger.error("Telegram error: %s", e)
            return False

    def send_alert(self, event: str, detail: str) -> bool:
        emoji = _ALERT_EMOJI.get(event, "⚠️")
        return self.send(f"{emoji} <b>REGIME TRADER ALERT</b>\n\n<b>{event.upper()}</b>\n{detail}")

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
        circuit_breaker: str,
        signals: List[dict],
        orders_placed: List[dict],
        positions: List[dict],
        paper_trading: bool,
        error: Optional[str] = None,
    ) -> bool:
        msg = daily_briefing_message(
            date=date,
            regime_label=regime_label,
            regime_prob=regime_prob,
            regime_stability=regime_stability,
            is_flickering=is_flickering,
            equity=equity,
            daily_pnl=daily_pnl,
            daily_pnl_pct=daily_pnl_pct,
            circuit_breaker=circuit_breaker,
            signals=signals,
            orders_placed=orders_placed,
            positions=positions,
            paper_trading=paper_trading,
            error=error,
        )
        return self.send(msg)

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
        positions: List[dict],
        stock_prices: List[dict],
        paper_trading: bool,
        hmm_age_days: int = 0,
        hmm_stale_max_days: int = 3,
    ) -> bool:
        msg = market_summary_message(
            date=date,
            market_status=market_status,
            next_open=next_open,
            regime_label=regime_label,
            regime_prob=regime_prob,
            regime_stability=regime_stability,
            is_flickering=is_flickering,
            equity=equity,
            positions=positions,
            stock_prices=stock_prices,
            paper_trading=paper_trading,
            hmm_age_days=hmm_age_days,
            hmm_stale_max_days=hmm_stale_max_days,
        )
        return self.send(msg)
