"""
Test Telegram bot connectivity and message formatting.

Usage:
    python -m pytest tests/test_telegram.py -v
    # or send a real test message:
    python tests/test_telegram.py
"""

import os
import sys
from datetime import datetime

import pytest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


class TestTelegramNotifier:
    """Tests for TelegramNotifier initialization, enable/disable logic, and message formatting."""

    def test_notifier_initializes(self):
        """TelegramNotifier must instantiate without raising an exception."""
        from monitoring.telegram_notifier import TelegramNotifier
        tg = TelegramNotifier()
        assert tg is not None

    def test_disabled_when_no_credentials(self, monkeypatch):
        """Notifier must report enabled=False when both credential environment variables are absent."""
        monkeypatch.delenv("TELEGRAM_BOT_TOKEN", raising=False)
        monkeypatch.delenv("TELEGRAM_CHAT_ID", raising=False)
        from monitoring.telegram_notifier import TelegramNotifier
        tg = TelegramNotifier()
        assert not tg.enabled

    def test_enabled_when_credentials_present(self, monkeypatch):
        """Notifier must report enabled=True when both credential environment variables are set."""
        monkeypatch.setenv("TELEGRAM_BOT_TOKEN", "123:abc")
        monkeypatch.setenv("TELEGRAM_CHAT_ID", "456")
        from monitoring.telegram_notifier import TelegramNotifier
        tg = TelegramNotifier()
        assert tg.enabled

    def test_send_returns_false_when_disabled(self):
        """send() must return False without error when the notifier is disabled."""
        from monitoring.telegram_notifier import TelegramNotifier
        tg = TelegramNotifier()
        if not tg.enabled:
            result = tg.send("test message")
            assert result is False

    def test_briefing_message_format(self, monkeypatch):
        """send_daily_briefing must return a bool, compose HTML, and never hit the real Telegram API."""
        from monitoring.telegram_notifier import TelegramNotifier

        tg = TelegramNotifier()
        captured: list[str] = []

        def fake_send(message: str) -> bool:
            captured.append(message)
            return True

        monkeypatch.setattr(tg, "send", fake_send)

        result = tg.send_daily_briefing(
            date=datetime(2026, 4, 11),
            regime_label="BULL",
            regime_prob=0.72,
            regime_stability=14,
            is_flickering=False,
            equity=105_230.0,
            daily_pnl=340.0,
            daily_pnl_pct=0.32,
            circuit_breaker="NORMAL",
            signals=[{"symbol": "SPY", "direction": "LONG", "alloc_pct": 95, "entry": 520.30}],
            orders_placed=[{"symbol": "SPY", "side": "BUY", "qty": 50, "price": 520.82}],
            positions=[{"symbol": "SPY", "shares": 200, "pnl_pct": 1.2}],
            paper_trading=True,
            news={"SPY": {"title": "Test headline", "source": "Test", "url": "https://example.com", "time_ago": "1h ago"}},
        )
        assert result is True
        assert len(captured) == 1
        body = captured[0]
        assert "HMM TRADER DAILY BRIEFING" in body
        assert "From Peak" not in body
        assert "TOP NEWS" in body
        assert "Test headline" in body


if __name__ == "__main__":
    """Send a real briefing via Telegram (same as production cron). Requires .env credentials."""
    import run_daily

    run_daily.run()
