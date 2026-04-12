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

    def test_briefing_message_format(self, capsys):
        """send_daily_briefing must return a bool and not raise for a complete set of daily metrics."""
        from monitoring.telegram_notifier import TelegramNotifier
        tg = TelegramNotifier()
        result = tg.send_daily_briefing(
            date=datetime(2026, 4, 11),
            regime_label="BULL",
            regime_prob=0.72,
            regime_stability=14,
            is_flickering=False,
            equity=105_230.0,
            daily_pnl=340.0,
            daily_pnl_pct=0.32,
            peak_dd_pct=-1.2,
            circuit_breaker="NORMAL",
            signals=[{"symbol": "SPY", "direction": "LONG", "alloc_pct": 95, "entry": 520.30, "stop": 508.0}],
            orders_placed=[{"symbol": "SPY", "side": "BUY", "qty": 50, "price": 520.82}],
            positions=[{"symbol": "SPY", "shares": 200, "pnl_pct": 1.2, "stop": 508.0}],
            paper_trading=True,
        )
        assert isinstance(result, bool)


if __name__ == "__main__":
    """Run the full daily pipeline and send the real Telegram message."""
    import run_daily
    run_daily.run()
