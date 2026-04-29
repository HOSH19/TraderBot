"""Tests for :class:`~monitoring.telegram_notifier.TelegramNotifier` wiring and HTML briefing.

Usage:
    python -m pytest tests/test_telegram.py -v
    python tests/test_telegram.py
"""

import os
import sys
from datetime import datetime

import pytest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


class TestTelegramNotifier:
    """Credential gating, ``send`` short-circuit, and patched briefing composition."""

    def test_notifier_initializes(self):
        """Constructor does not raise."""
        from monitoring.telegram_notifier import TelegramNotifier
        tg = TelegramNotifier()
        assert tg is not None

    def test_disabled_when_no_credentials(self, monkeypatch):
        """Missing env → ``enabled`` is false."""
        monkeypatch.delenv("TELEGRAM_BOT_TOKEN", raising=False)
        monkeypatch.delenv("TELEGRAM_CHAT_ID", raising=False)
        from monitoring.telegram_notifier import TelegramNotifier
        tg = TelegramNotifier()
        assert not tg.enabled

    def test_enabled_when_credentials_present(self, monkeypatch):
        """Both env vars set → ``enabled`` is true."""
        monkeypatch.setenv("TELEGRAM_BOT_TOKEN", "123:abc")
        monkeypatch.setenv("TELEGRAM_CHAT_ID", "456")
        from monitoring.telegram_notifier import TelegramNotifier
        tg = TelegramNotifier()
        assert tg.enabled

    def test_send_returns_false_when_disabled(self):
        """No network when disabled; ``send`` returns ``False``."""
        from monitoring.telegram_notifier import TelegramNotifier
        tg = TelegramNotifier()
        if not tg.enabled:
            result = tg.send("test message")
            assert result is False

    def test_briefing_message_format(self, monkeypatch):
        """``send_daily_briefing`` builds expected HTML sections via patched ``send``."""
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
        )
        assert result is True
        assert len(captured) == 1
        body = captured[0]
        assert "HMM TRADER DAILY BRIEFING" in body
        assert "From Peak" not in body


if __name__ == "__main__":
    # Sends a real cron-style message; requires working ``.env`` Telegram vars.
    import run_daily

    run_daily.run()
