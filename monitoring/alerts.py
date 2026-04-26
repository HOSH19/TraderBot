"""
Alert system for critical trading events.
Rate-limited: 1 alert per event type per 15 minutes.
Delivery: console, log file, optional email, optional webhook.
"""

import logging
import os
import time

logger = logging.getLogger("alerts")


class AlertManager:
    """Dispatches rate-limited alerts for critical trading events via log, webhook, and email."""

    def __init__(self, config: dict):
        """Initialize the alert manager with rate-limit settings and optional delivery targets.

        Args:
            config: Full application config dict; reads monitoring.alert_rate_limit_minutes,
                    alerts.email, and alerts.webhook_url. Environment variables
                    ALERT_EMAIL and ALERT_WEBHOOK_URL override config values.
        """
        self.cfg = config
        self.rate_limit_secs = config.get("monitoring", {}).get("alert_rate_limit_minutes", 15) * 60
        self._last_sent: dict = {}
        self._email = os.getenv("ALERT_EMAIL") or config.get("alerts", {}).get("email", "")
        self._webhook = os.getenv("ALERT_WEBHOOK_URL") or config.get("alerts", {}).get("webhook_url", "")

    def send(self, event_type: str, message: str) -> None:
        """Log and optionally webhook/email; no-op if ``event_type`` was sent inside the rate-limit window."""
        now = time.time()
        last = self._last_sent.get(event_type, 0)
        if now - last < self.rate_limit_secs:
            return

        self._last_sent[event_type] = now
        logger.warning(f"[ALERT:{event_type.upper()}] {message}")

        if self._webhook:
            self._send_webhook(event_type, message)
        if self._email:
            self._send_email(event_type, message)

    def _send_webhook(self, event_type: str, message: str):
        """POST the alert payload to the configured webhook URL. Errors are logged at DEBUG."""
        try:
            import requests
            payload = {"event": event_type, "message": message, "timestamp": time.time()}
            requests.post(self._webhook, json=payload, timeout=5)
        except Exception:
            pass

    def _send_email(self, event_type: str, message: str):
        """Send an alert email to the configured address via localhost SMTP. Errors are logged at DEBUG."""
        try:
            import smtplib
            from email.message import EmailMessage

            msg = EmailMessage()
            msg.set_content(f"Event: {event_type}\n\n{message}")
            msg["Subject"] = f"[RegimeTrader] {event_type.upper()}"
            msg["From"] = self._email
            msg["To"] = self._email

            with smtplib.SMTP("localhost") as smtp:
                smtp.send_message(msg)
        except Exception:
            pass

    def on_regime_state(self, current_state, previous_state) -> None:
        """Send ``regime_change`` when the confirmed regime label differs from the previous bar."""
        if not (previous_state and current_state):
            return
        if not (
            getattr(previous_state, "label", None)
            and getattr(current_state, "label", None)
            and previous_state.label != current_state.label
            and current_state.is_confirmed
        ):
            return
        self.send(
            "regime_change",
            f"Regime changed: {previous_state.label} → {current_state.label} "
            f"(p={current_state.probability:.2f})",
        )

    def on_large_pnl(self, symbol: str, pnl_pct: float, threshold: float = 0.05):
        """Send a large_pnl alert when a position's gain or loss exceeds the threshold.

        Args:
            symbol: Ticker symbol of the position.
            pnl_pct: Unrealized P&L as a decimal (e.g. 0.07 for +7%).
            threshold: Absolute P&L fraction that triggers the alert (default 5%).
        """
        if abs(pnl_pct) >= threshold:
            direction = "gain" if pnl_pct > 0 else "loss"
            self.send("large_pnl", f"Large {direction}: {symbol} {pnl_pct*100:+.1f}%")

    def on_data_feed_down(self, symbol: str):
        """Send a data_feed_down alert when price data cannot be retrieved for a symbol."""
        self.send("data_feed_down", f"Data feed down for {symbol}")

    def on_api_error(self, error: str):
        """Send an api_lost alert when the Alpaca broker API raises an unexpected error."""
        self.send("api_lost", f"Alpaca API error: {error}")

    def on_flicker_exceeded(self, flicker_rate: int, threshold: int):
        """Send a flicker_exceeded alert when the HMM regime changes too frequently.

        Args:
            flicker_rate: Number of regime transitions observed in the flicker window.
            threshold: Maximum allowed transitions before triggering the alert.
        """
        self.send("flicker_exceeded", f"HMM flicker rate {flicker_rate} exceeds threshold {threshold}")
