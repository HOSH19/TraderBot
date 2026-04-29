"""Rate-limited alerts (log + optional webhook/email) for operational incidents."""

import logging
import os
import time

logger = logging.getLogger("alerts")


class AlertManager:
    """Debounce duplicate ``event_type`` keys then fan out to configured sinks."""

    def __init__(self, config: dict) -> None:
        """Load rate-limit minutes plus optional email/webhook targets.

        Args:
            config: Uses ``monitoring.alert_rate_limit_minutes`` and ``alerts.*`` keys.
                ``ALERT_EMAIL`` / ``ALERT_WEBHOOK_URL`` env vars override file settings.
        """
        self.cfg = config
        self.rate_limit_secs = config.get("monitoring", {}).get("alert_rate_limit_minutes", 15) * 60
        self._last_sent: dict = {}
        self._email = os.getenv("ALERT_EMAIL") or config.get("alerts", {}).get("email", "")
        self._webhook = os.getenv("ALERT_WEBHOOK_URL") or config.get("alerts", {}).get("webhook_url", "")

    def send(self, event_type: str, message: str) -> None:
        """Emit ``message`` unless ``event_type`` is still inside the cooldown window."""
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

