"""
Structured JSON logging with rotating files.
Files: main.log, trades.log, alerts.log, regime.log
Each entry includes: timestamp, regime, probability, equity, positions, daily_pnl
"""

import json
import logging
import logging.handlers
import os

from core.timeutil import utc_now


class StructuredFormatter(logging.Formatter):
    """Logging formatter that serialises each log record as a JSON object."""

    def format(self, record: logging.LogRecord) -> str:
        """Serialize a log record to a JSON string.

        Always includes timestamp, level, logger name, and message.
        Appends optional trading fields (regime, probability, equity, positions,
        daily_pnl) when present on the record, plus exception info if applicable.
        """
        doc = {
            "timestamp": utc_now().isoformat(),
            "level": record.levelname,
            "logger": record.name,
            "message": record.getMessage(),
        }
        for key in ("regime", "probability", "equity", "positions", "daily_pnl"):
            if hasattr(record, key):
                doc[key] = getattr(record, key)
        if record.exc_info:
            doc["exception"] = self.formatException(record.exc_info)
        return json.dumps(doc)


def _make_rotating_handler(path: str, max_bytes: int, backup_count: int) -> logging.Handler:
    """Create a RotatingFileHandler with StructuredFormatter already attached.

    Args:
        path: Absolute or relative path to the log file.
        max_bytes: Maximum file size before rotation.
        backup_count: Number of rotated backups to retain.
    """
    handler = logging.handlers.RotatingFileHandler(
        path, maxBytes=max_bytes, backupCount=backup_count, encoding="utf-8"
    )
    handler.setFormatter(StructuredFormatter())
    return handler


def setup_structured_logging(config: dict):
    """Configure the root logger and per-domain rotating log files.

    Sets up a console handler (INFO+) and four rotating JSON log files:
    main.log (DEBUG+), trades.log, alerts.log, and regime.log.
    Log directory and rotation settings are read from config.monitoring.
    """
    monitoring_cfg = config.get("monitoring", {})
    log_dir = monitoring_cfg.get("log_dir", "logs")
    max_bytes = monitoring_cfg.get("log_max_bytes", 10 * 1024 * 1024)
    backup_count = monitoring_cfg.get("log_backup_count", 30)

    os.makedirs(log_dir, exist_ok=True)

    root = logging.getLogger()
    root.setLevel(logging.DEBUG)

    console = logging.StreamHandler()
    console.setLevel(logging.INFO)
    console.setFormatter(logging.Formatter("%(asctime)s [%(levelname)s] %(name)s: %(message)s"))
    root.addHandler(console)

    main_handler = _make_rotating_handler(os.path.join(log_dir, "main.log"), max_bytes, backup_count)
    main_handler.setLevel(logging.DEBUG)
    root.addHandler(main_handler)

    trade_logger = logging.getLogger("trades")
    trade_logger.addHandler(
        _make_rotating_handler(os.path.join(log_dir, "trades.log"), max_bytes, backup_count)
    )

    alert_logger = logging.getLogger("alerts")
    alert_logger.addHandler(
        _make_rotating_handler(os.path.join(log_dir, "alerts.log"), max_bytes, backup_count)
    )

    regime_logger = logging.getLogger("regime")
    regime_logger.addHandler(
        _make_rotating_handler(os.path.join(log_dir, "regime.log"), max_bytes, backup_count)
    )


def log_trade(symbol: str, direction: str, qty: float, price: float, regime: str, pnl: float = 0.0):
    """Write a structured trade entry to the trades logger.

    Args:
        symbol: Ticker of the traded instrument.
        direction: 'BUY' or 'SELL'.
        qty: Number of shares traded.
        price: Execution price per share.
        regime: Active regime label at the time of the trade.
        pnl: Realised P&L for the trade (default 0.0 for new entries).
    """
    logger = logging.getLogger("trades")
    logger.info(
        f"TRADE {direction} {qty} {symbol} @ ${price:.2f}",
        extra={"regime": regime, "pnl": pnl},
    )


def log_regime_change(old_regime: str, new_regime: str, probability: float, equity: float):
    """Write a WARNING-level regime transition entry to the regime logger.

    Args:
        old_regime: Previous regime label.
        new_regime: New confirmed regime label.
        probability: HMM confidence for the new regime (0–1).
        equity: Portfolio equity at the time of the transition.
    """
    logger = logging.getLogger("regime")
    logger.warning(
        f"REGIME CHANGE: {old_regime} → {new_regime} (p={probability:.2f})",
        extra={"regime": new_regime, "probability": probability, "equity": equity},
    )
