"""SQLite-backed state persistence: equity curve, regime history, trade log.

Replaces the flat state_snapshot.json. The snapshot table still provides
the same fast-read semantics for process restart recovery.
"""

import logging
import sqlite3
from contextlib import contextmanager
from datetime import datetime
from typing import Iterator, List, Optional

logger = logging.getLogger(__name__)

_SCHEMA = """
CREATE TABLE IF NOT EXISTS snapshot (
    key   TEXT PRIMARY KEY,
    value TEXT NOT NULL
);

CREATE TABLE IF NOT EXISTS equity_curve (
    ts     TEXT PRIMARY KEY,
    equity REAL NOT NULL,
    cash   REAL NOT NULL
);

CREATE TABLE IF NOT EXISTS regime_history (
    ts          TEXT PRIMARY KEY,
    label       TEXT NOT NULL,
    probability REAL NOT NULL,
    is_confirmed INTEGER NOT NULL
);

CREATE TABLE IF NOT EXISTS trade_log (
    id         INTEGER PRIMARY KEY AUTOINCREMENT,
    ts         TEXT NOT NULL,
    symbol     TEXT NOT NULL,
    side       TEXT NOT NULL,
    qty        REAL NOT NULL,
    price      REAL NOT NULL,
    order_id   TEXT,
    regime     TEXT,
    strategy   TEXT
);
"""


class StateStore:
    """Thin SQLite wrapper for regime-trader runtime state."""

    def __init__(self, db_path: str) -> None:
        self._path = db_path
        self._init_db()

    # ------------------------------------------------------------------ #
    # Snapshot (fast restart recovery)                                     #
    # ------------------------------------------------------------------ #

    def save_snapshot(self, portfolio, regime_state) -> None:
        import json
        from core.timeutil import utc_now
        data = {
            "timestamp": utc_now().isoformat(),
            "equity": portfolio.equity,
            "cash": portfolio.cash,
            "daily_start_equity": portfolio.daily_start_equity or portfolio.equity,
            "weekly_start_equity": portfolio.weekly_start_equity or portfolio.equity,
            "circuit_breaker_status": portfolio.circuit_breaker_status,
            "regime": regime_state.label if regime_state else "UNKNOWN",
            "regime_prob": float(regime_state.probability) if regime_state else 0.0,
        }
        with self._connect() as conn:
            conn.execute(
                "INSERT OR REPLACE INTO snapshot (key, value) VALUES ('state', ?)",
                (json.dumps(data),),
            )

    def load_snapshot(self) -> dict:
        import json
        with self._connect() as conn:
            row = conn.execute("SELECT value FROM snapshot WHERE key='state'").fetchone()
        return json.loads(row[0]) if row else {}

    # ------------------------------------------------------------------ #
    # Time-series tables                                                   #
    # ------------------------------------------------------------------ #

    def append_equity(self, ts: datetime, equity: float, cash: float) -> None:
        with self._connect() as conn:
            conn.execute(
                "INSERT OR REPLACE INTO equity_curve (ts, equity, cash) VALUES (?, ?, ?)",
                (ts.isoformat(), equity, cash),
            )

    def append_regime(self, ts: datetime, label: str, probability: float, is_confirmed: bool) -> None:
        with self._connect() as conn:
            conn.execute(
                "INSERT OR REPLACE INTO regime_history (ts, label, probability, is_confirmed) VALUES (?, ?, ?, ?)",
                (ts.isoformat(), label, probability, int(is_confirmed)),
            )

    def log_trade(
        self,
        ts: datetime,
        symbol: str,
        side: str,
        qty: float,
        price: float,
        order_id: Optional[str] = None,
        regime: Optional[str] = None,
        strategy: Optional[str] = None,
    ) -> None:
        with self._connect() as conn:
            conn.execute(
                "INSERT INTO trade_log (ts, symbol, side, qty, price, order_id, regime, strategy) "
                "VALUES (?, ?, ?, ?, ?, ?, ?, ?)",
                (ts.isoformat(), symbol, side, qty, price, order_id, regime, strategy),
            )

    def recent_equity(self, n: int = 252) -> List[dict]:
        with self._connect() as conn:
            rows = conn.execute(
                "SELECT ts, equity, cash FROM equity_curve ORDER BY ts DESC LIMIT ?", (n,)
            ).fetchall()
        return [{"ts": r[0], "equity": r[1], "cash": r[2]} for r in reversed(rows)]

    def recent_trades(self, n: int = 100) -> List[dict]:
        with self._connect() as conn:
            rows = conn.execute(
                "SELECT ts, symbol, side, qty, price, order_id, regime, strategy "
                "FROM trade_log ORDER BY id DESC LIMIT ?", (n,)
            ).fetchall()
        cols = ("ts", "symbol", "side", "qty", "price", "order_id", "regime", "strategy")
        return [dict(zip(cols, r)) for r in reversed(rows)]

    # ------------------------------------------------------------------ #
    # Private                                                              #
    # ------------------------------------------------------------------ #

    def _init_db(self) -> None:
        with self._connect() as conn:
            conn.executescript(_SCHEMA)

    @contextmanager
    def _connect(self) -> Iterator[sqlite3.Connection]:
        conn = sqlite3.connect(self._path, timeout=10)
        try:
            conn.execute("PRAGMA journal_mode=WAL")
            yield conn
            conn.commit()
        except Exception:
            conn.rollback()
            raise
        finally:
            conn.close()
