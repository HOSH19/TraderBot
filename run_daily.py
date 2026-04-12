"""
Daily cron entry point for Regime Trader.

Run this via cron OR manually at any time.

Behavior:
  - Market OPEN / weekday after close  → full pipeline: analyze, generate signals, place orders, briefing
  - Market CLOSED (weekend/holiday)    → analysis only: regime + stock prices, no orders, summary to Telegram

Cron schedule (runs at 4:05 PM ET = 21:05 UTC, Mon–Fri):
  5 21 * * 1-5 /path/to/venv/bin/python /path/to/regime-trader/run_daily.py

Run manually any time (e.g. on a weekend for a quick status check):
  python run_daily.py
"""

import json
import logging
import os
import sys
import traceback
from datetime import datetime, timedelta

import yaml
from dotenv import load_dotenv

load_dotenv()

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, BASE_DIR)

HMM_MODEL_FILE = os.path.join(BASE_DIR, "hmm_model.pkl")
STATE_SNAPSHOT_FILE = os.path.join(BASE_DIR, "state_snapshot.json")
LOG_DIR = os.path.join(BASE_DIR, "logs")


def load_config() -> dict:
    """Load and return the YAML settings from config/settings.yaml."""
    with open(os.path.join(BASE_DIR, "config", "settings.yaml")) as f:
        return yaml.safe_load(f)


def setup_logging(config: dict):
    """Initialize structured logging using the application config."""
    os.makedirs(LOG_DIR, exist_ok=True)
    from monitoring.logger import setup_structured_logging
    setup_structured_logging(config)


def _load_prev_snapshot() -> dict:
    """Load the last persisted state snapshot from disk, returning empty dict on failure."""
    if os.path.exists(STATE_SNAPSHOT_FILE):
        try:
            with open(STATE_SNAPSHOT_FILE) as f:
                return json.load(f)
        except Exception:
            pass
    return {}


def _save_snapshot(portfolio, regime_state, prev_snapshot: dict):
    """
    Persist portfolio and regime state to disk as a JSON snapshot.

    Carries forward the peak_equity, daily_start_equity, and weekly_start_equity
    values from the previous snapshot when they are not yet reset.
    """
    snapshot = {
        "timestamp": datetime.utcnow().isoformat(),
        "equity": portfolio.equity,
        "cash": portfolio.cash,
        "peak_equity": max(portfolio.equity, prev_snapshot.get("peak_equity", portfolio.equity)),
        "daily_start_equity": portfolio.daily_start_equity or portfolio.equity,
        "weekly_start_equity": portfolio.weekly_start_equity or portfolio.equity,
        "circuit_breaker_status": portfolio.circuit_breaker_status,
        "regime": regime_state.label if regime_state else "UNKNOWN",
        "regime_prob": float(regime_state.probability) if regime_state else 0.0,
    }
    with open(STATE_SNAPSHOT_FILE, "w") as f:
        json.dump(snapshot, f, indent=2)


def _fetch_bars(market_data, symbols, timeframe, logger):
    """
    Fetch ~10 years of historical OHLCV bars for each symbol.

    Returns a dict mapping symbol -> DataFrame. Symbols with no data are skipped.
    """
    bars_by_symbol = {}
    for sym in symbols:
        df = market_data.get_historical_bars(
            sym,
            timeframe=timeframe,
            start=datetime.utcnow() - timedelta(days=2500),
        )
        if not df.empty:
            bars_by_symbol[sym] = df
            logger.info(f"{sym}: {len(df)} bars, latest close ${df['close'].iloc[-1]:.2f}")
        else:
            logger.warning(f"No data for {sym}, skipping")
    return bars_by_symbol


def _load_or_train_hmm(config, primary_bars, market_open: bool, logger):
    """
    Load the saved HMM model or train a new one if absent or stale.

    Retraining only occurs on market-open runs; weekend/closed runs reuse a stale
    model for status-check purposes.
    """
    from core.hmm_engine import HMMEngine
    hmm = HMMEngine(config.get("hmm", {}))

    if os.path.exists(HMM_MODEL_FILE):
        hmm.load(HMM_MODEL_FILE)
        if hmm.is_stale(max_days=7) and market_open:
            logger.info("HMM stale — retraining (this takes ~2 mins)...")
            hmm.train(primary_bars)
            hmm.save(HMM_MODEL_FILE)
            logger.info("HMM retrained and saved")
        elif hmm.is_stale(max_days=7):
            logger.info(f"HMM is stale but market is closed — using existing model for status check")
    else:
        logger.info("No saved model — training from scratch (this takes ~2 mins)...")
        hmm.train(primary_bars)
        hmm.save(HMM_MODEL_FILE)
        logger.info("HMM trained and saved")

    return hmm


def _stock_price_summary(bars_by_symbol: dict) -> list:
    """Compute last close and weekly change for each symbol."""
    rows = []
    for sym, df in bars_by_symbol.items():
        if len(df) < 2:
            continue
        close = float(df["close"].iloc[-1])
        week_ago_close = float(df["close"].iloc[-6]) if len(df) >= 6 else float(df["close"].iloc[0])
        week_chg = (close - week_ago_close) / week_ago_close * 100
        rows.append({"symbol": sym, "close": close, "week_chg_pct": week_chg})
    return rows


def run():
    """
    Execute the daily trading pipeline.

    When the market is open (or it's a weekday after-close run), runs the full pipeline:
    regime detection, signal generation, risk checks, order placement, and Telegram briefing.
    When the market is closed (weekend/holiday), sends a summary-only Telegram message with
    regime and portfolio status and skips all order placement.
    """
    config = load_config()
    setup_logging(config)
    logger = logging.getLogger(__name__)

    symbols = config.get("broker", {}).get("symbols", ["SPY"])
    paper_trading = config.get("broker", {}).get("paper_trading", True)
    timeframe = config.get("broker", {}).get("timeframe", "1Day")

    logger.info(f"=== Run starting: {datetime.utcnow().isoformat()} ===")

    from broker.alpaca_client import AlpacaClient
    from broker.order_executor import OrderExecutor
    from broker.position_tracker import PositionTracker
    from core.regime_strategies import StrategyOrchestrator
    from core.risk_manager import PortfolioState, RiskManager
    from core.signal_generator import SignalGenerator
    from data.market_data import MarketData
    from data.news_fetcher import fetch_news_for_symbols
    from monitoring.telegram_notifier import TelegramNotifier

    telegram = TelegramNotifier()

    try:
        alpaca = AlpacaClient(config)
        if not alpaca.health_check():
            raise ConnectionError("Alpaca health check failed")

        clock = alpaca.get_clock()
        market_open = clock.is_open
        next_open_str = clock.next_open.strftime("%a %b %d %H:%M ET") if clock.next_open else "unknown"

        if market_open:
            market_status = "OPEN"
        else:
            now = datetime.utcnow()
            market_status = "CLOSED (weekend)" if now.weekday() >= 5 else "CLOSED (after hours)"

        logger.info(f"Market status: {market_status} | Next open: {next_open_str}")

        market_data = MarketData(alpaca)

        bars_by_symbol = _fetch_bars(market_data, symbols, timeframe, logger)
        if not bars_by_symbol:
            raise RuntimeError("No market data fetched for any symbol")

        primary_bars = bars_by_symbol[symbols[0]]

        hmm = _load_or_train_hmm(config, primary_bars, market_open, logger)

        regime_state = hmm.predict_regime_filtered(primary_bars)
        is_flickering = hmm.is_flickering()
        hmm_age_days = (datetime.utcnow() - hmm.training_date).days if hmm.training_date else 0

        logger.info(
            f"Regime: {regime_state.label} ({regime_state.probability*100:.0f}%) "
            f"stability={regime_state.consecutive_bars}bars flickering={is_flickering}"
        )

        account = alpaca.get_account()
        equity = float(account.equity)
        prev_snapshot = _load_prev_snapshot()

        portfolio = PortfolioState(equity=equity, cash=float(account.cash), buying_power=float(account.buying_power))
        portfolio.peak_equity = max(equity, prev_snapshot.get("peak_equity", equity))
        portfolio.daily_start_equity = prev_snapshot.get("daily_start_equity", equity)
        portfolio.weekly_start_equity = prev_snapshot.get("weekly_start_equity", equity)

        position_tracker = PositionTracker(alpaca, portfolio)
        position_tracker.sync_from_alpaca()

        positions_list = [
            {
                "symbol": sym,
                "shares": int(pos.shares),
                "pnl_pct": pos.unrealized_pnl_pct * 100,
                "stop": pos.stop_loss,
            }
            for sym, pos in portfolio.positions.items()
        ]

        logger.info("Fetching news headlines...")
        news = fetch_news_for_symbols(symbols)

        if not market_open:
            logger.info("Market closed — sending summary, skipping order placement")
            stock_prices = _stock_price_summary(bars_by_symbol)
            sent = telegram.send_market_summary(
                date=datetime.utcnow(),
                market_status=market_status,
                next_open=next_open_str,
                regime_label=regime_state.label,
                regime_prob=regime_state.probability,
                regime_stability=hmm.get_regime_stability(),
                is_flickering=is_flickering,
                equity=equity,
                peak_dd_pct=portfolio.drawdown_from_peak * 100,
                positions=positions_list,
                stock_prices=stock_prices,
                paper_trading=paper_trading,
                hmm_age_days=hmm_age_days,
                news=news,
            )
            if telegram.enabled and not sent:
                logger.error(
                    "Telegram summary was not delivered. Check logs for 'Telegram send failed' — "
                    "token/chat_id in GitHub secrets (same Environment as the workflow) or revoke/regenerate bot."
                )
            logger.info("=== Market closed run complete ===")
            return

        risk_manager = RiskManager(config)
        cb_action, cb_reason = risk_manager.circuit_breaker.check(portfolio)
        if cb_action in ("HALTED", "CLOSE_ALL_DAY", "CLOSE_ALL_WEEK"):
            logger.warning(f"Circuit breaker: {cb_action} — {cb_reason}")
            telegram.send_alert("circuit_breaker", f"{cb_action}: {cb_reason}")
            _save_snapshot(portfolio, regime_state, prev_snapshot)
            return

        orchestrator = StrategyOrchestrator(config.get("strategy", {}), hmm.regime_infos)
        signal_gen = SignalGenerator(hmm, orchestrator, config)

        current_allocations = {
            sym: pos.shares * pos.current_price / equity
            for sym, pos in portfolio.positions.items() if equity > 0
        }

        signals, _ = signal_gen.generate(symbols, bars_by_symbol, current_allocations)
        logger.info(f"Generated {len(signals)} signal(s)")

        order_executor = OrderExecutor(alpaca, dry_run=False)
        signal_dicts = []
        orders_placed = []

        for sig in signals:
            risk_decision = risk_manager.validate_signal(sig, portfolio)
            if risk_decision.approved and risk_decision.modified_signal:
                s = risk_decision.modified_signal
                order_id = order_executor.submit_order(sig, risk_decision)
                if order_id:
                    qty = int(equity * s.position_size_pct * s.leverage / s.entry_price)
                    orders_placed.append({
                        "symbol": s.symbol, "side": "BUY", "qty": qty,
                        "price": round(s.entry_price * 1.001, 2), "order_id": order_id,
                    })
                signal_dicts.append({
                    "symbol": s.symbol, "direction": s.direction,
                    "alloc_pct": s.position_size_pct * 100,
                    "entry": s.entry_price, "stop": s.stop_loss,
                })
            else:
                logger.info(f"Signal rejected: {sig.symbol} — {risk_decision.rejection_reason}")

        _save_snapshot(portfolio, regime_state, prev_snapshot)

        daily_pnl = equity - (portfolio.daily_start_equity or equity)
        daily_pnl_pct = (daily_pnl / portfolio.daily_start_equity * 100) if portfolio.daily_start_equity else 0.0

        sent = telegram.send_daily_briefing(
            date=datetime.utcnow(),
            regime_label=regime_state.label,
            regime_prob=regime_state.probability,
            regime_stability=hmm.get_regime_stability(),
            is_flickering=is_flickering,
            equity=equity,
            daily_pnl=daily_pnl,
            daily_pnl_pct=daily_pnl_pct,
            peak_dd_pct=portfolio.drawdown_from_peak * 100,
            circuit_breaker=portfolio.circuit_breaker_status,
            signals=signal_dicts,
            orders_placed=orders_placed,
            positions=positions_list,
            paper_trading=paper_trading,
            news=news,
        )
        if telegram.enabled and not sent:
            logger.error(
                "Telegram briefing was not delivered. Check logs for 'Telegram send failed' — "
                "token/chat_id in GitHub secrets (same Environment as the workflow)."
            )

        logger.info(f"=== Daily run complete. Orders placed: {len(orders_placed)} ===")

    except Exception as e:
        logger.critical(f"Run failed: {e}\n{traceback.format_exc()}")
        telegram.send_alert("error", f"Run failed:\n{str(e)}")
        sys.exit(1)


if __name__ == "__main__":
    run()
