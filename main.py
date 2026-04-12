"""
Regime Trader — Entry Point.

Usage:
  python main.py                        # Live/paper trading loop
  python main.py --dry-run              # Full pipeline, no orders placed
  python main.py --backtest --symbols SPY --start 2019-01-01 --end 2024-12-31
  python main.py --backtest --compare   # With benchmark comparisons
  python main.py --stress-test --symbols SPY --start 2019-01-01 --end 2024-12-31
  python main.py --train-only           # Train HMM and exit
  python main.py --dashboard            # Show dashboard for running instance
"""

import argparse
import json
import logging
import os
import signal
import sys
import time
from datetime import datetime, timedelta

import schedule
import yaml
from dotenv import load_dotenv

load_dotenv()

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, BASE_DIR)

STATE_SNAPSHOT_FILE = os.path.join(BASE_DIR, "state_snapshot.json")
HMM_MODEL_FILE = os.path.join(BASE_DIR, "hmm_model.pkl")
LOG_DIR = os.path.join(BASE_DIR, "logs")


def load_config() -> dict:
    """Load and return the YAML application config from config/settings.yaml."""
    cfg_path = os.path.join(BASE_DIR, "config", "settings.yaml")
    with open(cfg_path) as f:
        return yaml.safe_load(f)


def setup_logging(config: dict):
    """Initialize structured logging using settings from the application config."""
    os.makedirs(LOG_DIR, exist_ok=True)
    from monitoring.logger import setup_structured_logging
    setup_structured_logging(config)


def load_or_train_hmm(config: dict, market_data, symbols: list):
    """
    Return a ready-to-use HMMEngine, loading from disk or training from scratch.

    Loads the saved model if it exists and is fresh (≤7 days old). Otherwise
    fetches ~3 years of bars for the primary symbol and trains a new model, saving it.
    """
    from core.hmm_engine import HMMEngine

    hmm = HMMEngine(config.get("hmm", {}))
    primary_symbol = symbols[0]

    if os.path.exists(HMM_MODEL_FILE):
        hmm.load(HMM_MODEL_FILE)
        if not hmm.is_stale(max_days=7):
            logging.getLogger(__name__).info(f"Loaded HMM from {HMM_MODEL_FILE}")
            return hmm
        logging.getLogger(__name__).info("HMM model is stale (>7 days), retraining...")

    logging.getLogger(__name__).info(f"Training HMM on {primary_symbol}...")
    bars = market_data.get_historical_bars(
        primary_symbol,
        timeframe=config.get("broker", {}).get("timeframe", "1Day"),
        start=datetime.utcnow() - timedelta(days=1200),
    )
    if bars.empty:
        raise RuntimeError(f"No historical data returned for {primary_symbol}")

    hmm.train(bars)
    hmm.save(HMM_MODEL_FILE)
    return hmm


def save_state_snapshot(portfolio_state, regime_state):
    """Persist current portfolio and regime state to state_snapshot.json."""
    try:
        snap = {
            "timestamp": datetime.utcnow().isoformat(),
            "equity": portfolio_state.equity,
            "cash": portfolio_state.cash,
            "circuit_breaker_status": portfolio_state.circuit_breaker_status,
            "regime": regime_state.label if regime_state else "UNKNOWN",
            "regime_prob": float(regime_state.probability) if regime_state else 0.0,
        }
        with open(STATE_SNAPSHOT_FILE, "w") as f:
            json.dump(snap, f, indent=2)
    except Exception as e:
        logging.getLogger(__name__).error(f"save_state_snapshot failed: {e}")


def load_state_snapshot() -> dict:
    """Load the persisted state snapshot from disk, returning empty dict if missing or corrupt."""
    if os.path.exists(STATE_SNAPSHOT_FILE):
        try:
            with open(STATE_SNAPSHOT_FILE) as f:
                return json.load(f)
        except Exception:
            pass
    return {}


def print_session_summary(portfolio_state, start_time: datetime):
    """Print a summary of session duration, final equity, daily P&L, and circuit-breaker status."""
    elapsed = (datetime.utcnow() - start_time).total_seconds() / 3600
    print(f"\n{'='*50}")
    print(f"SESSION SUMMARY")
    print(f"Duration: {elapsed:.1f}h")
    print(f"Final equity: ${portfolio_state.equity:,.2f}")
    print(f"Daily P&L: ${portfolio_state.daily_pnl:+,.2f}")
    print(f"Circuit breaker: {portfolio_state.circuit_breaker_status}")
    print(f"{'='*50}\n")


def run_trading_loop(config: dict, dry_run: bool = False):
    """
    Run the live (or dry-run) trading loop, subscribing to real-time bar data.

    Initializes all components, streams bars via Alpaca, processes regime signals on
    each bar, manages circuit breakers, submits orders, and handles graceful shutdown
    on SIGINT/SIGTERM. Blocks until shutdown is requested.

    Args:
        config: Full application config dict.
        dry_run: If True, all order logic runs but no orders are actually submitted.
    """
    logger = logging.getLogger(__name__)
    from broker.alpaca_client import AlpacaClient
    from broker.order_executor import OrderExecutor
    from broker.position_tracker import PositionTracker
    from core.hmm_engine import HMMEngine
    from core.regime_strategies import StrategyOrchestrator
    from core.risk_manager import PortfolioState, RiskManager
    from core.signal_generator import SignalGenerator
    from data.market_data import MarketData
    from monitoring.alerts import AlertManager
    from monitoring.dashboard import Dashboard

    symbols = config.get("broker", {}).get("symbols", ["SPY"])
    timeframe = config.get("broker", {}).get("timeframe", "1Day")
    start_time = datetime.utcnow()
    last_regime_state = None
    shutdown_requested = False

    alpaca = AlpacaClient(config)

    if not alpaca.health_check():
        raise ConnectionError("Alpaca health check failed on startup")

    if not alpaca.is_market_open() and not dry_run:
        logger.info("Market is closed. Waiting for next open...")

    market_data = MarketData(alpaca)

    portfolio = PortfolioState(
        equity=0.0, cash=0.0, buying_power=0.0
    )
    position_tracker = PositionTracker(alpaca, portfolio)
    position_tracker.sync_from_alpaca()

    prev_snapshot = load_state_snapshot()
    if prev_snapshot:
        logger.info(f"Recovered state: equity=${prev_snapshot.get('equity', 0):,.2f} "
                    f"regime={prev_snapshot.get('regime', 'UNKNOWN')}")

    hmm = load_or_train_hmm(config, market_data, symbols)
    orchestrator = StrategyOrchestrator(config.get("strategy", {}), hmm.regime_infos)
    signal_gen = SignalGenerator(hmm, orchestrator, config)
    risk_manager = RiskManager(config)
    order_executor = OrderExecutor(alpaca, dry_run=dry_run)
    alert_manager = AlertManager(config)
    dashboard = Dashboard(config)

    bars_by_symbol: dict = {}
    for sym in symbols:
        df = market_data.get_historical_bars(
            sym, timeframe=timeframe,
            start=datetime.utcnow() - timedelta(days=1200),
        )
        if not df.empty:
            bars_by_symbol[sym] = df

    def on_bar(bar):
        """Process an incoming real-time bar: update history, detect regime, validate signals, and place orders."""
        nonlocal last_regime_state
        sym = bar.symbol if hasattr(bar, "symbol") else list(bars_by_symbol.keys())[0]

        if sym in bars_by_symbol:
            new_row = {
                "open": bar.open, "high": bar.high,
                "low": bar.low, "close": bar.close, "volume": bar.volume,
            }
            import pandas as pd
            new_df = pd.DataFrame([new_row], index=[bar.timestamp])
            bars_by_symbol[sym] = pd.concat([bars_by_symbol[sym], new_df])
        else:
            return

        if sym != symbols[0]:
            return

        current_allocations = {
            s: p.shares * p.current_price / portfolio.equity
            for s, p in portfolio.positions.items()
            if portfolio.equity > 0
        }

        signals, regime_state = signal_gen.generate(symbols, bars_by_symbol, current_allocations)
        last_regime_state = regime_state

        if regime_state:
            logger.info(
                f"Regime: {regime_state.label} (p={regime_state.probability:.2f}, "
                f"confirmed={regime_state.is_confirmed}, bars={regime_state.consecutive_bars})"
            )
            alert_manager.on_regime_state(regime_state, last_regime_state)

        cb_action, cb_reason = risk_manager.circuit_breaker.update(portfolio)
        if cb_action in ("CLOSE_ALL_DAY", "CLOSE_ALL_WEEK", "HALTED"):
            logger.warning(f"Circuit breaker triggered: {cb_action} — {cb_reason}")
            alert_manager.send("circuit_breaker", f"Circuit breaker: {cb_action} — {cb_reason}")
            if not dry_run:
                order_executor.close_all_positions()
            return

        for sig in signals:
            risk_decision = risk_manager.validate_signal(sig, portfolio)
            if risk_decision.approved:
                order_id = order_executor.submit_order(sig, risk_decision)
                if order_id:
                    logger.info(
                        f"Order submitted: {sig.symbol} {sig.direction} "
                        f"alloc={sig.position_size_pct*100:.0f}% lev={sig.leverage}x | {sig.reasoning}"
                    )
            else:
                logger.info(f"Signal rejected: {sig.symbol} — {risk_decision.rejection_reason}")

        for sym, pos in portfolio.positions.items():
            if sym in bars_by_symbol and len(bars_by_symbol[sym]) > 0:
                current_price = float(bars_by_symbol[sym]["close"].iloc[-1])
                position_tracker.update_position_price(sym, current_price)

        save_state_snapshot(portfolio, regime_state)
        dashboard.refresh(portfolio, regime_state, hmm, signals)

    def handle_shutdown(signum, frame):
        """Handle SIGINT/SIGTERM by stopping the stream, saving state, and exiting cleanly."""
        nonlocal shutdown_requested
        shutdown_requested = True
        logger.info("Shutdown signal received")
        market_data.stop_stream()
        save_state_snapshot(portfolio, last_regime_state)
        print_session_summary(portfolio, start_time)
        sys.exit(0)

    signal.signal(signal.SIGINT, handle_shutdown)
    signal.signal(signal.SIGTERM, handle_shutdown)

    def weekly_retrain():
        """Retrain the HMM model weekly and propagate updated regime info to downstream components."""
        nonlocal hmm, orchestrator
        logger.info("Weekly HMM retrain starting...")
        try:
            new_hmm = load_or_train_hmm(config, market_data, symbols)
            hmm = new_hmm
            orchestrator.update_regime_infos(hmm.regime_infos)
            signal_gen.hmm = hmm
            alert_manager.send("hmm_retrained", f"HMM retrained: n={hmm.n_regimes} BIC={hmm.bic_score:.1f}")
            logger.info("Weekly retrain complete")
        except Exception as e:
            logger.error(f"Weekly retrain failed: {e}")

    schedule.every().monday.at("09:00").do(weekly_retrain)
    schedule.every().day.at("00:00").do(risk_manager.reset_daily_counters)
    schedule.every().monday.at("00:00").do(risk_manager.reset_weekly_counters)
    schedule.every().monday.at("00:00").do(position_tracker.reset_weekly)
    schedule.every().day.at("00:00").do(position_tracker.reset_daily)

    market_data.subscribe_bars(symbols, timeframe, on_bar)
    logger.info("System online")
    print(f"\n{'='*50}\nREGIME TRADER ONLINE\nSymbols: {symbols}\nMode: {'DRY-RUN' if dry_run else 'PAPER' if config.get('broker',{}).get('paper_trading') else 'LIVE'}\n{'='*50}\n")

    while not shutdown_requested:
        schedule.run_pending()
        time.sleep(1)


def run_backtest(config: dict, args):
    """
    Fetch historical data, run the walk-forward backtester, and print results.

    Optionally runs stress tests and benchmark comparisons based on CLI args.
    Saves equity curve, trade log, and regime history CSVs to backtest_results/.

    Args:
        config: Full application config dict.
        args: Parsed argparse namespace with symbols, start, end, stress_test, and compare flags.
    """
    from broker.alpaca_client import AlpacaClient
    from backtest.backtester import WalkForwardBacktester
    from backtest.performance import print_report
    from backtest.stress_test import run_crash_injection, run_gap_risk, run_regime_misclassification, print_stress_report
    from data.market_data import MarketData

    logger = logging.getLogger(__name__)
    symbols = args.symbols or config.get("broker", {}).get("symbols", ["SPY"])
    start = datetime.fromisoformat(args.start) if args.start else datetime(2019, 1, 1)
    end = datetime.fromisoformat(args.end) if args.end else datetime(2024, 12, 31)

    alpaca = AlpacaClient(config)
    market_data = MarketData(alpaca)

    for symbol in symbols:
        logger.info(f"Running backtest for {symbol} {start.date()} → {end.date()}")
        bars = market_data.get_historical_bars(symbol, start=start, end=end)

        if bars.empty:
            logger.error(f"No data for {symbol}")
            continue

        backtester = WalkForwardBacktester(config)
        result = backtester.run(symbol, bars)

        print_report(result, bars, config.get("backtest", {}).get("risk_free_rate", 0.045))

        if args.stress_test:
            crash = run_crash_injection(bars, config)
            gap = run_gap_risk(bars, config)
            misclass = run_regime_misclassification(bars, config)
            print_stress_report(crash, gap, misclass)

        if hasattr(args, "compare") and args.compare:
            from backtest.performance import buy_and_hold_benchmark, sma200_benchmark, random_allocation_benchmark
            initial_capital = config.get("backtest", {}).get("initial_capital", 100_000)
            bah = buy_and_hold_benchmark(bars, initial_capital)
            sma = sma200_benchmark(bars, initial_capital)
            rand = random_allocation_benchmark(bars, initial_capital=initial_capital)
            print(f"\nBenchmarks for {symbol}:")
            print(f"  Buy & hold: {(bah.iloc[-1]/bah.iloc[0]-1)*100:.2f}%")
            print(f"  200 SMA:    {(sma.iloc[-1]/sma.iloc[0]-1)*100:.2f}%")
            print(f"  Random:     {rand['mean']*100:.2f}% ± {rand['std']*100:.2f}%")

        import csv, os
        out_dir = os.path.join(BASE_DIR, "backtest_results")
        os.makedirs(out_dir, exist_ok=True)
        result.equity_curve.to_csv(os.path.join(out_dir, f"{symbol}_equity_curve.csv"))
        if result.trade_log:
            import dataclasses
            with open(os.path.join(out_dir, f"{symbol}_trade_log.csv"), "w", newline="") as f:
                writer = csv.DictWriter(f, fieldnames=dataclasses.fields(result.trade_log[0]))
                writer.writeheader()
                for trade in result.trade_log:
                    writer.writerow(dataclasses.asdict(trade))
        if not result.regime_history.empty:
            result.regime_history.to_csv(os.path.join(out_dir, f"{symbol}_regime_history.csv"))
        logger.info(f"Results saved to {out_dir}/")


def main():
    """Parse CLI arguments and dispatch to the appropriate mode (backtest, train, dashboard, or live)."""
    parser = argparse.ArgumentParser(description="Regime Trader Bot")
    parser.add_argument("--dry-run", action="store_true", help="Full pipeline, no orders placed")
    parser.add_argument("--backtest", action="store_true", help="Run walk-forward backtester")
    parser.add_argument("--train-only", action="store_true", help="Train HMM and exit")
    parser.add_argument("--stress-test", action="store_true", help="Run stress tests")
    parser.add_argument("--compare", action="store_true", help="Include benchmark comparisons")
    parser.add_argument("--dashboard", action="store_true", help="Show dashboard for running instance")
    parser.add_argument("--symbols", nargs="+", help="Symbols to trade/backtest")
    parser.add_argument("--start", type=str, help="Backtest start date (YYYY-MM-DD)")
    parser.add_argument("--end", type=str, help="Backtest end date (YYYY-MM-DD)")
    args = parser.parse_args()

    config = load_config()
    setup_logging(config)
    logger = logging.getLogger(__name__)

    if args.backtest or args.stress_test:
        run_backtest(config, args)
        return

    if args.train_only:
        from broker.alpaca_client import AlpacaClient
        from data.market_data import MarketData
        symbols = args.symbols or config.get("broker", {}).get("symbols", ["SPY"])
        alpaca = AlpacaClient(config)
        market_data = MarketData(alpaca)
        hmm = load_or_train_hmm(config, market_data, symbols)
        logger.info(f"Training complete: n={hmm.n_regimes} BIC={hmm.bic_score:.2f}")
        return

    if args.dashboard:
        from monitoring.dashboard import Dashboard
        import json
        snap = load_state_snapshot()
        print(json.dumps(snap, indent=2))
        return

    try:
        run_trading_loop(config, dry_run=args.dry_run)
    except KeyboardInterrupt:
        logger.info("Interrupted by user")
    except Exception as e:
        import traceback
        logger.critical(f"Unhandled exception: {e}\n{traceback.format_exc()}")
        try:
            from monitoring.alerts import AlertManager
            alert_mgr = AlertManager(config)
            alert_mgr.send("system_error", f"Unhandled exception: {e}")
        except Exception:
            pass
        sys.exit(1)


if __name__ == "__main__":
    main()
