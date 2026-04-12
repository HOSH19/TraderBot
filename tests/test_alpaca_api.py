"""
Test your Alpaca API credentials and connectivity.
Run this first to confirm everything is set up correctly before running the bot.

Usage:
    python -m pytest tests/test_alpaca_api.py -v
    # or directly:
    python tests/test_alpaca_api.py
"""

import os
import sys

import pytest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


def _load_config():
    """Load the project settings.yaml config and return it as a dict."""
    import yaml
    cfg_path = os.path.join(os.path.dirname(__file__), "..", "config", "settings.yaml")
    with open(cfg_path) as f:
        return yaml.safe_load(f)


def _get_client():
    """Instantiate and return a live AlpacaClient using settings from the config file."""
    from broker.alpaca_client import AlpacaClient
    config = _load_config()
    return AlpacaClient(config)


class TestAlpacaCredentials:
    """Live connectivity tests for the Alpaca API; require valid credentials in .env."""

    def test_env_vars_present(self):
        """Check that API keys are set in environment."""
        from dotenv import load_dotenv
        load_dotenv()
        api_key = os.getenv("ALPACA_API_KEY", "")
        secret_key = os.getenv("ALPACA_SECRET_KEY", "")
        assert api_key and api_key != "your_key_here", (
            "ALPACA_API_KEY not set. Copy .env.example to .env and add your keys."
        )
        assert secret_key and secret_key != "your_secret_here", (
            "ALPACA_SECRET_KEY not set. Copy .env.example to .env and add your keys."
        )
        print(f"\n✓ API key found: {api_key[:4]}...{api_key[-4:]}")

    def test_client_connects(self):
        """Test that AlpacaClient initializes without error."""
        try:
            client = _get_client()
            assert client is not None
            print("\n✓ AlpacaClient initialized successfully")
        except ValueError as e:
            pytest.fail(f"Client connection failed: {e}")

    def test_health_check(self):
        """Verify account is active."""
        client = _get_client()
        healthy = client.health_check()
        assert healthy, "Alpaca health check failed — account may be inactive or credentials wrong"
        print("\n✓ Account is ACTIVE")

    def test_get_account(self):
        """Fetch account details and verify key fields."""
        client = _get_client()
        account = client.get_account()
        assert account is not None
        equity = float(account.equity)
        buying_power = float(account.buying_power)
        assert equity >= 0, "Account equity should be non-negative"
        print(f"\n✓ Account equity: ${equity:,.2f}")
        print(f"  Buying power:   ${buying_power:,.2f}")
        print(f"  Status:         {account.status}")
        print(f"  Paper trading:  {client.paper_trading}")

    def test_market_clock(self):
        """Check if market clock API is accessible."""
        client = _get_client()
        clock = client.get_clock()
        assert clock is not None
        print(f"\n✓ Market clock: is_open={clock.is_open}")
        print(f"  Next open:  {clock.next_open}")
        print(f"  Next close: {clock.next_close}")

    def test_get_positions(self):
        """Fetch current positions (may be empty for new accounts)."""
        client = _get_client()
        positions = client.get_positions()
        assert isinstance(positions, list)
        print(f"\n✓ Open positions: {len(positions)}")
        for p in positions:
            print(f"  {p.symbol}: {p.qty} shares @ ${p.avg_entry_price}")

    def test_historical_data(self):
        """Fetch historical bars for SPY."""
        from datetime import datetime, timedelta
        from data.market_data import MarketData
        client = _get_client()
        md = MarketData(client)
        bars = md.get_historical_bars(
            "SPY",
            timeframe="1Day",
            start=datetime.utcnow() - timedelta(days=30),
        )
        assert not bars.empty, "No historical data returned for SPY"
        assert "close" in bars.columns
        assert len(bars) >= 15, f"Expected at least 15 bars, got {len(bars)}"
        print(f"\n✓ Historical data for SPY: {len(bars)} bars")
        print(f"  Latest close: ${bars['close'].iloc[-1]:.2f}")
        print(f"  Date range: {bars.index[0].date()} → {bars.index[-1].date()}")

    def test_latest_quote(self):
        """Check bid/ask spread for SPY."""
        from data.market_data import MarketData
        client = _get_client()
        md = MarketData(client)
        quote = md.get_latest_quote("SPY")
        if quote:
            spread_pct = quote["spread_pct"] * 100
            print(f"\n✓ SPY quote: bid=${quote['bid']:.2f} ask=${quote['ask']:.2f} spread={spread_pct:.3f}%")
            assert spread_pct < 1.0, f"SPY spread {spread_pct:.3f}% seems unusually wide"
        else:
            pytest.skip("Could not fetch quote (market may be closed)")

    def test_paper_trading_mode(self):
        """Confirm paper trading mode is enabled (safety check)."""
        config = _load_config()
        assert config.get("broker", {}).get("paper_trading", True), (
            "paper_trading is set to False in settings.yaml. "
            "Make sure you intend to trade with real money."
        )
        print("\n✓ Paper trading mode confirmed")


if __name__ == "__main__":
    print("=" * 60)
    print("ALPACA API CONNECTION TEST")
    print("=" * 60)
    pytest.main([__file__, "-v", "--tb=short"])
