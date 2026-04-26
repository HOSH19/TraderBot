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

from core.timeutil import utc_now


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

    def test_client_connects(self):
        """Test that AlpacaClient initializes without error."""
        try:
            client = _get_client()
            assert client is not None
        except ValueError as e:
            pytest.fail(f"Client connection failed: {e}")

    def test_health_check(self):
        """Verify account is active."""
        client = _get_client()
        healthy = client.health_check()
        assert healthy, "Alpaca health check failed — account may be inactive or credentials wrong"

    def test_get_account(self):
        """Fetch account details and verify key fields."""
        client = _get_client()
        account = client.get_account()
        assert account is not None
        equity = float(account.equity)
        buying_power = float(account.buying_power)
        assert equity >= 0, "Account equity should be non-negative"
        assert buying_power >= 0

    def test_market_clock(self):
        """Check if market clock API is accessible."""
        client = _get_client()
        clock = client.get_clock()
        assert clock is not None

    def test_get_positions(self):
        """Fetch current positions (may be empty for new accounts)."""
        client = _get_client()
        positions = client.get_positions()
        assert isinstance(positions, list)

    def test_historical_data(self):
        """Fetch historical bars for SPY."""
        from datetime import timedelta
        from data.market_data import MarketData
        client = _get_client()
        md = MarketData(client)
        bars = md.get_historical_bars(
            "SPY",
            timeframe="1Day",
            start=utc_now() - timedelta(days=30),
        )
        assert not bars.empty, "No historical data returned for SPY"
        assert "close" in bars.columns
        assert len(bars) >= 15, f"Expected at least 15 bars, got {len(bars)}"

    def test_latest_quote(self):
        """Check bid/ask spread for SPY."""
        from data.market_data import MarketData
        client = _get_client()
        md = MarketData(client)
        quote = md.get_latest_quote("SPY")
        if quote:
            spread_pct = quote["spread_pct"] * 100
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


if __name__ == "__main__":
    pytest.main([__file__, "-q", "--tb=short"])
