# TraderBot 🤖

A fully automated, HMM-based regime trading bot that trades US equities via Alpaca's paper/live API. Runs as a daily cron job and delivers a Telegram briefing every evening including regime analysis, portfolio state, orders placed, and top news headlines per ticker.

**Philosophy: risk management > signal generation.**

The edge is not in predicting market direction — it's in being fully invested during calm markets and reducing exposure during turbulent ones. When you cut your worst drawdowns in half, compounding works in your favour over time.

---

## How It Works

```
Every weekday at 4:05 PM ET (cron)
        │
        ▼
data/market_data.py         Fetch daily OHLCV bars from Alpaca (IEX feed)
        │
        ▼
data/feature_engineering.py Compute 14 OHLCV features, rolling z-score normalised
        │
        ▼
core/hmm_engine.py          Gaussian HMM — BIC model selection, forward algorithm
        │                   (no look-ahead bias), regime labelling
        ▼
core/regime_strategies.py   Volatility-rank → allocation size (always LONG)
        │
        ▼
core/risk_manager.py        Circuit breakers — absolute veto power over all orders
        │
        ▼
broker/                     Alpaca client, LIMIT/bracket orders, position tracker
        │
        ▼
monitoring/                 Structured JSON logs, Telegram daily briefing + news
```

**The HMM is a volatility classifier, not a price predictor.** It detects calm vs turbulent market environments. The strategy layer uses this to set portfolio allocation — fully invested in calm markets, reduced in turbulent ones.

**Always LONG, never SHORT.** V-shaped recoveries happen fast and the HMM is 2–3 days late detecting them. Shorting during rebounds wipes out crash gains.

---

## Features

- **Hidden Markov Model** — Gaussian HMM with BIC model selection (tests 3–7 regimes), manual forward algorithm to eliminate look-ahead bias, regime stability filter and flicker detection
- **14 engineered features** — log returns (1/5/20 day), realised volatility, vol ratio, volume z-score, ADX, SMA slope, RSI z-score, SMA200 distance, ROC, normalised ATR — all 252-period rolling z-score standardised
- **Walk-forward backtesting** — in-sample 252 days, out-of-sample 126 days, fill delay (signal day N, execute day N+1 open), slippage
- **Volatility-ranked allocation** — three strategy tiers mapped to HMM regime volatility rank, uncertainty mode halves position sizes
- **Circuit breakers** — daily drawdown halt, weekly drawdown halt, peak drawdown hard stop — independent veto power
- **Alpaca integration** — paper and live trading, LIMIT orders, bracket OCO orders, stop tightening, exponential backoff reconnection
- **Telegram daily briefing** — regime, portfolio P&L, signals, orders placed, top news headline per ticker (NewsAPI)
- **Cron-based** — no persistent server required; adapts automatically between market-open (full pipeline) and market-closed (summary only) runs

---

## Quick Start

### 1. Clone and create environment

```bash
git clone https://github.com/HOSH19/TraderBot.git
cd TraderBot

conda create -n regime-trader python=3.11 -y
conda activate regime-trader
pip install -r requirements.txt
```

### 2. Set up credentials

```bash
cp .env.example .env
```

Edit `.env` and fill in your keys:

```env
ALPACA_API_KEY=your_key_here
ALPACA_SECRET_KEY=your_secret_here
ALPACA_PAPER=true

TELEGRAM_BOT_TOKEN=your_bot_token
TELEGRAM_CHAT_ID=your_chat_id

NEWSAPI_KEY=your_newsapi_key  
```

- **Alpaca keys** — free paper trading account at [alpaca.markets](https://alpaca.markets) → Paper Trading → API Keys
- **Telegram bot** — message [@BotFather](https://t.me/BotFather) → `/newbot` → copy token. Then visit `https://api.telegram.org/bot<TOKEN>/getUpdates` to get your `chat_id`
- **NewsAPI key** — free tier (100 req/day) at [newsapi.org/register](https://newsapi.org/register)

### 3. Configure your tickers

Edit `config/settings.yaml`:

```yaml
broker:
  symbols: [AAPL, TSLA, GOOGL, NVDA, AMD]
  paper_trading: true
```

### 4. Test your setup

```bash
# Test Alpaca API connection
python -m pytest tests/test_alpaca_api.py -v

# Send a real Telegram message with live data
python tests/test_telegram.py
```

### 5. Set up the daily cron job

```bash
crontab -e
```

Add this line (runs Mon–Fri at 4:05 PM ET = 21:05 UTC):

```
5 21 * * 1-5 /path/to/conda/envs/regime-trader/bin/python /path/to/TraderBot/run_daily.py >> /path/to/TraderBot/logs/cron.log 2>&1
```

That's it. The bot runs automatically every weekday evening.

---

## Running Manually

```bash
# Full pipeline run (works any time — adapts to market hours)
python run_daily.py

# Walk-forward backtest
python main.py --backtest --symbols AAPL --start 2020-01-01 --end 2024-12-31

# Backtest with benchmark comparison
python main.py --backtest --symbols AAPL --start 2020-01-01 --end 2024-12-31 --compare

# Stress test
python main.py --stress-test --symbols AAPL --start 2020-01-01 --end 2024-12-31

# Train HMM only
python main.py --train-only
```

---

## What the Telegram Message Looks Like

**Weekday (market open / after close) — full briefing:**
```
🤖 HMM TRADER DAILY BRIEFING
📅 Monday, Apr 14 2026  |  📄 PAPER

📊 REGIME
🐂 BULL (98% confidence)
Stability: 5 bars

💼 PORTFOLIO
Equity: $100,000.00
📈 Daily P&L: +$320.00 (+0.32%)
From Peak: 0.0%  ✅
Circuit Breaker: NORMAL ✅

🎯 TODAY'S SIGNALS
• AAPL: LONG 25% @ $261.00  stop $248.95
• NVDA: LONG 20% @ $189.50  stop $180.52

📋 ORDERS PLACED
• AAPL: BUY 95 shares @ $261.26 (LIMIT)

📦 OPEN POSITIONS
• TSLA: 40 shares  📈 +2.3%  stop $335.00

📰 TOP NEWS
• AAPL: Apple Unveils New AI Features... — Bloomberg  2h ago
• TSLA: Down 30% From Highs, Should You Buy?... — Motley Fool  4h ago
...

Next run: tomorrow after market close
```
