# HMM Algo Trader

Automated US-equity algorithmic trading system. A Hidden Markov Model classifies the market into volatility-ordered regimes; a multi-layer strategy stack maps those regimes to target long exposure, confirmed by technical indicators, sized by Kelly Criterion, and protected by ATR-based trailing stops.

The HMM is a **volatility / environment classifier**, not a return forecaster. The system is **long-only**.

---

## Architecture

```
Market Data
    │
    ▼
Feature Engineering ──► HMM Regime Detection
                              │
                        Stability Filter
                              │
                        Regime State (label + probability)
                              │
                    ┌─────────┴──────────┐
                    ▼                    ▼
           Technical Signal       Strategy Selection
             Filter                (vol-tier mapping)
           (RSI/MACD/BB)               │
                    └─────────┬──────────┘
                              ▼
                        Signal (direction + size)
                              │
                        Kelly Sizer + Risk Manager
                              │
                        Order Executor (Alpaca)
                              │
                        ATR Stop Manager (GTC orders)
```

---

## Modules

### `data/`

| File | Class / Function | Responsibility |
|------|-----------------|----------------|
| `market_data.py` | `MarketData` | Thin facade — composes fetcher and stream |
| `historical_fetcher.py` | `HistoricalFetcher` | OHLCV download, gap repair, snapshots |
| `stream_manager.py` | `StreamManager` | Alpaca WebSocket bar stream, daemon thread |
| `feature_engineering.py` | `get_feature_matrix()` | Price-derived + macro features, z-scored |
| `macro_fetcher.py` | `fetch_macro_df()` | VIX, yield curve, credit proxy via yfinance |

### `core/hmm/`

| File | Class / Function | Responsibility |
|------|-----------------|----------------|
| `engine.py` | `HMMEngine` | Coordinator — trains, infers, persists |
| `model_selector.py` | `ModelSelector` | BIC selection across candidate state counts |
| `stability_filter.py` | `StabilityFilter` | Debounce regime switches, track flicker rate |
| `regime_metadata.py` | `build_regime_infos()` | Extract return/vol stats from Viterbi paths |
| `forward_algorithm.py` | `forward_pass()` | Normalized forward recursion (no look-ahead) |
| `persistence.py` | `save()` / `load()` | Pickle serialization for model checkpoint |
| `gaussian_model.py` | `GaussianHMMModel` | Gaussian emission wrapper over hmmlearn |
| `student_t_model.py` | `StudentTHMMModel` | Student-t emission with custom EM |

### `core/signals/`

| File | Class / Function | Responsibility |
|------|-----------------|----------------|
| `indicators.py` | `rsi()`, `macd()`, `bollinger()`, `atr()` | Pure vectorized indicator functions |
| `technical_filter.py` | `TechnicalSignalFilter` | Gate and scale signals by regime tier |

### `core/strategies/`

| File | Class | Responsibility |
|------|-------|----------------|
| `orchestrator.py` | `StrategyOrchestrator` | Route regime to strategy, apply technical confirmation |
| `low_vol_bull.py` | `LowVolBullStrategy` | High allocation, elevated leverage |
| `mid_vol_cautious.py` | `MidVolCautiousStrategy` | Allocation scales with trend |
| `high_vol_defensive.py` | `HighVolDefensiveStrategy` | Reduced allocation, 1.0× leverage |

### `core/risk/`

| File | Class | Responsibility |
|------|-------|----------------|
| `risk_manager.py` | `RiskManager` | Pipeline: circuit breakers → Kelly sizing → exposure caps |
| `kelly_sizer.py` | `KellySizer` | Half-Kelly with correlation-aware reduce/reject |
| `stop_manager.py` | `StopManager` | ATR trailing stops as live GTC orders on Alpaca |
| `circuit_breaker.py` | `CircuitBreaker` | Intraday / weekly / peak drawdown gates |

### `monitoring/`

| File | Class / Function | Responsibility |
|------|-----------------|----------------|
| `telegram_notifier.py` | `TelegramNotifier` | Thin HTTP sender to Telegram Bot API |
| `messages.py` | `daily_briefing_message()`, `market_summary_message()` | Pure HTML message builders |
| `dashboard.py` | `Dashboard` | Rich TUI — refresh throttle and render coordination |
| `panels.py` | `regime_panel()`, `portfolio_panel()`, … | Pure Rich panel builder functions |
| `alerts.py` | `AlertManager` | Rate-limited alerts (webhook / email) |
| `state_store.py` | `StateStore` | SQLite: snapshot, equity curve, regime history, trade log |
| `logger.py` | `setup_structured_logging()` | JSON rotating log handler |

### `backtest/`

| File | Class | Responsibility |
|------|-------|----------------|
| `walk_forward_backtester.py` | `WalkForwardBacktester` | Rolling IS train → OOS simulation |
| `stress_test.py` | — | Crash injection, gap risk, regime misclassification |
| `performance.py` | — | Sharpe, Calmar, drawdown, attribution report |

---

## Feature Engineering

All features are built from OHLCV in `data/feature_engineering.py` using only information available **at or before** each bar. Every column is passed through a **252-day rolling z-score**.

| Feature | Construction |
|---------|-------------|
| `ret_1`, `ret_5`, `ret_20` | Log returns over 1, 5, 20 days |
| `realized_vol` | 20-day rolling std of daily log returns |
| `vol_ratio` | Short (5d) vs long (20d) realized vol ratio |
| `vol_norm` | Volume z-score vs 50-day rolling mean/std |
| `vol_trend` | First difference of 10-day volume SMA |
| `adx` | 14-period ADX (trend strength) |
| `sma50_slope` | One-day change in 50-day SMA of close |
| `rsi_zscore` | RSI(14) as a rolling z-score |
| `dist_sma200` | Fractional distance of close from 200-day SMA |
| `roc_10`, `roc_20` | Rate of change over 10 and 20 days |
| `norm_atr` | ATR(14) / close |

**Macro features** (when `use_macro_features: true`):

| Feature | Source | Captures |
|---------|--------|---------|
| `macro_vix` | `^VIX` | Market fear / implied vol |
| `macro_yield_spread` | `^TNX − ^IRX` | Yield curve steepness |
| `macro_credit_proxy` | `HYG − LQD` log-return diff | Credit stress |

---

## HMM Regime Detection

**Emission models** — selectable via `emission_type` in config:

- **Gaussian** — standard multivariate Gaussian HMM via hmmlearn.
- **Student-t** — custom EM using the Gaussian scale-mixture representation:

  > Student-t(ν) = ∫ Gaussian(x | μ, Σ/τ) · Gamma(τ | ν/2, ν/2) dτ

  Each observation gets a per-state auxiliary weight E[τ_{t,k}] = (ν + d) / (ν + δ_{t,k}). Outliers (high Mahalanobis distance δ) receive lower weight, shrinking their influence on the covariance M-step. With ν=4, tails match empirical equity returns.

**Model selection** — `ModelSelector` fits each candidate state count (3–7) with multiple random restarts and picks the lowest BIC.

**Live inference** — `forward_pass()` runs the normalized forward recursion only. Viterbi is never used on the streaming boundary to prevent look-ahead bias.

**Stability filter** — `StabilityFilter` requires a new state to persist for `stability_bars` consecutive bars before the confirmed regime flips. Confirmed switches within `flicker_window` are counted; exceeding `flicker_threshold` halves position sizes.

**Labels** — at training time, Viterbi assigns each state a return-ranked human label (BEAR → STRONG\_BEAR → … → BULL → STRONG\_BULL → EUPHORIA). Volatility rank selects a strategy tier.

---

## Technical Signal Layer

`TechnicalSignalFilter` adds a second confirmation gate after the HMM regime, applied per-symbol inside `StrategyOrchestrator`:

| Regime tier | Signal type | Indicators used |
|------------|-------------|----------------|
| Low-vol / bull | Momentum | RSI(14) in [50, 75] AND MACD histogram positive and growing |
| Mid-vol / neutral | Mean-reversion | Price at or below Bollinger Band (20, 2σ) midline |
| High-vol / defensive | None | Pass-through — position is already defensive |

Confirmation returns a `strength` float in [0, 1] that directly scales the signal's `position_size_pct`. A failed confirmation drops the signal entirely (size = 0).

---

## Risk Layer

### Kelly Criterion sizing

`KellySizer` applies **half-Kelly** (0.5f) before existing risk caps:

- Default priors: win\_rate = 0.52, payoff\_ratio = 1.5 (conservative until backtest data is available).
- **Correlation cap:** computes 20-day return correlation against all existing positions.
  - ≥ `correlation_reduce_threshold` (0.70): size multiplied by (1 − correlation).
  - ≥ `correlation_reject_threshold` (0.85): signal rejected entirely.

### Circuit breaker

`CircuitBreaker` enforces four independent gates:

| Gate | Threshold | Action |
|------|-----------|--------|
| Daily drawdown soft | 2% | Reduce size 50% |
| Daily drawdown hard | 3% | Close all + halt |
| Weekly drawdown soft | 5% | Reduce size 50% |
| Weekly drawdown hard | 7% | Close all + halt |
| Peak drawdown | 10% | Halt + write lock file |

### ATR trailing stops

`StopManager` places a **live GTC StopOrder on Alpaca** for every open position so the stop survives process restarts:

- ATR multiplier varies by regime vol tier: 1.5× (low), 2.0× (mid), 3.0× (high).
- On each bar, if the new ATR stop is higher than the current stop, the order is replaced (trailing tighter). The stop never widens.

---

## State Persistence

`StateStore` (`monitoring/state_store.py`) replaces the flat `state_snapshot.json` with a **SQLite database** (`state.db`, WAL mode):

| Table | Contents |
|-------|---------|
| `snapshot` | Latest equity, cash, regime, circuit-breaker status — fast restart recovery |
| `equity_curve` | Timestamped equity + cash rows |
| `regime_history` | Regime label, probability, confirmation flag per bar |
| `trade_log` | Every submitted order: symbol, side, qty, price, order\_id, regime, strategy |

---

## Configuration

All tunable parameters in `config/settings.yaml`:

```yaml
hmm:
  emission_type: student_t   # gaussian | student_t
  student_t_dof: 4
  use_macro_features: true
  n_candidates: [3, 4, 5, 6, 7]
  stale_max_days: 3

technical:
  rsi_period: 14
  rsi_bull_min: 50
  rsi_bull_max: 75
  macd_fast: 12
  macd_slow: 26
  bb_period: 20
  bb_std: 2.0

risk:
  max_risk_per_trade: 0.01
  max_single_position: 0.15
  correlation_reduce_threshold: 0.70
  correlation_reject_threshold: 0.85
  daily_dd_halt: 0.03
  weekly_dd_halt: 0.07
  max_dd_from_peak: 0.10
```

---

## Running

```bash
# Live / paper trading loop
python main.py

# Dry run (no orders placed)
python main.py --dry-run

# Daily cron (trade if market open, else send Telegram summary)
python run_daily.py

# Walk-forward backtest
python main.py --backtest --symbols SPY --start 2019-01-01 --end 2024-12-31

# Stress tests (crash injection, gap risk, regime misclassification)
python main.py --backtest --stress-test --symbols SPY

# Train HMM and exit
python main.py --train-only

# Tests
python -m pytest tests/ -v
```

---

## Paper → Live Gate

| Gate | Requirement |
|------|------------|
| Backtest | Sharpe > 0.8, max drawdown < 20% out-of-sample |
| Paper trading | 30+ trading days, equity within 15% of backtest projection |
| Slippage audit | Simulated vs actual fills within 0.3% per trade |
| Circuit breaker | Manual trigger confirmed on paper |
| Live | Switch Alpaca keys from paper to live endpoint |
