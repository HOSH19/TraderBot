# Regime Trader

Automated US-equity regime detection and allocation control. A Hidden Markov Model classifies the market into volatility-ordered states; a strategy layer maps those states to target long exposure.

The HMM is a **volatility/environment classifier**, not a return forecaster. The stack is long-only.

---

## How it works

**1. Feature engineering** (`data/feature_engineering.py`)

14 OHLCV-derived features (returns, realized vol, ADX, RSI, ATR, etc.) plus 3 optional macro features, all passed through a 252-day rolling z-score. Every feature is strictly causal — no future information.

**Macro features** (fetched via `yfinance`):
- `macro_vix` — CBOE Volatility Index level
- `macro_yield_spread` — 10-year minus 3-month Treasury yield (curve steepness)
- `macro_credit_proxy` — HYG minus LQD daily log-return (credit stress, duration-neutral)

**2. Regime detection** (`core/hmm/`)

- **Emission model:** Gaussian or Student-t (configured via `emission_type`). Student-t uses degrees-of-freedom ν=4, which matches empirical equity tail thickness and reduces the influence of crash days on covariance estimates via auxiliary scale-mixture weights.
- **Model selection:** BIC over 3–7 hidden states with multiple random restarts.
- **Live inference:** forward algorithm only — Viterbi is never used for real-time probabilities to avoid look-ahead bias.
- **Stability filter:** a new state must persist for `stability_bars` bars before the confirmed regime flips.
- **Persistence:** model is retrained on a configurable schedule (`stale_max_days`) and committed back to the repo.

**3. Strategy layer** (`core/strategies/`)

Each HMM state maps to a low / mid / high volatility strategy template by volatility rank. These set target allocation and leverage caps. Orders are suppressed inside a `rebalance_threshold` deadband.

---

## Configuration

All tunable parameters live in `config/settings.yaml`:

```yaml
hmm:
  emission_type: student_t   # gaussian | student_t
  student_t_dof: 4
  use_macro_features: true
  n_candidates: [3, 4, 5, 6, 7]
  stale_max_days: 3
```
