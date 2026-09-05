# Quant Lab

**Quant Lab** is a public research portfolio: small, testable Python modules for
systematic trading *ideas* — backtests, risk, options, pairs, and market-making —
shipped with offline notebooks. It is extracted from a private research monorepo
and deliberately stripped of live brokers, runners, and ops.

> **Disclaimer.** This repository is for **research and education only**.
> Nothing here is financial advice. Nothing here is a live trading system.
> Production broker clients, order routers, and daily runners are
> **intentionally excluded** (see [`docs/SANITIZATION.md`](docs/SANITIZATION.md)).

---

## Project checklist

| # | Project | Core modules | Notebook |
|---|---------|--------------|----------|
| 1 | **Momentum breakout backtest** | [`strategies/momentum.py`](src/quant_lab/strategies/momentum.py), [`backtest/engine.py`](src/quant_lab/backtest/engine.py) | [`notebooks/01_momentum_backtest.ipynb`](notebooks/01_momentum_backtest.ipynb) |
| 2 | **Options pricing (Black–Scholes)** | [`options/black_scholes.py`](src/quant_lab/options/black_scholes.py) | [`notebooks/02_options_pricing.ipynb`](notebooks/02_options_pricing.ipynb) |
| 3 | **Volatility targeting / sizing** | [`risk/sizing.py`](src/quant_lab/risk/sizing.py), [`risk/metrics.py`](src/quant_lab/risk/metrics.py) | [`notebooks/03_volatility_targeting.ipynb`](notebooks/03_volatility_targeting.ipynb) |
| 4 | **Pairs trading** | [`pairs/`](src/quant_lab/pairs/) (cointegration, spread, signals) | [`notebooks/04_pairs_trading.ipynb`](notebooks/04_pairs_trading.ipynb) |
| 5 | **Market making** | [`microstructure/market_maker.py`](src/quant_lab/microstructure/market_maker.py) | [`notebooks/05_market_making.ipynb`](notebooks/05_market_making.ipynb) |

Synthetic sample paths live in [`quant_lab.data`](src/quant_lab/data/sample.py) — no network required.

---

## Install

```bash
python -m venv .venv
source .venv/bin/activate   # Windows: .venv\Scripts\activate
pip install -e ".[dev]"
```

Optional notebook extras (matplotlib / Jupyter):

```bash
pip install -e ".[dev,notebooks]"
```

---

## Run tests

```bash
pytest -q
```

CI runs the same suite on push and pull requests
([`.github/workflows/ci.yml`](.github/workflows/ci.yml)).

---

## Package layout

```
src/quant_lab/
  backtest/          # vectorised engine + walk-forward helpers
  data/              # synthetic OHLCV, pairs, mid paths
  strategies/        # momentum / breakout signals
  risk/              # Sharpe, drawdown, VaR, vol targeting, Kelly
  options/           # Black–Scholes price, Greeks, IV
  pairs/             # Engle–Granger, z-score pairs strategy
  microstructure/    # inventory-aware MM simulator
notebooks/           # one demo per project (offline)
docs/SANITIZATION.md # what was excluded from the private monorepo
tests/               # pytest coverage for public APIs
```

---

## Design notes

- **Offline first** — demos and tests use seeded synthetic data.
- **Look-ahead aware** — backtests lag signals by one bar; costs are explicit.
- **Thin surface** — public `__init__` exports keep notebook imports short.
- **No broker code** — if you need live execution, build it elsewhere.

---

## License

MIT — see [`LICENSE`](LICENSE).
