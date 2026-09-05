# Sanitization notes

`quant-lab` is a **public, educational** extract from a private research monorepo
(`trading-strategies`). The goal is to ship readable research modules and demos
without exposing production wiring, credentials, or proprietary ops.

## Intentionally excluded

| Area | What was left out |
|------|-------------------|
| **Live brokers** | Interactive Brokers / other broker clients, session config, account adapters |
| **Runner / ops** | Daily runner, automation jobs, deploy workflows, Docker/IBC ops images |
| **Research trials** | Trial ledgers, study notebooks tied to private datasets, short-interest caches |
| **Generated configs** | Aggressive/defensive runner YAML, portfolio selection outputs, dashboard feeds |
| **Market data plumbing** | Databento/OPRA recorders, Edgar/13F scrapers, live chain stores |
| **Execution stack** | OMS, live order-state machines, fill reconciliation against a broker |
| **Internal docs / boards** | Prop-firm challenge boards, capital-path notes, chief-of-staff ops guides |

## What remains

Pure research building blocks that run offline on synthetic sample data:

- vectorised backtests and walk-forward helpers
- risk metrics and position sizing
- momentum / breakout strategies
- Black–Scholes pricing and Greeks
- pairs cointegration + z-score signals
- a simple market-making simulator

Production broker connectivity and live trading runners are **intentionally absent**.
This package is not a trading system.
