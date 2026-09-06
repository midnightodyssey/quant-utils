# Agent notes (quant-utils)

Public shared quant primitives. Private trading systems depend on this package;
this package must **never** depend on private repos.

## Scope

- Own: backtest engine, risk metrics, sizing, indicators, Black–Scholes,
  momentum strategies, pairs cointegration, market-making simulator, demos.
- Do **not** add: broker adapters, live runners, account credentials, ops
  host config, research ledgers, or anything that assumes a funded account.

## Dependency rule

`trading-strategies` (and other private repos) → `quant-utils` only.
Never import `framework.*`, runners, or private research packages from here.

## Pull requests

1. Prefer small, reviewable PRs with `pytest -q` green.
2. After `main` moves, update the PR branch (`gh pr update-branch` / the
   `update-open-prs-from-main` workflow) before marking ready.
3. Use the PR template checklist.

## Releases

Bump version in `pyproject.toml`, tag `vX.Y.Z`, and push the tag so private
repos can pin deliberately (see `trading-strategies` `requirements.txt`).

## Secrets

Never commit `.env`, API keys, IBKR/Databento credentials, or account IDs.
Synthetic/sample data only in this repo.
