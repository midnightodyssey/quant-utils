"""
quant_lab.backtest.engine
─────────────────────────────────────────────────────────────────────────────
Vectorised backtesting engine.

Design principles:
  - No loops: every calculation is a pandas/numpy vectorised operation
  - Pure functions: same inputs always produce same outputs
  - Realistic costs: slippage + commission on every trade
  - Signal lag: signals are shifted by 1 bar to prevent look-ahead bias
  - Integrates with risk metrics for full performance reporting
"""

from __future__ import annotations

import itertools
from dataclasses import dataclass, field
from typing import Any, Callable, Mapping, Sequence

import numpy as np
import pandas as pd

from quant_lab.risk.metrics import risk_summary, sharpe


TRADING_DAYS = 252


@dataclass
class BacktestResult:
    """
    Container for all backtest outputs.

    Attributes:
        returns:      net daily strategy returns (after costs)
        equity_curve: cumulative equity curve starting at 1.0
        positions:    lagged position series (what you actually held each day)
        trades:       number of times position changed (entry or exit)
        metrics:      dict of risk metrics from risk_summary()
    """

    returns: pd.Series
    equity_curve: pd.Series
    positions: pd.Series
    trades: int
    metrics: dict


def run_backtest(
    signals: pd.Series,
    prices: pd.Series,
    slippage: float = 0.0005,
    commission: float = 0.001,
    risk_free_rate: float = 0.05,
) -> BacktestResult:
    """
    Vectorised backtest engine.

    How it works:
        1. Align signals and prices to the same index
        2. Lag signals by 1 bar — prevents look-ahead bias
           (signal fires at close of day T → position held from T to T+1)
        3. Compute gross returns: position × next-day price return
        4. Identify position changes → apply slippage + commission
        5. Build equity curve: cumulative product of (1 + net_return)
        6. Pass cleaned returns to risk_summary() for full metrics

    Signal convention:
         1  = long  (hold for the next bar)
         0  = flat  (no position)
        -1  = short (short for the next bar)
        Fractional values allowed, e.g. 0.5 = half-sized long.

    Cost model:
        Each time position changes by Δ, you pay:
            cost = Δ × (2 × slippage + commission)
        Costs are deducted from gross returns on the same bar.

    Args:
        signals:        pd.Series of position signals (−1 to 1), indexed by date
        prices:         pd.Series of asset prices, same index as signals
        slippage:       one-way slippage as fraction (default 0.05% = 5 bps)
        commission:     round-trip commission as fraction (default 0.1% = 10 bps)
        risk_free_rate: annual risk-free rate used in Sharpe/Sortino

    Returns:
        BacktestResult with returns, equity curve, positions, trade count, metrics
    """
    signals, prices = signals.align(prices, join="inner")
    positions = signals.shift(1).fillna(0)

    price_returns = prices.pct_change()
    gross_returns = positions * price_returns

    position_changes = positions.diff().abs().fillna(0)
    trade_costs = position_changes * (2 * slippage + commission)
    net_returns = gross_returns - trade_costs

    equity_curve = (1 + net_returns).cumprod()
    trades = int((position_changes > 0).sum())
    metrics = risk_summary(net_returns.dropna(), risk_free_rate)

    return BacktestResult(
        returns=net_returns,
        equity_curve=equity_curve,
        positions=positions,
        trades=trades,
        metrics=metrics,
    )


def rolling_oos_windows(
    signals: pd.Series,
    prices: pd.Series,
    n_splits: int = 5,
    train_pct: float = 0.7,
    slippage: float = 0.0005,
    commission: float = 0.001,
    risk_free_rate: float = 0.05,
) -> list:
    """
    Evaluate a FIXED signal series on N sequential out-of-sample slices.

    This does NOT refit anything: the signals are handed in already computed,
    the train portion of each window is discarded unused, and the same
    parameters apply in every slice. That is a segmented backtest, not a
    walk-forward. Use walk_forward() when you need genuine refitting.

    Args:
        signals:    full signal series (all history)
        prices:     full price series (all history)
        n_splits:   number of windows (default 5)
        train_pct:  fraction of each window skipped as nominal training
        slippage:   per-side slippage
        commission: round-trip commission
        risk_free_rate: annual risk-free rate

    Returns:
        List of BacktestResult — one per test slice
    """
    n = len(signals)
    window_size = n // n_splits
    results = []

    for i in range(n_splits):
        start = i * window_size
        end = start + window_size if i < n_splits - 1 else n
        train_end = start + int(window_size * train_pct)

        test_signals = signals.iloc[train_end:end]
        test_prices = prices.iloc[train_end:end]

        if len(test_signals) < 2:
            continue

        result = run_backtest(
            test_signals, test_prices, slippage, commission, risk_free_rate
        )
        results.append(result)

    return results


@dataclass
class WalkForwardWindow:
    """One train → test step of a walk-forward."""

    index: int
    train_start: int
    train_end: int
    test_start: int
    test_end: int
    params: dict
    train_score: float
    candidates: pd.DataFrame
    result: BacktestResult


@dataclass
class WalkForwardResult:
    """Full walk-forward output with stitched out-of-sample returns."""

    windows: list
    returns: pd.Series
    equity_curve: pd.Series
    params_table: pd.DataFrame
    trials: int
    metrics: dict = field(default_factory=dict)

    @property
    def n_windows(self) -> int:
        return len(self.windows)

    def param_turnover(self) -> float:
        """Fraction of window-to-window steps where chosen parameters changed."""
        if len(self.windows) < 2:
            return 0.0
        changes = sum(
            1
            for a, b in zip(self.windows[:-1], self.windows[1:])
            if a.params != b.params
        )
        return changes / (len(self.windows) - 1)


def expand_grid(
    param_grid: Mapping[str, Sequence],
    valid: Callable[[dict], bool] | None = None,
) -> list:
    """Cartesian product of a parameter grid, optionally filtered."""
    keys = list(param_grid.keys())
    combos = [
        dict(zip(keys, values))
        for values in itertools.product(*[param_grid[k] for k in keys])
    ]
    if valid is not None:
        combos = [c for c in combos if valid(c)]
    return combos


def window_edges(n: int, n_splits: int = 5, train_pct: float = 0.5) -> list:
    """
    Integer boundaries for an anchored/rolling walk-forward.

    The first `train_pct` of the sample is the initial training block. What
    remains is cut into `n_splits` consecutive test blocks.

    Returns:
        List of (test_start, test_end) integer position pairs, test_end exclusive.
    """
    if n_splits < 1:
        raise ValueError(f"n_splits must be >= 1, got {n_splits}")
    if not 0.0 < train_pct < 1.0:
        raise ValueError(f"train_pct must be in (0, 1), got {train_pct}")

    initial_train = int(n * train_pct)
    remaining = n - initial_train
    if remaining < n_splits * 2:
        return []

    test_len = remaining // n_splits
    edges = []
    for i in range(n_splits):
        start = initial_train + i * test_len
        end = start + test_len if i < n_splits - 1 else n
        edges.append((start, end))
    return edges


def walk_forward(
    strategy_factory: Callable[..., Any],
    df: pd.DataFrame,
    param_grid: Mapping[str, Sequence],
    n_splits: int = 5,
    train_pct: float = 0.5,
    anchored: bool = True,
    score_fn: Callable[[pd.Series], float] | None = None,
    valid: Callable[[dict], bool] | None = None,
    price_col: str = "Close",
    slippage: float = 0.0005,
    commission: float = 0.001,
    risk_free_rate: float = 0.05,
) -> WalkForwardResult:
    """
    Walk-forward validation that refits parameters on each train block.

    Takes a strategy constructor and a parameter grid, searches the grid on
    each train block, and applies the winner to the following test block.
    Parameters are therefore a function of past data only.

    Args:
        strategy_factory: callable taking grid kwargs, returning an object
                          with .generate_signals(df)
        df:               OHLCV frame with a DatetimeIndex
        param_grid:       {name: [values]} searched on every train block
        n_splits:         number of test blocks
        train_pct:        fraction of the sample held as the initial train block
        anchored:         True → train on everything before the test block
                          False → train on a fixed-length trailing window
        score_fn:         train-block objective (defaults to Sharpe)
        valid:            predicate to drop impossible combinations
        price_col:        column of df used as the traded price
        slippage:         per-side slippage
        commission:       round-trip commission
        risk_free_rate:   annual risk-free rate

    Returns:
        WalkForwardResult
    """
    combos = expand_grid(param_grid, valid)
    if not combos:
        raise ValueError("param_grid is empty after applying `valid`")

    if score_fn is None:

        def score_fn(returns: pd.Series) -> float:
            return sharpe(returns.dropna(), risk_free_rate)

    prices = df[price_col].astype(float)
    n = len(df)
    edges = window_edges(n, n_splits, train_pct)

    initial_train = int(n * train_pct)
    windows = []
    trials = 0

    for i, (test_start, test_end) in enumerate(edges):
        train_start = 0 if anchored else max(0, test_start - initial_train)
        train_end = test_start

        train_df = df.iloc[:train_end]
        rows = []
        for params in combos:
            signals = strategy_factory(**params).generate_signals(train_df)
            fit_slice = slice(train_start, train_end)
            result = run_backtest(
                signals.iloc[fit_slice],
                prices.iloc[fit_slice],
                slippage,
                commission,
                risk_free_rate,
            )
            s = score_fn(result.returns)
            rows.append({**params, "train_score": float(s) if np.isfinite(s) else np.nan})
            trials += 1

        candidates = pd.DataFrame(rows)
        best_pos = candidates["train_score"].idxmax()
        if pd.isna(best_pos):
            best_pos = 0
        best = combos[int(best_pos)]

        visible = df.iloc[:test_end]
        test_signals = strategy_factory(**best).generate_signals(visible)
        result = run_backtest(
            test_signals.iloc[test_start:test_end],
            prices.iloc[test_start:test_end],
            slippage,
            commission,
            risk_free_rate,
        )

        windows.append(
            WalkForwardWindow(
                index=i,
                train_start=train_start,
                train_end=train_end,
                test_start=test_start,
                test_end=test_end,
                params=best,
                train_score=float(candidates.loc[int(best_pos), "train_score"]),
                candidates=candidates,
                result=result,
            )
        )

    if not windows:
        empty = pd.Series(dtype=float)
        return WalkForwardResult(
            windows=[],
            returns=empty,
            equity_curve=empty,
            params_table=pd.DataFrame(),
            trials=trials,
            metrics={},
        )

    stitched = pd.concat([w.result.returns for w in windows])
    stitched = stitched[~stitched.index.duplicated(keep="first")].sort_index()

    params_table = pd.DataFrame(
        [
            {
                "window": w.index,
                "train_start": df.index[w.train_start],
                "train_end": df.index[w.train_end - 1],
                "test_start": df.index[w.test_start],
                "test_end": df.index[w.test_end - 1],
                "train_bars": w.train_end - w.train_start,
                "test_bars": w.test_end - w.test_start,
                **w.params,
                "train_score": w.train_score,
                "test_sharpe": w.result.metrics.get("Sharpe Ratio", np.nan),
                "trades": w.result.trades,
            }
            for w in windows
        ]
    )

    return WalkForwardResult(
        windows=windows,
        returns=stitched,
        equity_curve=(1 + stitched).cumprod(),
        params_table=params_table,
        trials=trials,
        metrics=risk_summary(stitched.dropna(), risk_free_rate),
    )


def summary_table(result: BacktestResult) -> pd.Series:
    """
    Format a BacktestResult as a labelled pandas Series for display.

    Adds Trades and Final Equity beyond the standard risk metrics.
    """
    data = dict(result.metrics)
    data["Trades"] = float(result.trades)
    data["Final Equity"] = round(float(result.equity_curve.dropna().iloc[-1]), 4)
    return pd.Series(data)
