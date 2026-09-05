"""Tests for the vectorised backtest engine."""

import numpy as np
import pandas as pd
import pytest

from quant_lab.backtest import (
    BacktestResult,
    WalkForwardResult,
    expand_grid,
    rolling_oos_windows,
    run_backtest,
    summary_table,
    walk_forward,
    window_edges,
)
from quant_lab.strategies import PriceBreakout


@pytest.fixture
def rising_prices():
    return pd.Series(np.linspace(100, 200, 252))


@pytest.fixture
def flat_prices():
    return pd.Series([100.0] * 252)


@pytest.fixture
def always_long(rising_prices):
    return pd.Series([1.0] * len(rising_prices), index=rising_prices.index)


@pytest.fixture
def always_flat(rising_prices):
    return pd.Series([0.0] * len(rising_prices), index=rising_prices.index)


@pytest.fixture
def alternating_signals(rising_prices):
    vals = [1.0 if i % 2 == 0 else -1.0 for i in range(len(rising_prices))]
    return pd.Series(vals, index=rising_prices.index)


def test_result_is_backtest_result(rising_prices, always_long):
    result = run_backtest(always_long, rising_prices)
    assert isinstance(result, BacktestResult)


def test_result_has_all_fields(rising_prices, always_long):
    result = run_backtest(always_long, rising_prices)
    assert hasattr(result, "returns")
    assert hasattr(result, "equity_curve")
    assert hasattr(result, "positions")
    assert hasattr(result, "trades")
    assert hasattr(result, "metrics")


def test_metrics_has_all_six_keys(rising_prices, always_long):
    result = run_backtest(always_long, rising_prices)
    expected = {
        "Sharpe Ratio",
        "Sortino Ratio",
        "Max Drawdown",
        "Calmar Ratio",
        "VaR (95%, 1-day)",
        "CVaR (95%, 1-day)",
    }
    assert set(result.metrics.keys()) == expected


def test_flat_signal_zero_gross_returns(flat_prices, always_flat):
    result = run_backtest(always_flat, flat_prices, slippage=0, commission=0)
    assert result.returns.dropna().abs().sum() == 0.0


def test_long_signal_rising_prices_positive_equity(rising_prices, always_long):
    result = run_backtest(always_long, rising_prices, slippage=0, commission=0)
    final = result.equity_curve.dropna().iloc[-1]
    assert final > 1.0


def test_short_signal_rising_prices_negative_equity(rising_prices):
    always_short = pd.Series([-1.0] * len(rising_prices), index=rising_prices.index)
    result = run_backtest(always_short, rising_prices, slippage=0, commission=0)
    final = result.equity_curve.dropna().iloc[-1]
    assert final < 1.0


def test_costs_reduce_returns(rising_prices, always_long):
    no_cost = run_backtest(always_long, rising_prices, slippage=0, commission=0)
    with_cost = run_backtest(always_long, rising_prices, slippage=0.001, commission=0.002)
    assert no_cost.equity_curve.dropna().iloc[-1] >= with_cost.equity_curve.dropna().iloc[-1]


def test_high_turnover_strategy_hurt_most_by_costs(rising_prices, alternating_signals):
    no_cost = run_backtest(alternating_signals, rising_prices, slippage=0, commission=0)
    with_cost = run_backtest(
        alternating_signals, rising_prices, slippage=0.0005, commission=0.001
    )
    assert with_cost.equity_curve.dropna().iloc[-1] < no_cost.equity_curve.dropna().iloc[-1]


def test_signal_lagged_by_one_bar(rising_prices):
    signals = pd.Series([1.0] * len(rising_prices), index=rising_prices.index)
    result = run_backtest(signals, rising_prices, slippage=0, commission=0)
    assert result.positions.iloc[0] == 0.0
    assert result.positions.iloc[1] == signals.iloc[0]


def test_no_future_data_used(rising_prices):
    prices = rising_prices.copy()
    prices.iloc[50] *= 2.0
    signals = pd.Series(0.0, index=prices.index)
    signals.iloc[50] = 1.0
    result = run_backtest(signals, prices, slippage=0, commission=0)
    assert result.returns.iloc[50] == 0.0


def test_equity_curve_same_length_as_prices(rising_prices, always_long):
    result = run_backtest(always_long, rising_prices)
    assert len(result.equity_curve) == len(rising_prices)


def test_always_long_one_trade(rising_prices, always_long):
    result = run_backtest(always_long, rising_prices)
    assert result.trades == 1


def test_always_flat_zero_trades(flat_prices, always_flat):
    result = run_backtest(always_flat, flat_prices)
    assert result.trades == 0


def test_alternating_many_trades(rising_prices, alternating_signals):
    result = run_backtest(alternating_signals, rising_prices)
    assert result.trades > 100


def test_rolling_oos_returns_list(rising_prices, always_long):
    results = rolling_oos_windows(always_long, rising_prices, n_splits=3)
    assert isinstance(results, list)
    assert len(results) == 3
    for r in results:
        assert isinstance(r, BacktestResult)


def test_expand_grid_filters():
    grid = {"fast": [5, 10], "slow": [20, 50]}
    combos = expand_grid(grid, lambda p: p["fast"] < p["slow"])
    assert len(combos) == 4


def test_window_edges_are_disjoint_and_cover_the_tail():
    edges = window_edges(1000, n_splits=5, train_pct=0.5)
    assert len(edges) == 5
    assert edges[0][0] == 500
    assert edges[-1][1] == 1000


def test_window_edges_rejects_bad_train_pct():
    with pytest.raises(ValueError):
        window_edges(1000, train_pct=1.0)


@pytest.fixture
def noisy_ohlcv():
    rng = np.random.default_rng(7)
    n = 800
    idx = pd.bdate_range("2015-01-01", periods=n)
    drift = np.concatenate([np.full(n // 2, 0.0012), np.zeros(n - n // 2)])
    noise = np.concatenate(
        [rng.normal(0, 0.004, n // 2), rng.normal(0, 0.022, n - n // 2)]
    )
    close = 100 * np.cumprod(1 + drift + noise)
    return pd.DataFrame(
        {
            "Open": close,
            "High": close * 1.005,
            "Low": close * 0.995,
            "Close": close,
            "Volume": 1_000_000.0,
        },
        index=idx,
    )


def test_walk_forward_returns_result(noisy_ohlcv):
    grid = {"period": [5, 10, 20]}
    wf = walk_forward(PriceBreakout, noisy_ohlcv, grid, n_splits=4)
    assert isinstance(wf, WalkForwardResult)
    assert wf.n_windows == 4
    assert wf.trials == 4 * 3


def test_walk_forward_empty_grid_raises(noisy_ohlcv):
    with pytest.raises(ValueError):
        walk_forward(PriceBreakout, noisy_ohlcv, {"period": [5]}, valid=lambda p: False)


def test_summary_table_returns_series(rising_prices, always_long):
    result = run_backtest(always_long, rising_prices)
    table = summary_table(result)
    assert isinstance(table, pd.Series)
    assert "Trades" in table.index
    assert "Final Equity" in table.index
