"""Tests for momentum strategies."""

import numpy as np
import pandas as pd
import pytest

from quant_utils.backtest import BacktestResult
from quant_utils.strategies import ATRBreakout, PriceBreakout, Strategy


def make_df(close: pd.Series) -> pd.DataFrame:
    return pd.DataFrame(
        {
            "Open": close * 0.999,
            "High": close * 1.005,
            "Low": close * 0.995,
            "Close": close,
            "Volume": 1_000_000.0,
        }
    )


@pytest.fixture
def rising_df():
    return make_df(pd.Series(np.linspace(100, 200, 200)))


@pytest.fixture
def falling_df():
    return make_df(pd.Series(np.linspace(200, 100, 200)))


@pytest.fixture
def realistic_df():
    np.random.seed(42)
    close = pd.Series(100 + np.cumsum(np.random.randn(300) * 0.5 + 0.05))
    return make_df(close)


def test_strategies_are_strategy_subclasses():
    for cls in [PriceBreakout, ATRBreakout]:
        assert issubclass(cls, Strategy)


def test_strategy_names():
    assert PriceBreakout().name == "PriceBreakout"
    assert ATRBreakout().name == "ATRBreakout"


def test_base_class_is_abstract():
    with pytest.raises(TypeError):
        Strategy()


@pytest.mark.parametrize("strategy", [PriceBreakout(), ATRBreakout()], ids=lambda s: s.name)
def test_signals_return_series(strategy, realistic_df):
    signals = strategy.generate_signals(realistic_df)
    assert isinstance(signals, pd.Series)
    assert len(signals) == len(realistic_df)
    assert not signals.isna().any()
    assert signals.isin([-1.0, 0.0, 1.0]).all()


@pytest.mark.parametrize("strategy", [PriceBreakout(), ATRBreakout()], ids=lambda s: s.name)
def test_run_returns_backtest_result(strategy, realistic_df):
    result = strategy.run(realistic_df)
    assert isinstance(result, BacktestResult)
    for val in result.metrics.values():
        assert isinstance(val, float)


def test_price_breakout_zero_signals_during_warmup(rising_df):
    period = 20
    signals = PriceBreakout(period=period).generate_signals(rising_df)
    assert (signals.iloc[:period] == 0.0).all()


def test_price_breakout_long_on_sustained_rise(rising_df):
    signals = PriceBreakout(period=5).generate_signals(rising_df)
    warmed = signals.iloc[10:]
    assert (warmed == 1.0).mean() > 0.8


def test_price_breakout_short_on_sustained_fall(falling_df):
    signals = PriceBreakout(period=5).generate_signals(falling_df)
    warmed = signals.iloc[10:]
    assert (warmed == -1.0).mean() > 0.8
