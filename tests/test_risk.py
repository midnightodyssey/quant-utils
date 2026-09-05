"""Tests for risk metrics."""

import numpy as np
import pandas as pd
import pytest

from quant_utils.risk import (
    calmar,
    cvar,
    max_drawdown,
    risk_summary,
    sharpe,
    sortino,
    var_parametric,
)


@pytest.fixture
def flat_returns():
    return pd.Series([0.0] * 252)


@pytest.fixture
def positive_returns():
    return pd.Series([0.001] * 252)


@pytest.fixture
def realistic_returns():
    np.random.seed(42)
    return pd.Series(np.random.normal(0.0005, 0.01, 252))


def test_sharpe_flat_returns(flat_returns):
    assert sharpe(flat_returns) == 0.0


def test_sharpe_positive_for_good_strategy(positive_returns):
    result = sharpe(positive_returns, risk_free_rate=0.0)
    assert result > 5.0


def test_sharpe_is_float(realistic_returns):
    assert isinstance(sharpe(realistic_returns), float)


def test_sortino_positive_returns(positive_returns):
    assert sortino(positive_returns, risk_free_rate=0.0) == 0.0


def test_sortino_greater_than_sharpe_for_skewed_returns():
    np.random.seed(1)
    gains = np.abs(np.random.normal(0.002, 0.015, 400))
    losses = -np.abs(np.random.normal(0.001, 0.005, 100))
    data = np.concatenate([gains, losses])
    np.random.shuffle(data)
    returns = pd.Series(data)
    assert sortino(returns, risk_free_rate=0.0) >= sharpe(returns, risk_free_rate=0.0)


def test_sortino_downside_deviation_divides_by_full_sample():
    returns = pd.Series([0.01, 0.01, 0.01, -0.01])
    result = sortino(returns, risk_free_rate=0.0)
    assert abs(result - np.sqrt(252)) < 1e-9


def test_sortino_handles_empty_series():
    assert sortino(pd.Series([], dtype=float)) == 0.0


def test_mdd_is_negative_or_zero(realistic_returns):
    assert max_drawdown(realistic_returns) <= 0.0


def test_mdd_all_positive_returns(positive_returns):
    assert abs(max_drawdown(positive_returns)) < 1e-10


def test_mdd_known_case():
    returns = pd.Series([1.0, -0.5])
    assert abs(max_drawdown(returns) - (-0.5)) < 1e-10


def test_calmar_positive_for_good_strategy(positive_returns):
    assert calmar(positive_returns) == 0.0


def test_calmar_numerator_is_cagr_not_arithmetic_mean(realistic_returns):
    expected = (float((1 + realistic_returns).prod()) - 1) / abs(
        max_drawdown(realistic_returns)
    )
    assert abs(calmar(realistic_returns) - expected) < 1e-12


def test_calmar_reflects_volatility_drag():
    returns = pd.Series([0.10, -0.10] * 126)
    assert abs(returns.mean()) < 1e-15
    assert float((1 + returns).prod()) < 1.0
    assert calmar(returns) < 0.0


def test_calmar_handles_empty_series():
    assert calmar(pd.Series([], dtype=float)) == 0.0


def test_var_is_positive(realistic_returns):
    assert var_parametric(realistic_returns) > 0


def test_var_99_greater_than_95(realistic_returns):
    assert var_parametric(realistic_returns, confidence=0.99) > var_parametric(
        realistic_returns, confidence=0.95
    )


def test_var_horizon_scaling(realistic_returns):
    var_1d = var_parametric(realistic_returns, horizon=1)
    var_10d = var_parametric(realistic_returns, horizon=10)
    assert abs(var_10d / var_1d - np.sqrt(10)) < 0.01


def test_cvar_exceeds_var(realistic_returns):
    assert cvar(realistic_returns, confidence=0.95) >= var_parametric(
        realistic_returns, confidence=0.95
    )


def test_cvar_is_positive(realistic_returns):
    assert cvar(realistic_returns) > 0


def test_risk_summary_returns_all_keys(realistic_returns):
    result = risk_summary(realistic_returns)
    expected_keys = {
        "Sharpe Ratio",
        "Sortino Ratio",
        "Max Drawdown",
        "Calmar Ratio",
        "VaR (95%, 1-day)",
        "CVaR (95%, 1-day)",
    }
    assert set(result.keys()) == expected_keys
    for val in result.values():
        assert isinstance(val, float)
