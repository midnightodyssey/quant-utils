"""Tests for Black-Scholes pricing."""

from datetime import date

import pytest

from quant_lab.options import (
    OptionContract,
    black_scholes_greeks,
    black_scholes_price,
    implied_volatility,
    year_fraction_to_expiry,
)


def test_option_contract_dataclass() -> None:
    c = OptionContract(
        symbol="AAPL", option_type="call", strike=200.0, expiry=date(2026, 12, 18)
    )
    assert c.symbol == "AAPL"
    assert c.option_type == "call"
    assert c.multiplier == 100


def test_year_fraction_non_negative() -> None:
    t = year_fraction_to_expiry(expiry=date(2000, 1, 1), as_of=date(2026, 1, 1))
    assert t == 0.0


def test_black_scholes_prices_positive() -> None:
    call = black_scholes_price(
        spot=100,
        strike=100,
        time_to_expiry_years=1.0,
        risk_free_rate=0.03,
        volatility=0.2,
        option_type="call",
    )
    put = black_scholes_price(
        spot=100,
        strike=100,
        time_to_expiry_years=1.0,
        risk_free_rate=0.03,
        volatility=0.2,
        option_type="put",
    )
    assert call > 0
    assert put > 0


def test_put_call_parity_no_dividend() -> None:
    s, k, t, r, vol = 120.0, 110.0, 0.75, 0.04, 0.25
    call = black_scholes_price(s, k, t, r, vol, "call")
    put = black_scholes_price(s, k, t, r, vol, "put")
    lhs = call - put
    rhs = s - k * (2.718281828459045 ** (-r * t))
    assert lhs == pytest.approx(rhs, rel=1e-5)


def test_greeks_ranges() -> None:
    g_call = black_scholes_greeks(
        spot=100,
        strike=100,
        time_to_expiry_years=0.5,
        risk_free_rate=0.01,
        volatility=0.2,
        option_type="call",
    )
    g_put = black_scholes_greeks(
        spot=100,
        strike=100,
        time_to_expiry_years=0.5,
        risk_free_rate=0.01,
        volatility=0.2,
        option_type="put",
    )
    assert 0.0 < g_call.delta < 1.0
    assert -1.0 < g_put.delta < 0.0
    assert g_call.gamma > 0.0
    assert g_call.vega > 0.0


def test_implied_volatility_recovers_true_vol() -> None:
    s, k, t, r, true_vol = 100.0, 105.0, 0.8, 0.02, 0.32
    px = black_scholes_price(s, k, t, r, true_vol, "call")
    iv = implied_volatility(px, s, k, t, r, "call")
    assert iv == pytest.approx(true_vol, rel=1e-4)


def test_implied_volatility_out_of_bounds_raises() -> None:
    with pytest.raises(ValueError):
        implied_volatility(
            market_price=500.0,
            spot=100.0,
            strike=100.0,
            time_to_expiry_years=1.0,
            risk_free_rate=0.01,
            option_type="call",
        )
