"""
quant_utils.options.black_scholes

Core derivatives analytics for European option pricing and risk.

Scope:
- OptionContract dataclass
- Black-Scholes price (call/put)
- Closed-form Greeks (delta, gamma, vega, theta, rho)
- Implied volatility via bisection
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import date, datetime
from math import erf, exp, log, pi, sqrt
from typing import Literal

OptionType = Literal["call", "put"]


@dataclass(frozen=True)
class OptionContract:
    """Simple option contract descriptor."""

    symbol: str
    option_type: OptionType
    strike: float
    expiry: date
    multiplier: int = 100


@dataclass(frozen=True)
class Greeks:
    """Option sensitivity bundle (per 1 contract unit)."""

    delta: float
    gamma: float
    vega: float
    theta: float
    rho: float


def _norm_pdf(x: float) -> float:
    return exp(-0.5 * x * x) / sqrt(2.0 * pi)


def _norm_cdf(x: float) -> float:
    return 0.5 * (1.0 + erf(x / sqrt(2.0)))


def _validate_inputs(
    spot: float,
    strike: float,
    time_to_expiry_years: float,
    volatility: float,
    option_type: OptionType,
) -> None:
    if spot <= 0:
        raise ValueError(f"spot must be > 0, got {spot}")
    if strike <= 0:
        raise ValueError(f"strike must be > 0, got {strike}")
    if time_to_expiry_years < 0:
        raise ValueError(
            f"time_to_expiry_years must be >= 0, got {time_to_expiry_years}"
        )
    if volatility < 0:
        raise ValueError(f"volatility must be >= 0, got {volatility}")
    if option_type not in ("call", "put"):
        raise ValueError(f"option_type must be 'call' or 'put', got {option_type}")


def year_fraction_to_expiry(
    expiry: date, as_of: date | None = None, day_count: int = 365
) -> float:
    """Convert expiry date into year fraction using ACT/day_count convention."""
    if as_of is None:
        as_of = datetime.utcnow().date()
    days = (expiry - as_of).days
    return max(0.0, days / float(day_count))


def _deterministic_value(
    spot: float,
    strike: float,
    time_to_expiry_years: float,
    risk_free_rate: float,
    dividend_yield: float,
    option_type: OptionType,
) -> float:
    """
    Option value when the payoff is certain (sigma → 0+ or T → 0+).

    Uses the discounted forward payoff so put-call parity holds when r or q
    is non-zero and T > 0.
    """
    fwd_spot = spot * exp(-dividend_yield * time_to_expiry_years)
    pv_strike = strike * exp(-risk_free_rate * time_to_expiry_years)
    if option_type == "call":
        return max(fwd_spot - pv_strike, 0.0)
    return max(pv_strike - fwd_spot, 0.0)


def _d1_d2(
    spot: float,
    strike: float,
    time_to_expiry_years: float,
    risk_free_rate: float,
    volatility: float,
    dividend_yield: float,
) -> tuple[float, float]:
    vt = volatility * sqrt(time_to_expiry_years)
    d1 = (
        log(spot / strike)
        + (risk_free_rate - dividend_yield + 0.5 * volatility * volatility)
        * time_to_expiry_years
    ) / vt
    d2 = d1 - vt
    return d1, d2


def _deterministic_greeks(
    spot: float,
    strike: float,
    time_to_expiry_years: float,
    risk_free_rate: float,
    dividend_yield: float,
    option_type: OptionType,
) -> Greeks:
    """Greeks in the degenerate case (sigma → 0+ or T → 0+)."""
    disc_q = exp(-dividend_yield * time_to_expiry_years)
    disc_r = exp(-risk_free_rate * time_to_expiry_years)
    fwd_spot = spot * disc_q
    pv_strike = strike * disc_r

    in_the_money = (
        fwd_spot > pv_strike if option_type == "call" else pv_strike > fwd_spot
    )
    if not in_the_money:
        return Greeks(delta=0.0, gamma=0.0, vega=0.0, theta=0.0, rho=0.0)

    sign = 1.0 if option_type == "call" else -1.0
    return Greeks(
        delta=sign * disc_q,
        gamma=0.0,
        vega=0.0,
        theta=sign * (dividend_yield * fwd_spot - risk_free_rate * pv_strike),
        rho=sign * strike * time_to_expiry_years * disc_r,
    )


def black_scholes_price(
    spot: float,
    strike: float,
    time_to_expiry_years: float,
    risk_free_rate: float,
    volatility: float,
    option_type: OptionType,
    dividend_yield: float = 0.0,
) -> float:
    """Black-Scholes-Merton option price for European call/put."""
    _validate_inputs(spot, strike, time_to_expiry_years, volatility, option_type)

    if time_to_expiry_years == 0.0 or volatility == 0.0:
        return _deterministic_value(
            spot, strike, time_to_expiry_years, risk_free_rate, dividend_yield, option_type
        )

    d1, d2 = _d1_d2(
        spot, strike, time_to_expiry_years, risk_free_rate, volatility, dividend_yield
    )

    disc_q = exp(-dividend_yield * time_to_expiry_years)
    disc_r = exp(-risk_free_rate * time_to_expiry_years)

    if option_type == "call":
        return spot * disc_q * _norm_cdf(d1) - strike * disc_r * _norm_cdf(d2)
    return strike * disc_r * _norm_cdf(-d2) - spot * disc_q * _norm_cdf(-d1)


def black_scholes_greeks(
    spot: float,
    strike: float,
    time_to_expiry_years: float,
    risk_free_rate: float,
    volatility: float,
    option_type: OptionType,
    dividend_yield: float = 0.0,
) -> Greeks:
    """Closed-form Black-Scholes Greeks (annual theta, vega/rho per 1.00 move)."""
    _validate_inputs(spot, strike, time_to_expiry_years, volatility, option_type)

    if time_to_expiry_years == 0.0 or volatility == 0.0:
        return _deterministic_greeks(
            spot, strike, time_to_expiry_years, risk_free_rate, dividend_yield, option_type
        )

    d1, d2 = _d1_d2(
        spot, strike, time_to_expiry_years, risk_free_rate, volatility, dividend_yield
    )
    disc_q = exp(-dividend_yield * time_to_expiry_years)
    disc_r = exp(-risk_free_rate * time_to_expiry_years)
    pdf_d1 = _norm_pdf(d1)

    gamma = disc_q * pdf_d1 / (spot * volatility * sqrt(time_to_expiry_years))
    vega = spot * disc_q * pdf_d1 * sqrt(time_to_expiry_years)

    if option_type == "call":
        delta = disc_q * _norm_cdf(d1)
        theta = (
            -(spot * disc_q * pdf_d1 * volatility) / (2.0 * sqrt(time_to_expiry_years))
            - risk_free_rate * strike * disc_r * _norm_cdf(d2)
            + dividend_yield * spot * disc_q * _norm_cdf(d1)
        )
        rho = strike * time_to_expiry_years * disc_r * _norm_cdf(d2)
    else:
        delta = disc_q * (_norm_cdf(d1) - 1.0)
        theta = (
            -(spot * disc_q * pdf_d1 * volatility) / (2.0 * sqrt(time_to_expiry_years))
            + risk_free_rate * strike * disc_r * _norm_cdf(-d2)
            - dividend_yield * spot * disc_q * _norm_cdf(-d1)
        )
        rho = -strike * time_to_expiry_years * disc_r * _norm_cdf(-d2)

    return Greeks(delta=delta, gamma=gamma, vega=vega, theta=theta, rho=rho)


def implied_volatility(
    market_price: float,
    spot: float,
    strike: float,
    time_to_expiry_years: float,
    risk_free_rate: float,
    option_type: OptionType,
    dividend_yield: float = 0.0,
    low: float = 1e-6,
    high: float = 5.0,
    tol: float = 1e-6,
    max_iter: int = 200,
    vol_tol: float = 1e-8,
    max_vol_band: float | None = 0.01,
) -> float:
    """
    Solve implied volatility using bisection on Black-Scholes price.

    Bisection runs to a volatility bracket of ``vol_tol``. Where vega is
    small enough that ``tol`` of price spans more than ``max_vol_band`` of
    volatility, raises rather than returning a confident-looking number.
    """
    _validate_inputs(spot, strike, time_to_expiry_years, 0.0, option_type)
    if market_price < 0:
        raise ValueError(f"market_price must be >= 0, got {market_price}")

    if time_to_expiry_years == 0.0:
        raise ValueError(
            "implied volatility is undefined at time_to_expiry_years=0; "
            "an expired option carries no volatility information"
        )

    lo = low
    hi = high

    plo = black_scholes_price(
        spot, strike, time_to_expiry_years, risk_free_rate, lo, option_type, dividend_yield
    )
    phi = black_scholes_price(
        spot, strike, time_to_expiry_years, risk_free_rate, hi, option_type, dividend_yield
    )

    if market_price < plo - tol or market_price > phi + tol:
        raise ValueError(
            "market_price is outside model bounds for provided inputs; "
            f"price={market_price}, bounds=[{plo}, {phi}]"
        )

    for _ in range(max_iter):
        if hi - lo <= vol_tol:
            break
        mid = 0.5 * (lo + hi)
        pmid = black_scholes_price(
            spot, strike, time_to_expiry_years, risk_free_rate, mid, option_type, dividend_yield
        )
        if pmid > market_price:
            hi = mid
        else:
            lo = mid

    solved = 0.5 * (lo + hi)

    if max_vol_band is not None:
        vega = black_scholes_greeks(
            spot, strike, time_to_expiry_years, risk_free_rate, solved, option_type, dividend_yield
        ).vega
        band = float("inf") if vega <= 0.0 else 2.0 * tol / vega
        if band > max_vol_band:
            raise ValueError(
                "market_price does not identify an implied volatility: vega="
                f"{vega:.3e} means a price tolerance of {tol:g} spans {band:.3g} "
                f"of volatility (max_vol_band={max_vol_band:g})."
            )

    return solved
