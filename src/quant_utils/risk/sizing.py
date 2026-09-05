"""
quant_utils.risk.sizing
─────────────────────────────────────────────────────────────────────────────
Position sizing utilities.

Position sizing answers: "Given a signal, how MUCH do I trade?"
Correct sizing turns a mediocre strategy into a profitable one; wrong
sizing bankrupts even a great strategy.

Three methods in increasing sophistication:
  1. Fixed Fraction — simple and robust
  2. Kelly Criterion — mathematically optimal, requires accurate win stats
  3. Volatility Targeting — institutional standard for multi-asset portfolios
"""

from __future__ import annotations


def fixed_fraction(
    capital: float,
    risk_pct: float,
    stop_pct: float,
    price: float,
) -> int:
    """
    Fixed Fraction — risk a fixed % of capital per trade.

    Formula:
        risk_amount    = capital × risk_pct
        stop_distance  = price × stop_pct
        shares         = risk_amount / stop_distance

    Example:
        capital = 100,000, risk 1% per trade = 1,000 risk
        price = 50, stop 2% below entry = 1.00 per share
        → buy 1,000 shares
        If stop is hit: 1,000 × 1.00 loss = 1,000 = exactly 1% of capital

    Args:
        capital:  total trading capital
        risk_pct: fraction of capital to risk per trade (e.g. 0.01 = 1%)
        stop_pct: stop loss distance as fraction of price (e.g. 0.02 = 2%)
        price:    current asset price

    Returns:
        Number of shares/units to trade (integer, always floored)
    """
    if stop_pct <= 0 or price <= 0:
        return 0
    risk_amount = capital * risk_pct
    stop_distance = price * stop_pct
    return int(risk_amount / stop_distance)


def kelly(
    win_rate: float,
    avg_win: float,
    avg_loss: float,
) -> float:
    """
    Kelly Criterion — the mathematically optimal fraction to bet.

    Formula:
        f* = win_rate / avg_loss - (1 - win_rate) / avg_win

    Why it matters:
        Kelly maximises the long-run geometric growth rate of capital.
        Betting more than Kelly causes ruin even with a positive-EV strategy.
        Betting less is safer but suboptimal.

    Practical usage:
        Professionals often use "half Kelly" (f* / 2) because win-rate and
        average P&L estimates are noisy, and full Kelly has large variance.

    Args:
        win_rate: fraction of trades that are winners (0.0 to 1.0)
        avg_win:  average P&L on winning trades (positive, e.g. 0.02)
        avg_loss: average P&L on losing trades (positive, e.g. 0.01)

    Returns:
        Optimal fraction of capital to wager per trade (float)
    """
    if avg_win <= 0 or avg_loss <= 0:
        return 0.0
    loss_rate = 1.0 - win_rate
    return float(win_rate / avg_loss - loss_rate / avg_win)


def vol_target(
    capital: float,
    target_vol: float,
    asset_vol: float,
    price: float,
) -> int:
    """
    Volatility Targeting — size positions to achieve a target portfolio vol.

    The institutional standard. Ensures consistent risk regardless of which
    assets you are trading.

    Formula:
        position_value = capital × (target_vol / asset_vol)
        shares         = position_value / price

    Intuition:
        A 20% vol asset gets half the position of a 10% vol asset, because
        it contributes twice the risk per unit invested.

    Args:
        capital:    total trading capital
        target_vol: target annualised portfolio volatility (e.g. 0.10 = 10%)
        asset_vol:  estimated annualised volatility of the asset
        price:      current price of the asset

    Returns:
        Number of shares/units to hold (integer, floored)
    """
    if asset_vol <= 0 or price <= 0:
        return 0
    position_value = capital * (target_vol / asset_vol)
    return int(position_value / price)
