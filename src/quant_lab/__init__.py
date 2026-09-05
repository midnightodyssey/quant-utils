"""
quant-lab — educational quantitative trading toolkit.

Public API re-exports the most commonly used entry points.
"""

from quant_lab.backtest import BacktestResult, run_backtest, summary_table, walk_forward
from quant_lab.data import make_ohlcv
from quant_lab.indicators import atr, bollinger_bands, ema, macd, rsi, sma
from quant_lab.options import (
    Greeks,
    OptionContract,
    black_scholes_greeks,
    black_scholes_price,
    implied_volatility,
)
from quant_lab.risk import (
    calmar,
    fixed_fraction,
    kelly,
    max_drawdown,
    risk_summary,
    sharpe,
    sortino,
    vol_target,
)
from quant_lab.strategies import ATRBreakout, PriceBreakout, Strategy

__all__ = [
    "ATRBreakout",
    "BacktestResult",
    "Greeks",
    "OptionContract",
    "PriceBreakout",
    "Strategy",
    "atr",
    "black_scholes_greeks",
    "black_scholes_price",
    "bollinger_bands",
    "calmar",
    "ema",
    "fixed_fraction",
    "implied_volatility",
    "kelly",
    "macd",
    "make_ohlcv",
    "max_drawdown",
    "risk_summary",
    "rsi",
    "run_backtest",
    "sharpe",
    "sma",
    "sortino",
    "summary_table",
    "vol_target",
    "walk_forward",
]

__version__ = "0.1.0"
