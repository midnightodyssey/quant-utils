"""
quant-lab — educational quantitative trading toolkit.

Public API re-exports the most commonly used entry points.
"""

from quant_lab.backtest import BacktestResult, run_backtest, summary_table, walk_forward
from quant_lab.data import make_ohlcv
from quant_lab.indicators import atr, bollinger_bands, ema, macd, rsi, sma
from quant_lab.microstructure import (
    MarketMakerConfig,
    MarketMakerResult,
    MarketMakerSimulator,
    simulate_mid_path,
)
from quant_lab.options import (
    Greeks,
    OptionContract,
    black_scholes_greeks,
    black_scholes_price,
    implied_volatility,
)
from quant_lab.pairs import (
    CointegrationResult,
    PairsSignalConfig,
    PairsStrategy,
    engle_granger,
    generate_signals,
    rolling_zscore,
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
    "CointegrationResult",
    "Greeks",
    "MarketMakerConfig",
    "MarketMakerResult",
    "MarketMakerSimulator",
    "OptionContract",
    "PairsSignalConfig",
    "PairsStrategy",
    "PriceBreakout",
    "Strategy",
    "atr",
    "black_scholes_greeks",
    "black_scholes_price",
    "bollinger_bands",
    "calmar",
    "ema",
    "engle_granger",
    "fixed_fraction",
    "generate_signals",
    "implied_volatility",
    "kelly",
    "macd",
    "make_ohlcv",
    "max_drawdown",
    "risk_summary",
    "rolling_zscore",
    "rsi",
    "run_backtest",
    "sharpe",
    "simulate_mid_path",
    "sma",
    "sortino",
    "summary_table",
    "vol_target",
    "walk_forward",
]

__version__ = "0.1.0"
