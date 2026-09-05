"""Pairs trading via Engle–Granger cointegration and z-score signals."""

from quant_lab.pairs.cointegration import (
    CointegrationResult,
    engle_granger,
    ols_hedge_ratio,
)
from quant_lab.pairs.spread import build_spread, rolling_zscore
from quant_lab.pairs.strategy import PairsSignalConfig, PairsStrategy, generate_signals

__all__ = [
    "CointegrationResult",
    "engle_granger",
    "ols_hedge_ratio",
    "build_spread",
    "rolling_zscore",
    "PairsSignalConfig",
    "PairsStrategy",
    "generate_signals",
]
