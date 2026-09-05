"""Pairs trading via Engle–Granger cointegration and z-score signals."""

from quant_utils.pairs.cointegration import (
    CointegrationResult,
    engle_granger,
    ols_hedge_ratio,
)
from quant_utils.pairs.spread import build_spread, rolling_zscore
from quant_utils.pairs.strategy import PairsSignalConfig, PairsStrategy, generate_signals

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
