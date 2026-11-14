"""Pairs trading via Engle–Granger cointegration and z-score signals."""

from quant_utils.pairs.cointegration import (
    CointegrationResult,
    eg_critical_value,
    eg_pvalue,
    engle_granger,
    ols_hedge_ratio,
)
from quant_utils.pairs.errors import PairsError
from quant_utils.pairs.selection import (
    PairCandidate,
    PairSelection,
    candidate_pairs,
    select_pairs,
)
from quant_utils.pairs.spread import (
    build_spread,
    half_life,
    rolling_hedge_ratio,
    rolling_zscore,
    zscore,
)
from quant_utils.pairs.strategy import (
    PairsSignalConfig,
    PairsStrategy,
    generate_signals,
    positions_from_z,
)

__all__ = [
    "CointegrationResult",
    "PairCandidate",
    "PairSelection",
    "PairsError",
    "PairsSignalConfig",
    "PairsStrategy",
    "build_spread",
    "candidate_pairs",
    "eg_critical_value",
    "eg_pvalue",
    "engle_granger",
    "generate_signals",
    "half_life",
    "ols_hedge_ratio",
    "positions_from_z",
    "rolling_hedge_ratio",
    "rolling_zscore",
    "select_pairs",
    "zscore",
]
