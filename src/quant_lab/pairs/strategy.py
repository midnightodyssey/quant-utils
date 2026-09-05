"""Z-score entry/exit signals and a small pairs strategy wrapper."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np
import pandas as pd

from quant_lab.pairs.cointegration import engle_granger
from quant_lab.pairs.spread import build_spread, rolling_zscore


@dataclass(frozen=True)
class PairsSignalConfig:
    """Thresholds for mean-reversion entries on the residual z-score."""

    entry_z: float = 2.0
    exit_z: float = 0.5
    z_window: int = 60
    use_log: bool = True


def generate_signals(
    zscore: pd.Series | np.ndarray,
    *,
    entry_z: float = 2.0,
    exit_z: float = 0.5,
) -> pd.Series:
    """Map a z-score series to positions in ``{+1, -1, 0}``.

    Rules (stateful):
    * Enter **long spread** (+1) when ``z < -entry_z``.
    * Enter **short spread** (-1) when ``z > +entry_z``.
    * Exit to flat when ``|z| < exit_z``.
    * Otherwise hold the previous position.

    Long spread means long ``y`` / short ``b x`` in the Engle–Granger residual.
    """
    if entry_z <= 0 or exit_z < 0:
        raise ValueError("entry_z must be > 0 and exit_z >= 0")
    if exit_z >= entry_z:
        raise ValueError("exit_z must be strictly below entry_z")

    z = pd.Series(zscore, dtype=float)
    pos = np.zeros(len(z), dtype=int)
    state = 0
    for i, value in enumerate(z.to_numpy()):
        if not np.isfinite(value):
            pos[i] = state
            continue
        if state == 0:
            if value < -entry_z:
                state = 1
            elif value > entry_z:
                state = -1
        else:
            if abs(value) < exit_z:
                state = 0
        pos[i] = state
    return pd.Series(pos, index=z.index, name="position")


def spread_pnl(
    spread: pd.Series | np.ndarray,
    position: pd.Series | np.ndarray,
) -> pd.Series:
    """Simple PnL on the residual: ``position_{t-1} * Δspread_t``.

    Educational mark-to-market on the cointegration residual (not notionals).
    """
    s = pd.Series(spread, dtype=float)
    p = pd.Series(position, dtype=float).reindex(s.index).fillna(0.0)
    ret = s.diff().fillna(0.0)
    pnl = (p.shift(1).fillna(0.0) * ret).rename("pnl")
    return pnl


class PairsStrategy:
    """Fit cointegration on a train window, trade z-score on a test window.

    Compatible with a simple backtest loop if ``quant_lab.backtest`` exists:
    expose ``fit`` / ``generate`` / ``positions``. Otherwise use
    :meth:`run` for self-contained residual PnL.
    """

    def __init__(self, config: PairsSignalConfig | None = None) -> None:
        self.config = config or PairsSignalConfig()
        self.hedge_ratio_: float | None = None
        self.intercept_: float | None = None
        self.cointegration_: Any = None

    def fit(self, y: pd.Series, x: pd.Series) -> PairsStrategy:
        """Estimate hedge ratio via Engle–Granger on the provided sample."""
        result = engle_granger(y, x, use_log=self.config.use_log)
        self.cointegration_ = result
        self.hedge_ratio_ = result.hedge_ratio
        self.intercept_ = result.intercept
        return self

    def generate(self, y: pd.Series, x: pd.Series) -> pd.DataFrame:
        """Build spread, z-score, and position signals for ``y`` / ``x``."""
        if self.hedge_ratio_ is None or self.intercept_ is None:
            raise RuntimeError("call fit() before generate()")
        spread = build_spread(
            y,
            x,
            self.hedge_ratio_,
            intercept=self.intercept_,
            use_log=self.config.use_log,
        )
        z = rolling_zscore(spread, window=self.config.z_window)
        pos = generate_signals(
            z,
            entry_z=self.config.entry_z,
            exit_z=self.config.exit_z,
        )
        pnl = spread_pnl(spread, pos)
        return pd.DataFrame(
            {
                "spread": spread,
                "zscore": z,
                "position": pos,
                "pnl": pnl,
            },
            index=spread.index,
        )

    def run(self, y: pd.Series, x: pd.Series) -> pd.DataFrame:
        """Fit on the full sample then generate signals (demo convenience)."""
        return self.fit(y, x).generate(y, x)

    # Alias used by some simple backtest adapters
    def positions(self, y: pd.Series, x: pd.Series) -> pd.Series:
        return self.generate(y, x)["position"]
