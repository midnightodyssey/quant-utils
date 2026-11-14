"""Z-score entry/exit signals and a small pairs strategy wrapper.

Timing: a position reported at bar ``t`` is a decision taken at the close of
``t`` using data up to and including ``t``. It earns PnL from bar ``t + 1``
(one-bar signal delay; see :func:`spread_pnl`).
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Any

import numpy as np
import pandas as pd

from quant_utils.pairs.cointegration import engle_granger
from quant_utils.pairs.errors import PairsError
from quant_utils.pairs.spread import build_spread, rolling_zscore


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
    * A non-finite ``z`` (warm-up or missing data) forces **flat**: an
      unobservable signal is not a reason to stay in a trade.

    Long spread means long ``y`` / short ``b x`` in the Engle–Granger residual.
    For stops, a time stop and lock-out use :func:`positions_from_z`.
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
            state = 0
            pos[i] = 0
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


def positions_from_z(
    z: pd.Series | np.ndarray,
    *,
    entry_z: float,
    exit_z: float,
    stop_z: float,
    max_hold: int,
) -> pd.Series:
    """Stateful entry / exit / stop / time-stop machine on a z-score series.

    All four parameters are required: there are deliberately no defaults, so a
    threshold is always a conscious choice. Requires
    ``0 <= exit_z < entry_z < stop_z`` and ``max_hold >= 1``.

    * +1 (long spread) opens at ``z <= -entry_z``; -1 at ``z >= +entry_z``.
    * Closes when the spread has reverted to ``|z| <= exit_z``.
    * A **stop** (adverse ``|z| >= stop_z``) or **time stop** (``max_hold`` bars
      held) flattens and *locks out* the pair until ``|z| < exit_z``, so a
      stopped pair cannot re-enter immediately. A flat bar with
      ``|z| >= stop_z`` also locks.
    * A non-finite ``z`` goes flat and resets the holding counter; a lock-out in
      force stays in force.

    The position at bar t uses ``z`` up to t only; execute at t + 1.
    """
    if not (0 <= exit_z < entry_z < stop_z):
        raise PairsError("require 0 <= exit_z < entry_z < stop_z")
    if max_hold < 1:
        raise PairsError("max_hold must be >= 1")
    zs = pd.Series(z, dtype=float)
    out = np.zeros(len(zs), dtype=np.int64)
    pos = held = 0
    locked = False
    for i, zi in enumerate(zs.to_numpy()):
        if not math.isfinite(zi):
            pos = held = 0
            continue
        if pos != 0:
            adverse = zi <= -stop_z if pos > 0 else zi >= stop_z
            reverted = zi >= -exit_z if pos > 0 else zi <= exit_z
            if adverse or held >= max_hold:
                pos, held, locked = 0, 0, True
            elif reverted:
                pos, held = 0, 0
        elif locked:
            locked = abs(zi) >= exit_z
        elif abs(zi) >= stop_z:
            locked = True
        elif zi <= -entry_z:
            pos = 1
        elif zi >= entry_z:
            pos = -1
        if pos != 0:
            held += 1
        out[i] = pos
    return pd.Series(out, index=zs.index, name="position")


def spread_pnl(
    spread: pd.Series | np.ndarray,
    position: pd.Series | np.ndarray,
) -> pd.Series:
    """Simple PnL on the residual: ``position_{t-1} * Δspread_t``.

    This is the one-bar signal delay: the position decided at the close of
    ``t - 1`` earns the move into ``t``. Educational mark-to-market on the
    cointegration residual (not notionals).
    """
    s = pd.Series(spread, dtype=float)
    p = pd.Series(position, dtype=float).reindex(s.index).fillna(0.0)
    ret = s.diff().fillna(0.0)
    pnl = (p.shift(1).fillna(0.0) * ret).rename("pnl")
    return pnl


class PairsStrategy:
    """Fit the hedge on a formation window, trade only afterwards.

    The hedge ratio is estimated by Engle–Granger on the formation window and
    then frozen. Bars inside the formation window are always flat. Positions
    are close-of-bar decisions; :func:`spread_pnl` applies the one-bar delay
    (a position at ``t`` earns the spread change from ``t`` to ``t + 1``).

    Use :meth:`fit` then :meth:`generate` for an explicit split (pass
    ``generate`` data that does not overlap the fit sample if you want an
    out-of-sample result), or :meth:`run` for a self-contained version.
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

    def generate(self, y: pd.Series, x: pd.Series, *, trade_from: int = 0) -> pd.DataFrame:
        """Build spread, z-score and position signals for ``y`` / ``x``.

        Positions before row ``trade_from`` are forced flat.
        """
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
        if trade_from > 0:
            pos.iloc[:trade_from] = 0
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

    def run(self, y: pd.Series, x: pd.Series, *, formation_window: int) -> pd.DataFrame:
        """Fit on the first ``formation_window`` bars; trade only after them.

        ``formation_window`` is required. The hedge ratio never sees a bar it
        later trades, so there is no look-ahead (v0.1.0 fitted on the full
        sample). The z-score of the first traded bars uses formation-window
        spread history, which is past data.
        """
        if formation_window < 3 or formation_window >= len(y):
            raise PairsError("formation_window must leave bars to trade after it")
        self.fit(y.iloc[:formation_window], x.iloc[:formation_window])
        return self.generate(y, x, trade_from=formation_window)

    # Alias used by some simple backtest adapters
    def positions(self, y: pd.Series, x: pd.Series, *, trade_from: int = 0) -> pd.Series:
        return self.generate(y, x, trade_from=trade_from)["position"]
