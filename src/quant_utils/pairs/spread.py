"""Spread construction and trailing statistics for pairs trading.

Causality
---------
Every rolling statistic here is *trailing* and **includes bar t**. A value at
bar ``t`` is therefore known at the close of ``t``. Anything that acts on it can
execute no earlier than bar ``t + 1``; the caller applies that one-bar delay
(:func:`quant_utils.pairs.strategy.spread_pnl` does so for residual PnL).
"""

from __future__ import annotations

import math

import numpy as np
import pandas as pd

from quant_utils.pairs.errors import PairsError

#: Fewest observations the half-life estimator will run on by default.
MIN_OBS = 30


def build_spread(
    y: pd.Series | np.ndarray,
    x: pd.Series | np.ndarray,
    hedge_ratio: float,
    *,
    intercept: float = 0.0,
    use_log: bool = True,
) -> pd.Series:
    """Residual spread ``e_t = y*_t - a - b x*_t``.

    By default ``y*`` / ``x*`` are log prices (matching :func:`ols_hedge_ratio`).
    The resulting series is the Engle–Granger residual used for z-scoring.
    """
    y_s = pd.Series(y, dtype=float)
    x_s = pd.Series(x, dtype=float)
    if isinstance(y, pd.Series) and isinstance(x, pd.Series):
        frame = pd.concat([y_s.rename("y"), x_s.rename("x")], axis=1, join="inner").dropna()
        y_s, x_s = frame["y"], frame["x"]
    else:
        n = min(len(y_s), len(x_s))
        y_s = y_s.iloc[:n].reset_index(drop=True)
        x_s = x_s.iloc[:n].reset_index(drop=True)

    if use_log:
        if (y_s <= 0).any() or (x_s <= 0).any():
            raise ValueError("log spread requires strictly positive prices")
        y_v = np.log(y_s.to_numpy(dtype=float))
        x_v = np.log(x_s.to_numpy(dtype=float))
    else:
        y_v = y_s.to_numpy(dtype=float)
        x_v = x_s.to_numpy(dtype=float)

    spread = pd.Series(y_v - intercept - hedge_ratio * x_v, index=y_s.index, name="spread")
    return spread


def rolling_hedge_ratio(
    y: pd.Series, x: pd.Series, window: int
) -> tuple[pd.Series, pd.Series]:
    """Trailing-window OLS ``y = alpha + beta * x``; returns ``(alpha, beta)``.

    The window ends at (and includes) bar t, so the value at t uses no later
    data. The first ``window - 1`` values are NaN. ``y`` and ``x`` are aligned
    by index; pass log prices for an elasticity-style beta.
    """
    if window < 2:
        raise PairsError("window must be >= 2")
    cov = y.rolling(window).cov(x)
    var = x.rolling(window).var()
    beta = cov / var
    alpha = y.rolling(window).mean() - beta * x.rolling(window).mean()
    return alpha, beta


def zscore(spread: pd.Series | np.ndarray, window: int, *, min_periods: int | None = None) -> pd.Series:
    """Trailing z-score ``(e - mean) / std`` over ``window`` bars (ddof=1).

    The window ends at and includes bar t (see the module docstring). A zero
    standard deviation yields NaN rather than ``inf``.
    """
    if window < 2:
        raise PairsError("window must be >= 2")
    s = pd.Series(spread, dtype=float).rename("spread")
    periods = window if min_periods is None else min_periods
    roll = s.rolling(window=window, min_periods=periods)
    z = (s - roll.mean()) / roll.std(ddof=1)
    return z.replace([np.inf, -np.inf], np.nan).rename("zscore")


def rolling_zscore(
    spread: pd.Series | np.ndarray,
    window: int,
    *,
    min_periods: int | None = None,
) -> pd.Series:
    """Alias of :func:`zscore`, kept for backward compatibility.

    The window includes the current observation, so a decision made on this
    z-score is a close-of-bar-t decision to be executed at t + 1.
    """
    return zscore(spread, window, min_periods=min_periods)


def half_life(spread: pd.Series | np.ndarray, *, min_obs: int = MIN_OBS) -> float:
    """Ornstein–Uhlenbeck half-life in bars, from ``ds_t = a + b s_{t-1}``.

    With ``phi = 1 + b`` the half-life is ``-ln 2 / ln phi``. Returns ``inf``
    when ``phi >= 1`` (no mean reversion) and ``0.0`` when ``phi <= 0`` (the
    spread reverts within a bar). This is the exact discrete form; the common
    ``-ln 2 / b`` shortcut understates fast reversion.
    """
    s = np.asarray(spread, dtype=float)
    s = s[np.isfinite(s)]
    if len(s) < max(min_obs, 3):
        raise PairsError(f"need at least {max(min_obs, 3)} finite observations")
    lag, diff = s[:-1], np.diff(s)
    var = float(((lag - lag.mean()) ** 2).sum())
    if var == 0.0:
        raise PairsError("spread has zero variance")
    b = float(((lag - lag.mean()) * (diff - diff.mean())).sum() / var)
    phi = 1.0 + b
    if phi >= 1.0:
        return math.inf
    if phi <= 0.0:
        return 0.0
    return -math.log(2.0) / math.log(phi)
