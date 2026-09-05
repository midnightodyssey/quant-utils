"""Spread construction and rolling z-scores for pairs trading."""

from __future__ import annotations

import numpy as np
import pandas as pd


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


def rolling_zscore(
    spread: pd.Series | np.ndarray,
    window: int,
    *,
    min_periods: int | None = None,
) -> pd.Series:
    """Rolling z-score of the spread: ``(e - μ) / σ`` over ``window``.

    Uses a trailing window that includes the current observation (standard for
    simple demos). For strict no-lookahead research, shift the result by one.
    """
    if window < 2:
        raise ValueError("window must be >= 2")
    s = pd.Series(spread, dtype=float).rename("spread")
    min_periods = window if min_periods is None else min_periods
    mu = s.rolling(window=window, min_periods=min_periods).mean()
    sigma = s.rolling(window=window, min_periods=min_periods).std(ddof=1)
    z = (s - mu) / sigma
    # Avoid inf when sigma collapses
    z = z.replace([np.inf, -np.inf], np.nan).rename("zscore")
    return z
