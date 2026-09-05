"""
quant_utils.indicators
─────────────────────────────────────────────────────────────────────────────
All indicator functions follow the same contract:
  - Input:  pandas Series (or multiple Series for OHLC indicators)
  - Output: pandas Series aligned to the same index
  - No side effects — pure functions
"""

from __future__ import annotations

import numpy as np
import pandas as pd


def sma(series: pd.Series, period: int) -> pd.Series:
    """
    Simple Moving Average — equal weight to every bar in the window.

    The first (period - 1) values are NaN because there aren't enough bars yet.
    """
    return series.rolling(period).mean()


def ema(series: pd.Series, period: int) -> pd.Series:
    """
    Exponential Moving Average — more weight to recent bars.

    Uses the recursive formula with multiplier = 2 / (period + 1).
    """
    return series.ewm(span=period, adjust=False).mean()


def wma(series: pd.Series, period: int) -> pd.Series:
    """
    Weighted Moving Average — linearly increasing weight toward the most recent bar.

    Weights = [1, 2, 3, ..., period]. Most recent bar gets weight `period`.
    """
    weights = np.arange(1, period + 1)
    return series.rolling(period).apply(
        lambda x: np.dot(x, weights) / weights.sum(),
        raw=True,
    )


def rsi(series: pd.Series, period: int = 14) -> pd.Series:
    """
    Relative Strength Index — measures speed and magnitude of price changes.
    Oscillates between 0 and 100.

    Interpretation:
        RSI > 70  →  overbought (potential short signal)
        RSI < 30  →  oversold  (potential long signal)
    """
    delta = series.diff()
    gain = delta.clip(lower=0)
    loss = (-delta).clip(lower=0)
    avg_gain = gain.rolling(period).mean()
    avg_loss = loss.rolling(period).mean()
    rs = avg_gain / avg_loss
    return 100 - (100 / (1 + rs))


def macd(
    series: pd.Series,
    fast: int = 12,
    slow: int = 26,
    signal: int = 9,
) -> tuple[pd.Series, pd.Series, pd.Series]:
    """
    Moving Average Convergence Divergence.

    Returns (macd_line, signal_line, histogram).
    """
    fast_ema = ema(series, fast)
    slow_ema = ema(series, slow)
    macd_line = fast_ema - slow_ema
    signal_line = macd_line.ewm(span=signal, adjust=False).mean()
    histogram = macd_line - signal_line
    return macd_line, signal_line, histogram


def bollinger_bands(
    series: pd.Series,
    period: int = 20,
    std_dev: float = 2.0,
) -> tuple[pd.Series, pd.Series, pd.Series]:
    """
    Bollinger Bands — volatility envelope around a moving average.

    Returns (upper, middle, lower) bands.
    """
    middle = sma(series, period)
    sigma = series.rolling(period).std()
    upper = middle + std_dev * sigma
    lower = middle - std_dev * sigma
    return upper, middle, lower


def atr(
    high: pd.Series,
    low: pd.Series,
    close: pd.Series,
    period: int = 14,
) -> pd.Series:
    """
    Average True Range — measures market volatility using full price range.

    True Range is the largest of:
      1. High − Low
      2. |High − Previous Close|
      3. |Low  − Previous Close|

    Useful for adaptive stop distances (e.g. 1.5× or 2× ATR).
    """
    prev_close = close.shift(1)
    tr = pd.concat(
        [
            high - low,
            (high - prev_close).abs(),
            (low - prev_close).abs(),
        ],
        axis=1,
    ).max(axis=1)
    return tr.rolling(period).mean()
