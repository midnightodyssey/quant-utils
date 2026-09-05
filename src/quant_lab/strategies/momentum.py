"""
quant_lab.strategies.momentum
─────────────────────────────────────────────────────────────────────────────
Momentum / breakout strategies — buy strength, sell weakness.

The core assumption: assets that have been moving strongly in one direction
tend to continue moving in that direction (trend persistence).

Academically: the momentum premium is one of the most well-documented
anomalies in finance (Jegadeesh & Titman, 1993).
"""

from __future__ import annotations

import pandas as pd

from quant_lab.indicators import atr
from quant_lab.strategies.base import Strategy


class PriceBreakout(Strategy):
    """
    Price Breakout Momentum — channel breakout in the Turtle Trading style.

    How it works:
        Long  (+1): today's close breaks ABOVE the N-day high
        Short (-1): today's close breaks BELOW the N-day low
        Flat  (0):  price is within its N-day range

    Look-ahead note:
        We use shift(1) on the rolling high/low so we're comparing today's
        close to the range of the PREVIOUS N days.

    Args:
        period: channel lookback in bars (default 20)
    """

    def __init__(self, period: int = 20):
        self.period = period

    def generate_signals(self, df: pd.DataFrame) -> pd.Series:
        rolling_high = df["Close"].shift(1).rolling(self.period).max()
        rolling_low = df["Close"].shift(1).rolling(self.period).min()

        close = df["Close"]
        signal = pd.Series(0.0, index=df.index)

        signal[close > rolling_high] = 1.0
        signal[close < rolling_low] = -1.0
        signal[rolling_high.isna()] = 0.0

        return signal


class ATRBreakout(Strategy):
    """
    ATR Breakout — channel breakout with a volatility filter.

    Same entry as PriceBreakout, but only enter when current ATR is above
    its own rolling average. This filters false breakouts in quiet markets.

    Args:
        period:     channel lookback and ATR MA period (default 20)
        atr_period: ATR smoothing period (default 14)
    """

    def __init__(self, period: int = 20, atr_period: int = 14):
        self.period = period
        self.atr_period = atr_period

    def generate_signals(self, df: pd.DataFrame) -> pd.Series:
        rolling_high = df["Close"].shift(1).rolling(self.period).max()
        rolling_low = df["Close"].shift(1).rolling(self.period).min()

        atr_vals = atr(df["High"], df["Low"], df["Close"], self.atr_period)
        atr_avg = atr_vals.rolling(self.period).mean()
        high_vol = atr_vals > atr_avg

        close = df["Close"]
        signal = pd.Series(0.0, index=df.index)

        signal[(close > rolling_high) & high_vol] = 1.0
        signal[(close < rolling_low) & high_vol] = -1.0
        signal[rolling_high.isna() | atr_avg.isna()] = 0.0

        return signal
