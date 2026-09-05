"""Vectorised backtesting engine."""

from quant_lab.backtest.engine import (
    BacktestResult,
    WalkForwardResult,
    WalkForwardWindow,
    expand_grid,
    rolling_oos_windows,
    run_backtest,
    summary_table,
    walk_forward,
    window_edges,
)

__all__ = [
    "BacktestResult",
    "WalkForwardResult",
    "WalkForwardWindow",
    "expand_grid",
    "rolling_oos_windows",
    "run_backtest",
    "summary_table",
    "walk_forward",
    "window_edges",
]
