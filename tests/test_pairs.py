"""Tests for Engle–Granger pairs trading utilities."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from quant_lab.data.sample import make_cointegrated_pair
from quant_lab.pairs.cointegration import (
    engle_granger,
    ols_hedge_ratio,
    residual_half_life,
)
from quant_lab.pairs.spread import build_spread, rolling_zscore
from quant_lab.pairs.strategy import (
    PairsSignalConfig,
    PairsStrategy,
    generate_signals,
    spread_pnl,
)


def test_ols_hedge_ratio_recovers_known_beta():
    pair = make_cointegrated_pair(n=800, hedge_ratio=0.75, sigma_spread=0.01, seed=1)
    beta, intercept, resid = ols_hedge_ratio(pair["y"], pair["x"], use_log=True)
    assert beta == pytest.approx(0.75, abs=0.05)
    assert len(resid) == len(pair)
    assert np.isfinite(resid).all()


def test_engle_granger_detects_cointegrated_pair():
    pair = make_cointegrated_pair(n=600, hedge_ratio=0.9, half_life=15.0, seed=2)
    result = engle_granger(pair["y"], pair["x"], use_log=True, significance=0.05)
    assert result.is_cointegrated
    assert result.adf_stat < result.critical_values[0.05]
    assert result.use_log is True
    assert "cointegrated" in result.summary().lower() or "REJECT" in result.summary()


def test_engle_granger_rejects_independent_walks():
    rng = np.random.default_rng(99)
    n = 400
    x = 100 * np.exp(np.cumsum(rng.normal(0, 0.01, n)))
    y = 100 * np.exp(np.cumsum(rng.normal(0, 0.01, n)))
    result = engle_granger(y, x, use_log=True, significance=0.01)
    # Independent I(1) series should usually fail at 1%; allow rare false positives
    # by checking ADF is not extremely negative.
    assert result.adf_stat > result.critical_values[0.01] or not result.is_cointegrated


def test_residual_half_life_is_finite_for_mr_spread():
    pair = make_cointegrated_pair(n=500, half_life=25.0, seed=3)
    _, _, resid = ols_hedge_ratio(pair["y"], pair["x"], use_log=True)
    hl = residual_half_life(resid)
    assert np.isfinite(hl)
    assert 5.0 < hl < 80.0


def test_build_spread_and_zscore_shapes():
    pair = make_cointegrated_pair(n=200, seed=4)
    beta, a, _ = ols_hedge_ratio(pair["y"], pair["x"])
    spread = build_spread(pair["y"], pair["x"], beta, intercept=a, use_log=True)
    z = rolling_zscore(spread, window=30)
    assert len(spread) == 200
    assert z.isna().sum() >= 29  # warm-up
    assert z.dropna().std() == pytest.approx(1.0, abs=0.3)


def test_generate_signals_entry_exit_rules():
    # Hand-crafted z path: enter short, hold, exit, enter long, exit
    z = pd.Series(
        [0.0, 2.5, 1.5, 0.3, 0.0, -2.5, -1.0, -0.2, 0.0],
        dtype=float,
    )
    pos = generate_signals(z, entry_z=2.0, exit_z=0.5)
    assert list(pos) == [0, -1, -1, 0, 0, 1, 1, 0, 0]


def test_generate_signals_rejects_bad_thresholds():
    z = pd.Series([0.0, 1.0])
    with pytest.raises(ValueError):
        generate_signals(z, entry_z=1.0, exit_z=1.0)


def test_pairs_strategy_run_produces_pnl():
    pair = make_cointegrated_pair(n=400, half_life=12.0, sigma_spread=0.03, seed=5)
    strat = PairsStrategy(
        PairsSignalConfig(entry_z=1.5, exit_z=0.4, z_window=40, use_log=True)
    )
    out = strat.run(pair["y"], pair["x"])
    assert {"spread", "zscore", "position", "pnl"} <= set(out.columns)
    assert set(out["position"].dropna().unique()).issubset({-1, 0, 1})
    # PnL should be defined and finite
    assert np.isfinite(out["pnl"].fillna(0)).all()
    assert strat.hedge_ratio_ is not None


def test_spread_pnl_uses_lagged_position():
    spread = pd.Series([0.0, 1.0, 1.5, 1.0])
    position = pd.Series([0, 1, 1, 0])
    pnl = spread_pnl(spread, position)
    # position at t=1 is 0 (shifted), t=2 uses pos=1 → Δ=0.5, t=3 uses pos=1 → Δ=-0.5
    assert pnl.iloc[2] == pytest.approx(0.5)
    assert pnl.iloc[3] == pytest.approx(-0.5)
