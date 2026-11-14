"""Tests for Engle–Granger pairs trading utilities."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from quant_utils.data.sample import make_cointegrated_pair
from quant_utils.pairs.cointegration import (
    engle_granger,
    ols_hedge_ratio,
    residual_half_life,
)
from quant_utils.pairs.spread import build_spread, rolling_zscore
from quant_utils.pairs.strategy import (
    PairsSignalConfig,
    PairsStrategy,
    generate_signals,
    spread_pnl,
)


def test_ols_hedge_ratio_recovers_known_beta():
    pair = make_cointegrated_pair(n=800, hedge_ratio=0.75, sigma_spread=0.01, seed=1)
    beta, _, resid = ols_hedge_ratio(pair["y"], pair["x"], use_log=True)
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
    out = strat.run(pair["y"], pair["x"], formation_window=200)
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


# --- Regression tests for the v0.1.0 defects (each failed before the fix) ----


def _independent_logs(n: int, seed: int) -> tuple[pd.Series, pd.Series]:
    rng = np.random.default_rng(seed)
    y = pd.Series(np.cumsum(rng.normal(0, 0.01, n)))
    x = pd.Series(np.cumsum(rng.normal(0, 0.01, n)))
    return y, x


def test_regression_eg_uses_cointegration_critical_values():
    """v0.1.0 used plain-ADF values (5% = -2.86); EG needs about -3.34."""
    y, x = _independent_logs(400, 0)
    res = engle_granger(y, x, use_log=False)
    assert res.critical_values[0.05] < -3.30
    assert res.critical_values[0.01] < -3.85
    assert res.critical_values[0.10] < -3.00


def test_regression_eg_false_positive_rate_is_nominal():
    """v0.1.0 rejected the null on ~1 in 4 independent walks at the 5% level."""
    n_sims, rejects = 400, 0
    for seed in range(n_sims):
        y, x = _independent_logs(250, seed)
        rejects += engle_granger(y, x, use_log=False, significance=0.05).is_cointegrated
    rate = rejects / n_sims
    assert 0.02 <= rate <= 0.09, rate


def test_regression_nan_zscore_goes_flat():
    """v0.1.0 held the previous position through a NaN z-score."""
    z = pd.Series([0.0, -2.5, np.nan, -1.0, -2.5])
    assert list(generate_signals(z, entry_z=2.0, exit_z=0.5)) == [0, 1, 0, 0, 1]


def test_regression_run_does_not_fit_on_the_trading_sample():
    """v0.1.0 run() fitted the hedge on the full sample and traded that sample."""
    pair = make_cointegrated_pair(n=500, half_life=12.0, sigma_spread=0.03, seed=5)
    y, x = pair["y"], pair["x"]
    cfg = PairsSignalConfig(entry_z=1.5, exit_z=0.4, z_window=40, use_log=True)
    formation = 250
    base = PairsStrategy(cfg)
    out = base.run(y, x, formation_window=formation)
    # No position inside the formation window.
    assert (out["position"].iloc[:formation] == 0).all()
    # Perturbing data after the cut never changes the fitted hedge ...
    y2, x2 = y.copy(), x.copy()
    rng = np.random.default_rng(1)
    y2.iloc[formation + 50 :] *= np.exp(rng.normal(0, 0.2, len(y) - formation - 50))
    pert = PairsStrategy(cfg)
    out2 = pert.run(y2, x2, formation_window=formation)
    assert pert.hedge_ratio_ == base.hedge_ratio_
    # ... nor any position taken before the perturbation.
    pd.testing.assert_series_equal(
        out["position"].iloc[: formation + 50], out2["position"].iloc[: formation + 50]
    )
