"""Tests for the generic pairs primitives and the Engle–Granger test.

Synthetic data only. statsmodels is an optional verification dependency: the
cross-check tests skip when it is not installed.
"""

from __future__ import annotations

import itertools
import math

import numpy as np
import pandas as pd
import pytest

from quant_utils.pairs import (
    PairsError,
    candidate_pairs,
    eg_critical_value,
    eg_pvalue,
    engle_granger,
    half_life,
    positions_from_z,
    rolling_hedge_ratio,
    select_pairs,
    zscore,
)
from quant_utils.pairs.cointegration import _MACKINNON_N2_C

BETA = 1.3
ALPHA = 0.5


def _planted(n=600, beta=BETA, theta=0.05, seed=0):
    """Log-price pair: x a random walk, y = alpha + beta*x + OU(theta)."""
    rng = np.random.default_rng(seed)
    x = np.cumsum(rng.normal(0.0, 0.02, n)) + 4.0
    phi = np.exp(-theta)
    u = np.zeros(n)
    for t in range(1, n):
        u[t] = phi * u[t - 1] + rng.normal(0.0, 0.01)
    idx = pd.bdate_range("2020-01-01", periods=n)
    return (
        pd.Series(ALPHA + beta * x + u, index=idx, name="Y"),
        pd.Series(x, index=idx, name="X"),
    )


def _independent(n, seed):
    rng = np.random.default_rng(seed)
    idx = pd.bdate_range("2020-01-01", periods=n)
    y = pd.Series(np.cumsum(rng.normal(0, 0.01, n)) + 4.0, index=idx)
    x = pd.Series(np.cumsum(rng.normal(0, 0.01, n)) + 4.0, index=idx)
    return y, x


# -- Engle-Granger -----------------------------------------------------------


def test_planted_pair_is_detected():
    y, x = _planted(theta=0.1)
    res = engle_granger(y, x, use_log=False)
    assert res.adf_stat < res.critical_values[0.05]
    assert res.adf_pvalue_approx < 0.01
    assert res.is_cointegrated
    assert res.nobs == len(y)


def test_critical_values_known_asymptotes():
    assert eg_critical_value(0.05, 10**9) == pytest.approx(-3.33613, abs=1e-4)
    assert eg_critical_value(0.01, 10**9) == pytest.approx(-3.89644, abs=1e-4)
    assert eg_critical_value(0.10, 10**9) == pytest.approx(-3.04445, abs=1e-4)


def test_pvalue_monotone_bounded_and_consistent_with_critical_values():
    taus = np.linspace(-8.0, 0.0, 41)
    ps = [eg_pvalue(float(t), 500) for t in taus]
    assert all(0.0 < p <= 1.0 for p in ps)
    assert all(a <= b for a, b in itertools.pairwise(ps))
    for level in (0.01, 0.05, 0.10):
        assert eg_pvalue(eg_critical_value(level, 500), 500) == pytest.approx(level, rel=1e-9)


def test_engle_granger_input_validation():
    y, x = _planted()
    with pytest.raises(PairsError):
        engle_granger(y.iloc[:20], x.iloc[:20], use_log=False)
    with pytest.raises(PairsError):
        engle_granger(y, x, max_lag=3)  # max_lag needs autolag
    with pytest.raises(PairsError):
        engle_granger(y, x, autolag="hqic")  # type: ignore[arg-type]


def test_autolag_picks_a_lag_in_range():
    y, x = _planted(theta=0.1)
    res = engle_granger(y, x, use_log=False, autolag="aic", max_lag=6)
    assert 0 <= res.adf_lags <= 6
    res_b = engle_granger(y, x, use_log=False, autolag="bic", max_lag=6)
    assert res_b.adf_lags <= res.adf_lags + 6


# -- Cross-check vs statsmodels (verification dependency only) ---------------


def test_critical_table_matches_statsmodels():
    av = pytest.importorskip("statsmodels.tsa.adfvalues")
    table = av.tau_c_2010[1]  # N = 2 variables, constant
    for row, level in zip(table, (0.01, 0.05, 0.10), strict=True):
        assert tuple(row) == pytest.approx(_MACKINNON_N2_C[level])


@pytest.mark.parametrize("seed", range(4))
@pytest.mark.parametrize("planted", [True, False])
def test_tau_matches_statsmodels_coint(seed, planted):
    sm = pytest.importorskip("statsmodels.tsa.stattools")
    y, x = _planted(n=400, theta=0.08, seed=seed) if planted else _independent(400, seed)
    for lags in (0, 1, 3):
        ours = engle_granger(y, x, use_log=False, lags=lags)
        tau, _, _ = sm.coint(y, x, trend="c", maxlag=lags, autolag=None)
        assert ours.adf_stat == pytest.approx(tau, abs=1e-8)
    aic = engle_granger(y, x, use_log=False, autolag="aic")
    tau, _, _ = sm.coint(y, x, trend="c", autolag="aic")
    assert aic.adf_stat == pytest.approx(tau, abs=1e-6)


# -- Hedge ratio / half-life / z-score --------------------------------------


def test_rolling_hedge_ratio_recovers_beta_and_is_trailing():
    y, x = _planted(n=1500)
    _, beta = rolling_hedge_ratio(y, x, window=300)
    assert beta.iloc[:299].isna().all()
    err = (beta.dropna() - BETA).abs()
    assert err.mean() < 0.06
    # trailing: changing the future leaves the past unchanged
    y2 = y.copy()
    y2.iloc[1000:] += 5.0
    _, beta2 = rolling_hedge_ratio(y2, x, window=300)
    pd.testing.assert_series_equal(beta.iloc[:1000], beta2.iloc[:1000])


def test_half_life_recovers_planted_theta():
    theta = 0.05
    y, x = _planted(n=4000, theta=theta, seed=3)
    resid = y - ALPHA - BETA * x
    assert half_life(resid) == pytest.approx(math.log(2) / theta, rel=0.15)


def test_half_life_edge_cases():
    y, _ = _independent(2000, 1)
    assert half_life(y) > 100
    alternating = pd.Series([1.0, -1.0] * 30)  # phi = -1 -> reverts within a bar
    assert half_life(alternating) == 0.0
    with pytest.raises(PairsError):
        half_life(pd.Series(np.arange(10.0)))


def test_zscore_is_trailing_and_standardised():
    s = pd.Series(np.random.default_rng(0).normal(size=300))
    z = zscore(s, window=60)
    assert z.iloc[:59].isna().all()
    w = s.iloc[-60:]
    assert z.iloc[-1] == pytest.approx((w.iloc[-1] - w.mean()) / w.std(ddof=1))
    with pytest.raises(PairsError):
        zscore(s, window=1)


# -- Position state machine --------------------------------------------------

_KW = {"entry_z": 2.0, "exit_z": 0.5, "stop_z": 4.0, "max_hold": 40}


def test_positions_require_all_parameters():
    z = pd.Series([0.0, 1.0])
    for missing in _KW:
        kw = {k: v for k, v in _KW.items() if k != missing}
        with pytest.raises(TypeError):
            positions_from_z(z, **kw)  # type: ignore[arg-type]
    with pytest.raises(PairsError):
        positions_from_z(z, **{**_KW, "exit_z": 2.5})
    with pytest.raises(PairsError):
        positions_from_z(z, **{**_KW, "stop_z": 1.5})
    with pytest.raises(PairsError):
        positions_from_z(z, **{**_KW, "max_hold": 0})


def test_positions_enter_exit_and_stop_with_lockout():
    z = pd.Series([0, -1, -2.1, -1.5, -0.4, 0, 2.2, 3.0, 4.5, 3.0, 2.5, 0.2, 2.5])
    pos = positions_from_z(z, **_KW).tolist()
    assert pos == [0, 0, 1, 1, 0, 0, -1, -1, 0, 0, 0, 0, -1]


def test_time_stop_flattens_and_locks():
    z = pd.Series([-2.5, -2.4, -2.3, -2.3, -2.3, -0.1, -2.5])
    pos = positions_from_z(z, **{**_KW, "max_hold": 3}).tolist()
    assert pos == [1, 1, 1, 0, 0, 0, 1]


def test_nan_z_is_flat():
    z = pd.Series([np.nan, np.nan, -3.0, np.nan, -3.0])
    assert positions_from_z(z, **_KW).tolist() == [0, 0, 1, 0, 1]


def test_positions_have_no_lookahead():
    rng = np.random.default_rng(5)
    z = pd.Series(rng.normal(0, 2, 400))
    base = positions_from_z(z, **_KW)
    z2 = z.copy()
    z2.iloc[201:] = rng.normal(0, 3, 199)
    pert = positions_from_z(z2, **_KW)
    pd.testing.assert_series_equal(base.iloc[:201], pert.iloc[:201])


# -- Selection / Bonferroni --------------------------------------------------

_SEL = {"half_life_min": 2.0, "half_life_max": 40.0, "beta_min": 0.5, "beta_max": 2.0}


def _panel():
    y, x = _planted(n=500, theta=0.1, seed=1)
    ya, xa = _independent(500, 21)
    zz, _ = _independent(500, 22)
    cols = {"AAA": np.exp(y.values), "BBB": np.exp(x.values)}
    cols.update({"CCC": np.exp(ya.values), "DDD": np.exp(xa.values), "EEE": np.exp(zz.values)})
    return pd.DataFrame(cols, index=y.index)


def test_candidate_pairs_counts_and_rejects_ambiguity():
    cands = candidate_pairs({"a": ["A", "B", "C"], "b": ["D", "E"], "c": ["F"]})
    assert len(cands) == 4
    assert all(a < b for a, b in cands)
    with pytest.raises(PairsError):
        candidate_pairs({"a": ["A", "B"], "b": ["B", "C"]})


def test_select_pairs_bonferroni_count_and_planted_pair():
    panel = _panel()
    cands = candidate_pairs({"g": list(panel.columns)})
    sel = select_pairs(panel, cands, **_SEL)
    assert sel.n_candidates == len(cands) == 10
    assert sel.n_tested == 10
    assert sel.alpha_adjusted == pytest.approx(0.05 / 10)
    assert ("AAA", "BBB") in {(c.symbol_a, c.symbol_b) for c in sel.selected}
    assert len(sel.selected) <= 2
    assert all(c.pvalue <= sel.alpha_adjusted for c in sel.selected)


def test_select_pairs_counts_untestable_candidates():
    sel = select_pairs(_panel(), [("AAA", "BBB"), ("AAA", "MISSING")], **_SEL)
    assert (sel.n_candidates, sel.n_tested) == (2, 1)


def test_select_pairs_bounds_are_required_and_enforced():
    panel = _panel()
    with pytest.raises(TypeError):
        select_pairs(panel, [("AAA", "BBB")])  # type: ignore[call-arg]
    assert len(select_pairs(panel, [("AAA", "BBB")], **_SEL).selected) == 1
    slow = select_pairs(panel, [("AAA", "BBB")], **{**_SEL, "half_life_max": 3.0})
    assert slow.n_candidates == 1 and slow.selected == ()
    narrow = select_pairs(panel, [("AAA", "BBB")], **{**_SEL, "beta_max": 1.0})
    assert narrow.selected == ()
