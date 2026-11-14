"""Engle–Granger cointegration test.

Design choices
--------------
* **Hedge ratio on log prices by default.** Regressing ``log(y)`` on ``log(x)``
  yields an elasticity-style beta that is scale-invariant and standard in
  textbook pairs trading. Pass ``use_log=False`` to fit on raw prices (or to
  pass log prices you have already computed).
* **Residual test is a no-intercept ADF with cointegration critical values.**
  The first-step OLS already contains a constant, so the residual has mean
  zero and the second-step ADF omits the intercept. Its null distribution is
  *not* the Dickey-Fuller one: the residual was chosen to look stationary, so
  the critical values are more negative (5% at about -3.34 versus -2.86). They
  come from the MacKinnon (2010) response surface for ``N = 2`` variables,
  ``cv(T) = b0 + b1/T + b2/T^2``.
* **Fixed ADF lags by default** (``lags=1``). Optionally pick the lag by AIC or
  BIC with ``autolag``; selection happens on a common sample, as in
  ``statsmodels``.
* **No statsmodels at runtime.** Everything is numpy/scipy/pandas.
  ``statsmodels.tsa.stattools.coint`` is used only in the tests, when
  installed, to verify tau and the critical-value table.

p-value method
--------------
:func:`eg_pvalue` interpolates ``log p`` linearly between the finite-sample
1%, 5% and 10% critical values; beyond them the nearest segment's slope is
extended. It is an **approximation**, not an exact p-value. Measured against
the asymptotic MacKinnon (1994) p-values used by ``statsmodels``:

* between roughly the 1% and 10% points it agrees to within about 15%
  (relative), with no consistent sign;
* below the 1% point (tau about -4 and beyond) it **overstates** p, by a factor
  of about 1.2 near the 1% point, growing to several-fold at 0.05% and orders
  of magnitude in the far tail. It is therefore conservative (too hard to
  reject) when a Bonferroni threshold is well below 1%. The extrapolated value
  is a loose upper bound there, not an estimate: do not read a tiny p as
  precise;
* it clips to ``[1e-12, 1]``, and above the 10% point it extends the 5%-10%
  slope, which also overstates p (conservative).

For a Bonferroni gate this errs toward fewer selections, never more.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Literal

import numpy as np
import pandas as pd
from scipy import stats

from quant_utils.pairs.errors import PairsError
from quant_utils.pairs.spread import MIN_OBS, half_life

# MacKinnon (2010) response-surface coefficients for the cointegration
# residual test, N = 2 variables, constant, no trend:
#   cv(T) = b0 + b1 / T + b2 / T^2 + b3 / T^3.
# (MacKinnon, "Critical Values for Cointegration Tests", Queen's Economics
# Working Paper 1227; also tabulated in statsmodels' ``tsa.adfvalues``
# ``tau_c_2010``, N=2 row.) The tests compare this table with statsmodels.
_MACKINNON_N2_C: dict[float, tuple[float, float, float, float]] = {
    0.01: (-3.89644, -10.9519, -33.527, 0.0),
    0.05: (-3.33613, -6.1101, -6.823, 0.0),
    0.10: (-3.04445, -4.2412, -2.720, 0.0),
}

# Plain-ADF (constant) critical values, used by :func:`adf_statistic` only.
_ADF_CRITICAL = {
    0.01: -3.43,
    0.05: -2.86,
    0.10: -2.57,
}

#: Default number of lagged differences in the residual ADF regression.
DEFAULT_ADF_LAGS = 1


@dataclass(frozen=True)
class CointegrationResult:
    """Outcome of an Engle–Granger two-step test on a price pair.

    ``adf_stat`` is the no-intercept ADF t-statistic on the OLS residual (more
    negative is stronger evidence). ``critical_values`` are the finite-sample
    cointegration critical values for this sample size, keyed 0.01/0.05/0.10.
    ``adf_pvalue_approx`` comes from :func:`eg_pvalue`.
    """

    hedge_ratio: float
    intercept: float
    residuals: pd.Series
    adf_stat: float
    adf_pvalue_approx: float
    critical_values: dict[float, float]
    is_cointegrated: bool
    use_log: bool
    nobs: int
    adf_lags: int

    def summary(self) -> str:
        flag = "REJECT unit root (likely cointegrated)" if self.is_cointegrated else (
            "FAIL to reject unit root (not cointegrated at chosen level)"
        )
        return (
            f"Engle–Granger ({'log' if self.use_log else 'level'} prices)\n"
            f"  hedge_ratio={self.hedge_ratio:.6f}  intercept={self.intercept:.6f}\n"
            f"  tau={self.adf_stat:.4f}  approx_p={self.adf_pvalue_approx:.4g}  "
            f"lags={self.adf_lags}  n={self.nobs}\n"
            f"  {flag}"
        )


def eg_critical_value(level: float, nobs: int) -> float:
    """Finite-sample Engle–Granger critical value (N=2, constant) for ``level``.

    ``level`` is 0.01, 0.05 or 0.10; ``nobs`` is the effective sample size of
    the residual ADF regression.
    """
    if level not in _MACKINNON_N2_C:
        raise PairsError("level must be one of 0.01, 0.05, 0.10")
    if nobs < 1:
        raise PairsError("nobs must be positive")
    b0, b1, b2, b3 = _MACKINNON_N2_C[level]
    return b0 + b1 / nobs + b2 / nobs**2 + b3 / nobs**3


def eg_pvalue(tau: float, nobs: int) -> float:
    """Approximate Engle–Granger p-value for a residual-ADF statistic ``tau``.

    Log-linear interpolation in ``log p`` between the 1%, 5% and 10% finite-
    sample critical values; outside that range the nearest segment's slope is
    extended. See the module docstring for how this compares with exact
    asymptotic p-values. In short: close between the 1% and 10% points, and
    **conservative** (overstated) below 1%, which matters when a Bonferroni
    threshold is far below 1%.
    """
    (c1, l1), (c5, l5), (c10, l10) = (
        (eg_critical_value(lv, nobs), math.log(lv)) for lv in (0.01, 0.05, 0.10)
    )
    if tau <= c5:
        logp = l5 + (l5 - l1) / (c5 - c1) * (tau - c5)
    else:
        logp = l5 + (l10 - l5) / (c10 - c5) * (tau - c5)
    return float(min(1.0, max(1e-12, math.exp(logp))))


def ols_hedge_ratio(
    y: pd.Series | np.ndarray,
    x: pd.Series | np.ndarray,
    *,
    use_log: bool = True,
) -> tuple[float, float, pd.Series]:
    """Fit ``y = a + b x + e`` via OLS (optionally on log prices).

    Parameters
    ----------
    y, x:
        Aligned price levels (positive if ``use_log``).
    use_log:
        If True (default), regress ``log(y)`` on ``log(x)``.

    Returns
    -------
    hedge_ratio, intercept, residuals
        Residuals are indexed like the input series when pandas is used,
        otherwise a default ``RangeIndex``.
    """
    y_s, x_s = _align_pair(y, x)
    if use_log:
        if (y_s <= 0).any() or (x_s <= 0).any():
            raise ValueError("log-price OLS requires strictly positive prices")
        y_fit = np.log(y_s.to_numpy(dtype=float))
        x_fit = np.log(x_s.to_numpy(dtype=float))
    else:
        y_fit = y_s.to_numpy(dtype=float)
        x_fit = x_s.to_numpy(dtype=float)

    # Design matrix: [1, x]
    X = np.column_stack([np.ones(len(x_fit)), x_fit])
    beta, *_ = np.linalg.lstsq(X, y_fit, rcond=None)
    intercept = float(beta[0])
    hedge_ratio = float(beta[1])
    resid = pd.Series(y_fit - X @ beta, index=y_s.index, name="residual")
    return hedge_ratio, intercept, resid


def adf_statistic(
    series: pd.Series | np.ndarray,
    *,
    max_lag: int | None = None,
) -> tuple[float, float, int, dict[float, float]]:
    """Constant-only Augmented Dickey–Fuller test statistic (plain ADF).

    This tests a series that is itself a candidate unit root. It is **not**
    valid for an OLS residual: use :func:`engle_granger`, which applies the
    no-intercept regression and the cointegration critical values.

    Regresses ``Δy_t = a + φ y_{t-1} + Σ γ_i Δy_{t-i} + ε``.
    The unit-root null is ``φ = 0``. Lag length defaults to Schwert's rule
    ``floor(12 * (T/100)^{1/4})``.

    Returns
    -------
    adf_stat, approx_pvalue, used_lags, critical_values
    """
    y = np.asarray(series, dtype=float)
    y = y[np.isfinite(y)]
    n = len(y)
    if n < 20:
        raise ValueError(f"need at least 20 observations for ADF, got {n}")

    if max_lag is None:
        max_lag = int(np.floor(12.0 * (n / 100.0) ** 0.25))
    max_lag = int(max(0, min(max_lag, n // 4)))

    dy = np.diff(y)
    y_lag = y[:-1]

    if max_lag > 0:
        rows = []
        targets = []
        for t in range(max_lag, len(dy)):
            row = [1.0, y_lag[t]]
            for i in range(1, max_lag + 1):
                row.append(dy[t - i])
            rows.append(row)
            targets.append(dy[t])
        X = np.asarray(rows, dtype=float)
        target = np.asarray(targets, dtype=float)
    else:
        X = np.column_stack([np.ones(len(dy)), y_lag])
        target = dy

    beta, *_ = np.linalg.lstsq(X, target, rcond=None)
    resid = target - X @ beta
    dof = len(target) - X.shape[1]
    if dof <= 0:
        raise ValueError("insufficient degrees of freedom for ADF")
    sigma2 = float(np.sum(resid**2) / dof)
    xtx_inv = np.linalg.inv(X.T @ X)
    se_phi = float(np.sqrt(sigma2 * xtx_inv[1, 1]))
    if se_phi == 0.0:
        raise ValueError("degenerate ADF regression (zero SE on phi)")
    adf_stat = float(beta[1] / se_phi)

    # Rough interpolation against the plain-ADF critical values; educational.
    approx_p = _approx_adf_pvalue(adf_stat)
    return adf_stat, approx_p, max_lag, dict(_ADF_CRITICAL)


def _adf_nc_fit(u: np.ndarray, lags: int, trim: int) -> tuple[float, int, float]:
    """No-intercept ADF ``du_t = phi u_{t-1} + sum g_k du_{t-k} + e``.

    ``trim`` (>= ``lags``) drops the first ``trim`` differences so that models
    with different lag counts can be compared on one sample. Returns
    ``(tau, n_eff, ssr)``.
    """
    du = np.diff(u)
    n = len(du) - trim
    dep = du[trim:]
    cols = [u[trim : trim + n]]  # u_{t-1}
    cols += [du[trim - k : trim - k + n] for k in range(1, lags + 1)]
    mat = np.column_stack(cols)
    dof = n - mat.shape[1]
    if n < 10 or dof <= 0:
        raise PairsError("too few observations for the ADF regression")
    coef, *_ = np.linalg.lstsq(mat, dep, rcond=None)
    resid = dep - mat @ coef
    ssr = float(resid @ resid)
    if ssr == 0.0:
        raise PairsError("degenerate ADF regression (zero residual variance)")
    se = math.sqrt(ssr / dof * np.linalg.inv(mat.T @ mat)[0, 0])
    return float(coef[0] / se), n, ssr


def _select_lag(u: np.ndarray, max_lag: int, criterion: str) -> int:
    best_lag, best_ic = 0, math.inf
    for k in range(max_lag + 1):
        _, n, ssr = _adf_nc_fit(u, k, max_lag)
        penalty = 2.0 * (k + 1) if criterion == "aic" else (k + 1) * math.log(n)
        ic = n * math.log(ssr / n) + penalty
        if ic < best_ic:
            best_lag, best_ic = k, ic
    return best_lag


def engle_granger(
    y: pd.Series | np.ndarray,
    x: pd.Series | np.ndarray,
    *,
    use_log: bool = True,
    significance: Literal[0.01, 0.05, 0.10] = 0.05,
    lags: int = DEFAULT_ADF_LAGS,
    autolag: Literal["aic", "bic"] | None = None,
    max_lag: int | None = None,
) -> CointegrationResult:
    """Two-step Engle–Granger cointegration test (``y`` regressed on ``x``).

    1. OLS ``y = a + b x`` (log prices if ``use_log``).
    2. No-intercept ADF on the residual, judged against the finite-sample
       cointegration critical values (see the module docstring).

    The test is ordered: ``x`` is the regressor. Do not test both orders and
    keep the better; that double-counts a trial.

    Parameters
    ----------
    lags:
        Fixed number of lagged differences in the ADF regression (default 1).
        Ignored when ``autolag`` is set.
    autolag:
        ``None`` (default) uses ``lags``. ``"aic"`` / ``"bic"`` choose the lag
        in ``0..max_lag`` by that information criterion.
    max_lag:
        Upper bound for ``autolag`` (default: ``ceil(12 (T/100)^{1/4})``
        capped at ``T // 2 - 1``). Only valid together with ``autolag``.

    Notes
    -----
    v0.1.0 ran a plain constant-ADF with plain-ADF critical values and a
    Schwert-rule lag; that was anti-conservative. ``max_lag`` used to set the
    fixed lag; use ``lags`` for that now.
    """
    if significance not in (0.01, 0.05, 0.10):
        raise PairsError("significance must be one of 0.01, 0.05, 0.10")
    if autolag not in (None, "aic", "bic"):
        raise PairsError("autolag must be None, 'aic' or 'bic'")
    if max_lag is not None and autolag is None:
        raise PairsError("max_lag only applies with autolag; use lags= for a fixed lag")
    if lags < 0:
        raise PairsError("lags must be >= 0")

    hedge_ratio, intercept, residuals = ols_hedge_ratio(y, x, use_log=use_log)
    u = residuals.to_numpy(dtype=float)
    if len(u) < MIN_OBS:
        raise PairsError(f"need at least {MIN_OBS} observations, got {len(u)}")

    if autolag is not None:
        cap = math.ceil(12.0 * (len(u) / 100.0) ** 0.25) if max_lag is None else max_lag
        cap = max(0, min(cap, len(u) // 2 - 1))
        lags = _select_lag(u, cap, autolag)

    tau, n_eff, _ = _adf_nc_fit(u, lags, lags)
    crit = {lv: eg_critical_value(lv, n_eff) for lv in (0.01, 0.05, 0.10)}
    return CointegrationResult(
        hedge_ratio=hedge_ratio,
        intercept=intercept,
        residuals=residuals,
        adf_stat=tau,
        adf_pvalue_approx=eg_pvalue(tau, n_eff),
        critical_values=crit,
        is_cointegrated=tau < crit[float(significance)],
        use_log=use_log,
        nobs=len(u),
        adf_lags=lags,
    )


def residual_half_life(residuals: pd.Series | np.ndarray) -> float:
    """Ornstein–Uhlenbeck half-life of the residual (see :func:`half_life`).

    Uses the exact discrete form ``-ln 2 / ln(1 + b)``; v0.1.0 used
    ``-ln 2 / b``, which understates fast reversion. Returns ``inf`` when the
    residual does not mean-revert.
    """
    return half_life(residuals, min_obs=3)


def _align_pair(
    y: pd.Series | np.ndarray,
    x: pd.Series | np.ndarray,
) -> tuple[pd.Series, pd.Series]:
    y_s = pd.Series(y, dtype=float).rename("y")
    x_s = pd.Series(x, dtype=float).rename("x")
    if isinstance(y, pd.Series) and isinstance(x, pd.Series):
        aligned = pd.concat([y_s, x_s], axis=1, join="inner").dropna()
        if aligned.empty:
            raise ValueError("no overlapping non-null observations")
        return aligned["y"], aligned["x"]
    n = min(len(y_s), len(x_s))
    y_s = y_s.iloc[:n].reset_index(drop=True)
    x_s = x_s.iloc[:n].reset_index(drop=True)
    mask = y_s.notna() & x_s.notna()
    return y_s[mask], x_s[mask]


def _approx_adf_pvalue(stat: float) -> float:
    """Piecewise-linear interpolation against plain-ADF critical values."""
    points = sorted(_ADF_CRITICAL.items())  # (p, crit) ascending p
    crit_p = sorted(((c, p) for p, c in points), key=lambda t: t[0])
    if stat <= crit_p[0][0]:
        return 0.01
    if stat >= -1.6:  # well above 10% critical ≈ -2.57
        return float(min(0.99, stats.norm.sf(-stat / 2.0)))
    for i in range(len(crit_p) - 1):
        c0, p0 = crit_p[i]
        c1, p1 = crit_p[i + 1]
        if c0 <= stat <= c1:
            w = (stat - c0) / (c1 - c0) if c1 != c0 else 0.0
            return float(p0 + w * (p1 - p0))
    c_lo, p_lo = crit_p[-1]
    w = (stat - c_lo) / (-1.6 - c_lo) if -1.6 != c_lo else 1.0
    return float(min(0.99, p_lo + w * (0.5 - p_lo)))
