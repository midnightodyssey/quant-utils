"""Engle–Granger cointegration helpers.

Design choices
--------------
* **Hedge ratio on log prices by default.** Regressing ``log(y)`` on ``log(x)``
  yields an elasticity-style beta that is scale-invariant and standard in
  textbook pairs trading. Pass ``use_log=False`` to fit on raw prices instead.
* **ADF without statsmodels.** We implement a constant-only Augmented
  Dickey–Fuller regression with OLS and MacKinnon approximate critical values.
  This keeps dependencies to numpy/scipy/pandas while remaining educationally
  faithful to Engle–Granger step 2. For production research prefer
  ``statsmodels.tsa.stattools.adfuller``.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

import numpy as np
import pandas as pd
from scipy import stats

# MacKinnon (1991/2010) approximate critical values for ADF with constant,
# no trend, large-n limit. Used as a teaching reference, not a full lookup table.
_ADF_CRITICAL = {
    0.01: -3.43,
    0.05: -2.86,
    0.10: -2.57,
}


@dataclass(frozen=True)
class CointegrationResult:
    """Outcome of an Engle–Granger two-step test on a price pair."""

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
            f"  ADF={self.adf_stat:.4f}  approx_p={self.adf_pvalue_approx:.4f}  "
            f"lags={self.adf_lags}  n={self.nobs}\n"
            f"  {flag}"
        )


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
    """Constant-only Augmented Dickey–Fuller test statistic.

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

    # Align for lagged differences: drop first max_lag of dy / y_lag
    if max_lag > 0:
        rows = []
        targets = []
        for t in range(max_lag, len(dy)):
            # y_{t} level is y_lag[t] corresponding to dy[t] = y[t+1]-y[t]
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

    beta, residuals, _, _ = np.linalg.lstsq(X, target, rcond=None)
    resid = target - X @ beta
    dof = len(target) - X.shape[1]
    if dof <= 0:
        raise ValueError("insufficient degrees of freedom for ADF")
    sigma2 = float(np.sum(resid**2) / dof)
    # Covariance of OLS: σ² (X'X)^{-1}
    xtx_inv = np.linalg.inv(X.T @ X)
    se_phi = float(np.sqrt(sigma2 * xtx_inv[1, 1]))
    if se_phi == 0.0:
        raise ValueError("degenerate ADF regression (zero SE on phi)")
    adf_stat = float(beta[1] / se_phi)

    # Rough MacKinnon-style p-value via interpolation against critical values.
    # Maps more-negative stats to smaller p-values; educational approximation only.
    approx_p = _approx_adf_pvalue(adf_stat)
    return adf_stat, approx_p, max_lag, dict(_ADF_CRITICAL)


def engle_granger(
    y: pd.Series | np.ndarray,
    x: pd.Series | np.ndarray,
    *,
    use_log: bool = True,
    significance: Literal[0.01, 0.05, 0.10] = 0.05,
    max_lag: int | None = None,
) -> CointegrationResult:
    """Two-step Engle–Granger cointegration test.

    1. OLS hedge ratio (log or level prices).
    2. ADF on the residual spread; reject unit root ⇒ evidence of cointegration.
    """
    hedge_ratio, intercept, residuals = ols_hedge_ratio(y, x, use_log=use_log)
    adf_stat, approx_p, lags, crit = adf_statistic(residuals, max_lag=max_lag)
    threshold = crit[float(significance)]
    is_coint = adf_stat < threshold
    return CointegrationResult(
        hedge_ratio=hedge_ratio,
        intercept=intercept,
        residuals=residuals,
        adf_stat=adf_stat,
        adf_pvalue_approx=approx_p,
        critical_values=crit,
        is_cointegrated=is_coint,
        use_log=use_log,
        nobs=len(residuals),
        adf_lags=lags,
    )


def residual_half_life(residuals: pd.Series | np.ndarray) -> float:
    """Ornstein–Uhlenbeck half-life estimate from AR(1) on the residual.

    ``Δe_t = λ e_{t-1} + ε`` ⇒ half-life = ``-ln(2) / λ`` when ``λ < 0``.
    Returns ``inf`` if the residual does not mean-revert under this fit.
    """
    e = np.asarray(residuals, dtype=float)
    e = e[np.isfinite(e)]
    if len(e) < 3:
        raise ValueError("need at least 3 residual observations")
    lag = e[:-1]
    diff = np.diff(e)
    X = np.column_stack([np.ones(len(lag)), lag])
    beta, *_ = np.linalg.lstsq(X, diff, rcond=None)
    lam = float(beta[1])
    if lam >= 0:
        return float("inf")
    return float(-np.log(2.0) / lam)


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
    """Piecewise-linear interpolation against MacKinnon critical values."""
    # More negative ⇒ smaller p. Clamp outside [0.01, 0.99] for sanity.
    points = sorted(_ADF_CRITICAL.items())  # (p, crit) ascending p
    # Work in (crit, p) space sorted by crit ascending (more negative first)
    crit_p = sorted(((c, p) for p, c in points), key=lambda t: t[0])
    if stat <= crit_p[0][0]:
        return 0.01
    if stat >= -1.6:  # well above 10% critical ≈ -2.57
        # Soft map towards 1 using a normal-ish tail heuristic
        return float(min(0.99, stats.norm.sf(-stat / 2.0)))
    # Interpolate between tabulated points
    for i in range(len(crit_p) - 1):
        c0, p0 = crit_p[i]
        c1, p1 = crit_p[i + 1]
        if c0 <= stat <= c1:
            w = (stat - c0) / (c1 - c0) if c1 != c0 else 0.0
            return float(p0 + w * (p1 - p0))
    # Between 10% crit and -1.6
    c_lo, p_lo = crit_p[-1]
    w = (stat - c_lo) / (-1.6 - c_lo) if -1.6 != c_lo else 1.0
    return float(min(0.99, p_lo + w * (0.5 - p_lo)))
