"""
quant_utils.data.sample

Generate synthetic series for demos and tests — no network required at import.

Includes OHLCV bars plus cointegrated pairs and mid-price paths used by the
pairs-trading and market-making modules.
"""

from __future__ import annotations

import numpy as np
import pandas as pd


def make_ohlcv(
    n: int = 252,
    start_price: float = 100.0,
    drift: float = 0.0003,
    vol: float = 0.01,
    seed: int = 42,
    start: str = "2020-01-01",
) -> pd.DataFrame:
    """
    Generate a synthetic OHLCV DataFrame with a DatetimeIndex (business days).

    Uses a geometric random walk for Close, then derives High/Low/Open with
    modest intraday ranges. Volume is constant for simplicity.

    Args:
        n:           number of bars
        start_price: initial close price
        drift:       mean daily log-return
        vol:         daily return standard deviation
        seed:        RNG seed for reproducibility
        start:       start date string for the index

    Returns:
        DataFrame with columns Open, High, Low, Close, Volume
    """
    rng = np.random.default_rng(seed)
    returns = rng.normal(drift, vol, n)
    close = start_price * np.cumprod(1.0 + returns)
    idx = pd.bdate_range(start, periods=n)

    high = close * (1.0 + np.abs(rng.normal(0.0, 0.003, n)))
    low = close * (1.0 - np.abs(rng.normal(0.0, 0.003, n)))
    open_ = close * (1.0 + rng.normal(0.0, 0.001, n))
    # Ensure High/Low bracket Open and Close
    high = np.maximum(high, np.maximum(open_, close))
    low = np.minimum(low, np.minimum(open_, close))

    return pd.DataFrame(
        {
            "Open": open_,
            "High": high,
            "Low": low,
            "Close": close,
            "Volume": np.full(n, 1_000_000.0),
        },
        index=idx,
    )


def make_trending_ohlcv(
    n: int = 200,
    start_price: float = 100.0,
    end_price: float = 200.0,
    start: str = "2020-01-01",
) -> pd.DataFrame:
    """Deterministic linearly rising Close series wrapped as OHLCV."""
    close = np.linspace(start_price, end_price, n)
    idx = pd.bdate_range(start, periods=n)
    return pd.DataFrame(
        {
            "Open": close * 0.999,
            "High": close * 1.005,
            "Low": close * 0.995,
            "Close": close,
            "Volume": np.full(n, 1_000_000.0),
        },
        index=idx,
    )


def make_cointegrated_pair(
    n: int = 500,
    *,
    hedge_ratio: float = 0.8,
    intercept: float = 0.0,
    x0: float = 100.0,
    sigma_x: float = 0.01,
    sigma_spread: float = 0.02,
    half_life: float = 20.0,
    seed: int | None = 42,
) -> pd.DataFrame:
    """Generate a synthetic cointegrated log-price pair.

    Construction
    ------------
    * ``x`` follows a random walk in log space (GBM drift-free).
    * Residual ``e_t`` is AR(1) with mean-reversion calibrated to ``half_life``.
    * ``log(y) = intercept + hedge_ratio * log(x) + e``.

    Returns columns ``x``, ``y`` as positive price levels.
    """
    if n < 2:
        raise ValueError("n must be >= 2")
    if half_life <= 0:
        raise ValueError("half_life must be > 0")

    rng = np.random.default_rng(seed)
    # AR(1) coefficient from half-life: hl = -ln(2) / ln(φ) ⇒ φ = 2^{-1/hl}
    phi = float(2.0 ** (-1.0 / half_life))
    innov_scale = sigma_spread * np.sqrt(max(1e-12, 1.0 - phi**2))

    log_x = np.empty(n, dtype=float)
    log_x[0] = np.log(x0)
    for t in range(1, n):
        log_x[t] = log_x[t - 1] + rng.normal(0.0, sigma_x)

    e = np.empty(n, dtype=float)
    e[0] = rng.normal(0.0, sigma_spread)
    for t in range(1, n):
        e[t] = phi * e[t - 1] + rng.normal(0.0, innov_scale)

    log_y = intercept + hedge_ratio * log_x + e
    return pd.DataFrame(
        {
            "x": np.exp(log_x),
            "y": np.exp(log_y),
        }
    )


def make_gbm_mid(
    n_steps: int = 252,
    *,
    x0: float = 100.0,
    mu: float = 0.0,
    sigma: float = 0.2,
    dt: float = 1 / 252,
    seed: int | None = 42,
) -> pd.Series:
    """GBM mid-price path (wrapper around the microstructure helper)."""
    from quant_utils.microstructure.market_maker import simulate_mid_path

    return simulate_mid_path(
        n_steps, x0=x0, mu=mu, sigma=sigma, dt=dt, process="gbm", seed=seed
    )


def make_mean_reverting_mid(
    n_steps: int = 252,
    *,
    x0: float = 100.0,
    kappa: float = 5.0,
    theta: float | None = None,
    sigma: float = 0.2,
    dt: float = 1 / 252,
    seed: int | None = 42,
) -> pd.Series:
    """Mean-reverting mid-price path for inventory-risk demos."""
    from quant_utils.microstructure.market_maker import simulate_mid_path

    return simulate_mid_path(
        n_steps,
        x0=x0,
        sigma=sigma,
        dt=dt,
        process="ou",
        kappa=kappa,
        theta=theta,
        seed=seed,
    )
