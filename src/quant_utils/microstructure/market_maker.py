"""Avellaneda–Stoikov-lite market-making simulator.

Educational single-asset quote engine:

* Mid price from GBM or OU (mean-reverting) path.
* Quotes: ``bid = mid - half_spread - skew``, ``ask = mid + half_spread - skew``
  where inventory skew pushes quotes down when long (encourage sells) and up
  when short.
* Fill model: each step, independent Poisson/Bernoulli fill attempts on the
  bid and ask with base intensity scaled by how "competitive" the quote is
  relative to the mid move (price crossing / touching quotes).
* Tracks inventory, cash, mark-to-market PnL, and a coarse split between
  earned spread and inventory (mark) PnL.
* Hard inventory limit plus optional quadratic inventory risk penalty in the
  reported objective (does not alter quotes beyond the skew term).
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
import pandas as pd


@dataclass(frozen=True)
class MarketMakerConfig:
    """Quote, fill, and risk parameters for the simulator."""

    half_spread: float = 0.05
    inventory_skew: float = 0.01
    """Quote shift per unit of inventory (Avellaneda–Stoikov reservation tilt)."""
    max_inventory: int = 20
    fill_prob: float = 0.15
    """Base probability of a one-lot fill on each side per step when quote is live."""
    touch_boost: float = 0.35
    """Added fill probability when mid crosses/touches the quote."""
    inventory_penalty: float = 0.001
    """Quadratic risk penalty weight: ``penalty * inventory^2`` per step."""
    lot_size: float = 1.0
    seed: int | None = 42


@dataclass
class MarketMakerResult:
    """Time series and summary statistics from a simulation run."""

    path: pd.DataFrame
    total_pnl: float
    spread_pnl: float
    inventory_pnl: float
    risk_penalty: float
    final_inventory: float
    config: MarketMakerConfig = field(repr=False)

    def summary(self) -> str:
        return (
            f"MM PnL={self.total_pnl:.4f}  "
            f"spread={self.spread_pnl:.4f}  "
            f"inventory={self.inventory_pnl:.4f}  "
            f"penalty={self.risk_penalty:.4f}  "
            f"final_q={self.final_inventory:.1f}"
        )


def simulate_mid_path(
    n_steps: int,
    *,
    x0: float = 100.0,
    mu: float = 0.0,
    sigma: float = 0.2,
    dt: float = 1 / 252,
    process: str = "gbm",
    kappa: float = 5.0,
    theta: float | None = None,
    seed: int | None = 42,
) -> pd.Series:
    """Simulate a mid-price path.

    Parameters
    ----------
    process:
        ``"gbm"`` — geometric Brownian motion.
        ``"ou"`` / ``"mean_reverting"`` — Ornstein–Uhlenbeck on the log-price
        (mean-reverting mid), useful for inventory-risk demos.
    """
    if n_steps < 1:
        raise ValueError("n_steps must be >= 1")
    rng = np.random.default_rng(seed)
    theta = x0 if theta is None else theta

    mids = np.empty(n_steps + 1, dtype=float)
    mids[0] = float(x0)

    if process.lower() in {"ou", "mean_reverting", "mean-reverting"}:
        # OU on price level with reflecting floor to keep mid positive
        for t in range(n_steps):
            shock = rng.normal() * sigma * np.sqrt(dt) * x0
            mids[t + 1] = mids[t] + kappa * (theta - mids[t]) * dt + shock
            mids[t + 1] = max(mids[t + 1], 1e-6)
    elif process.lower() == "gbm":
        for t in range(n_steps):
            z = rng.normal()
            mids[t + 1] = mids[t] * np.exp((mu - 0.5 * sigma**2) * dt + sigma * np.sqrt(dt) * z)
    else:
        raise ValueError(f"unknown process={process!r}; use 'gbm' or 'ou'")

    return pd.Series(mids, name="mid")


class MarketMakerSimulator:
    """Discrete-time single-asset market maker with inventory skew."""

    def __init__(self, config: MarketMakerConfig | None = None) -> None:
        self.config = config or MarketMakerConfig()

    def run(self, mid: pd.Series | np.ndarray) -> MarketMakerResult:
        """Simulate quoting and fills along a mid-price path."""
        cfg = self.config
        mid_s = pd.Series(mid, dtype=float).reset_index(drop=True)
        if len(mid_s) < 2:
            raise ValueError("mid path needs at least 2 points")
        if cfg.half_spread < 0:
            raise ValueError("half_spread must be >= 0")
        if not 0.0 <= cfg.fill_prob <= 1.0:
            raise ValueError("fill_prob must be in [0, 1]")

        rng = np.random.default_rng(cfg.seed)
        n = len(mid_s)
        inventory = 0.0
        cash = 0.0
        spread_pnl_cum = 0.0

        rows: list[dict[str, float]] = []
        prev_mid = float(mid_s.iloc[0])

        for t in range(n):
            m = float(mid_s.iloc[t])
            # Reservation / skew: long inventory → lower quotes (attract sells)
            skew = cfg.inventory_skew * inventory
            bid = m - cfg.half_spread - skew
            ask = m + cfg.half_spread - skew
            if bid >= ask:
                # Extreme skew: collapse to a one-tick book around mid
                mid_tick = max(cfg.half_spread, 1e-8)
                bid = m - mid_tick
                ask = m + mid_tick

            bid_live = inventory < cfg.max_inventory
            ask_live = inventory > -cfg.max_inventory

            bid_fill = 0.0
            ask_fill = 0.0

            # Fill model: base Bernoulli intensity + boost when mid crosses quote
            if bid_live:
                p_bid = cfg.fill_prob
                if m <= bid or prev_mid > bid >= m:
                    p_bid = min(1.0, p_bid + cfg.touch_boost)
                if rng.random() < p_bid:
                    bid_fill = cfg.lot_size
                    cash -= bid * bid_fill
                    inventory += bid_fill
                    # Half-spread earned vs contemporaneous mid
                    spread_pnl_cum += (m - bid) * bid_fill

            if ask_live:
                p_ask = cfg.fill_prob
                if m >= ask or prev_mid < ask <= m:
                    p_ask = min(1.0, p_ask + cfg.touch_boost)
                if rng.random() < p_ask:
                    ask_fill = cfg.lot_size
                    cash += ask * ask_fill
                    inventory -= ask_fill
                    spread_pnl_cum += (ask - m) * ask_fill

            mtm = cash + inventory * m
            inv_pnl = mtm - spread_pnl_cum
            penalty = cfg.inventory_penalty * (inventory**2)
            objective = mtm - penalty

            rows.append(
                {
                    "mid": m,
                    "bid": bid,
                    "ask": ask,
                    "skew": skew,
                    "bid_fill": bid_fill,
                    "ask_fill": ask_fill,
                    "inventory": inventory,
                    "cash": cash,
                    "mtm_pnl": mtm,
                    "spread_pnl": spread_pnl_cum,
                    "inventory_pnl": inv_pnl,
                    "risk_penalty": penalty,
                    "objective": objective,
                }
            )
            prev_mid = m

        path = pd.DataFrame(rows)
        final_mtm = float(path["mtm_pnl"].iloc[-1])
        final_spread = float(path["spread_pnl"].iloc[-1])
        final_inv_pnl = float(path["inventory_pnl"].iloc[-1])
        total_penalty = float(path["risk_penalty"].sum())

        return MarketMakerResult(
            path=path,
            total_pnl=final_mtm,
            spread_pnl=final_spread,
            inventory_pnl=final_inv_pnl,
            risk_penalty=total_penalty,
            final_inventory=float(inventory),
            config=cfg,
        )
