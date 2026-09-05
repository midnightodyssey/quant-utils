"""Tests for the educational market-making simulator."""

from __future__ import annotations

import numpy as np
import pytest

from quant_utils.data.sample import make_gbm_mid, make_mean_reverting_mid
from quant_utils.microstructure.market_maker import (
    MarketMakerConfig,
    MarketMakerSimulator,
    simulate_mid_path,
)


def test_simulate_gbm_and_ou_paths():
    gbm = simulate_mid_path(100, process="gbm", seed=1)
    ou = simulate_mid_path(100, process="ou", seed=1, kappa=8.0)
    assert len(gbm) == 101
    assert len(ou) == 101
    assert (gbm > 0).all()
    assert (ou > 0).all()
    sample_gbm = make_gbm_mid(50, seed=2)
    sample_ou = make_mean_reverting_mid(50, seed=2)
    assert len(sample_gbm) == 51
    assert len(sample_ou) == 51


def test_quotes_respect_spread_and_skew_direction():
    mid = make_gbm_mid(30, seed=3)
    cfg = MarketMakerConfig(
        half_spread=0.10,
        inventory_skew=0.05,
        fill_prob=0.0,  # no fills → inventory stays 0 → skew 0
        seed=3,
    )
    result = MarketMakerSimulator(cfg).run(mid)
    path = result.path
    assert np.allclose(path["ask"] - path["bid"], 2 * cfg.half_spread)
    assert np.allclose(path["bid"], path["mid"] - cfg.half_spread)
    assert np.allclose(path["ask"], path["mid"] + cfg.half_spread)


def test_long_inventory_skews_quotes_down():
    # Quotes use start-of-step inventory; fills update inventory afterward.
    # So positive inventory at t shows up as positive skew on quotes at t+1.
    mid = np.full(40, 100.0)
    cfg = MarketMakerConfig(
        half_spread=0.05,
        inventory_skew=0.02,
        fill_prob=0.9,
        touch_boost=0.0,
        max_inventory=5,
        seed=7,
    )
    result = MarketMakerSimulator(cfg).run(mid)
    path = result.path
    prev_inv = path["inventory"].shift(1, fill_value=0.0)
    skewed = path[prev_inv > 0]
    assert len(skewed) > 0
    assert (skewed["skew"] > 0).all()
    # bid = mid - half - skew ⇒ bid < mid - half when long
    assert (skewed["bid"] < skewed["mid"] - cfg.half_spread + 1e-12).all()


def test_inventory_respects_hard_limit():
    mid = make_mean_reverting_mid(200, seed=8, sigma=0.05)
    cfg = MarketMakerConfig(
        half_spread=0.02,
        inventory_skew=0.001,
        fill_prob=0.5,
        touch_boost=0.3,
        max_inventory=3,
        lot_size=1.0,
        seed=8,
    )
    result = MarketMakerSimulator(cfg).run(mid)
    assert result.path["inventory"].max() <= cfg.max_inventory
    assert result.path["inventory"].min() >= -cfg.max_inventory


def test_pnl_attribution_identity():
    mid = make_gbm_mid(100, seed=9, sigma=0.15)
    cfg = MarketMakerConfig(half_spread=0.05, fill_prob=0.25, seed=9)
    result = MarketMakerSimulator(cfg).run(mid)
    # mtm ≈ spread_pnl + inventory_pnl by construction
    mtm = result.path["mtm_pnl"]
    reconstructed = result.path["spread_pnl"] + result.path["inventory_pnl"]
    assert np.allclose(mtm, reconstructed, atol=1e-8)
    assert result.total_pnl == pytest.approx(mtm.iloc[-1])
    assert "MM PnL" in result.summary()


def test_risk_penalty_nonnegative():
    mid = make_gbm_mid(80, seed=10)
    cfg = MarketMakerConfig(inventory_penalty=0.01, fill_prob=0.3, seed=10)
    result = MarketMakerSimulator(cfg).run(mid)
    assert (result.path["risk_penalty"] >= 0).all()
    assert result.risk_penalty >= 0


def test_unknown_process_raises():
    with pytest.raises(ValueError, match="unknown process"):
        simulate_mid_path(10, process="jump")
