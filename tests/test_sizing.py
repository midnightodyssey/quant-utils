"""Tests for position sizing helpers."""

from quant_lab.risk.sizing import fixed_fraction, kelly, vol_target


def test_fixed_fraction_known_value():
    result = fixed_fraction(capital=100_000, risk_pct=0.01, stop_pct=0.02, price=50.0)
    assert result == 1000


def test_fixed_fraction_scales_with_capital():
    s1 = fixed_fraction(capital=100_000, risk_pct=0.01, stop_pct=0.02, price=50.0)
    s2 = fixed_fraction(capital=200_000, risk_pct=0.01, stop_pct=0.02, price=50.0)
    assert s2 == 2 * s1


def test_fixed_fraction_zero_stop_returns_zero():
    assert fixed_fraction(capital=100_000, risk_pct=0.01, stop_pct=0.0, price=50.0) == 0


def test_fixed_fraction_zero_price_returns_zero():
    assert fixed_fraction(capital=100_000, risk_pct=0.01, stop_pct=0.02, price=0.0) == 0


def test_fixed_fraction_returns_integer():
    assert isinstance(
        fixed_fraction(capital=100_000, risk_pct=0.01, stop_pct=0.03, price=47.0), int
    )


def test_fixed_fraction_wider_stop_smaller_position():
    tight = fixed_fraction(capital=100_000, risk_pct=0.01, stop_pct=0.01, price=50.0)
    wide = fixed_fraction(capital=100_000, risk_pct=0.01, stop_pct=0.02, price=50.0)
    assert tight > wide


def test_kelly_positive_for_good_edge():
    assert kelly(win_rate=0.6, avg_win=0.02, avg_loss=0.02) > 0


def test_kelly_zero_for_breakeven():
    assert abs(kelly(win_rate=0.5, avg_win=0.01, avg_loss=0.01)) < 1e-10


def test_kelly_negative_for_no_edge():
    assert kelly(win_rate=0.4, avg_win=0.01, avg_loss=0.01) < 0


def test_kelly_zero_avg_win_returns_zero():
    assert kelly(win_rate=0.6, avg_win=0.0, avg_loss=0.01) == 0.0


def test_kelly_returns_float():
    assert isinstance(kelly(win_rate=0.55, avg_win=0.015, avg_loss=0.01), float)


def test_kelly_higher_win_rate_bigger_bet():
    k1 = kelly(win_rate=0.55, avg_win=0.02, avg_loss=0.02)
    k2 = kelly(win_rate=0.65, avg_win=0.02, avg_loss=0.02)
    assert k2 > k1


def test_vol_target_known_value():
    result = vol_target(capital=100_000, target_vol=0.10, asset_vol=0.20, price=50.0)
    assert result == 1000


def test_vol_target_higher_vol_smaller_position():
    low_vol = vol_target(capital=100_000, target_vol=0.10, asset_vol=0.10, price=50.0)
    high_vol = vol_target(capital=100_000, target_vol=0.10, asset_vol=0.20, price=50.0)
    assert low_vol > high_vol


def test_vol_target_zero_asset_vol_returns_zero():
    assert vol_target(capital=100_000, target_vol=0.10, asset_vol=0.0, price=50.0) == 0


def test_vol_target_returns_integer():
    assert isinstance(
        vol_target(capital=100_000, target_vol=0.10, asset_vol=0.15, price=50.0), int
    )


def test_vol_target_scales_with_capital():
    s1 = vol_target(capital=100_000, target_vol=0.10, asset_vol=0.20, price=50.0)
    s2 = vol_target(capital=200_000, target_vol=0.10, asset_vol=0.20, price=50.0)
    assert s2 == 2 * s1
