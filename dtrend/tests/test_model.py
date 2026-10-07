import numpy as np
import pandas as pd
import pytest

from dtrend.config import RuleConfig, config_from_dict, load_config
from dtrend.data.panel import synthetic_panel, with_universe
from dtrend.model import forecasts as fc
from dtrend.model import rules
from dtrend.model.system import run_model
from dtrend.research.analysis import sharpe_se
from dtrend.research.backtest import apply_buffer, simulate
from dtrend.research.runner import run_research


@pytest.fixture(scope="module")
def cfg():
    return config_from_dict({"forecast": {"scalar_min_periods": 20}, "data": {"universe_size": 10}})


@pytest.fixture(scope="module")
def panel(cfg):
    return with_universe(synthetic_panel(n_symbols=14, n_days=700), cfg.data)


@pytest.fixture(scope="module")
def model(panel, cfg):
    return run_model(panel, cfg)


def test_example_config_loads():
    cfg = load_config("configs/dtrend.toml")
    assert [leg.name for leg in cfg.legs] == ["ts_momentum", "xs_momentum", "xs_carry"]
    assert sum(len(leg.rules) for leg in cfg.legs) == 21


@pytest.mark.parametrize(
    "raw,msg",
    [
        ({"legs": [{"name": "a", "mode": "zz", "rules": [{"kind": "ewmac", "lookback": 8}]}]}, "mode"),
        ({"legs": [{"name": "a", "mode": "ts", "rules": [{"kind": "foo", "lookback": 8}]}]}, "unknown rule"),
        ({"risk": {"leg_weighting": "fixed"}}, "leg_weights"),
        ({"risk": {"nope": 1}}, "unknown keys"),
    ],
)
def test_config_validation(raw, msg):
    with pytest.raises(ValueError, match=msg):
        config_from_dict(raw)


def test_rule_shapes():
    idx = pd.date_range("2024-01-01", periods=60, freq="D", tz="UTC")
    up = pd.DataFrame({"X": np.linspace(100, 160, 60)}, index=idx)
    assert rules.breakout(up, 20)["X"].iloc[-1] == pytest.approx(20.0)  # at the top of the range
    assert rules.bollinger(up, 20)["X"].iloc[-1] > 0
    sigma = rules.daily_vol(up)
    assert rules.ewmac(up, 8, sigma)["X"].dropna().iloc[-1] > 0
    funding = pd.DataFrame({"X": 0.0003}, index=idx)
    carry = rules.carry(funding, 3, pd.DataFrame({"X": 0.5}, index=idx), 365)
    assert carry["X"].iloc[-1] == pytest.approx(-0.0003 * 365 / 0.5)  # positive funding: longs pay


def test_xs_demean_sums_to_zero(panel):
    raw = panel.returns().rolling(10).sum()
    xs = fc.cross_sectional_demean(raw, panel.universe, 3)
    sums = xs.sum(axis=1)[xs.notna().sum(axis=1) > 0]
    assert np.allclose(sums, 0.0, atol=1e-12)


def test_forecasts_scaled_and_capped(model, cfg):
    for leg in model.legs.values():
        for f in leg.rule_forecasts.values():
            assert f.abs().max().max() <= cfg.forecast.cap + 1e-9
            late = f.iloc[300:]
            assert 6.0 < late.abs().stack().mean() < 16.0  # average |forecast| near 10 (expanding estimate)
        assert leg.forecast.abs().max().max() <= cfg.forecast.cap + 1e-9


def test_model_is_causal(panel, cfg, model):
    for t in (panel.index[120], panel.index[400], panel.index[-1]):
        part = run_model(panel.until(t), cfg)
        np.testing.assert_allclose(part.target.iloc[-1].to_numpy(), model.target.loc[t].to_numpy(), atol=1e-12)


def test_positions_only_in_universe(panel, model):
    assert (model.target.where(~panel.universe, 0.0).abs().to_numpy() < 1e-15).all()


def test_vol_targeting_hits_tau(panel, cfg):
    res = run_research(panel, cfg)
    vol = res.summary.loc["D-TREND", "vol_pct_yr"] / 100
    assert 0.6 * cfg.risk.target_vol < vol < 1.5 * cfg.risk.target_vol


def test_equal_risk_scales_low_vol_leg_up(model):
    scale = model.leg_scale.iloc[-1]
    vols = pd.Series({k: v.returns.iloc[-365:].std() for k, v in model.legs.items()})
    # The lowest-vol leg gets the largest scale (unless both hit a bound).
    assert scale[vols.idxmin()] >= scale[vols.idxmax()]


def test_buffer():
    held = np.array([0.0, 0.10, 0.30, -0.5])
    opt = np.array([0.20, 0.12, 0.20, -0.2])
    width = np.array([0.05, 0.05, 0.05, 0.05])
    np.testing.assert_allclose(apply_buffer(held, opt, width), [0.15, 0.10, 0.25, -0.25])


def test_simulate_costs_and_funding():
    idx = pd.date_range("2024-01-01", periods=3, freq="D", tz="UTC")
    close = pd.DataFrame({"X": [100.0, 110.0, 110.0]}, index=idx)
    funding = pd.DataFrame({"X": [0.0, 0.001, 0.001]}, index=idx)
    target = pd.DataFrame({"X": [1.0, 1.0, 1.0]}, index=idx)
    buffer = pd.DataFrame({"X": [0.5, 0.5, 0.5]}, index=idx)
    sim = simulate(target, buffer, close, funding, cost_per_trade=0.001)
    # Day 0: buy from 0 to the lower buffer edge 0.5 -> cost 0.0005.
    assert sim.costs.iloc[0] == pytest.approx(0.0005)
    # Day 1: +10% price on ~0.5 weight, minus 0.1% funding on that weight.
    w0 = 0.5 / (1 - 0.0005)
    assert sim.returns_gross.iloc[1] == pytest.approx(w0 * 0.10 - w0 * 0.001)
    assert sim.funding.iloc[1] > 0


def test_sharpe_standard_error_matches_thread():
    # Thread: net Sharpe 1.306 +- 0.572 over 2021-01 .. 2026-09 (~5.7 years).
    assert sharpe_se(1.306, 5.7) == pytest.approx(0.572, abs=0.01)


def test_rule_smoothing_default():
    assert rules.smoothing_span(RuleConfig("breakout", 80)) == 20  # Carver L/4
    assert rules.smoothing_span(RuleConfig("ewmac", 16, smooth=0)) == 0
