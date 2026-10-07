import numpy as np
import pandas as pd
import pytest

from dtrend.config import config_from_dict
from dtrend.data.panel import synthetic_panel, with_universe
from dtrend.nautilus.backtest import run_nautilus_backtest
from dtrend.research.runner import run_research


@pytest.fixture(scope="module")
def setup():
    cfg = config_from_dict(
        {
            "execution": {"starting_equity": 1_000_000},
            "forecast": {"scalar_min_periods": 20},
            "data": {"universe_size": 6},
        }
    )
    panel = with_universe(synthetic_panel(n_symbols=8, n_days=300), cfg.data)
    return cfg, panel


def test_nautilus_matches_research(setup, tmp_path):
    cfg, panel = setup
    nt = run_nautilus_backtest(panel, cfg, work_dir=tmp_path)
    research = run_research(panel, cfg)
    ref = cfg.execution.starting_equity * (1 + research.books["D-TREND"].returns_net).cumprod()
    # Nautilus equity at close T is measured before trading at T; research equity is after the cost of T.
    nt_ret = nt.equity.pct_change().dropna()
    ref_ret = ref.pct_change().reindex(nt_ret.index)
    mask = nt_ret.abs() > 0
    assert np.corrcoef(nt_ret[mask], ref_ret[mask])[0, 1] > 0.999
    last = nt.equity.index[-1]
    assert nt.equity.iloc[-1] == pytest.approx(ref.loc[last], rel=0.01)
    assert nt.funding_paid != 0.0


def test_live_equivalent_matches_precomputed(setup, tmp_path):
    cfg, panel = setup
    small = with_universe(synthetic_panel(n_symbols=5, n_days=160), cfg.data)
    a = run_nautilus_backtest(small, cfg, mode="precomputed", work_dir=tmp_path / "a")
    b = run_nautilus_backtest(small, cfg, mode="live_equivalent", work_dir=tmp_path / "b")
    assert len(a.decisions) == len(b.decisions) > 0
    pd.testing.assert_series_equal(a.decisions["side"], b.decisions["side"])
    np.testing.assert_allclose(a.decisions["qty"], b.decisions["qty"], rtol=1e-2)  # bar prices are rounded to tick
    assert a.equity.iloc[-1] == pytest.approx(b.equity.iloc[-1], rel=1e-4)


def test_funding_settlement_matches_research_ledger(tmp_path):
    cfg = config_from_dict(
        {"execution": {"starting_equity": 1_000_000}, "forecast": {"scalar_min_periods": 20}, "data": {"universe_size": 5}}
    )
    panel = with_universe(synthetic_panel(n_symbols=5, n_days=200), cfg.data)
    panel.funding.loc[:, :] = 0.0
    assert run_nautilus_backtest(panel, cfg, work_dir=tmp_path / "zero").funding_paid == 0.0

    panel.funding.loc[:, :] = 0.001  # every coin, every day: longs pay, shorts receive
    nt = run_nautilus_backtest(panel, cfg, work_dir=tmp_path / "pos")
    sim = run_research(panel, cfg).books["D-TREND"]
    equity_prev = (cfg.execution.starting_equity * (1 + sim.returns_net).cumprod()).shift(1)
    expected = float((sim.funding * equity_prev).loc[: nt.equity.index[-1]].sum())
    assert expected != 0.0
    assert nt.funding_paid == pytest.approx(expected, rel=0.02)


def test_live_node_config_builds_without_network():
    from dtrend.nautilus.live import build_node_config

    cfg = config_from_dict({"data": {"symbols": ["BTCUSDT", "ETHUSDT"]}})
    node_cfg, strat_cfg = build_node_config(cfg, ["BTCUSDT", "ETHUSDT"], environment="testnet")
    assert strat_cfg.dry_run is True
    assert strat_cfg.instrument_ids == ["BTCUSDT-PERP.BINANCE", "ETHUSDT-PERP.BINANCE"]
    assert "BINANCE" in node_cfg.data_clients
