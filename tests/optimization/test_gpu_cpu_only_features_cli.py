"""A real (tiny) GPU optimization with fork-only CPU features enabled must complete.

Slow (several minutes on a laptop GPU); skipped where torch/GPU is unavailable, as in CI.
"""

import json
import pickle
import sys

import pytest
from test_hsl_offline_runtime import offline_cli_config
from test_gpu_hsl_cli import add_offline_coin

torch = pytest.importorskip("torch")
pytestmark = pytest.mark.skipif(
    not (torch.backends.mps.is_available() or torch.cuda.is_available()),
    reason="GPU unavailable",
)


@pytest.mark.asyncio
async def test_gpu_optimizer_runs_with_cpu_only_features_and_pins_their_bounds(
    tmp_path, monkeypatch
):
    import msgpack
    from optimize import main

    cfg = offline_cli_config(tmp_path, monkeypatch, "coin")
    add_offline_coin(tmp_path, cfg)
    carton = tmp_path / "carton.csv"
    carton.write_text("date,BTC,ETH\n2023-12-31,1,1\n2024-01-01,1,0\n2024-01-02,1,1\n")
    cfg["live"]["strategy_kind"] = "ema_anchor"
    cfg["live"]["approved_coins"] = {"long": ["BTC", "ETH"], "short": []}
    cfg["backtest"].update(organillo_mode=True, organillo_carton_path=str(carton))
    risk = cfg["bot"]["long"]["risk"]
    risk.update(
        n_positions=2,
        divergence_filter_enabled=True,
        divergence_extended_horizons=True,
        divergence_zscore_threshold=2.5,
        divergence_delay_multiplier=30.0,
        divergence_we_cap_pct=0.15,
        time_stop_max_age_days=0.02,
        time_stop_close_pct=0.2,
        time_stop_close_we_max=0.5,
    )
    bounds = cfg["optimize"]["bounds"]["long"]["risk"]
    bounds.update(
        divergence_zscore_threshold=[2.0, 3.5, 0.25],
        divergence_delay_multiplier=[2.0, 40.0, 2.0],
        time_stop_max_age_days=[0.01, 0.05, 0.01],
        time_stop_close_pct=[0.1, 0.5, 0.05],
    )
    bounds["n_positions"] = [2, 2]
    cfg["optimize"].update(
        backend="gpu", iters=32, n_cpus=1, population_size=4,
        scoring=[{"metric": "adg_usd", "goal": "max"}], limits=[], seed=42,
        enable_overrides=[], compress_results_file=False, write_all_results=True,
    )
    cfg["optimize"]["gpu"].update(
        population_size=4, batch_size=4, exact_workers=1, max_pending_exact=2,
        validate_per_generation=2, drift_probes=1,
    )
    path = tmp_path / "config.json"
    path.write_text(json.dumps(cfg))
    monkeypatch.setattr(sys, "argv", ["optimize", str(path), "--suite", "n"])
    with pytest.raises(SystemExit) as finished:
        await main()
    assert finished.value.code == 0
    (artifact,) = (tmp_path / "optimize_results").rglob("all_results.bin")
    with artifact.open("rb") as handle:
        records = list(msgpack.Unpacker(handle, raw=False))
    assert records
    for record in records:
        saved = record["bot"]["long"]["risk"]
        assert saved["divergence_zscore_threshold"] == 2.5
        assert saved["divergence_delay_multiplier"] == 30.0
        assert saved["time_stop_max_age_days"] == 0.02
        assert saved["time_stop_close_pct"] == 0.2
        assert record["backtest"]["organillo_mode"] is True
    state = pickle.loads((artifact.parent / "checkpoint.pkl").read_bytes())
    assert state["halt_reason"] is None
