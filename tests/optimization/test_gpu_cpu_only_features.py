"""GPU screening does not model fork-only CPU features; it must warn, not fail."""

import logging

import pytest

import optimization.backends.gpu_backend as gpu_backend
from config.schema import get_template_config
from optimization.backends.gpu_backend import (
    _validate_gpu_static_scope,
    pin_cpu_only_feature_bounds,
)


@pytest.fixture(autouse=True)
def _fresh_warnings(monkeypatch):
    monkeypatch.setattr(gpu_backend, "_WARNED_CPU_ONLY_FEATURES", set())


def _trailing_config():
    config = get_template_config()
    config["live"]["strategy_kind"] = "trailing_martingale"
    return config


def test_plain_trailing_martingale_config_is_accepted_without_warning(caplog):
    with caplog.at_level(logging.WARNING):
        assert _validate_gpu_static_scope(_trailing_config()) == "trailing_martingale"
    assert "does not model" not in caplog.text


@pytest.mark.parametrize(
    "mutate,expected",
    [
        (lambda c: c["backtest"].update(organillo_mode=True), "organillo_mode"),
        (lambda c: c["bot"]["long"]["risk"].update(divergence_filter_enabled=True), "divergence"),
        (lambda c: c["bot"]["short"]["risk"].update(time_stop_max_age_days=1.0), "time-based stops"),
        (
            lambda c: c.update(
                coin_overrides={"BTC": {"bot": {"long": {"risk": {"time_stop_max_age_days": 2.0}}}}}
            ),
            "coin_overrides time-based stops",
        ),
    ],
)
def test_cpu_only_features_warn_instead_of_failing(caplog, mutate, expected):
    config = _trailing_config()
    mutate(config)
    with caplog.at_level(logging.WARNING):
        _validate_gpu_static_scope(config)
        _validate_gpu_static_scope(config)
    assert expected in caplog.text
    assert caplog.text.count("GPU screening does not model") == 1


def test_searchable_cpu_only_bounds_are_pinned_to_configured_values(caplog):
    config = _trailing_config()
    risk = config["bot"]["long"]["risk"]
    risk.update(divergence_zscore_threshold=2.5, time_stop_max_age_days=9.0)
    bounds = config["optimize"]["bounds"]["long"]["risk"]
    bounds.update(
        divergence_zscore_threshold=[2.0, 3.5, 0.25],
        time_stop_max_age_days=[0.5, 7.0, 0.5],
        divergence_we_cap_pct=[0.15, 0.15],
    )
    with caplog.at_level(logging.WARNING):
        pinned = pin_cpu_only_feature_bounds(config)
    assert bounds["divergence_zscore_threshold"] == [2.5, 2.5]
    # Out-of-range configured values are clamped into the original bounds.
    assert bounds["time_stop_max_age_days"] == [7.0, 7.0]
    assert bounds["divergence_we_cap_pct"] == [0.15, 0.15]
    assert sorted(pinned) == [
        "long.risk.divergence_zscore_threshold=2.5",
        "long.risk.time_stop_max_age_days=7",
    ]
    assert "pinned to configured values" in caplog.text


def test_organillo_suite_overrides_are_accepted_by_gpu_scope():
    config = _trailing_config()
    config["backtest"]["organillo_mode"] = True
    gpu_backend._validate_gpu_suite_override_paths(
        config,
        label="s1",
        overrides={
            "backtest.organillo_mode": True,
            "backtest.organillo_carton_path": "x.csv",
            "backtest.organillo_carton_hash": "a" * 64,
        },
    )


def test_ema_anchor_alone_needs_no_warning_because_gpu_models_its_cap(caplog):
    config = _trailing_config()
    config["live"]["strategy_kind"] = "ema_anchor"
    with caplog.at_level(logging.WARNING):
        assert _validate_gpu_static_scope(config) == "ema_anchor"
    assert "does not model" not in caplog.text
