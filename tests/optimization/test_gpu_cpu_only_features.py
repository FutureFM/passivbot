"""GPU kernels do not model fork-only CPU features; the GPU backend must reject them."""

import pytest

from config.schema import get_template_config
from optimization.backends.gpu_backend import _validate_gpu_static_scope


def _trailing_config():
    config = get_template_config()
    config["live"]["strategy_kind"] = "trailing_martingale"
    return config


def test_plain_trailing_martingale_config_is_accepted():
    assert _validate_gpu_static_scope(_trailing_config()) == "trailing_martingale"


@pytest.mark.parametrize(
    "mutate,match",
    [
        (lambda c: c["backtest"].update(organillo_mode=True), "organillo_mode"),
        (lambda c: c["bot"]["long"]["risk"].update(divergence_filter_enabled=True), "divergence"),
        (lambda c: c["bot"]["short"]["risk"].update(time_stop_max_age_days=1.0), "time_stop"),
        (
            lambda c: c["optimize"]["bounds"]["long"]["risk"].update(
                time_stop_max_age_days=[0.0, 3.0, 0.5]
            ),
            "time_stop",
        ),
        (
            lambda c: c.update(
                coin_overrides={"BTC": {"bot": {"long": {"risk": {"time_stop_max_age_days": 2.0}}}}}
            ),
            "coin_overrides.BTC",
        ),
    ],
)
def test_cpu_only_features_are_rejected(mutate, match):
    config = _trailing_config()
    mutate(config)
    with pytest.raises(ValueError, match=match):
        _validate_gpu_static_scope(config)


def test_ema_anchor_is_accepted_with_exposure_cap_warning(caplog):
    config = _trailing_config()
    config["live"]["strategy_kind"] = "ema_anchor"
    with caplog.at_level("WARNING"):
        assert _validate_gpu_static_scope(config) == "ema_anchor"
    assert "exposure cap" in caplog.text
