import numpy as np
import pytest

from divergence_inputs import collect_divergence_rocs, roc_from_candles
from config.schema import get_template_config
from config.validate import validate_config


def candles_with_closes(end_ts, close_now=1.0, close_before=100.0, horizons=(240, 60, 15, 5, 0)):
    rows = []
    for horizon in horizons:
        ts = end_ts - horizon * 60_000
        close = close_now if horizon == 0 else close_before
        rows.append((ts, close, close, close, close, 1.0))
    return np.array(rows, dtype=[("ts", "i8"), ("o", "f4"), ("h", "f4"),
                                 ("l", "f4"), ("c", "f4"), ("bv", "f4")])


def test_roc_uses_exact_completed_minute_and_elapsed_horizons():
    end_ts = 300 * 60_000
    assert roc_from_candles(candles_with_closes(end_ts), end_ts) == [-99.0] * 4 + [None] * 2
    stale = candles_with_closes(end_ts - 60_000)
    assert roc_from_candles(stale, end_ts) == [None] * 6
    gap = candles_with_closes(end_ts)
    gap = gap[gap["ts"] != end_ts - 15 * 60_000]
    assert roc_from_candles(gap, end_ts) == [-99.0, None, -99.0, -99.0, None, None]


def test_extended_horizons_add_one_and_three_day_roc_only_when_enabled():
    end_ts = 5000 * 60_000
    candles = candles_with_closes(end_ts, horizons=(4320, 1440, 240, 60, 15, 5, 0))
    assert roc_from_candles(candles, end_ts) == [-99.0] * 4 + [None] * 2
    assert roc_from_candles(candles, end_ts, extended=True) == [-99.0] * 6


@pytest.mark.asyncio
async def test_live_inputs_require_three_symbols_and_log_missing_coverage(caplog):
    end_ts = 300 * 60_000

    class Manager:
        def __init__(self):
            self.calls = []

        async def get_candles(self, symbol, **kwargs):
            self.calls.append((symbol, kwargs))
            if symbol == "C":
                raise RuntimeError("unavailable")
            return candles_with_closes(end_ts)

    cm = Manager()
    now_ms = end_ts + 60_000
    assert await collect_divergence_rocs(cm, ["A", "B"], now_ms) == {
        "A": [None] * 6, "B": [None] * 6
    }
    assert cm.calls == []
    result = await collect_divergence_rocs(cm, ["A", "B", "C"], now_ms)
    assert result["A"] == [-99.0] * 4 + [None] * 2
    assert result["B"] == [-99.0] * 4 + [None] * 2
    assert result["C"] == [None] * 6
    assert "candle input unavailable" in caplog.text
    assert all(kwargs["end_ts"] == end_ts for _, kwargs in cm.calls)
    assert all(kwargs["start_ts"] == end_ts - 240 * 60_000 for _, kwargs in cm.calls)
    cm.calls.clear()
    await collect_divergence_rocs(cm, ["A", "B", "C"], now_ms, extended=True)
    assert all(kwargs["start_ts"] == end_ts - 4320 * 60_000 for _, kwargs in cm.calls)


@pytest.mark.parametrize(
    "key,value",
    [
        ("divergence_zscore_threshold", -1.0),
        ("divergence_breadth_threshold_pct", 101.0),
        ("divergence_delay_multiplier", 0.5),
        ("divergence_we_cap_pct", 0.0),
        ("divergence_min_timeframes", 5),
        ("divergence_extended_horizons", 1),
    ],
)
def test_invalid_divergence_configuration_is_rejected(key, value):
    config = get_template_config()
    config["bot"]["long"]["risk"][key] = value
    with pytest.raises(ValueError, match=key):
        validate_config(config)


def test_extended_horizons_allow_up_to_six_required_timeframes():
    config = get_template_config()
    risk = config["bot"]["long"]["risk"]
    risk["divergence_extended_horizons"] = True
    risk["divergence_min_timeframes"] = 6
    validate_config(config)
    risk["divergence_min_timeframes"] = 7
    with pytest.raises(ValueError, match="divergence_min_timeframes must be 1..6"):
        validate_config(config)


@pytest.mark.parametrize(
    "enabled,multiplier,ceiling,expected",
    [(False, 30.0, None, 10.0), (True, 1.0, None, 10.0), (True, 30.0, None, 1440.0), (True, 30.0, 120.0, 120.0)],
)
def test_live_cooldown_lookback_covers_divergence_extension(enabled, multiplier, ceiling, expected):
    from types import SimpleNamespace
    from passivbot import Passivbot

    values = {
        "risk_entry_cooldown_minutes": 10.0,
        "entry_cooldown_min_duration_minutes": 0.0,
        "entry_cooldown_max_duration_minutes": ceiling,
        "entry_cooldown_weights_minutes": {"exposure_ratio": 0.0, "adverse_directionality": 0.0},
        "divergence_filter_enabled": enabled,
        "divergence_delay_multiplier": multiplier,
    }
    bot = SimpleNamespace(bp=lambda pside, key, symbol=None: values[key])
    assert Passivbot._entry_cooldown_horizon(bot, "long") == expected
