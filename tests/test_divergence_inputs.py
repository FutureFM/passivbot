import numpy as np
import pytest

from divergence_inputs import (
    collect_divergence_inputs,
    collect_divergence_rocs,
    hold_history_minutes,
    roc_from_candles,
)
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


@pytest.mark.asyncio
async def test_hold_history_rows_cover_window_plus_hold_in_symbol_order():
    assert hold_history_minutes(0.0) == 0
    assert hold_history_minutes(720.0) == 780
    end_ts = 10_000 * 60_000
    history_minutes = 5

    class Manager:
        def __init__(self):
            self.calls = []

        async def get_candles(self, symbol, **kwargs):
            self.calls.append(kwargs)
            ts = np.arange(kwargs["start_ts"], kwargs["end_ts"] + 1, 60_000)
            # Symbol A halves at the last two minutes; others are flat.
            close = np.where((symbol == "A") & (ts >= end_ts - 60_000), 50.0, 100.0)
            return np.array(list(zip(ts, close, close, close, close, close)),
                            dtype=[("ts", "i8"), ("o", "f4"), ("h", "f4"),
                                   ("l", "f4"), ("c", "f4"), ("bv", "f4")])

    cm = Manager()
    rocs, history = await collect_divergence_inputs(
        cm, ["A", "B", "C"], end_ts + 60_000, history_minutes=history_minutes
    )
    assert all(kw["start_ts"] == end_ts - (240 + history_minutes - 1) * 60_000 for kw in cm.calls)
    assert len(history) == history_minutes and all(len(step) == 3 for step in history)
    # The last history row is the latest ROC row; older rows precede the drop.
    assert history[-1] == [rocs["A"], rocs["B"], rocs["C"]]
    assert history[-1][0][0] == pytest.approx(-50.0)
    assert history[0][0][0] == pytest.approx(0.0)
    rocs_only, empty = await collect_divergence_inputs(cm, ["A", "B", "C"], end_ts + 60_000)
    assert empty == [] and rocs_only == rocs


@pytest.mark.parametrize(
    "key,value",
    [
        ("divergence_zscore_threshold", -1.0),
        ("divergence_breadth_threshold_pct", 101.0),
        ("divergence_delay_multiplier", 0.5),
        ("divergence_we_cap_pct", 0.0),
        ("divergence_min_timeframes", 5),
        ("divergence_extended_horizons", 1),
        ("divergence_hold_minutes", -1.0),
        ("divergence_persistence_pct", 101.0),
        ("divergence_delay_add_minutes", -1.0),
        ("divergence_market_delay_add_minutes", -5.0),
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
        "divergence_delay_add_minutes": 0.0,
        "divergence_market_delay_add_minutes": 0.0,
    }
    bot = SimpleNamespace(bp=lambda pside, key, symbol=None: values[key])
    assert Passivbot._entry_cooldown_horizon(bot, "long") == expected


def test_live_cooldown_lookback_covers_additive_divergence_delay_with_zero_cooldown():
    from types import SimpleNamespace
    from passivbot import Passivbot

    values = {
        "risk_entry_cooldown_minutes": 0.0,
        "entry_cooldown_min_duration_minutes": 0.0,
        "entry_cooldown_max_duration_minutes": None,
        "entry_cooldown_weights_minutes": {"exposure_ratio": 0.0, "adverse_directionality": 0.0},
        "divergence_filter_enabled": True,
        "divergence_delay_multiplier": 30.0,
        "divergence_delay_add_minutes": 0.0,
        "divergence_market_delay_add_minutes": 15.0,
    }
    bot = SimpleNamespace(bp=lambda pside, key, symbol=None: values[key])
    assert Passivbot._entry_cooldown_horizon(bot, "long") == 1440.0
    values["divergence_market_delay_add_minutes"] = 0.0
    assert Passivbot._entry_cooldown_horizon(bot, "long") == 0.0


def test_uses_day_horizons_for_extended_or_long_horizon():
    from divergence_inputs import uses_day_horizons

    assert not uses_day_horizons({"divergence_filter_enabled": False, "divergence_extended_horizons": True})
    assert uses_day_horizons({"divergence_filter_enabled": True, "divergence_extended_horizons": True})
    assert uses_day_horizons({"divergence_filter_enabled": True, "divergence_long_horizon_minutes": 4320.0})
    assert not uses_day_horizons({"divergence_filter_enabled": True, "divergence_long_horizon_minutes": 0.0})


def test_keep_last_good_state_restores_missing_candles_within_ttl():
    from divergence_inputs import LAST_GOOD_STATE_TTL_MS, keep_last_good_state

    cache = {}
    good = {"A": [-50.0, -40.0, None, None, None, None], "B": [0.0, 0.0, None, None, None, None]}
    held = {"A": [0.8, 0.0], "B": [0.0, 0.0]}
    assert keep_last_good_state(cache, good, held, 1_000_000) == []

    # A's latest candle is missing next cycle: its last good row and latch are restored.
    rocs = {"A": [None] * 6, "B": [1.0, 1.0, None, None, None, None]}
    held = {"A": [0.0, 0.0], "B": [0.0, 0.0]}
    assert keep_last_good_state(cache, rocs, held, 1_060_000) == ["A"]
    assert rocs["A"] == [-50.0, -40.0, None, None, None, None]
    assert held["A"] == [0.8, 0.0]

    # Beyond the TTL the coin is left without signal (and the stale entry dropped).
    rocs = {"A": [None] * 6, "B": [1.0, 1.0, None, None, None, None]}
    held = {}
    assert keep_last_good_state(cache, rocs, held, 1_000_000 + LAST_GOOD_STATE_TTL_MS + 1) == []
    assert rocs["A"] == [None] * 6
    assert "A" not in cache


def test_keep_last_good_state_keeps_partial_rows_of_recent_listings():
    from divergence_inputs import keep_last_good_state

    cache = {"A": (0, [-50.0] * 6, [1.0, 0.0])}
    # A recent listing has short horizons only: it is live data, not a failure.
    rocs = {"A": [-1.0, None, None, None, None, None]}
    held = {"A": [0.0, 0.0]}
    assert keep_last_good_state(cache, rocs, held, 60_000) == []
    assert rocs["A"][0] == -1.0
