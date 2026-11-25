import pytest

from close_decay_inputs import (
    DAY_MS,
    ONE_MINUTE_MS,
    DailySma20Cache,
    age_consumed,
    crash_history_minutes,
    daily_sma20_from_candles,
    sma20_consumed,
)
from config.schema import get_template_config
from config.validate import validate_config

NOW = 40 * DAY_MS + 5 * 3_600_000  # five hours into UTC day 40


def day_close_candles(closes_by_day, minute_of_day=DAY_MS - ONE_MINUTE_MS):
    return [[day * DAY_MS + minute_of_day, 0, 0, 0, close, 1.0] for day, close in closes_by_day.items()]


def test_sma20_averages_last_twenty_completed_days_only():
    candles = day_close_candles({d: float(d) for d in range(20, 40)})
    # The still-open current day must not enter the average (no lookahead).
    candles.append([40 * DAY_MS + 60_000, 0, 0, 0, 1e9, 1.0])
    assert daily_sma20_from_candles(candles, NOW) == pytest.approx(sum(range(20, 40)) / 20)


def test_sma20_unknown_when_any_day_is_missing():
    closes = {d: 10.0 for d in range(20, 40) if d != 33}
    assert daily_sma20_from_candles(day_close_candles(closes), NOW) is None
    assert daily_sma20_from_candles([], NOW) is None


def test_sma20_carries_close_forward_within_the_day_like_the_backtest():
    closes = {d: 10.0 for d in range(20, 40)}
    candles = day_close_candles(closes, minute_of_day=DAY_MS - 5 * ONE_MINUTE_MS)
    assert daily_sma20_from_candles(candles, NOW) == pytest.approx(10.0)
    # A close from the previous day must not stand in for a day without candles.
    early = day_close_candles({d: 10.0 for d in range(20, 40) if d != 39})
    early.append([38 * DAY_MS + 100, 0, 0, 0, 10.0, 1.0])
    assert daily_sma20_from_candles(early, NOW) is None


def test_sma20_ignores_invalid_closes():
    closes = {d: 10.0 for d in range(20, 40)}
    candles = day_close_candles(closes)
    candles.append([39 * DAY_MS + DAY_MS - ONE_MINUTE_MS + 0, 0, 0, 0, float("nan"), 1.0])
    assert daily_sma20_from_candles(candles, NOW) == pytest.approx(10.0)


class _FakeCm:
    def __init__(self, candles):
        self.candles = candles
        self.calls = 0

    async def get_candles(self, symbol, **kwargs):
        self.calls += 1
        assert kwargs["timeframe"] == "1m" and kwargs["end_ts"] < 40 * DAY_MS
        if isinstance(self.candles, Exception):
            raise self.candles
        return self.candles


@pytest.mark.asyncio
async def test_cache_refetches_only_on_a_new_day_and_failures_stay_unknown():
    cm = _FakeCm(day_close_candles({d: 10.0 for d in range(20, 40)}))
    cache = DailySma20Cache()
    assert await cache.collect(cm, ["A"], NOW) == {"A": 10.0}
    assert await cache.collect(cm, ["A"], NOW + 3_600_000) == {"A": 10.0}
    assert cm.calls == 1
    failing = _FakeCm(RuntimeError("down"))
    assert await DailySma20Cache().collect(failing, ["A"], NOW) == {"A": None}


def test_consumption_flags_and_history_length():
    assert not sma20_consumed({}) and sma20_consumed({"tp_decay_gate_sma20": True})
    assert sma20_consumed({"tp_sma20_above_boost": 0.1})
    assert not age_consumed({}) and age_consumed({"tp_decay_time_full_hours": 168.0})
    assert crash_history_minutes(0.0) == 2
    assert crash_history_minutes(30.2) == 33


@pytest.mark.parametrize(
    "key,value",
    [
        ("market_crash_timeframe_minutes", 30.0),
        ("market_crash_min_drop_pct", -1.0),
        ("market_crash_breadth_threshold_pct", 101.0),
        ("market_crash_cooldown_minutes", -1.0),
        ("market_crash_we_cap_pct", 0.0),
        ("market_crash_we_cap_pct", 1.5),
        ("market_crash_full_breadth_pct", -1.0),
        ("market_crash_full_breadth_pct", 101.0),
        ("market_crash_freeze_initial_entries", 1),
        ("risky_offload_we_threshold", -0.1),
        ("risky_offload_close_pct", 1.5),
        ("risky_offload_markup_pct", -1.0),
        ("tp_decay_floor_pct", float("nan")),
        ("tp_decay_time_start_hours", -1.0),
        ("tp_decay_we_start_ratio", -0.1),
        ("tp_decay_gate_sma20", "yes"),
        ("tp_sma20_above_boost", -0.1),
    ],
)
def test_invalid_crash_offload_decay_configuration_is_rejected(key, value):
    config = get_template_config()
    config["bot"]["long"]["risk"][key] = value
    with pytest.raises(ValueError, match=key):
        validate_config(config)


def test_valid_crash_offload_decay_configuration_passes():
    config = get_template_config()
    risk = config["bot"]["short"]["risk"]
    risk.update(
        tp_decay_time_start_hours=10.0,
        tp_decay_time_full_hours=10.0,  # step ramp
        tp_decay_floor_pct=-0.01,
        market_crash_timeframe_minutes=60.0,
        market_crash_full_breadth_pct=0.0,  # binary severity
        risky_offload_we_threshold=0.8,
    )
    validate_config(config)


class _AgeStub:
    def __init__(self, manager, full_hours=168.0):
        self.positions = {"A": {"long": {"size": 1.0}, "short": {"size": 0.0}}}
        self.qty_steps = {"A": 0.1}
        self.c_mults = {"A": 1.0}
        self._pnls_manager = manager
        self._fill_history_coverage_status = lambda **_: {"ready": True}
        self._full_hours = full_hours

    def bp(self, pside, key, symbol=None):
        return self._full_hours if key == "tp_decay_time_full_hours" else 0.0


def test_position_open_timestamp_comes_from_fills_and_is_unknown_without_proof(monkeypatch, caplog):
    import passivbot as pb

    stub = _AgeStub(manager=None)
    out = pb.Passivbot._get_position_open_timestamps(stub, ["A"])
    assert out == {"A": {"long": None, "short": None}}

    class _Manager:
        def get_events(self):
            return []

    stub = _AgeStub(_Manager())
    monkeypatch.setattr(pb, "reconstruct_episodes", lambda *a, **k: {("A", "long"): None})
    assert pb.Passivbot._get_position_open_timestamps(stub, ["A"])["A"]["long"] is None
    monkeypatch.setattr(
        pb, "reconstruct_episodes",
        lambda *a, **k: {("A", "long"): {"opened_timestamp_ms": 123, "count": 1, "time_stop": None}},
    )
    assert pb.Passivbot._get_position_open_timestamps(stub, ["A"])["A"]["long"] == 123
    # Time trigger disabled: no fill replay is needed at all.
    monkeypatch.setattr(pb, "reconstruct_episodes", lambda *a, **k: 1 / 0)
    off = pb.Passivbot._get_position_open_timestamps(_AgeStub(_Manager(), 0.0), ["A"])
    assert off["A"]["long"] is None
