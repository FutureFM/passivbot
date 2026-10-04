import numpy as np
import pytest

from divergence_inputs import collect_divergence_rocs, roc_from_candles
from config.schema import get_template_config
from config.validate import validate_config


def candles_with_closes(end_ts, close_now=1.0, close_before=100.0):
    rows = []
    for horizon in (240, 60, 15, 5, 0):
        ts = end_ts - horizon * 60_000
        close = close_now if horizon == 0 else close_before
        rows.append((ts, close, close, close, close, 1.0))
    return np.array(rows, dtype=[("ts", "i8"), ("o", "f4"), ("h", "f4"),
                                 ("l", "f4"), ("c", "f4"), ("bv", "f4")])


def test_roc_uses_exact_completed_minute_and_elapsed_horizons():
    end_ts = 300 * 60_000
    assert roc_from_candles(candles_with_closes(end_ts), end_ts) == [-99.0] * 4
    stale = candles_with_closes(end_ts - 60_000)
    assert roc_from_candles(stale, end_ts) == [None] * 4
    gap = candles_with_closes(end_ts)
    gap = gap[gap["ts"] != end_ts - 15 * 60_000]
    assert roc_from_candles(gap, end_ts) == [-99.0, None, -99.0, -99.0]


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
        "A": [None] * 4, "B": [None] * 4
    }
    assert cm.calls == []
    result = await collect_divergence_rocs(cm, ["A", "B", "C"], now_ms)
    assert result["A"] == [-99.0] * 4
    assert result["B"] == [-99.0] * 4
    assert result["C"] == [None] * 4
    assert "candle input unavailable" in caplog.text
    assert all(kwargs["end_ts"] == end_ts for _, kwargs in cm.calls)


@pytest.mark.parametrize(
    "key,value",
    [
        ("divergence_zscore_threshold", -1.0),
        ("divergence_breadth_threshold_pct", 101.0),
        ("divergence_delay_multiplier", 0.5),
        ("divergence_we_cap_pct", 0.0),
        ("divergence_min_timeframes", 5),
    ],
)
def test_invalid_divergence_configuration_is_rejected(key, value):
    config = get_template_config()
    config["bot"]["long"]["risk"][key] = value
    with pytest.raises(ValueError, match=key):
        validate_config(config)
