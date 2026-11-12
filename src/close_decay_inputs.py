"""Live inputs for the TP-decay close adjustment and the market-crash detector.

Everything here is rebuilt from exchange facts (fills) and completed candles on every cycle;
nothing is persisted. Missing evidence yields ``None`` and Rust then applies no decay: an
unknown age never ages a position and an unknown SMA20 keeps the gate closed.
"""

from __future__ import annotations

import asyncio
import bisect
import logging
import math

ONE_MINUTE_MS = 60_000
DAY_MS = 86_400_000
SMA_DAYS = 20


def sma20_consumed(bot_params: dict) -> bool:
    """True when the SMA20 gate or boost of these (Rust-keyed) bot params is enabled."""
    return bool(bot_params.get("tp_decay_gate_sma20")) or float(
        bot_params.get("tp_sma20_above_boost") or 0.0
    ) > 0.0


def age_consumed(bot_params: dict) -> bool:
    """True when the TP-decay time trigger is enabled."""
    return float(bot_params.get("tp_decay_time_full_hours") or 0.0) > 0.0


def crash_history_minutes(cooldown_minutes: float) -> int:
    """Contiguous 1m history that reproduces the crash severity exactly (Rust contract)."""
    return int(math.ceil(max(float(cooldown_minutes or 0.0), 0.0))) + 2


def daily_sma20_from_candles(candles, now_ms: int) -> float | None:
    """Mean of the last 20 completed UTC daily closes, or None if any is unavailable.

    The close of a UTC day is the close of the last 1m candle at or before that day's final
    minute (a gap carries the previous close forward within the same day), which matches the
    backtest definition. ``candles`` are 1m rows ``[ts, o, h, l, c, v]``.
    """
    rows = {}
    for row in candles:
        if len(row) < 5:
            continue
        ts, close = int(row[0]), float(row[4])
        if math.isfinite(close) and close > 0.0:
            rows[ts] = close
    if not rows:
        return None
    stamps = sorted(rows)
    today = int(now_ms) // DAY_MS
    total = 0.0
    for days_back in range(1, SMA_DAYS + 1):
        day_start = (today - days_back) * DAY_MS
        last_minute = day_start + DAY_MS - ONE_MINUTE_MS
        i = bisect.bisect_right(stamps, last_minute) - 1
        if i < 0 or stamps[i] < day_start:
            return None
        total += rows[stamps[i]]
    return total / SMA_DAYS


class DailySma20Cache:
    """Per-(symbol, UTC day) memo. Losing it only costs a refetch, never a decision change."""

    def __init__(self):
        self._values: dict[str, tuple[int, float | None]] = {}

    async def collect(self, cm, symbols, now_ms: int) -> dict[str, float | None]:
        day = int(now_ms) // DAY_MS
        out: dict[str, float | None] = {}
        todo = []
        for symbol in symbols:
            cached = self._values.get(symbol)
            if cached is not None and cached[0] == day and cached[1] is not None:
                out[symbol] = cached[1]
            else:
                todo.append(symbol)
        if not todo:
            return out
        end_ts = day * DAY_MS - ONE_MINUTE_MS
        start_ts = day * DAY_MS - SMA_DAYS * DAY_MS
        results = await asyncio.gather(
            *(
                cm.get_candles(
                    symbol,
                    start_ts=start_ts,
                    end_ts=end_ts,
                    strict=False,
                    timeframe="1m",
                    standardize=False,
                )
                for symbol in todo
            ),
            return_exceptions=True,
        )
        for symbol, result in zip(todo, results):
            if isinstance(result, asyncio.CancelledError):
                raise result
            if isinstance(result, BaseException):
                logging.warning(
                    "[tp_decay] daily candles unavailable for %s: %s; SMA20 treated as unknown",
                    symbol,
                    type(result).__name__,
                )
                out[symbol] = None
                continue
            value = daily_sma20_from_candles(result, now_ms)
            self._values[symbol] = (day, value)
            out[symbol] = value
        return out
