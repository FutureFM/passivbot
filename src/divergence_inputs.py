"""Completed-candle inputs for Rust's cross-asset divergence detector."""

import asyncio
import logging
import math


HORIZONS_MINUTES = (5, 15, 60, 240, 1440, 4320)
# The 1d/3d horizons are evaluated only with `divergence_extended_horizons`.
BASE_HORIZON_COUNT = 4
ONE_MINUTE_MS = 60_000
# Trailing window a flag must persist in before `divergence_hold_minutes` latches it
# (mirrors PERSISTENCE_WINDOW_MINUTES in passivbot-rust/src/divergence.rs).
PERSISTENCE_WINDOW_MINUTES = 60


def active_horizons(extended: bool) -> tuple[int, ...]:
    return HORIZONS_MINUTES if extended else HORIZONS_MINUTES[:BASE_HORIZON_COUNT]


def _closes_by_ts(candles) -> dict[int, float]:
    closes = {}
    for row in candles:
        if len(row) < 5:
            continue
        ts, close = int(row[0]), float(row[4])
        if math.isfinite(close) and close > 0.0:
            closes[ts] = close
    return closes


def roc_from_candles(candles, end_ts: int, *, extended: bool = False) -> list[float | None]:
    """Calculate aligned percentage changes without treating sparse bars as elapsed minutes."""
    return _roc_from_closes(_closes_by_ts(candles), end_ts, extended=extended)


def _roc_from_closes(closes: dict[int, float], end_ts: int, *, extended: bool) -> list[float | None]:
    result = [None] * len(HORIZONS_MINUTES)
    current = closes.get(end_ts)
    if current is None:
        return result
    for tf, horizon in enumerate(active_horizons(extended)):
        previous = closes.get(end_ts - horizon * ONE_MINUTE_MS)
        if previous is not None:
            result[tf] = (current / previous - 1.0) * 100.0
    return result


def hold_history_minutes(hold_minutes: float) -> int:
    """Completed minutes needed to rebuild the hold latch exactly (0 when disabled)."""
    if not hold_minutes or hold_minutes <= 0.0:
        return 0
    return PERSISTENCE_WINDOW_MINUTES + int(math.ceil(hold_minutes))


async def collect_divergence_rocs(
    cm, symbols: list[str], now_ms: int, *, extended: bool = False
) -> dict[str, list[float | None]]:
    """Use the same completed minute for every symbol; missing coverage is explicit."""
    rocs, _history = await collect_divergence_inputs(cm, symbols, now_ms, extended=extended)
    return rocs


async def collect_divergence_inputs(
    cm, symbols: list[str], now_ms: int, *, extended: bool = False, history_minutes: int = 0
) -> tuple[dict[str, list[float | None]], list[list[list[float | None]]]]:
    """Latest ROC rows plus, when `history_minutes` > 0, the trailing per-minute ROC history
    (oldest first, one row per symbol in `symbols` order) used to rebuild the hold latch."""
    empty = [None] * len(HORIZONS_MINUTES)
    horizons = active_horizons(extended)
    if len(symbols) < 3:
        logging.warning("[divergence] fewer than three symbols; protection inactive")
        return {symbol: empty.copy() for symbol in symbols}, []
    end_ts = (now_ms // ONE_MINUTE_MS - 1) * ONE_MINUTE_MS
    history_minutes = max(0, int(history_minutes))
    start_ts = end_ts - (max(horizons) + max(history_minutes - 1, 0)) * ONE_MINUTE_MS
    results = await asyncio.gather(
        *(
            cm.get_candles(
                symbol, start_ts=start_ts, end_ts=end_ts,
                max_age_ms=120_000, strict=False, timeframe="1m",
            )
            for symbol in symbols
        ),
        return_exceptions=True,
    )
    rocs = {}
    closes_by_symbol = {}
    for symbol, result in zip(symbols, results):
        if isinstance(result, BaseException):
            if isinstance(result, asyncio.CancelledError):
                raise result
            logging.warning("[divergence] candle input unavailable for %s: %s", symbol, type(result).__name__)
            rocs[symbol] = empty.copy()
            closes_by_symbol[symbol] = {}
        else:
            closes_by_symbol[symbol] = _closes_by_ts(result)
            rocs[symbol] = _roc_from_closes(closes_by_symbol[symbol], end_ts, extended=extended)
    if not any(
        sum(row[tf] is not None for row in rocs.values()) >= 3 for tf in range(len(horizons))
    ):
        logging.warning("[divergence] fewer than three valid symbols per horizon; protection inactive")
    history = [
        [
            _roc_from_closes(closes_by_symbol[symbol], ts, extended=extended)
            for symbol in symbols
        ]
        for ts in range(
            end_ts - (history_minutes - 1) * ONE_MINUTE_MS, end_ts + 1, ONE_MINUTE_MS
        )
    ] if history_minutes else []
    return rocs, history
