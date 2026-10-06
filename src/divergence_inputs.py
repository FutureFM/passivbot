"""Completed-candle inputs for Rust's cross-asset divergence detector."""

import asyncio
import logging
import math


HORIZONS_MINUTES = (5, 15, 60, 240, 1440, 4320)
# The 1d/3d horizons are evaluated only with `divergence_extended_horizons`.
BASE_HORIZON_COUNT = 4
ONE_MINUTE_MS = 60_000


def active_horizons(extended: bool) -> tuple[int, ...]:
    return HORIZONS_MINUTES if extended else HORIZONS_MINUTES[:BASE_HORIZON_COUNT]


def roc_from_candles(candles, end_ts: int, *, extended: bool = False) -> list[float | None]:
    """Calculate aligned percentage changes without treating sparse bars as elapsed minutes."""
    closes = {}
    for row in candles:
        if len(row) < 5:
            continue
        ts, close = int(row[0]), float(row[4])
        if math.isfinite(close) and close > 0.0:
            closes[ts] = close
    result = [None] * len(HORIZONS_MINUTES)
    current = closes.get(end_ts)
    if current is None:
        return result
    for tf, horizon in enumerate(active_horizons(extended)):
        previous = closes.get(end_ts - horizon * ONE_MINUTE_MS)
        if previous is not None:
            result[tf] = (current / previous - 1.0) * 100.0
    return result


async def collect_divergence_rocs(
    cm, symbols: list[str], now_ms: int, *, extended: bool = False
) -> dict[str, list[float | None]]:
    """Use the same completed minute for every symbol; missing coverage is explicit."""
    empty = [None] * len(HORIZONS_MINUTES)
    horizons = active_horizons(extended)
    if len(symbols) < 3:
        logging.warning("[divergence] fewer than three symbols; protection inactive")
        return {symbol: empty.copy() for symbol in symbols}
    end_ts = (now_ms // ONE_MINUTE_MS - 1) * ONE_MINUTE_MS
    start_ts = end_ts - max(horizons) * ONE_MINUTE_MS
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
    for symbol, result in zip(symbols, results):
        if isinstance(result, BaseException):
            if isinstance(result, asyncio.CancelledError):
                raise result
            logging.warning("[divergence] candle input unavailable for %s: %s", symbol, type(result).__name__)
            rocs[symbol] = empty.copy()
        else:
            rocs[symbol] = roc_from_candles(result, end_ts, extended=extended)
    if not any(
        sum(row[tf] is not None for row in rocs.values()) >= 3 for tf in range(len(horizons))
    ):
        logging.warning("[divergence] fewer than three valid symbols per horizon; protection inactive")
    return rocs
