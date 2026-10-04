from types import SimpleNamespace
from unittest.mock import AsyncMock, Mock

import pytest

from passivbot import Passivbot


def _event(symbol, pside, qty, psize, *, c_mult=1.0, timestamp=1_000):
    return SimpleNamespace(
        symbol=symbol,
        position_side=pside,
        qty=qty,
        psize=psize,
        c_mult=c_mult,
        timestamp=timestamp,
    )


def _bot(events, long_size, short_size=0.0, *, c_mult=1.0, coverage_ready=True):
    symbol = "BTC/USDT:USDT"
    return SimpleNamespace(
        positions={symbol: {
            "long": {"size": long_size},
            "short": {"size": short_size},
        }},
        qty_steps={symbol: 0.01},
        c_mults={symbol: c_mult},
        _pnls_manager=SimpleNamespace(get_events=lambda: events),
        _fill_history_coverage_status=lambda **kwargs: {"ready": coverage_ready},
        bp=lambda pside, key, coin: {
            "risk_entry_cooldown_minutes": 1.0,
            "risk_entry_cooldown_factor_per_fill": 2.0,
        }[key],
    )


def test_count_rebuilds_open_long_and_short_episodes_from_fills():
    symbol = "BTC/USDT:USDT"
    events = [
        _event(symbol, "long", 1.0, 1.0),
        _event(symbol, "short", -2.0, 2.0),
        _event(symbol, "long", 0.2, 1.2),
        _event(symbol, "long", -0.4, 0.8),
        _event(symbol, "short", -0.5, 2.5),
        _event(symbol, "short", 1.0, 1.5),
        _event(symbol, "long", 0.1, 0.9),
    ]
    counts = Passivbot._get_entry_fill_counts(_bot(events, 0.9, 1.5), [symbol])
    assert counts[symbol] == {"long": 3, "short": 2}


def test_count_resets_after_flat_and_rejects_incomplete_episode():
    symbol = "BTC/USDT:USDT"
    complete = [
        _event(symbol, "long", 1.0, 1.0),
        _event(symbol, "long", -1.0, 0.0),
        _event(symbol, "long", 0.3, 0.3),
    ]
    assert Passivbot._get_entry_fill_counts(_bot(complete, 0.3), [symbol])[symbol]["long"] == 1
    incomplete = [_event(symbol, "long", 0.1, 0.1)]
    assert Passivbot._get_entry_fill_counts(_bot(incomplete, 1.1), [symbol])[symbol]["long"] is None


def test_count_requires_proven_history_even_when_cached_size_matches():
    symbol = "BTC/USDT:USDT"
    # A partial cache can misidentify its first entry as the position opening.
    partial = [_event(symbol, "long", 0.1, 0.1)]
    bot = _bot(partial, 0.1, coverage_ready=False)
    assert Passivbot._get_entry_fill_counts(bot, [symbol])[symbol]["long"] is None


def test_count_uses_contract_multiplier_for_position_parity():
    symbol = "BTC/USDT:USDT"
    events = [
        _event(symbol, "long", 2.0, 0.2, c_mult=0.1),
        _event(symbol, "long", 1.0, 0.3, c_mult=0.1),
    ]
    bot = _bot(events, 3.0, c_mult=0.1)
    assert Passivbot._get_entry_fill_counts(bot, [symbol])[symbol]["long"] == 2


@pytest.mark.asyncio
async def test_missing_episode_schedules_exchange_history_repair_without_waiting():
    manager = SimpleNamespace(
        refresh_for_lookback=AsyncMock(),
        refresh=AsyncMock(),
        set_history_scope=Mock(),
    )
    bot = SimpleNamespace(
        _pnls_manager=manager,
        _position_history_anchor_timestamp_ms=Mock(return_value=3_600_000),
    )
    counts = {"BTC/USDT:USDT": {"long": None, "short": 0}}
    Passivbot._schedule_entry_fill_count_history_repair(bot, counts, 4_000_000)
    assert manager.refresh_for_lookback.await_count == 0
    await bot._entry_fill_count_repair_task
    manager.refresh_for_lookback.assert_awaited_once_with(start_ms=3_540_000)
    Passivbot._schedule_entry_fill_count_history_repair(bot, counts, 4_300_000)
    await bot._entry_fill_count_repair_task
    manager.refresh.assert_awaited_once_with(start_ms=None, end_ms=None)
    manager.set_history_scope.assert_called_once_with("all")
