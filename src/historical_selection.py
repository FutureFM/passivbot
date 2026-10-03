"""Daily, point-in-time entry eligibility for offline backtests and optimization."""

from __future__ import annotations

import csv
import gzip
import hashlib
import io
import logging
from copy import deepcopy
from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path

import numpy as np
import pandas as pd

DAY_MS = 86_400_000


@dataclass(frozen=True)
class HistoricalSelection:
    timestamps: np.ndarray
    eligible: np.ndarray
    coins: tuple[str, ...]
    content_hash: str


@dataclass(frozen=True)
class PreparedSelection:
    eligible: np.ndarray
    row_indices: np.ndarray
    content_hash: str

    def share(self, manager):
        return SharedSelection(
            manager.create_from(self.eligible)[0],
            manager.create_from(self.row_indices)[0],
            self.content_hash,
        )


class SharedSelection:
    """Pickle only descriptors; attach lazily in each optimizer worker."""

    def __init__(self, eligible_spec, rows_spec, content_hash):
        self.eligible_spec = eligible_spec
        self.rows_spec = rows_spec
        self.content_hash = content_hash
        self._attachments = []

    def arrays(self):
        from shared_arrays import attach_shared_array

        if not self._attachments:
            self._attachments = [
                attach_shared_array(self.eligible_spec),
                attach_shared_array(self.rows_spec),
            ]
            for attachment in self._attachments:
                attachment.array.flags.writeable = False
        return PreparedSelection(
            self._attachments[0].array,
            self._attachments[1].array,
            self.content_hash,
        )

    def close(self):
        attachments, self._attachments = self._attachments, []
        for attachment in attachments:
            attachment.close()

    def __getstate__(self):
        return self.eligible_spec, self.rows_spec, self.content_hash

    def __setstate__(self, state):
        self.__init__(*state)


def _coin_key(value):
    # PB8 keeps exact/exchange-scoped market identities lossless.
    from backtest_universe import normalize_backtest_coin

    return normalize_backtest_coin(value)


@lru_cache(maxsize=16)
def _load(path, mtime_ns, size):
    raw = Path(path).read_bytes()
    if path.endswith(".gz"):
        raw = gzip.decompress(raw)
    frame = pd.read_csv(io.BytesIO(raw), index_col=0)
    if frame.empty or not len(frame.columns):
        raise ValueError(f"Organillo carton is empty: {path}")
    dates = pd.to_datetime(frame.index, utc=True, errors="raise")
    timestamps = np.asarray(dates.as_unit("ms").asi8, dtype=np.int64)
    if np.any(timestamps % DAY_MS) or np.any(np.diff(timestamps) <= 0):
        raise ValueError("Organillo dates must be distinct, increasing UTC midnights")
    # Inspect the original header: pandas otherwise renames duplicate columns.
    headers = next(csv.reader(io.StringIO(raw.decode("utf-8-sig"))))[1:]
    if any(not header.strip() for header in headers):
        raise ValueError("Organillo columns must have market identities")
    coins = tuple(_coin_key(c) for c in headers)
    if len(set(coins)) != len(coins):
        raise ValueError("Organillo columns must resolve to distinct market identities")
    values = frame.to_numpy()
    if not np.isin(values, (0, 1)).all():
        raise ValueError("Organillo values must be 0 or 1")
    eligible = np.ascontiguousarray(values, dtype=np.uint8)
    timestamps.flags.writeable = eligible.flags.writeable = False
    digest = hashlib.sha256(raw).hexdigest()
    logging.info("Organillo loaded | hash=%s days=%d coins=%d", digest[:12], len(dates), len(coins))
    return HistoricalSelection(timestamps, eligible, coins, digest)


def load_selection(config):
    bt = config["backtest"]
    if not bt.get("organillo_mode", False):
        return None
    path = bt.get("organillo_carton_path")
    if not isinstance(path, str) or not path.strip():
        raise ValueError("backtest.organillo_carton_path is required in Organillo mode")
    source = Path(path).expanduser().resolve()
    stat = source.stat()
    selection = _load(str(source), stat.st_mtime_ns, stat.st_size)
    expected = bt.get("organillo_carton_hash")
    if expected and expected != selection.content_hash:
        raise ValueError("Organillo carton changed after preparation; restart with the new carton")
    return selection


def _date_ms(value):
    if str(value).lower() in ("now", "today"):
        return int(pd.Timestamp.now(tz="UTC").value // 1_000_000)
    return int(pd.to_datetime(value, utc=True).value // 1_000_000)


def prepare_config(config, *, allowed_coins=None):
    """Prepare a simulation-only copy; the carton replaces today's approved list."""
    selection = load_selection(config)
    if selection is None:
        return config
    result = deepcopy(config)
    bt = result["backtest"]
    start, end = _date_ms(bt["start_date"]), _date_ms(bt["end_date"])
    if start < selection.timestamps[0] or end >= selection.timestamps[-1] + DAY_MS:
        raise ValueError("Organillo carton does not cover the requested backtest dates")
    first = np.searchsorted(selection.timestamps, start, side="right") - 1
    last = np.searchsorted(selection.timestamps, end, side="right")
    required_days = end // DAY_MS - start // DAY_MS + 1
    if last - first != required_days:
        raise ValueError("Organillo carton has missing dates in the requested period")
    selected = selection.eligible[first:last].any(axis=0)
    coins = [c for c, active in zip(selection.coins, selected) if active]
    if allowed_coins is not None:
        allowed = {_coin_key(c) for c in allowed_coins}
        coins = [c for c in coins if c in allowed]
    if not coins:
        raise ValueError("Organillo has no selected coins in the requested universe/period")
    bt["organillo_carton_hash"] = selection.content_hash
    result["live"]["approved_coins"] = {"long": coins.copy(), "short": coins.copy()}
    # Subsequent PB8 formatting reuses this source snapshot, including in suites.
    result.setdefault("_coins_sources", {})["approved_coins"] = deepcopy(
        result["live"]["approved_coins"]
    )
    return result


def prepare_arrays(config, coins, timestamps, *, column_indices=None, n_columns=None):
    selection = load_selection(config)
    if selection is None:
        return None
    if timestamps is None:
        raise ValueError("Organillo requires actual HLCV timestamps")
    ts = np.asarray(timestamps, dtype=np.int64)
    indices = np.searchsorted(selection.timestamps, ts, side="right") - 1
    operative = ts >= _date_ms(config["backtest"]["start_date"])
    if np.any(operative & ((indices < 0) | (ts >= selection.timestamps[-1] + DAY_MS))):
        raise ValueError("Organillo carton does not cover the operational HLCV timestamps")
    day_matches = selection.timestamps[np.maximum(indices, 0)] == (ts // DAY_MS) * DAY_MS
    if np.any(operative & ~day_matches):
        raise ValueError("Organillo carton has missing dates in the operational HLCV timestamps")
    interval_ms = int(config["backtest"]["candle_interval_minutes"]) * 60_000
    if DAY_MS % interval_ms or np.any(ts % interval_ms):
        raise ValueError("Organillo requires candle boundaries aligned to UTC midnight")
    indices = np.ascontiguousarray(indices, dtype=np.int32)
    columns = list(range(len(coins))) if column_indices is None else list(column_indices)
    width = len(coins) if n_columns is None else n_columns
    if len(columns) != len(coins) or len(set(columns)) != len(columns):
        raise ValueError("Organillo column indices must match the dataset coins one-to-one")
    if any(col < 0 or col >= width for col in columns):
        raise ValueError("Organillo column index exceeds the physical HLCV width")
    eligible = np.zeros((len(selection.timestamps), width), dtype=np.uint8)
    lookup = {coin: i for i, coin in enumerate(selection.coins)}
    for coin, col in zip(coins, columns):
        key = _coin_key(coin)
        if key not in lookup:
            raise ValueError(f"Organillo has no column for dataset coin {coin}")
        eligible[:, col] = selection.eligible[:, lookup[key]]
    indices.flags.writeable = eligible.flags.writeable = False
    return PreparedSelection(eligible, indices, selection.content_hash)


def prepare_suite(config, scenarios):
    """Resolve per-scenario cartons before preparing the shared union dataset."""
    from dataclasses import replace

    result = prepare_config(config)
    resolved = []
    for scenario in scenarios:
        candidate = deepcopy(config)
        bt = candidate["backtest"]
        bt["start_date"] = scenario.start_date or bt["start_date"]
        bt["end_date"] = scenario.end_date or bt["end_date"]
        # Overrides are canonical dotted paths in PB8 suites.
        for key, value in (scenario.overrides or {}).items():
            if key.startswith("backtest.organillo_"):
                bt[key.split(".", 1)[1]] = value
        # A scenario may select a different file than the base.
        bt["organillo_carton_hash"] = None
        candidate = prepare_config(candidate, allowed_coins=scenario.coins)
        if candidate["backtest"].get("organillo_mode", False):
            overrides = dict(scenario.overrides or {})
            overrides["backtest.organillo_carton_hash"] = candidate["backtest"]["organillo_carton_hash"]
            resolved.append(replace(scenario, coins=candidate["live"]["approved_coins"]["long"], overrides=overrides))
        else:
            resolved.append(scenario)
    return result, resolved
