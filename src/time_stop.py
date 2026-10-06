"""Exchange evidence for time stops; order sizing and expiry remain Rust-owned."""
import math
import re
import struct
from decimal import Decimal
from uuid import uuid4

_TIME_STOP_ID = re.compile(r"0x(001e|001f)([0-9a-fA-F]{16})")
_TYPE_MARKER = re.compile(r"0x(?:001e|001f)")
_COMPACT = re.compile(r"t([LS])([0-9A-Za-z])([0-9A-Za-z])([0-9A-Za-z]+)$")
_BASE62 = "0123456789ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz"


def _base62(value):
    out = ""
    while value:
        value, digit = divmod(value, 62)
        out = _BASE62[digit] + out
    return out or "0"


def _unbase62(value):
    total = 0
    for digit in value:
        total = total * 62 + _BASE62.index(digit)
    return total


def time_stop_type_id(custom_id):
    marker = _TYPE_MARKER.search(str(custom_id))
    if marker:
        return int(marker.group()[2:], 16)
    compact = _COMPACT.search(str(custom_id))
    return (30 if compact.group(1) == "L" else 31) if compact else None


def encode_target(custom_id: str, target_size: float) -> str:
    """Retain broker prefix/type/uniqueness while embedding a native-qty target."""
    if not math.isfinite(target_size) or target_size < 0.0:
        raise ValueError("invalid time-stop target size")
    marker = _TYPE_MARKER.search(custom_id)
    if marker is None:
        raise ValueError("time-stop type marker is unavailable")
    end = marker.end()
    if len(custom_id) - end >= 20:
        return custom_id[:end] + struct.pack("!d", target_size).hex() + custom_id[end + 16:]
    # Short broker-prefixed IDs (e.g. OKX) retain the entire broker prefix.
    # Decimal mantissa/exponent encoding preserves the exact round-trip float.
    value = Decimal(str(target_size)).normalize()
    digits = value.as_tuple()
    exponent = digits.exponent
    mantissa = _base62(int("".join(map(str, digits.digits))))
    side = "L" if marker.group() == "0x001e" else "S"
    if not -30 <= exponent <= 31 or len(mantissa) >= 62:
        raise ValueError("time-stop target cannot fit this exchange client order id")
    token = "t" + side + _BASE62[len(mantissa)] + _BASE62[exponent + 30] + mantissa
    available = len(custom_id) - marker.start() - len(token)
    if available < 2:
        raise ValueError("client order id has insufficient space for exact time-stop evidence")
    nonce = _base62(int(uuid4().hex, 16)).zfill(available)[-available:]
    return custom_id[:marker.start()] + token + nonce


def decode_target(custom_id: str, pside: str) -> float:
    match = _TIME_STOP_ID.search(custom_id)
    if match is not None:
        if match.group(1) != ("001e" if pside == "long" else "001f"):
            raise ValueError("time-stop target side mismatch")
        target = struct.unpack("!d", bytes.fromhex(match.group(2)))[0]
    else:
        compact = _COMPACT.search(custom_id)
        if compact is None or compact.group(1) != ("L" if pside == "long" else "S"):
            raise ValueError("time-stop target evidence is unavailable")
        length = _BASE62.index(compact.group(2))
        exponent = _BASE62.index(compact.group(3)) - 30
        payload = compact.group(4)
        if length < 1 or len(payload) < length + 2:
            raise ValueError("truncated time-stop evidence")
        target = float(Decimal(_unbase62(payload[:length])).scaleb(exponent))
    if not math.isfinite(target) or target < 0.0:
        raise ValueError("invalid time-stop target evidence")
    return target


def reconstruct_episodes(events, positions, qty_steps, c_mults, coverage, keys, *, time_stop_keys=()):
    """Replay canonical after-states once for cooldown counts and time-stop facts.

    None denotes an unproven current episode. Fill coverage and authoritative
    position confirmation must precede consumption; no observation-time anchor.
    """
    keys = set(keys)
    time_stop_keys = set(time_stop_keys)
    state = {}
    for event in events:
        key = (str(event.symbol), str(event.position_side).lower())
        if key not in keys:
            continue
        symbol, pside = key
        multiplier = float(event.c_mult)
        delta = float(event.qty) * multiplier * (1 if pside == "long" else -1)
        after = float(event.psize)
        epsilon = max(float(qty_steps[symbol]) * multiplier * 0.25, 1e-10)
        before = after - delta
        row = state.get(key)
        if not all(math.isfinite(v) for v in (delta, after, multiplier)) or after < -epsilon or multiplier <= 0:
            state.pop(key, None)
            continue
        if row is not None and abs(before - row['last_size']) > epsilon:
            state.pop(key, None)
            row = None
        if after <= epsilon:
            state.pop(key, None)
            continue
        if before <= epsilon and delta > 0:
            row = dict(opened=int(event.timestamp), anchor=int(event.timestamp), count=0,
                       pending=None, grid_ref=None, time_ready=True, last_size=after)
            state[key] = row
        if row is None:
            continue
        row['last_size'] = after
        if delta > 0:
            row['count'] += 1
            row['grid_ref'] = None
        if key not in time_stop_keys:
            continue
        pb_type = str(getattr(event, 'pb_order_type', '') or '')
        cid = str(getattr(event, 'client_order_id', '') or '')
        temporal = pb_type == f'close_time_stop_{pside}' or time_stop_type_id(cid) is not None
        if temporal:
            if delta >= 0:
                row['time_ready'] = False
                continue
            try:
                target = decode_target(cid, pside) * multiplier
            except ValueError:
                row['time_ready'] = False
                continue
            if target > before + epsilon or (row['pending'] is not None and target > row['pending'] + epsilon):
                row['time_ready'] = False
                continue
            row['pending'] = target
            price = float(event.price)
            if not math.isfinite(price) or price <= 0:
                row['time_ready'] = False
            else:
                row['grid_ref'] = price
        elif delta < 0 and (pb_type == f'close_expired_{pside}'
                           or (pb_type in ('', 'unknown') and (not cid or '0x' in cid))):
            # Missing attribution might conceal a temporal execution/reset.
            row['time_ready'] = False
        if row['pending'] is not None and after <= row['pending'] + epsilon:
            row['anchor'] = int(event.timestamp)
            row['pending'] = None
    result = {}
    for key in keys:
        symbol, pside = key
        row = state.get(key)
        current = abs(float(positions[symbol][pside]['size'])) * float(c_mults[symbol])
        epsilon = max(float(qty_steps[symbol]) * float(c_mults[symbol]) * 0.25, 1e-10)
        if (row is None or row['count'] < 1 or abs(current - row['last_size']) > epsilon
                or not coverage(start_ms=max(0, row['opened'] - 1)).get('ready', False)):
            result[key] = None
            continue
        result[key] = dict(count=row['count'], opened_timestamp_ms=row['opened'], time_stop=None if not row['time_ready'] else {
            'anchor_timestamp_ms': row['anchor'],
            'pending_target_size': None if row['pending'] is None else row['pending'] / float(c_mults[symbol]),
            'grid_ref_price': row['grid_ref'],
        })
    return result
