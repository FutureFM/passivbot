"""Binance USDT-M perpetual instruments for Nautilus, and data conversion (panel -> Bars, funding)."""

from __future__ import annotations

import json
import math
from decimal import Decimal
from pathlib import Path

import numpy as np
import pandas as pd
from nautilus_trader.model.currencies import USDT
from nautilus_trader.model.data import Bar, BarType, FundingRateUpdate
from nautilus_trader.model.identifiers import InstrumentId, Symbol, Venue
from nautilus_trader.model.instruments import CryptoPerpetual
from nautilus_trader.model.objects import Money, Price, Quantity
from nautilus_trader.model.currencies import Currency

from dtrend.data.panel import Panel

BAR_SPEC = {"1d": "1-DAY", "4h": "4-HOUR", "1h": "1-HOUR"}


def instrument_id(symbol: str, venue: str = "BINANCE") -> InstrumentId:
    """Nautilus Binance adapter convention for USD-M perpetuals: BTCUSDT-PERP.BINANCE."""
    return InstrumentId.from_str(f"{symbol}-PERP.{venue}")


def symbol_of(iid: InstrumentId) -> str:
    return iid.symbol.value.removesuffix("-PERP")


def bar_type(symbol: str, interval: str = "1d", venue: str = "BINANCE") -> BarType:
    return BarType.from_str(f"{symbol}-PERP.{venue}-{BAR_SPEC[interval]}-LAST-EXTERNAL")


def _precision(step: float) -> int:
    return max(0, -int(math.floor(math.log10(step) + 1e-9)))


def infer_specs(close: pd.Series) -> dict:
    """Tick / step sizes when no exchange info is available: fine enough for any traded size."""
    px = close.dropna()
    lo, hi = float(px.min()), float(px.max())
    price_precision = int(min(9, max(2, 5 - math.floor(math.log10(lo)))))
    size_precision = int(min(9, max(0, math.ceil(math.log10(hi * 2)))))
    return {
        "price_precision": price_precision,
        "size_precision": size_precision,
        "tick_size": 10.0**-price_precision,
        "step_size": 10.0**-size_precision,
        "min_notional": 5.0,
    }


def specs_from_exchange_info(path: str | Path) -> dict[str, dict]:
    """Parse /fapi/v1/exchangeInfo filters into tick size, step size and min notional per symbol."""
    raw = json.loads(Path(path).read_text())
    out = {}
    for s in raw.get("symbols", []):
        filters = {f["filterType"]: f for f in s.get("filters", [])}
        tick = float(filters.get("PRICE_FILTER", {}).get("tickSize", 0) or 0)
        step = float(filters.get("LOT_SIZE", {}).get("stepSize", 0) or 0)
        if tick <= 0 or step <= 0:
            continue
        out[s["symbol"]] = {
            "price_precision": _precision(tick),
            "size_precision": _precision(step),
            "tick_size": tick,
            "step_size": step,
            "min_notional": float(filters.get("MIN_NOTIONAL", {}).get("notional", 5.0)),
        }
    return out


def make_perpetual(symbol: str, specs: dict, fee: float, venue: str = "BINANCE", margin_init: float = 0.10) -> CryptoPerpetual:
    base = symbol.removesuffix("USDT")
    base_ccy = Currency.from_str(base) if base.isalnum() and len(base) <= 16 else USDT
    pp, sp = specs["price_precision"], specs["size_precision"]
    return CryptoPerpetual(
        instrument_id=instrument_id(symbol, venue),
        raw_symbol=Symbol(symbol),
        base_currency=base_ccy,
        quote_currency=USDT,
        settlement_currency=USDT,
        is_inverse=False,
        price_precision=pp,
        size_precision=sp,
        price_increment=Price(specs["tick_size"], pp),
        size_increment=Quantity(specs["step_size"], sp),
        ts_event=0,
        ts_init=0,
        min_notional=Money(specs["min_notional"], USDT),
        margin_init=Decimal(str(margin_init)),
        margin_maint=Decimal(str(margin_init / 2)),
        maker_fee=Decimal(str(fee)),
        taker_fee=Decimal(str(fee)),
    )


def panel_to_bars(panel: Panel, symbol: str, instrument: CryptoPerpetual, interval: str = "1d") -> list[Bar]:
    """Bars stamped at the bar CLOSE time (the panel index). Open = previous close."""
    close = panel.close[symbol]
    valid = close.notna()
    c = close[valid]
    o = c.shift(1).fillna(c)
    h = np.maximum(panel.high[symbol][valid].fillna(c), np.maximum(o, c))
    lo = np.minimum(panel.low[symbol][valid].fillna(c), np.minimum(o, c))
    vol = (panel.quote_volume[symbol][valid].fillna(0.0) / c).clip(lower=0.0)
    bt = bar_type(symbol, interval, instrument.id.venue.value)
    pp, sp = instrument.price_precision, instrument.size_precision
    bars = []
    for ts, oo, hh, ll, cc, vv in zip(c.index, o, h, lo, c, vol):
        ns = int(ts.value)
        bars.append(
            Bar(
                bt,
                Price(round(oo, pp), pp),
                Price(round(hh, pp), pp),
                Price(round(ll, pp), pp),
                Price(round(cc, pp), pp),
                Quantity(round(vv, sp), sp),
                ns,
                ns,
            )
        )
    return bars


def funding_events(funding_raw: pd.Series, iid: InstrumentId) -> list[FundingRateUpdate]:
    """Raw funding events (one per settlement) as Nautilus FundingRateUpdate data."""
    out = []
    for ts, rate in funding_raw.dropna().items():
        ns = int(pd.Timestamp(ts).value)
        out.append(FundingRateUpdate(iid, Decimal(str(rate)), ns, ns))
    return out


def panel_funding_events(panel: Panel, symbol: str) -> pd.Series:
    """Per-bar funding sum as one event at the bar close (used with synthetic panels)."""
    s = panel.funding[symbol].where(panel.close[symbol].notna()).dropna()
    return s[s != 0.0]
