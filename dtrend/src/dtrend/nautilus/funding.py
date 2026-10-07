"""Perpetual funding settlement for the Nautilus backtest.

The Nautilus simulated exchange (1.231) does not settle perpetual funding. This module does:
before the exchange processes the bar that closes at T, it settles every funding event with
time <= T on the position held at that moment (the position set at the previous close).
A long pays `rate * notional`; a short receives it. Notional uses the last known close.
This is the same convention as the research backtest.
"""

from __future__ import annotations

import bisect

import pandas as pd
from nautilus_trader.backtest.config import SimulationModuleConfig
from nautilus_trader.backtest.modules import SimulationModule
from nautilus_trader.model.currencies import USDT
from nautilus_trader.model.data import Bar
from nautilus_trader.model.identifiers import InstrumentId
from nautilus_trader.model.objects import Money


class FundingSettlementModule(SimulationModule):
    def __init__(self, schedule: dict[InstrumentId, pd.Series], config: SimulationModuleConfig | None = None):
        super().__init__(config or SimulationModuleConfig())
        self._times: dict[InstrumentId, list[int]] = {}
        self._rates: dict[InstrumentId, list[float]] = {}
        for iid, series in schedule.items():
            s = series.dropna().sort_index()
            self._times[iid] = [int(pd.Timestamp(t).value) for t in s.index]
            self._rates[iid] = [float(r) for r in s.to_numpy()]
        self._cursor: dict[InstrumentId, int] = {iid: 0 for iid in self._times}
        self._last_close: dict[InstrumentId, float] = {}
        self.total_paid: float = 0.0
        self.payments: list[tuple[int, str, float]] = []

    def pre_process(self, data) -> None:
        if not isinstance(data, Bar):
            return
        iid = data.bar_type.instrument_id
        times = self._times.get(iid)
        if times:
            start = self._cursor[iid]
            end = bisect.bisect_right(times, data.ts_event, lo=start)
            if end > start:
                rate = sum(self._rates[iid][start:end])
                self._cursor[iid] = end
                self._settle(iid, rate, data)
        self._last_close[iid] = data.close.as_double()

    def _settle(self, iid: InstrumentId, rate: float, bar: Bar) -> None:
        qty = 0.0
        for position in self.exchange.cache.positions_open(instrument_id=iid):
            qty += position.signed_qty
        if qty == 0.0:
            return
        price = self._last_close.get(iid, bar.open.as_double())
        payment = qty * price * rate  # > 0: the account pays
        self.total_paid += payment
        self.payments.append((bar.ts_event, iid.value, payment))
        self.exchange.adjust_account(Money(-payment, USDT))

    def process(self, ts_now: int) -> None:
        pass

    def log_diagnostics(self, logger) -> None:
        logger.info(f"Funding paid (net, USDT): {self.total_paid:,.2f} over {len(self.payments)} settlements")

    def reset(self) -> None:
        self._cursor = {iid: 0 for iid in self._times}
        self._last_close.clear()
        self.total_paid = 0.0
        self.payments.clear()
