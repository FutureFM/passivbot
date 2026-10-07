"""D-TREND as a Nautilus Trader strategy (backtest and live).

At each bar close T the strategy:
  1. collects the bars of every instrument for T (a time alert fires shortly after T),
  2. gets the optimal target weights and buffer widths for T:
       - `targets_path` set: read them from a precomputed parquet (fast backtest), or
       - else: run `dtrend.model.system.run_model` on the price/funding history it holds
         (live mode; same function as the research backtest),
  3. computes equity (balance + unrealized PnL) and the held weight of each instrument,
  4. applies the Carver buffer and sends market orders for the difference.

All sizing decisions use only exchange state (positions, balance) plus market data, so a restart
reproduces the same orders.
"""

from __future__ import annotations

from datetime import timedelta
from decimal import Decimal

import numpy as np
import pandas as pd
from nautilus_trader.config import StrategyConfig
from nautilus_trader.model.currencies import USDT
from nautilus_trader.model.data import Bar, BarType, FundingRateUpdate
from nautilus_trader.model.enums import OrderSide, TimeInForce
from nautilus_trader.model.identifiers import InstrumentId, Venue
from nautilus_trader.trading.strategy import Strategy

from dtrend.config import config_from_dict
from dtrend.data.panel import Panel, aggregate_funding, compute_universe
from dtrend.model.system import run_model
from dtrend.research.backtest import apply_buffer


class DTrendStrategyConfig(StrategyConfig, frozen=True):
    instrument_ids: list[str]
    bar_types: list[str]
    dtrend: dict  # raw D-TREND config dict (same schema as the TOML file)
    targets_path: str | None = None  # parquet with columns target/<SYM> and buffer/<SYM>
    history_path: str | None = None  # optional data store dir to seed history (live)
    rebalance_delay_secs: int = 30
    max_history_bars: int = 0  # 0 = keep all; live may cap it for speed
    dry_run: bool = False  # log orders, do not send


class DTrendStrategy(Strategy):
    def __init__(self, config: DTrendStrategyConfig):
        super().__init__(config)
        self.cfg = config_from_dict(config.dtrend)
        self.instrument_ids = [InstrumentId.from_str(s) for s in config.instrument_ids]
        self.bar_types = [BarType.from_str(s) for s in config.bar_types]
        self.symbols = {iid: iid.symbol.value.removesuffix("-PERP") for iid in self.instrument_ids}
        self.venue = Venue(self.cfg.execution.venue)
        self._rows: dict[int, dict[str, tuple[float, float, float, float]]] = {}
        self._funding: dict[str, list[tuple[int, float]]] = {s: [] for s in self.symbols.values()}
        self._history: pd.DataFrame | None = None  # MultiIndex columns (field, symbol)
        self._seed_funding: pd.DataFrame | None = None  # per-bar funding from the data store
        self._scheduled: set[int] = set()
        self._targets: pd.DataFrame | None = None
        self._buffers: pd.DataFrame | None = None
        self.rebalances = 0
        self.decisions: list[dict] = []
        self.equity_log: list[tuple[pd.Timestamp, float]] = []

    # ------------------------------------------------------------------ lifecycle

    def on_start(self) -> None:
        if self.config.targets_path:
            df = pd.read_parquet(self.config.targets_path)
            self._targets = df["target"]
            self._buffers = df["buffer"]
        if self.config.history_path:
            self._seed_history(self.config.history_path)
        for bt in self.bar_types:
            self.subscribe_bars(bt)
        if not self.config.targets_path:
            for iid in self.instrument_ids:
                self.subscribe_funding_rates(iid)

    def on_stop(self) -> None:
        self.log.info(f"D-TREND stopped after {self.rebalances} rebalances")

    # ------------------------------------------------------------------ data

    def on_bar(self, bar: Bar) -> None:
        ts = bar.ts_event
        sym = self.symbols[bar.bar_type.instrument_id]
        self._rows.setdefault(ts, {})[sym] = (
            bar.close.as_double(),
            bar.high.as_double(),
            bar.low.as_double(),
            bar.volume.as_double() * bar.close.as_double(),
        )
        if ts not in self._scheduled:
            self._scheduled.add(ts)
            alert_ns = ts + self.config.rebalance_delay_secs * 1_000_000_000
            self.clock.set_time_alert_ns(f"rebalance-{ts}", alert_ns, lambda event, t=ts: self._rebalance(t))

    def on_funding_rate(self, update: FundingRateUpdate) -> None:
        sym = self.symbols.get(update.instrument_id)
        if sym is not None:
            self._funding[sym].append((update.ts_event, float(update.rate)))

    def _seed_history(self, path: str) -> None:
        from dtrend.data.panel import load_panel
        import dataclasses

        data_cfg = dataclasses.replace(self.cfg.data, data_dir=path, symbols=list(self.symbols.values()))
        panel = load_panel(data_cfg)
        self._history = pd.concat(
            {"close": panel.close, "high": panel.high, "low": panel.low, "quote_volume": panel.quote_volume}, axis=1
        )
        self._seed_funding = panel.funding

    def _append_rows(self, upto: int) -> None:
        done = sorted(t for t in self._rows if t <= upto)
        if not done:
            return
        recs = {}
        for t in done:
            row = self._rows.pop(t)
            for sym, (c, h, lo, qv) in row.items():
                recs[(pd.Timestamp(t, tz="UTC"), sym)] = (c, h, lo, qv)
        df = pd.DataFrame.from_dict(recs, orient="index", columns=["close", "high", "low", "quote_volume"])
        df.index = pd.MultiIndex.from_tuples(df.index)
        wide = df.unstack(level=1)
        wide.index.name = None
        fresh = wide.reindex(columns=pd.MultiIndex.from_product([["close", "high", "low", "quote_volume"], list(self.symbols.values())]))
        if self._history is None:
            self._history = fresh
        else:
            fresh = fresh[~fresh.index.isin(self._history.index)]
            self._history = pd.concat([self._history, fresh]).sort_index()
        if self.config.max_history_bars:
            self._history = self._history.iloc[-self.config.max_history_bars :]

    def _panel(self) -> Panel:
        h = self._history
        close = h["close"]
        bar = close.index.to_series().diff().median() if len(close) > 1 else pd.Timedelta(days=1)
        seed = self._seed_funding
        seed_end = seed.index[-1] if seed is not None and len(seed) else None
        funding = {}
        for sym in close.columns:
            col = pd.Series(0.0, index=close.index)
            if seed is not None and sym in seed.columns:
                col = seed[sym].reindex(close.index).fillna(0.0)
            events = self._funding.get(sym, [])
            if events:
                s = pd.Series([r for _, r in events], index=pd.to_datetime([t for t, _ in events], unit="ns", utc=True))
                live = aggregate_funding(s.groupby(level=0).last(), close.index, bar)
                newer = close.index > seed_end if seed_end is not None else np.ones(len(close.index), dtype=bool)
                col = col.where(~newer, live)
            funding[sym] = col
        funding = pd.DataFrame(funding, index=close.index).where(close.notna(), 0.0).fillna(0.0)
        d = self.cfg.data
        universe = compute_universe(
            close, h["quote_volume"], d.universe_size, d.universe_lookback_days, d.min_history_days, d.universe_rebalance
        )
        return Panel(close, h["high"], h["low"], h["quote_volume"], funding, universe)

    # ------------------------------------------------------------------ trading

    def _target_row(self, ts: pd.Timestamp) -> tuple[pd.Series, pd.Series] | None:
        if self._targets is not None:
            if ts not in self._targets.index:
                return None
            return self._targets.loc[ts], self._buffers.loc[ts]
        panel = self._panel()
        model = run_model(panel, self.cfg)
        return model.target.iloc[-1], model.buffer(self.cfg.execution.buffer_fraction).iloc[-1]

    def _equity(self) -> float:
        account = self.portfolio.account(self.venue)
        if account is None:
            return 0.0
        total = account.balance_total(USDT)
        equity = total.as_double() if total is not None else 0.0
        for money in self.portfolio.unrealized_pnls(self.venue).values():
            equity += money.as_double()
        return equity

    def _rebalance(self, ts_ns: int) -> None:
        self._append_rows(ts_ns)
        ts = pd.Timestamp(ts_ns, tz="UTC")
        row = self._target_row(ts)
        if row is None:
            return
        target, buffer = row
        equity = self._equity()
        if equity <= 0:
            self.log.error(f"non-positive equity {equity}; no rebalance")
            return
        self.rebalances += 1
        self.equity_log.append((ts, equity))
        for iid in self.instrument_ids:
            sym = self.symbols[iid]
            instrument = self.cache.instrument(iid)
            last = self.cache.bar(self._bar_type(iid))
            if instrument is None or last is None or last.ts_event != ts_ns:
                continue  # no fresh price for this instrument at T: it cannot trade now
            price = last.close.as_double()
            qty_now = float(self.portfolio.net_position(iid))
            held_w = qty_now * price / equity
            tgt = float(np.nan_to_num(target.get(sym, 0.0)))
            buf = float(np.nan_to_num(buffer.get(sym, 0.0)))
            new_w = float(apply_buffer(np.array([held_w]), np.array([tgt]), np.array([buf]))[0])
            delta_qty = (new_w - held_w) * equity / price
            self._send(instrument, delta_qty, price, ts, qty_now)

    def _bar_type(self, iid: InstrumentId) -> BarType:
        return next(bt for bt in self.bar_types if bt.instrument_id == iid)

    def _send(self, instrument, delta_qty: float, price: float, ts: pd.Timestamp, qty_now: float) -> None:
        step = float(instrument.size_increment)
        units = np.floor(abs(delta_qty) / step)
        size = units * step
        min_notional = float(instrument.min_notional) if instrument.min_notional is not None else 0.0
        min_notional = max(min_notional, self.cfg.execution.min_trade_notional)
        closing_all = qty_now != 0.0 and abs(abs(delta_qty) - abs(qty_now)) < step and np.sign(delta_qty) != np.sign(qty_now)
        if size <= 0 or (size * price < min_notional and not closing_all):
            return
        side = OrderSide.BUY if delta_qty > 0 else OrderSide.SELL
        self.decisions.append({"ts": ts, "instrument": instrument.id.value, "side": side.name, "qty": size, "price": price})
        if self.config.dry_run:
            self.log.info(f"[dry-run] {side.name} {size} {instrument.id} @~{price}")
            return
        order = self.order_factory.market(
            instrument_id=instrument.id,
            order_side=side,
            quantity=instrument.make_qty(Decimal(str(size))),
            time_in_force=TimeInForce.GTC,
        )
        self.submit_order(order)
