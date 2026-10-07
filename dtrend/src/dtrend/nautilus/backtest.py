"""Event-driven D-TREND backtest on the Nautilus BacktestEngine.

Two modes:
- precomputed (default): run the model once on the full panel, write targets to parquet, and let the
  strategy read the row for each close. The model is causal (see tests), so this equals the
  live computation and is much faster.
- live_equivalent: the strategy rebuilds the panel from the bars and funding it received and runs
  the model at every close, exactly as in live trading. Slow; use it to validate the live path.
"""

from __future__ import annotations

import logging
import tempfile
from dataclasses import dataclass
from decimal import Decimal
from pathlib import Path

import pandas as pd
from nautilus_trader.backtest.engine import BacktestEngine, BacktestEngineConfig
from nautilus_trader.config import LoggingConfig
from nautilus_trader.model.currencies import USDT
from nautilus_trader.model.enums import AccountType, OmsType
from nautilus_trader.model.identifiers import TraderId, Venue
from nautilus_trader.model.objects import Money

from dtrend.config import DTrendConfig, to_dict
from dtrend.data.panel import Panel
from dtrend.model.system import run_model
from dtrend.nautilus.funding import FundingSettlementModule
from dtrend.nautilus.instruments import (
    bar_type,
    funding_events,
    infer_specs,
    make_perpetual,
    panel_funding_events,
    panel_to_bars,
    specs_from_exchange_info,
)
from dtrend.nautilus.strategy import DTrendStrategy, DTrendStrategyConfig

log = logging.getLogger(__name__)


@dataclass
class NautilusResult:
    equity: pd.Series
    funding_paid: float
    orders: pd.DataFrame
    positions: pd.DataFrame
    decisions: pd.DataFrame
    rebalances: int


def write_targets(panel: Panel, cfg: DTrendConfig, path: Path) -> Path:
    model = run_model(panel, cfg)
    df = pd.concat({"target": model.target, "buffer": model.buffer(cfg.execution.buffer_fraction)}, axis=1)
    df.to_parquet(path)
    return path


def run_nautilus_backtest(
    panel: Panel,
    cfg: DTrendConfig,
    *,
    mode: str = "precomputed",
    raw_funding: dict[str, pd.Series] | None = None,
    exchange_info: str | Path | None = None,
    work_dir: str | Path | None = None,
    log_level: str = "ERROR",
) -> NautilusResult:
    if mode not in ("precomputed", "live_equivalent"):
        raise ValueError("mode must be 'precomputed' or 'live_equivalent'")
    ex = cfg.execution
    venue = Venue(ex.venue)
    work = Path(work_dir or tempfile.mkdtemp(prefix="dtrend_nt_"))
    work.mkdir(parents=True, exist_ok=True)

    engine = BacktestEngine(
        BacktestEngineConfig(trader_id=TraderId("DTREND-001"), logging=LoggingConfig(log_level=log_level, bypass_logging=log_level == "OFF"))
    )
    symbols = [s for s in panel.close.columns if panel.close[s].notna().any()]
    specs_by_symbol = specs_from_exchange_info(exchange_info) if exchange_info else {}
    instruments = {s: make_perpetual(s, specs_by_symbol.get(s) or infer_specs(panel.close[s]), ex.cost_per_trade, ex.venue) for s in symbols}

    schedule = {}
    for s, inst in instruments.items():
        if raw_funding is not None and s in raw_funding:
            schedule[inst.id] = raw_funding[s]
        else:
            schedule[inst.id] = panel_funding_events(panel, s)
    funding_module = FundingSettlementModule(schedule)
    engine.add_venue(
        venue=venue,
        oms_type=OmsType.NETTING,
        account_type=AccountType.MARGIN,
        base_currency=USDT,
        starting_balances=[Money(ex.starting_equity, USDT)],
        default_leverage=Decimal(str(ex.default_leverage)),
        modules=[funding_module],
    )
    for s, inst in instruments.items():
        engine.add_instrument(inst)
        engine.add_data(panel_to_bars(panel, s, inst, cfg.data.interval))
        if mode == "live_equivalent":
            events = funding_events(schedule[inst.id], inst.id)
            if events:
                engine.add_data(events)

    targets_path = None
    if mode == "precomputed":
        targets_path = str(write_targets(panel, cfg, work / "targets.parquet"))
    strat_cfg = DTrendStrategyConfig(
        instrument_ids=[inst.id.value for inst in instruments.values()],
        bar_types=[str(bar_type(s, cfg.data.interval, ex.venue)) for s in instruments],
        dtrend=to_dict(cfg),
        targets_path=targets_path,
        rebalance_delay_secs=1,
    )
    strategy = DTrendStrategy(strat_cfg)
    engine.add_strategy(strategy)
    engine.run()

    equity = pd.Series(dict(strategy.equity_log), dtype=float)
    result = NautilusResult(
        equity=equity,
        funding_paid=funding_module.total_paid,
        orders=engine.trader.generate_order_fills_report(),
        positions=engine.trader.generate_positions_report(),
        decisions=pd.DataFrame(strategy.decisions),
        rebalances=strategy.rebalances,
    )
    engine.dispose()
    return result
