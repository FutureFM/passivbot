"""D-TREND live node on Binance USDT-M futures (Nautilus Binance adapter).

SAFETY
- `dry_run=True` is the default: the strategy logs the orders it would send and sends nothing.
- `environment` selects LIVE, TESTNET or DEMO. Use TESTNET or DEMO first.
- API keys come from the environment (BINANCE_API_KEY / BINANCE_API_SECRET, or the
  TESTNET variants that the Nautilus adapter reads). Never put keys in the config file.

The strategy seeds its history from the local data store (`dtrend download` keeps it current),
then appends the daily bars and funding updates it receives. It rebalances shortly after each
daily close (00:00 UTC + `rebalance_delay_secs`).
"""

from __future__ import annotations

from pathlib import Path

from nautilus_trader.adapters.binance import BINANCE, BinanceLiveDataClientFactory, BinanceLiveExecClientFactory
from nautilus_trader.adapters.binance.common.enums import BinanceAccountType, BinanceEnvironment
from nautilus_trader.adapters.binance.config import BinanceDataClientConfig, BinanceExecClientConfig
from nautilus_trader.config import (
    CacheConfig,
    InstrumentProviderConfig,
    LiveExecEngineConfig,
    LoggingConfig,
    TradingNodeConfig,
)
from nautilus_trader.live.node import TradingNode
from nautilus_trader.model.identifiers import TraderId

from dtrend.config import DTrendConfig, to_dict
from dtrend.data.panel import load_panel
from dtrend.nautilus.instruments import bar_type, instrument_id
from dtrend.nautilus.strategy import DTrendStrategy, DTrendStrategyConfig


def live_symbols(cfg: DTrendConfig) -> list[str]:
    """Symbols to subscribe: the explicit list, or the current top-N universe from the store."""
    if cfg.data.symbols:
        return list(cfg.data.symbols)
    panel = load_panel(cfg.data)
    last = panel.universe.iloc[-1]
    return sorted(last[last].index)


def build_node_config(
    cfg: DTrendConfig,
    symbols: list[str],
    *,
    environment: str = "TESTNET",
    dry_run: bool = True,
    trader_id: str = "DTREND-001",
) -> tuple[TradingNodeConfig, DTrendStrategyConfig]:
    env = BinanceEnvironment[environment.upper()]
    ids = [instrument_id(s, cfg.execution.venue) for s in symbols]
    provider = InstrumentProviderConfig(load_ids=frozenset(ids))
    strategy = DTrendStrategyConfig(
        instrument_ids=[i.value for i in ids],
        bar_types=[str(bar_type(s, cfg.data.interval, cfg.execution.venue)) for s in symbols],
        dtrend=to_dict(cfg),
        history_path=str(Path(cfg.data.data_dir)),
        rebalance_delay_secs=60,
        max_history_bars=1500,
        dry_run=dry_run,
    )
    return TradingNodeConfig(
        trader_id=TraderId(trader_id),
        logging=LoggingConfig(log_level="INFO"),
        cache=CacheConfig(),
        exec_engine=LiveExecEngineConfig(reconciliation=True, reconciliation_lookback_mins=1440),
        data_clients={
            BINANCE: BinanceDataClientConfig(
                account_type=BinanceAccountType.USDT_FUTURES, environment=env, instrument_provider=provider
            )
        },
        exec_clients={
            BINANCE: BinanceExecClientConfig(
                account_type=BinanceAccountType.USDT_FUTURES, environment=env, instrument_provider=provider
            )
        },
        strategies=[],
    ), strategy


def run_live(cfg: DTrendConfig, *, environment: str = "TESTNET", dry_run: bool = True) -> None:
    symbols = live_symbols(cfg)
    node_cfg, strat_cfg = build_node_config(cfg, symbols, environment=environment, dry_run=dry_run)
    node = TradingNode(config=node_cfg)
    node.trader.add_strategy(DTrendStrategy(strat_cfg))
    node.add_data_client_factory(BINANCE, BinanceLiveDataClientFactory)
    node.add_exec_client_factory(BINANCE, BinanceLiveExecClientFactory)
    node.build()
    try:
        node.run()
    finally:
        node.dispose()
