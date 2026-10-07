"""Configuration for the D-TREND replica.

One TOML file drives the research backtest, the Nautilus backtest and the Nautilus live node.
Every value has a default, so a config only names what differs. Defaults follow the thread:
three legs (TS momentum, XS momentum, XS carry), vol target 50 %, top-50 Binance perps,
9.5 bp cost per trade, forecasts scaled to |10| and capped at 20.
"""

from __future__ import annotations

import tomllib
from dataclasses import dataclass, field, fields, is_dataclass
from pathlib import Path
from typing import Any

RULE_KINDS = ("ewmac", "breakout", "bollinger", "returns", "carry")
LEG_MODES = ("ts", "xs")


@dataclass
class RuleConfig:
    """One trading rule variation: a factor with one lookback."""

    kind: str
    lookback: int
    # EWMA span applied to the raw forecast. None = Carver default (lookback / 4, min 2).
    # Set 0 to switch smoothing off (the EWMAC rule is already smooth).
    smooth: int | None = None
    weight: float = 1.0

    @property
    def name(self) -> str:
        return f"{self.kind}_{self.lookback}"


@dataclass
class LegConfig:
    """One leg = one standalone system (own forecasts, own FDM, own IDM)."""

    name: str
    mode: str  # "ts" = time series, "xs" = cross-sectional (demeaned across the universe)
    rules: list[RuleConfig]
    enabled: bool = True


@dataclass
class DataConfig:
    data_dir: str = "data/binance"
    interval: str = "1d"
    start: str = "2021-01-01"
    end: str | None = None
    # Explicit symbol list. Empty = every USDT perp found in the store (survivorship free).
    symbols: list[str] = field(default_factory=list)
    exclude: list[str] = field(
        default_factory=lambda: ["USDCUSDT", "FDUSDUSDT", "BTCDOMUSDT", "DEFIUSDT", "FOOTBALLUSDT", "BLUEBIRDUSDT"]
    )
    # Universe = top N by trailing quote volume (market-cap proxy; Binance has no market cap).
    universe_size: int = 50
    universe_lookback_days: int = 30
    universe_rebalance: str = "MS"  # pandas offset alias: month start
    min_history_days: int = 60
    benchmark: str = "BTCUSDT"


@dataclass
class ForecastConfig:
    target_abs_forecast: float = 10.0
    cap: float = 20.0
    # Pooled expanding estimate of the forecast scalar needs this many dates first.
    scalar_min_periods: int = 60
    max_fdm: float = 2.5
    fdm_lookback_days: int = 500
    xs_min_instruments: int = 5
    # Optional fixed forecast scalars per rule name (for example from a frozen calibration).
    fixed_scalars: dict[str, float] = field(default_factory=dict)


@dataclass
class RiskConfig:
    target_vol: float = 0.50  # annual vol target tau
    vol_span: int = 35  # EWMA span (days) for instrument vol
    vol_floor: float = 0.20  # annual floor for instrument vol
    annualization: int = 365
    max_idm: float = 2.5
    idm_lookback_days: int = 365
    corr_min_periods: int = 60
    # Leg weighting: "equal_risk" = weight legs by 1 / realized leg vol (Narang),
    # "fixed" = use leg_weights as given.
    leg_weighting: str = "equal_risk"
    leg_weights: dict[str, float] = field(default_factory=dict)
    leg_vol_lookback_days: int = 365
    leg_scale_bounds: tuple[float, float] = (0.5, 2.0)
    max_leg_dm: float = 2.5
    max_gross_leverage: float = 5.0
    recalc_every: str = "MS"  # how often IDM / FDM / leg weights are re-estimated


@dataclass
class ExecutionConfig:
    cost_per_trade: float = 0.00095  # 9.5 bp of traded notional
    buffer_fraction: float = 0.10  # Carver position buffer, fraction of the average position
    starting_equity: float = 10_000.0
    min_trade_notional: float = 5.0
    venue: str = "BINANCE"
    default_leverage: float = 10.0


@dataclass
class AnalysisConfig:
    decile_horizons: list[int] = field(default_factory=lambda: list(range(1, 9)))
    speed_limit_sr: float = 0.15  # Carver speed limit: drop rules costing more SR per year
    apply_speed_limit: bool = False


@dataclass
class DTrendConfig:
    data: DataConfig = field(default_factory=DataConfig)
    forecast: ForecastConfig = field(default_factory=ForecastConfig)
    risk: RiskConfig = field(default_factory=RiskConfig)
    execution: ExecutionConfig = field(default_factory=ExecutionConfig)
    analysis: AnalysisConfig = field(default_factory=AnalysisConfig)
    legs: list[LegConfig] = field(default_factory=list)

    @property
    def active_legs(self) -> list[LegConfig]:
        return [leg for leg in self.legs if leg.enabled and leg.rules]


def default_legs() -> list[LegConfig]:
    """The thread's three legs: 4 TS factors, the same 4 as XS momentum, and XS carry."""

    def factors() -> list[RuleConfig]:
        return [
            RuleConfig("ewmac", 16, smooth=0),
            RuleConfig("ewmac", 32, smooth=0),
            RuleConfig("ewmac", 64, smooth=0),
            RuleConfig("breakout", 20),
            RuleConfig("breakout", 40),
            RuleConfig("breakout", 80),
            RuleConfig("bollinger", 20),
            RuleConfig("bollinger", 40),
            RuleConfig("returns", 20),
            RuleConfig("returns", 60),
        ]

    return [
        LegConfig("ts_momentum", "ts", factors()),
        LegConfig("xs_momentum", "xs", factors()),
        LegConfig("xs_carry", "xs", [RuleConfig("carry", 3, smooth=0)]),
    ]


_SECTIONS = {
    "data": DataConfig,
    "forecast": ForecastConfig,
    "risk": RiskConfig,
    "execution": ExecutionConfig,
    "analysis": AnalysisConfig,
}


def _build(cls, raw: dict[str, Any]):
    known = {f.name for f in fields(cls)}
    unknown = set(raw) - known
    if unknown:
        raise ValueError(f"unknown keys for {cls.__name__}: {sorted(unknown)}")
    return cls(**raw)


def config_from_dict(raw: dict[str, Any]) -> DTrendConfig:
    raw = dict(raw)
    legs_raw = raw.pop("legs", None)
    unknown = set(raw) - set(_SECTIONS)
    if unknown:
        raise ValueError(f"unknown config sections: {sorted(unknown)}")
    cfg = DTrendConfig(**{k: _build(_SECTIONS[k], v) for k, v in raw.items()})
    if legs_raw is None:
        cfg.legs = default_legs()
    else:
        cfg.legs = []
        for leg in legs_raw:
            leg = dict(leg)
            rules = [_build(RuleConfig, r) for r in leg.pop("rules", [])]
            cfg.legs.append(_build(LegConfig, {**leg, "rules": rules}))
    if isinstance(cfg.risk.leg_scale_bounds, list):
        cfg.risk.leg_scale_bounds = tuple(cfg.risk.leg_scale_bounds)
    validate_config(cfg)
    return cfg


def load_config(path: str | Path) -> DTrendConfig:
    with open(path, "rb") as f:
        return config_from_dict(tomllib.load(f))


def validate_config(cfg: DTrendConfig) -> None:
    if not cfg.active_legs:
        raise ValueError("config has no enabled legs with rules")
    names = [leg.name for leg in cfg.legs]
    if len(names) != len(set(names)):
        raise ValueError(f"leg names must be unique: {names}")
    for leg in cfg.legs:
        if leg.mode not in LEG_MODES:
            raise ValueError(f"leg {leg.name}: mode must be one of {LEG_MODES}")
        rule_names = [r.name for r in leg.rules]
        if len(rule_names) != len(set(rule_names)):
            raise ValueError(f"leg {leg.name}: duplicate rules {rule_names}")
        for r in leg.rules:
            if r.kind not in RULE_KINDS:
                raise ValueError(f"leg {leg.name}: unknown rule kind {r.kind!r}; known: {RULE_KINDS}")
            if r.lookback < 1:
                raise ValueError(f"leg {leg.name}: rule {r.name} lookback must be >= 1")
            if r.weight < 0:
                raise ValueError(f"leg {leg.name}: rule {r.name} weight must be >= 0")
    if cfg.risk.target_vol <= 0:
        raise ValueError("risk.target_vol must be > 0")
    if cfg.risk.leg_weighting not in ("equal_risk", "fixed"):
        raise ValueError("risk.leg_weighting must be 'equal_risk' or 'fixed'")
    if cfg.risk.leg_weighting == "fixed":
        missing = {leg.name for leg in cfg.active_legs} - set(cfg.risk.leg_weights)
        if missing:
            raise ValueError(f"risk.leg_weights missing legs: {sorted(missing)}")
    lo, hi = cfg.risk.leg_scale_bounds
    if not 0 < lo <= 1 <= hi:
        raise ValueError("risk.leg_scale_bounds must satisfy 0 < low <= 1 <= high")
    if not 0 <= cfg.execution.buffer_fraction < 1:
        raise ValueError("execution.buffer_fraction must be in [0, 1)")
    if cfg.forecast.cap <= cfg.forecast.target_abs_forecast:
        raise ValueError("forecast.cap must be > forecast.target_abs_forecast")


def to_dict(obj) -> Any:
    if is_dataclass(obj):
        return {f.name: to_dict(getattr(obj, f.name)) for f in fields(obj)}
    if isinstance(obj, (list, tuple)):
        return [to_dict(x) for x in obj]
    return obj
