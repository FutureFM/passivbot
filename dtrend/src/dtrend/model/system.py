"""The D-TREND system: rules -> leg forecasts -> vol-targeted positions -> equal-risk combined book.

Sizing per leg (Carver, "each leg as its own system"):
    position_i (fraction of equity) = forecast_i / 10 * tau * IDM_leg * w_i / sigma_i
with w_i = 1 / N (equal instrument weights over the current universe) and sigma_i the annual
instrument vol. Position-level vol targeting comes from 1 / sigma_i; strategy-level vol targeting
comes from tau * IDM and from the leg scale below.

Combined book (equal weighting by risk, Narang):
    W_i = DM_legs * sum_k omega_k * g_k * position_k,i
with g_k = tau / realized_vol(leg k) (bounded), omega_k = 1 / K (or fixed weights), and
DM_legs = 1 / sqrt(omega' C omega) over the scaled leg returns. Every estimate is causal.

`run_model` is a pure function of the panel. The research backtest, the Nautilus backtest and the
Nautilus live strategy all call it, so the three paths trade the same targets.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
import pandas as pd

from dtrend.config import DTrendConfig, LegConfig
from dtrend.data.panel import Panel
from dtrend.model import forecasts as fc
from dtrend.model.rules import daily_vol, raw_forecast


@dataclass
class LegOutput:
    name: str
    rule_forecasts: dict[str, pd.DataFrame]
    forecast: pd.DataFrame  # combined leg forecast, capped
    fdm: pd.Series
    idm: pd.Series
    position: pd.DataFrame  # fraction of equity, standalone leg at tau * IDM
    average_position: pd.DataFrame  # position at forecast = +10 (for buffers)
    returns: pd.Series  # standalone gross leg returns (unbuffered, before costs)
    scalars: dict[str, pd.Series] = field(default_factory=dict)


@dataclass
class ModelOutput:
    legs: dict[str, LegOutput]
    sigma_annual: pd.DataFrame
    instrument_weight: pd.DataFrame
    leg_scale: pd.DataFrame  # g_k per date
    leg_weight: pd.DataFrame  # omega_k per date
    leg_dm: pd.Series
    target: pd.DataFrame  # combined optimal position, fraction of equity
    average_position: pd.DataFrame  # combined position at forecast +10
    combined_forecast: pd.DataFrame  # target expressed in forecast units (10 = average)

    def buffer(self, fraction: float) -> pd.DataFrame:
        return self.average_position.abs() * fraction


def bars_per_year(cfg: DTrendConfig) -> int:
    hours = {"1d": 24, "4h": 4, "1h": 1}[cfg.data.interval]
    return int(cfg.risk.annualization * 24 / hours)


def _subsystem_returns(unit_position: pd.DataFrame, total_return: pd.DataFrame) -> pd.DataFrame:
    return unit_position.shift(1) * total_return


def instrument_diversification_multiplier(
    subsystem: pd.DataFrame, universe: pd.DataFrame, dates: pd.DatetimeIndex, lookback: int, min_periods: int, cap: float
) -> pd.Series:
    """IDM with equal instrument weights: sqrt(N / (1 + (N - 1) * avg_corr)), corr floored at 0.

    Correlations are those of the leg's per-instrument subsystem returns, so the IDM is right for
    both directional (TS) and market-neutral (XS) legs.
    """
    index = subsystem.index
    out = pd.Series(np.nan, index=index)
    for ts in dates:
        loc = index.get_loc(ts)
        members = universe.columns[universe.iloc[loc].to_numpy()]
        n = len(members)
        if n < 2:
            out.loc[ts] = 1.0
            continue
        window = subsystem.iloc[max(0, loc - lookback + 1) : loc + 1][members]
        corr = window.corr(min_periods=min_periods).to_numpy()
        upper = corr[np.triu_indices(n, k=1)]
        upper = upper[~np.isnan(upper)]
        if len(upper) == 0:
            out.loc[ts] = 1.0
            continue
        rho = float(np.clip(upper, 0.0, 1.0).mean())
        out.loc[ts] = min(cap, float(np.sqrt(n / (1.0 + (n - 1) * rho))))
    return out.ffill().fillna(1.0)


def _leg(
    leg: LegConfig,
    cfg: DTrendConfig,
    panel: Panel,
    sigma: pd.DataFrame,
    sigma_annual: pd.DataFrame,
    inst_weight: pd.DataFrame,
    total_return: pd.DataFrame,
    dates: pd.DatetimeIndex,
) -> LegOutput:
    f = cfg.forecast
    universe = panel.universe
    rule_fc: dict[str, pd.DataFrame] = {}
    scalars: dict[str, pd.Series] = {}
    bpy = bars_per_year(cfg)
    for rule in leg.rules:
        raw = raw_forecast(rule, panel.close, panel.funding, sigma, sigma_annual, bpy)
        if leg.mode == "xs":
            raw = fc.cross_sectional_demean(raw, universe, f.xs_min_instruments)
        scaled, scalar = fc.scale_and_cap(
            raw, universe, f.target_abs_forecast, f.cap, f.scalar_min_periods, f.fixed_scalars.get(f"{leg.name}.{rule.name}")
        )
        rule_fc[rule.name] = scaled
        scalars[rule.name] = scalar
    weights = {r.name: r.weight for r in leg.rules}
    fdm = fc.forecast_diversification_multiplier(rule_fc, weights, dates, f.fdm_lookback_days, f.max_fdm)
    forecast = fc.combine_forecasts(rule_fc, weights, fdm, f.cap)

    tau = cfg.risk.target_vol
    unit = (forecast / f.target_abs_forecast) * tau / sigma_annual
    subsystem = _subsystem_returns(unit.fillna(0.0), total_return)
    idm = instrument_diversification_multiplier(
        subsystem, universe, dates, cfg.risk.idm_lookback_days, cfg.risk.corr_min_periods, cfg.risk.max_idm
    )
    average_position = (tau * inst_weight / sigma_annual).mul(idm, axis=0).where(universe, 0.0).fillna(0.0)
    position = (forecast / f.target_abs_forecast * average_position).fillna(0.0)
    leg_returns = (position.shift(1) * total_return).sum(axis=1, min_count=1).fillna(0.0)
    return LegOutput(leg.name, rule_fc, forecast, fdm, idm, position, average_position, leg_returns, scalars)


def _leg_combination(
    cfg: DTrendConfig, legs: dict[str, LegOutput], dates: pd.DatetimeIndex
) -> tuple[pd.DataFrame, pd.DataFrame, pd.Series]:
    r = cfg.risk
    names = list(legs)
    index = next(iter(legs.values())).returns.index
    rets = pd.DataFrame({n: legs[n].returns for n in names})
    bpy = bars_per_year(cfg)
    lo, hi = r.leg_scale_bounds

    if r.leg_weighting == "fixed":
        w = np.array([r.leg_weights[n] for n in names], dtype=float)
        weight = pd.DataFrame(np.tile(w / w.sum(), (len(index), 1)), index=index, columns=names)
        scale = pd.DataFrame(1.0, index=index, columns=names)
    else:
        weight = pd.DataFrame(1.0 / len(names), index=index, columns=names)
        realized = rets.rolling(r.leg_vol_lookback_days, min_periods=r.corr_min_periods).std() * np.sqrt(bpy)
        scale = (r.target_vol / realized.replace(0.0, np.nan)).clip(lo, hi)
        scale = scale.fillna(1.0)
        scale.loc[~scale.index.isin(dates)] = np.nan
        scale = scale.ffill()

    dm = pd.Series(np.nan, index=index)
    scaled_rets = rets * scale.shift(1).fillna(1.0)
    for ts in dates:
        loc = index.get_loc(ts)
        window = scaled_rets.iloc[max(0, loc - r.leg_vol_lookback_days + 1) : loc + 1]
        if len(names) == 1 or window.dropna().shape[0] < r.corr_min_periods:
            dm.loc[ts] = 1.0
            continue
        dm.loc[ts] = fc.diversification_multiplier(window.corr().to_numpy(), weight.loc[ts].to_numpy(), r.max_leg_dm)
    return scale, weight, dm.ffill().fillna(1.0)


def run_model(panel: Panel, cfg: DTrendConfig) -> ModelOutput:
    r = cfg.risk
    sigma = daily_vol(panel.close, r.vol_span)
    bpy = bars_per_year(cfg)
    sigma_annual = (sigma * np.sqrt(bpy)).clip(lower=r.vol_floor)
    universe = panel.universe & sigma.notna()
    panel = Panel(panel.close, panel.high, panel.low, panel.quote_volume, panel.funding, universe)
    n_active = universe.sum(axis=1).replace(0, np.nan)
    inst_weight = universe.astype(float).div(n_active, axis=0).fillna(0.0)
    total_return = panel.returns().fillna(0.0) - panel.funding.fillna(0.0)
    dates = fc.recalc_dates(panel.index, r.recalc_every)

    legs = {
        leg.name: _leg(leg, cfg, panel, sigma, sigma_annual, inst_weight, total_return, dates) for leg in cfg.active_legs
    }
    scale, weight, dm = _leg_combination(cfg, legs, dates)

    target = None
    average = None
    for name, leg in legs.items():
        k = (weight[name] * scale[name] * dm).to_numpy()[:, None]
        part = leg.position * k
        avg_part = leg.average_position * k
        target = part if target is None else target + part
        average = avg_part if average is None else average + avg_part

    gross = target.abs().sum(axis=1)
    shrink = (r.max_gross_leverage / gross.replace(0.0, np.nan)).clip(upper=1.0).fillna(1.0)
    target = target.mul(shrink, axis=0)
    average = average.mul(shrink, axis=0)
    combined_forecast = (cfg.forecast.target_abs_forecast * target / average.replace(0.0, np.nan)).where(universe)
    return ModelOutput(legs, sigma_annual, inst_weight, scale, weight, dm, target, average, combined_forecast)


def standalone_leg_model(model: ModelOutput, leg: str) -> tuple[pd.DataFrame, pd.DataFrame]:
    """(target, average position) for one leg traded as its own system."""
    out = model.legs[leg]
    return out.position, out.average_position
