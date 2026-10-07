"""Forecast scaling, cross-sectional demeaning, capping and combination (Carver framework)."""

from __future__ import annotations

import numpy as np
import pandas as pd


def cross_sectional_demean(raw: pd.DataFrame, universe: pd.DataFrame, min_instruments: int) -> pd.DataFrame:
    """XS version of a rule: subtract the universe average at each date."""
    masked = raw.where(universe)
    count = masked.notna().sum(axis=1)
    demeaned = masked.sub(masked.mean(axis=1), axis=0)
    demeaned.loc[count < min_instruments] = np.nan
    return demeaned


def forecast_scalar(raw: pd.DataFrame, universe: pd.DataFrame, target: float, min_periods: int) -> pd.Series:
    """Causal, pooled scalar so that the average |forecast| over the universe is `target`.

    Per date: cross-sectional mean |raw| over universe members. Then an expanding mean over dates.
    """
    cs_abs = raw.abs().where(universe).mean(axis=1)
    pooled = cs_abs.expanding(min_periods=min_periods).mean()
    return target / pooled.replace(0.0, np.nan)


def scale_and_cap(
    raw: pd.DataFrame, universe: pd.DataFrame, target: float, cap: float, min_periods: int, fixed: float | None
) -> tuple[pd.DataFrame, pd.Series]:
    if fixed is not None:
        scalar = pd.Series(float(fixed), index=raw.index)
    else:
        scalar = forecast_scalar(raw, universe, target, min_periods)
    scaled = raw.mul(scalar, axis=0).clip(-cap, cap).where(universe)
    return scaled, scalar


def recalc_dates(index: pd.DatetimeIndex, freq: str) -> pd.DatetimeIndex:
    """Dates at which slow estimates (FDM, IDM, leg weights) are refreshed. Always includes index[0]."""
    marks = pd.date_range(index[0].normalize(), index[-1], freq=freq, tz=index.tz)
    pos = np.unique(np.clip(index.searchsorted(marks), 0, len(index) - 1))
    dates = index[pos]
    if len(dates) == 0 or dates[0] != index[0]:
        dates = dates.insert(0, index[0])
    return dates


def diversification_multiplier(corr: np.ndarray, weights: np.ndarray, cap: float) -> float:
    """1 / sqrt(w' C w) with negative correlations floored at 0 (Carver), capped."""
    c = np.nan_to_num(np.clip(corr, 0.0, 1.0), nan=0.0)
    np.fill_diagonal(c, 1.0)
    w = weights / weights.sum()
    var = float(w @ c @ w)
    if var <= 0:
        return 1.0
    return float(min(cap, 1.0 / np.sqrt(var)))


def forecast_diversification_multiplier(
    forecasts: dict[str, pd.DataFrame], weights: dict[str, float], dates: pd.DatetimeIndex, lookback: int, cap: float
) -> pd.Series:
    """FDM per date: rule forecasts pooled across instruments, correlation over a trailing window."""
    names = list(forecasts)
    index = next(iter(forecasts.values())).index
    out = pd.Series(np.nan, index=index)
    if len(names) == 1:
        return pd.Series(1.0, index=index)
    w = np.array([weights[n] for n in names], dtype=float)
    for ts in dates:
        loc = index.get_loc(ts)
        window = slice(max(0, loc - lookback + 1), loc + 1)
        stacked = pd.DataFrame({n: forecasts[n].iloc[window].stack(future_stack=True) for n in names}).dropna()
        if len(stacked) < 30:
            out.loc[ts] = 1.0
            continue
        out.loc[ts] = diversification_multiplier(stacked.corr().to_numpy(), w, cap)
    return out.ffill().fillna(1.0)


def combine_forecasts(
    forecasts: dict[str, pd.DataFrame], weights: dict[str, float], fdm: pd.Series, cap: float
) -> pd.DataFrame:
    """Weighted sum of rule forecasts x FDM, capped. Missing rules count as 0 (Carver)."""
    total_w = sum(weights.values())
    any_valid = None
    combined = None
    for name, fc in forecasts.items():
        part = fc.fillna(0.0) * (weights[name] / total_w)
        combined = part if combined is None else combined + part
        valid = fc.notna()
        any_valid = valid if any_valid is None else any_valid | valid
    return combined.mul(fdm, axis=0).clip(-cap, cap).where(any_valid)
