"""Performance statistics and forecast diagnostics, as shown in the thread.

- Sharpe gross / net, with standard error sqrt((1 + SR^2 / 2) / years)
- return %/yr, trading cost %/yr, turnover (Carver: traded / average gross position, per year)
- per-year tables, max drawdown
- forecast deciles vs forward vol-adjusted total return for T+1..T+8
- information coefficient (rank correlation) per rule
- Carver "speed limit": annual cost of each rule in Sharpe units
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from dtrend.research.backtest import SimResult


def sharpe(r: pd.Series, bars_per_year: int) -> float:
    r = r.dropna()
    sd = r.std()
    return float(r.mean() / sd * np.sqrt(bars_per_year)) if sd > 0 else float("nan")


def sharpe_se(sr: float, years: float) -> float:
    return float(np.sqrt((1.0 + 0.5 * sr * sr) / years)) if years > 0 else float("nan")


def max_drawdown(r: pd.Series) -> float:
    eq = (1.0 + r.fillna(0.0)).cumprod()
    return float((eq / eq.cummax() - 1.0).min())


def summarize(sim: SimResult, bars_per_year: int, start: pd.Timestamp | None = None) -> dict[str, float]:
    sel = slice(start, None)
    net = sim.returns_net.loc[sel]
    gross = sim.returns_gross.loc[sel]
    years = len(net) / bars_per_year
    sr_net = sharpe(net, bars_per_year)
    sr_gross = sharpe(gross, bars_per_year)
    avg_gross_pos = sim.gross_exposure.loc[sel].mean()
    traded = sim.traded.loc[sel].sum()
    eq = (1.0 + net).cumprod()
    return {
        "sharpe_gross": sr_gross,
        "sharpe_net": sr_net,
        "sharpe_net_se": sharpe_se(sr_net, years),
        "return_gross_pct_yr": 100 * gross.mean() * bars_per_year,
        "return_net_pct_yr": 100 * net.mean() * bars_per_year,
        "cagr_pct": 100 * (float(eq.iloc[-1]) ** (1 / years) - 1) if years > 0 and eq.iloc[-1] > 0 else float("nan"),
        "vol_pct_yr": 100 * net.std() * np.sqrt(bars_per_year),
        "cost_pct_yr": 100 * sim.costs.loc[sel].sum() / years if years else float("nan"),
        "funding_pct_yr": -100 * sim.funding.loc[sel].sum() / years if years else float("nan"),
        "turnover_x_yr": traded / avg_gross_pos / years if avg_gross_pos > 0 and years else float("nan"),
        "traded_over_equity_x_yr": traded / years if years else float("nan"),
        "avg_gross_leverage": float(avg_gross_pos),
        "max_drawdown_pct": 100 * max_drawdown(net),
        "years": years,
    }


def yearly_sharpe(returns: dict[str, pd.Series], bars_per_year: int) -> pd.DataFrame:
    rows = {}
    for name, r in returns.items():
        by_year = r.groupby(r.index.year).apply(lambda x: sharpe(x, bars_per_year))
        rows[name] = by_year
        rows[name].loc["full"] = sharpe(r, bars_per_year)
    return pd.DataFrame(rows)


def forward_vol_adjusted_returns(
    close: pd.DataFrame, funding: pd.DataFrame, sigma_annual: pd.DataFrame, horizon: int, bars_per_year: int
) -> pd.DataFrame:
    """Total return a long earns on bar t+h (price minus funding), / annual vol at t, x bars/year."""
    total = close.pct_change(fill_method=None) - funding
    return total.shift(-horizon) / sigma_annual * bars_per_year


def forecast_deciles(
    forecast: pd.DataFrame,
    close: pd.DataFrame,
    funding: pd.DataFrame,
    sigma_annual: pd.DataFrame,
    horizons: list[int],
    bars_per_year: int,
) -> pd.DataFrame:
    """Mean forward vol-adjusted return per forecast decile, pooled over (coin, date) readings."""
    stacked_fc = forecast.stack(future_stack=True).dropna()
    rows = {}
    for h in horizons:
        fwd = forward_vol_adjusted_returns(close, funding, sigma_annual, h, bars_per_year).stack(future_stack=True)
        df = pd.DataFrame({"f": stacked_fc, "y": fwd}).dropna()
        if len(df) < 10:
            continue
        df["decile"] = pd.qcut(df["f"].rank(method="first"), 10, labels=False)
        rows[f"T+{h}"] = df.groupby("decile")["y"].mean()
    return pd.DataFrame(rows).T


def information_coefficient(
    forecast: pd.DataFrame, close: pd.DataFrame, funding: pd.DataFrame, sigma_annual: pd.DataFrame, bars_per_year: int
) -> dict[str, float]:
    """Daily cross-sectional Spearman IC vs next-bar vol-adjusted total return, and its t-stat.

    Pooled IC (one rank correlation over all readings) also scores TS rules that are flat across coins.
    """
    fwd = forward_vol_adjusted_returns(close, funding, sigma_annual, 1, bars_per_year)
    f_rank = forecast.rank(axis=1)
    y_rank = fwd.where(forecast.notna()).rank(axis=1)
    daily = f_rank.corrwith(y_rank, axis=1, method="pearson").dropna()
    pooled = pd.DataFrame({"f": forecast.stack(future_stack=True), "y": fwd.stack(future_stack=True)}).dropna()
    pooled_ic = float(pooled["f"].rank().corr(pooled["y"].rank())) if len(pooled) > 10 else float("nan")
    n = len(daily)
    mean = float(daily.mean()) if n else float("nan")
    t = float(mean / daily.std() * np.sqrt(n)) if n > 1 and daily.std() > 0 else float("nan")
    return {"ic_xs_mean": mean, "ic_xs_t": t, "ic_pooled": pooled_ic, "days": n}


def rule_cost_sr(
    forecast: pd.DataFrame, sigma_annual: pd.DataFrame, cost_per_trade: float, target_abs: float, bars_per_year: int
) -> float:
    """Annual trading cost of a rule in Sharpe units, for one instrument at average size.

    Position (fraction of equity, one instrument at tau) = F / 10 * tau / sigma. Annual cost
    = cost * sum |delta position| / years; in Sharpe units divide by tau, so tau cancels.
    """
    unit = (forecast / target_abs) / sigma_annual
    yearly = unit.diff().abs().sum() / (unit.notna().sum() / bars_per_year)
    yearly = yearly.replace([np.inf, -np.inf], np.nan).dropna()
    return float(cost_per_trade * yearly.mean()) if len(yearly) else float("nan")
