"""Fast vectorized-per-day research backtest with Carver buffering, costs and funding.

Timing: at close t the book trades from its held (drifted) weights to the buffered target, paying
`cost_per_trade` on the traded notional. Over (t, t+1] it earns the price return and pays funding.
Weights are fractions of equity, so the equity curve compounds ("sized off the equity at close t-1").
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd


@dataclass
class SimResult:
    returns_net: pd.Series
    returns_gross: pd.Series
    costs: pd.Series
    funding: pd.Series
    traded: pd.Series  # sum |trade| per bar, fraction of equity
    gross_exposure: pd.Series  # sum |weight| after trading
    weights: pd.DataFrame  # held weights after trading at each close

    def equity(self, start: float, net: bool = True) -> pd.Series:
        r = self.returns_net if net else self.returns_gross
        return start * (1.0 + r).cumprod()


def apply_buffer(held: np.ndarray, optimal: np.ndarray, width: np.ndarray) -> np.ndarray:
    """Carver buffer: no trade inside [optimal - width, optimal + width]; else trade to the nearest edge."""
    lower = optimal - width
    upper = optimal + width
    return np.where(held < lower, lower, np.where(held > upper, upper, held))


def simulate(
    target: pd.DataFrame,
    buffer: pd.DataFrame,
    close: pd.DataFrame,
    funding: pd.DataFrame,
    cost_per_trade: float,
) -> SimResult:
    index = target.index
    cols = target.columns
    tgt = target.reindex(columns=cols).fillna(0.0).to_numpy()
    buf = buffer.reindex(index=index, columns=cols).fillna(0.0).to_numpy()
    px = close.reindex(index=index, columns=cols).to_numpy()
    fund = funding.reindex(index=index, columns=cols).fillna(0.0).to_numpy()
    tradable = ~np.isnan(px)
    ret = np.zeros_like(px)
    with np.errstate(invalid="ignore", divide="ignore"):
        ret[1:] = np.where(tradable[1:] & tradable[:-1], px[1:] / px[:-1] - 1.0, 0.0)

    n, m = tgt.shape
    held = np.zeros(m)
    out_w = np.zeros((n, m))
    r_net = np.zeros(n)
    r_gross = np.zeros(n)
    costs = np.zeros(n)
    fund_paid = np.zeros(n)
    traded = np.zeros(n)
    gross = np.zeros(n)
    for t in range(n):
        if t > 0:
            pnl = float(held @ ret[t])
            fpay = float(held @ fund[t])
            port = pnl - fpay
            r_gross[t] = port
            fund_paid[t] = fpay
            held = held * (1.0 + ret[t]) / (1.0 + port) if 1.0 + port > 0 else np.zeros(m)
        # A symbol without a price cannot trade; a held one that lost its price (delisted) is closed.
        new = apply_buffer(held, tgt[t], buf[t])
        new = np.where(tradable[t], new, 0.0)
        trade = np.abs(new - held)
        c = cost_per_trade * float(trade.sum())
        costs[t] = c
        traded[t] = float(trade.sum())
        r_net[t] = (1.0 + r_gross[t]) * (1.0 - c) - 1.0 if t > 0 else -c
        held = new / (1.0 - c) if c < 1 else np.zeros(m)  # equity shrinks by the cost; notional does not
        out_w[t] = held
        gross[t] = float(np.abs(held).sum())
    s = lambda a: pd.Series(a, index=index)  # noqa: E731
    return SimResult(
        returns_net=s(r_net),
        returns_gross=s(r_gross),
        costs=s(costs),
        funding=s(fund_paid),
        traded=s(traded),
        gross_exposure=s(gross),
        weights=pd.DataFrame(out_w, index=index, columns=cols),
    )


def buy_and_hold(close: pd.Series) -> pd.Series:
    return close.pct_change(fill_method=None).fillna(0.0)
