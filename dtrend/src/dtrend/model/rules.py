"""Raw trading rules. Each returns a wide (time x symbol) raw forecast, before scaling.

All rules are causal: the value at close t uses data up to close t only.
Rules from the thread:
  1. bollinger  distance to the 2SD Bollinger band
  2. breakout   Rob Carver breakout (distance inside the N-day range)
  3. returns    vol-normalized N-day return
  4. ewmac      EWMA crossover (fast N, slow 4N) / price vol
  carry         average funding over N days (default 3), annualized, / instrument vol
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from dtrend.config import RuleConfig


def daily_vol(close: pd.DataFrame, span: int = 35) -> pd.DataFrame:
    """EWMA standard deviation of simple returns (per bar)."""
    return close.pct_change(fill_method=None).ewm(span=span, min_periods=max(10, span // 3)).std()


def ewmac(close: pd.DataFrame, fast: int, sigma: pd.DataFrame) -> pd.DataFrame:
    fast_ma = close.ewm(span=fast, min_periods=fast).mean()
    slow_ma = close.ewm(span=4 * fast, min_periods=4 * fast).mean()
    return (fast_ma - slow_ma) / (sigma * close)


def breakout(close: pd.DataFrame, lookback: int) -> pd.DataFrame:
    """Carver breakout: 40 * (price - range mid) / range width. Range in [-20, +20]."""
    roll_max = close.rolling(lookback, min_periods=max(2, lookback // 2)).max()
    roll_min = close.rolling(lookback, min_periods=max(2, lookback // 2)).min()
    width = (roll_max - roll_min).replace(0.0, np.nan)
    return 40.0 * (close - (roll_max + roll_min) / 2.0) / width


def bollinger(close: pd.DataFrame, lookback: int) -> pd.DataFrame:
    """Distance from the N-day mean in units of the 2SD band: +1 at the upper band, -1 at the lower."""
    mean = close.rolling(lookback, min_periods=max(2, lookback // 2)).mean()
    std = close.rolling(lookback, min_periods=max(2, lookback // 2)).std().replace(0.0, np.nan)
    return (close - mean) / (2.0 * std)


def returns(close: pd.DataFrame, lookback: int, sigma: pd.DataFrame) -> pd.DataFrame:
    log_ret = np.log(close / close.shift(lookback))
    return log_ret / (sigma * np.sqrt(lookback))


def carry(funding: pd.DataFrame, avg_bars: int, sigma_annual: pd.DataFrame, bars_per_year: int) -> pd.DataFrame:
    """Annualized carry a LONG earns (= minus funding paid) divided by annual vol."""
    avg = funding.rolling(avg_bars, min_periods=1).mean()
    return -avg * bars_per_year / sigma_annual


def smoothing_span(rule: RuleConfig) -> int:
    if rule.smooth is not None:
        return int(rule.smooth)
    return max(2, rule.lookback // 4)


def raw_forecast(
    rule: RuleConfig,
    close: pd.DataFrame,
    funding: pd.DataFrame,
    sigma: pd.DataFrame,
    sigma_annual: pd.DataFrame,
    bars_per_year: int,
) -> pd.DataFrame:
    if rule.kind == "ewmac":
        raw = ewmac(close, rule.lookback, sigma)
    elif rule.kind == "breakout":
        raw = breakout(close, rule.lookback)
    elif rule.kind == "bollinger":
        raw = bollinger(close, rule.lookback)
    elif rule.kind == "returns":
        raw = returns(close, rule.lookback, sigma)
    elif rule.kind == "carry":
        raw = carry(funding.where(close.notna()), rule.lookback, sigma_annual, bars_per_year)
    else:  # validated in config
        raise ValueError(rule.kind)
    span = smoothing_span(rule)
    if span > 1:
        raw = raw.ewm(span=span, min_periods=1).mean().where(raw.notna())
    return raw.replace([np.inf, -np.inf], np.nan)
