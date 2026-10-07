"""QuantStats HTML tearsheet of a backtest's daily strategy-equity returns."""

import logging
import math
import os

import pandas as pd

QUANTSTATS_REPORT_FILENAME = "quantstats_report.html"
# Crypto perpetuals trade every calendar day.
PERIODS_PER_YEAR = 365


def daily_strategy_returns(bal_eq: pd.DataFrame) -> pd.Series:
    """Daily close-to-close returns of `strategy_equity` (UTC calendar days)."""
    if "strategy_equity" not in bal_eq.columns:
        raise ValueError("balance/equity frame has no strategy_equity column")
    equity = pd.to_numeric(bal_eq["strategy_equity"], errors="raise")
    index = pd.DatetimeIndex(pd.to_datetime(bal_eq.index))
    daily = pd.Series(equity.to_numpy(), index=index).resample("1D").last().dropna()
    returns = daily.pct_change().dropna()
    returns.name = "strategy"
    return returns


def daily_btc_returns(bal_eq: pd.DataFrame) -> pd.Series | None:
    """Daily BTC/USD returns implied by the simulation's own BTC conversion.

    `usd_total_balance / btc_total_balance` is the BTC price used by the backtest, so the
    benchmark needs no download. Returns None when that price is unavailable or constant.
    """
    if not {"usd_total_balance", "btc_total_balance"}.issubset(bal_eq.columns):
        return None
    usd = pd.to_numeric(bal_eq["usd_total_balance"], errors="coerce")
    btc = pd.to_numeric(bal_eq["btc_total_balance"], errors="coerce")
    price = (usd / btc).where((usd > 0) & (btc > 0))
    index = pd.DatetimeIndex(pd.to_datetime(bal_eq.index))
    daily = pd.Series(price.to_numpy(), index=index).resample("1D").last().dropna()
    daily = daily[daily.map(math.isfinite)]
    if len(daily) < 3 or daily.max() == daily.min():
        return None
    returns = daily.pct_change().dropna()
    returns.name = "BTC"
    return returns


def write_quantstats_report(bal_eq: pd.DataFrame, results_path: str, *, title: str) -> str | None:
    """Write the tearsheet next to the other backtest artifacts.

    Returns the written path, or None when the report cannot be produced; the reason is
    logged. The tearsheet is an optional artifact, so it never fails the backtest.
    """
    try:
        import quantstats as qs
    except ImportError:
        logging.warning(
            "quantstats is not installed; skipping %s (pip install -r requirements-full.txt)",
            QUANTSTATS_REPORT_FILENAME,
        )
        return None
    if "strategy_equity" not in bal_eq.columns:
        logging.warning("skipping %s: no strategy_equity series", QUANTSTATS_REPORT_FILENAME)
        return None
    returns = daily_strategy_returns(bal_eq)
    if len(returns) < 2:
        logging.info(
            "skipping %s: fewer than two daily strategy returns", QUANTSTATS_REPORT_FILENAME
        )
        return None
    path = os.path.join(results_path, QUANTSTATS_REPORT_FILENAME)
    # A ticker string would make quantstats download market data; pass the series instead.
    benchmark = daily_btc_returns(bal_eq)
    if benchmark is None:
        logging.info("%s: no BTC price series; writing it without benchmark", QUANTSTATS_REPORT_FILENAME)
    try:
        qs.reports.html(
            returns,
            benchmark=benchmark,
            output=path,
            title=title,
            periods_per_year=PERIODS_PER_YEAR,
            compounded=True,
        )
    except Exception as exc:  # optional artifact: report it, keep the backtest results
        logging.warning(
            "failed to write %s: %s: %s", QUANTSTATS_REPORT_FILENAME, type(exc).__name__, exc
        )
        return None
    return path
