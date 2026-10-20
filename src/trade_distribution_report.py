"""Distribution of per-trade PnL in a backtest, normalized by account balance.

A trade is one round trip of one position: it starts with the first fill of a (coin, side)
while flat and ends with the fill that brings the position size back to exactly zero.
Its PnL is the realized PnL plus signed fees of every fill in it, and it is normalized by
the wallet balance just before its first fill. Positions still open at the end of the
backtest are counted but excluded from the distribution.

Normalized results are compared across time and across runs without the bias of a growing
balance, so aggregates such as "sum of trade %" weight every trade by its size relative to
the account at the time, not by its USD amount.

Usable as a library from `backtest.post_process` or standalone on existing results:

    python src/trade_distribution_report.py backtests/<exchange>/<run_dir>
    python src/trade_distribution_report.py <run_a> <run_b> ... --compare out.html
"""

import argparse
import base64
import html
import io
import json
import logging
import math
import os

import numpy as np
import pandas as pd

TRADE_DISTRIBUTION_REPORT_FILENAME = "trade_distribution.html"
TRADE_DISTRIBUTION_SUMMARY_FILENAME = "trade_distribution.json"
TRADES_FILENAME = "trades.csv"

WIN_COLOR = "#2a78d6"
LOSS_COLOR = "#e34948"
NEUTRAL_COLOR = "#8a8984"
INK_MUTED = "#6b6a65"
GRID_COLOR = "#e4e3df"
# Fixed categorical order for comparing runs; never cycled.
SERIES_COLORS = (
    "#2a78d6", "#eb6834", "#1baf7a", "#eda100",
    "#e87ba4", "#008300", "#4a3aa7", "#e34948",
)
PERCENTILES = (1, 5, 10, 25, 50, 75, 90, 95, 99)
LOSS_THRESHOLDS_PCT = (0.1, 0.25, 0.5, 1.0, 2.0, 5.0)
WIN_THRESHOLDS_PCT = (0.1, 0.25, 0.5, 1.0, 2.0, 5.0)
TOP_TRADE_FRACTIONS = (0.001, 0.005, 0.01, 0.05)
ENTRY_BUCKETS = ((1, 1), (2, 2), (3, 4), (5, 8), (9, 16), (17, 32), (33, None))


def _pside_from_type(fill_type: str) -> str:
    if fill_type.endswith("_long"):
        return "long"
    if fill_type.endswith("_short"):
        return "short"
    raise ValueError(f"cannot infer position side from fill type {fill_type!r}")


# ---------------------------------------------------------------------------- trades


def reconstruct_trades(fdf: pd.DataFrame) -> pd.DataFrame:
    """One row per round trip (coin, pside), in order of close time.

    `fdf` is the processed fills frame written to fills.csv (rows in simulation order).
    """
    columns = [
        "coin", "pside", "open_time", "close_time", "duration_hours", "n_entries",
        "n_closes", "pnl", "fees", "net_pnl", "balance_at_open", "net_pnl_pct",
        "max_wallet_exposure", "exit_type", "closed",
    ]
    if fdf.empty:
        return pd.DataFrame(columns=columns)
    df = fdf.reset_index(drop=True).copy()
    df["_order"] = np.arange(len(df))
    df["timestamp"] = pd.to_datetime(df["timestamp"])
    df["pside"] = df["type"].astype(str).map(_pside_from_type)
    df["net"] = df["pnl"] + df["fee_paid"]
    df = df.sort_values(["coin", "pside", "_order"], kind="stable")
    flat_after = df["psize"].eq(0.0)
    # A new trade starts on the fill following a flat position (or the group's first fill).
    starts = flat_after.groupby([df["coin"], df["pside"]]).shift(1, fill_value=True)
    df["_trade"] = starts.astype(int).cumsum()
    df["_is_entry"] = df["type"].astype(str).str.startswith("entry")

    g = df.groupby("_trade", sort=False)
    first = g.head(1).set_index("_trade")
    last = g.tail(1).set_index("_trade")
    trades = pd.DataFrame(
        {
            "coin": first["coin"],
            "pside": first["pside"],
            "open_time": first["timestamp"],
            "close_time": last["timestamp"],
            "n_entries": g["_is_entry"].sum(),
            "n_closes": g["_is_entry"].size() - g["_is_entry"].sum(),
            "pnl": g["pnl"].sum(),
            "fees": g["fee_paid"].sum(),
            "net_pnl": g["net"].sum(),
            # usd_total_balance is the balance after the fill; undo the first fill's effect.
            "balance_at_open": first["usd_total_balance"] - first["net"],
            "max_wallet_exposure": g["wallet_exposure"].max(),
            "exit_type": last["type"],
            "closed": last["psize"].eq(0.0),
        }
    )
    trades["duration_hours"] = (
        trades["close_time"] - trades["open_time"]
    ).dt.total_seconds() / 3600.0
    trades["net_pnl_pct"] = np.where(
        trades["balance_at_open"] > 0.0,
        trades["net_pnl"] / trades["balance_at_open"] * 100.0,
        np.nan,
    )
    trades.loc[~trades["closed"], "exit_type"] = "open"
    return trades.sort_values(["close_time", "coin", "pside"], kind="stable")[
        columns
    ].reset_index(drop=True)


def _closed(trades: pd.DataFrame) -> pd.DataFrame:
    return trades[trades["closed"].astype(bool)].dropna(subset=["net_pnl_pct"])


# ---------------------------------------------------------------------------- statistics


def _finite(x) -> float | None:
    if x is None:
        return None
    x = float(x)
    return x if math.isfinite(x) else None


def _quantile_skew(pct: pd.Series) -> float | None:
    """Kelly skewness in [-1, 1]: robust to the single outliers that dominate moment skew."""
    p10, p50, p90 = np.percentile(pct, [10, 50, 90])
    return float((p90 + p10 - 2.0 * p50) / (p90 - p10)) if p90 > p10 else None


def _max_streak(mask: np.ndarray) -> int:
    best = run = 0
    for value in mask:
        run = run + 1 if value else 0
        best = max(best, run)
    return best


def _tail_share(values: pd.Series, total: float, frac: float) -> float | None:
    if total <= 0.0 or values.empty:
        return None
    k = max(1, int(math.ceil(len(values) * frac)))
    return float(values.abs().nlargest(k).sum() / total)


def brief_stats(pct: pd.Series) -> dict:
    """Compact per-group statistics used by the breakdown tables."""
    n = len(pct)
    if n == 0:
        return {"n_trades": 0}
    wins, losses = pct[pct > 0.0], pct[pct < 0.0]
    return {
        "n_trades": int(n),
        "win_rate": float(len(wins) / n),
        "mean_win_pct": float(wins.mean()) if len(wins) else None,
        "mean_loss_pct": float(losses.mean()) if len(losses) else None,
        "mean_pct": float(pct.mean()),
        "worst_pct": float(pct.min()),
        "sum_pct": float(pct.sum()),
    }


def distribution_stats(trades: pd.DataFrame) -> dict:
    """Statistics of closed trades' net PnL as % of balance at open."""
    closed = _closed(trades)
    pct = closed["net_pnl_pct"]
    n = len(pct)
    stats = {"n_trades": int(n), "n_open_at_end": int((~trades["closed"].astype(bool)).sum())}
    if n == 0:
        return stats
    wins = pct[pct > 0.0]
    losses = pct[pct < 0.0]
    gross_win = float(wins.sum())
    gross_loss = float(-losses.sum())
    mean_win = float(wins.mean()) if len(wins) else 0.0
    mean_loss = float(losses.mean()) if len(losses) else 0.0
    total = float(pct.sum())
    durations = closed["duration_hours"]

    sorted_desc = np.sort(pct.to_numpy())[::-1]
    without_best = {}
    for frac in TOP_TRADE_FRACTIONS:
        k = max(1, int(math.ceil(n * frac)))
        without_best[f"{frac * 100:g}%"] = {
            "n_removed": k,
            "sum_pct_after": float(sorted_desc[k:].sum()),
        }

    def ladder(values: pd.Series, thresholds, sign: float, gross: float) -> list[dict]:
        rows = []
        for threshold in thresholds:
            hit = values[sign * values >= threshold]
            rows.append(
                {
                    "threshold_pct": threshold,
                    "n_trades": int(len(hit)),
                    "share_of_trades": float(len(hit) / n),
                    "share_of_gross": float(hit.abs().sum() / gross) if gross > 0.0 else None,
                }
            )
        return rows

    order = closed.sort_values("close_time", kind="stable")["net_pnl_pct"].to_numpy()
    stats.update(
        {
            "win_rate": len(wins) / n,
            "n_wins": int(len(wins)),
            "n_losses": int(len(losses)),
            "n_breakeven": int(n - len(wins) - len(losses)),
            "mean_pct": float(pct.mean()),
            "median_pct": float(pct.median()),
            "std_pct": _finite(pct.std()),
            "skew": _finite(pct.skew()) if n > 2 else None,
            "quantile_skew_p10_p90": _quantile_skew(pct),
            "excess_kurtosis": _finite(pct.kurt()) if n > 3 else None,
            "mean_win_pct": mean_win,
            "median_win_pct": float(wins.median()) if len(wins) else 0.0,
            "mean_loss_pct": mean_loss,
            "median_loss_pct": float(losses.median()) if len(losses) else 0.0,
            "payoff_ratio": mean_win / -mean_loss if mean_loss < 0.0 else None,
            # Win rate at which the observed mean win and mean loss would net to zero.
            "breakeven_win_rate": (
                -mean_loss / (mean_win - mean_loss) if mean_loss < 0.0 < mean_win else None
            ),
            "profit_factor": gross_win / gross_loss if gross_loss > 0.0 else None,
            "largest_win_pct": float(pct.max()),
            "largest_loss_pct": float(pct.min()),
            "worst_loss_in_mean_wins": (
                -float(pct.min()) / mean_win if mean_win > 0.0 and pct.min() < 0.0 else None
            ),
            "percentiles_pct": {f"p{p}": float(np.percentile(pct, p)) for p in PERCENTILES},
            "loss_share_worst_1pct_of_losses": _tail_share(losses, gross_loss, 0.01),
            "loss_share_worst_10pct_of_losses": _tail_share(losses, gross_loss, 0.10),
            "win_share_best_1pct_of_wins": _tail_share(wins, gross_win, 0.01),
            "win_share_best_10pct_of_wins": _tail_share(wins, gross_win, 0.10),
            "sum_pct": total,
            "sum_pct_without_best": without_best,
            "loss_ladder": ladder(losses, LOSS_THRESHOLDS_PCT, -1.0, gross_loss),
            "win_ladder": ladder(wins, WIN_THRESHOLDS_PCT, 1.0, gross_win),
            "max_consecutive_losses": _max_streak(order < 0.0),
            "max_consecutive_wins": _max_streak(order > 0.0),
            "sum_net_pnl_usd": float(closed["net_pnl"].sum()),
            "median_duration_hours": _finite(durations.median()),
            "median_duration_hours_wins": _finite(durations[pct > 0.0].median()),
            "median_duration_hours_losses": _finite(durations[pct < 0.0].median()),
            "p95_duration_hours": _finite(durations.quantile(0.95)),
            "mean_entries_per_trade": float(closed["n_entries"].mean()),
        }
    )
    return stats


def daily_trade_counts(trades: pd.DataFrame, start=None, end=None) -> pd.Series:
    """Closed trades per UTC calendar day, including days without any close."""
    times = pd.to_datetime(_closed(trades)["close_time"])
    if start is None:
        start = pd.to_datetime(trades["open_time"]).min() if len(trades) else None
    if end is None:
        end = times.max() if len(times) else None
    if start is None or end is None or pd.isna(start) or pd.isna(end):
        return pd.Series(dtype=int, name="closed_trades")
    days = pd.date_range(pd.Timestamp(start).normalize(), pd.Timestamp(end).normalize(), freq="1D")
    counts = times.dt.normalize().value_counts().reindex(days, fill_value=0).sort_index()
    counts.name = "closed_trades"
    return counts


def daily_stats(counts: pd.Series) -> dict:
    if counts.empty:
        return {"n_days": 0}
    return {
        "n_days": int(len(counts)),
        "mean_trades_per_day": float(counts.mean()),
        "median_trades_per_day": float(counts.median()),
        "p90_trades_per_day": float(counts.quantile(0.9)),
        "max_trades_per_day": int(counts.max()),
        "days_without_trades_share": float((counts == 0).mean()),
    }


def extreme_days(trades: pd.DataFrame, n: int = 5) -> dict:
    """Days whose closed trades summed to the largest losses/gains (correlated outcomes)."""
    closed = _closed(trades)
    if closed.empty:
        return {"worst": [], "best": []}
    day = pd.to_datetime(closed["close_time"]).dt.normalize()
    grp = closed.groupby(day)["net_pnl_pct"]
    frame = pd.DataFrame(
        {"sum_pct": grp.sum(), "n_trades": grp.size(), "n_losses": grp.apply(lambda s: int((s < 0).sum()))}
    )

    def rows(sub):
        return [
            {"date": idx.strftime("%Y-%m-%d"), "sum_pct": float(r.sum_pct),
             "n_trades": int(r.n_trades), "n_losses": int(r.n_losses)}
            for idx, r in sub.iterrows()
        ]

    return {
        "worst": rows(frame.nsmallest(n, "sum_pct")),
        "best": rows(frame.nlargest(n, "sum_pct")),
    }


def breakdown(trades: pd.DataFrame, key: pd.Series, label: str) -> list[dict]:
    closed = _closed(trades)
    rows = []
    for value, grp in closed.groupby(key.loc[closed.index], sort=True, observed=True):
        rows.append({label: value, **brief_stats(grp["net_pnl_pct"])})
    return rows


def _entries_bucket(n_entries: pd.Series) -> pd.Series:
    labels = pd.Series(index=n_entries.index, dtype=object)
    for lo, hi in ENTRY_BUCKETS:
        mask = (n_entries >= lo) & (True if hi is None else n_entries <= hi)
        labels[mask] = f"{lo}+" if hi is None else (str(lo) if lo == hi else f"{lo}-{hi}")
    order = [f"{lo}+" if hi is None else (str(lo) if lo == hi else f"{lo}-{hi}") for lo, hi in ENTRY_BUCKETS]
    return pd.Categorical(labels, categories=order, ordered=True)


def build_summary(trades: pd.DataFrame, start=None, end=None) -> tuple[dict, pd.Series]:
    counts = daily_trade_counts(trades, start, end)
    closed_times = pd.to_datetime(trades["close_time"])
    entries_bucket = pd.Series(_entries_bucket(trades["n_entries"]), index=trades.index)
    by_entries = [
        row for row in breakdown(trades, entries_bucket, "entries") if row["n_trades"] > 0
    ]
    by_coin = breakdown(trades, trades["coin"], "coin")
    summary = {
        "definition": (
            "trade = round trip of one (coin, side) position from flat to flat; "
            "pct = (realized pnl + fees) / wallet balance before its first fill * 100"
        ),
        "all": distribution_stats(trades),
        "daily": daily_stats(counts),
        "extreme_days": extreme_days(trades),
        "by_year": [
            {**row, "year": int(row["year"])}
            for row in breakdown(trades, closed_times.dt.year, "year")
        ],
        "by_entries": [{**row, "entries": str(row["entries"])} for row in by_entries],
        "by_coin": sorted(by_coin, key=lambda r: r["sum_pct"]),
        "exit_types": sorted(
            breakdown(trades, trades["exit_type"].astype(str), "exit_type"),
            key=lambda r: -r["n_trades"],
        ),
    }
    for pside in ("long", "short"):
        side = trades[trades["pside"] == pside]
        if len(side):
            summary[pside] = distribution_stats(side)
            summary[f"daily_{pside}"] = daily_stats(daily_trade_counts(side, start, end))
    summary["insights"] = insights(summary)
    return summary, counts


def insights(summary: dict) -> list[str]:
    """Plain-language observations derived from the summary."""
    s = summary["all"]
    if not s.get("n_trades"):
        return []
    out = []
    mw, ml, wr = s["mean_win_pct"], s["mean_loss_pct"], s["win_rate"]
    if s.get("breakeven_win_rate") is not None:
        margin = (wr - s["breakeven_win_rate"]) * 100
        out.append(
            f"The average loss ({ml:+.3f}%) is {-ml / mw:.1f}× the average win ({mw:+.3f}%), so "
            f"the strategy needs a {s['breakeven_win_rate'] * 100:.1f}% win rate to break even; "
            f"it has {wr * 100:.1f}% ({margin:+.1f} points of margin)."
        )
    if s.get("worst_loss_in_mean_wins"):
        out.append(
            f"The worst trade lost {-s['largest_loss_pct']:.2f}% of the balance, which takes "
            f"{s['worst_loss_in_mean_wins']:,.0f} average wins to repay."
        )
    if s.get("loss_share_worst_10pct_of_losses") is not None:
        n_tail = max(1, math.ceil(s["n_losses"] * 0.1))
        out.append(
            f"The worst 10% of losing trades ({n_tail:,}) account for "
            f"{s['loss_share_worst_10pct_of_losses'] * 100:.0f}% of all losses; the worst 1% for "
            f"{s['loss_share_worst_1pct_of_losses'] * 100:.0f}%."
        )
    total = s["sum_pct"]
    one = s["sum_pct_without_best"].get("1%")
    if one and total > 0:
        kept = one["sum_pct_after"] / total
        out.append(
            f"Removing the best 1% of trades ({one['n_removed']:,}) leaves {kept * 100:.0f}% of "
            f"the summed trade result"
            + (" — profit depends heavily on a few outliers." if kept < 0.5 else ".")
        )
    by_entries = summary.get("by_entries") or []
    negative = [r for r in by_entries if r["sum_pct"] < 0.0]
    if negative:
        n_neg = sum(r["n_trades"] for r in negative)
        out.append(
            f"Trades with {', '.join(r['entries'] for r in negative)} entry fills "
            f"({n_neg:,} trades, {n_neg / s['n_trades'] * 100:.0f}%) lose in aggregate: "
            f"{sum(r['sum_pct'] for r in negative):+.1f}% summed, against "
            f"{sum(r['sum_pct'] for r in by_entries if r['sum_pct'] >= 0.0):+.1f}% from the rest."
        )
    elif len(by_entries) >= 2:
        worst_b = min(by_entries, key=lambda r: r["win_rate"])
        out.append(
            f"Every entry-depth group is profitable in aggregate; the lowest win rate is "
            f"{worst_b['win_rate'] * 100:.0f}% for trades with {worst_b['entries']} entry fills."
        )
    dw, dl = s.get("median_duration_hours_wins"), s.get("median_duration_hours_losses")
    if dw and dl:
        out.append(
            f"Losing trades last {dl / dw:.1f}× as long as winning ones "
            f"(median {dl:.1f} h vs {dw:.1f} h)."
        )
    worst = summary.get("extreme_days", {}).get("worst") or []
    if worst:
        w = worst[0]
        out.append(
            f"Worst day: {w['date']}, {w['n_losses']} losing of {w['n_trades']} trades closed, "
            f"summing {w['sum_pct']:+.2f}% of balance; losses cluster when coins move together."
        )
    out.append(
        f"Longest losing streak: {s['max_consecutive_losses']} consecutive trades (by close "
        f"time, all coins)."
    )
    years = [r for r in summary.get("by_year", []) if r["n_trades"] >= 30]
    if len(years) >= 2:
        best = max(years, key=lambda r: r["mean_pct"])
        worst_y = min(years, key=lambda r: r["mean_pct"])
        out.append(
            f"Expectancy per trade ranges from {worst_y['mean_pct']:+.4f}% ({worst_y['year']}) "
            f"to {best['mean_pct']:+.4f}% ({best['year']})."
        )
    return out


# ---------------------------------------------------------------------------- figures


def _png(fig) -> str:
    buf = io.BytesIO()
    fig.savefig(buf, format="png", dpi=110, bbox_inches="tight")
    return base64.b64encode(buf.getvalue()).decode("ascii")


def _subplots(*args, figsize, **kwargs):
    # Figure objects need no pyplot state or backend switch, so the report never
    # interferes with other plots open in the same process.
    from matplotlib.figure import Figure

    fig = Figure(figsize=figsize)
    return fig, fig.subplots(*args, **kwargs)


def _scoped_style(func):
    """Apply the report's font settings only while its figures are built."""
    import functools

    @functools.wraps(func)
    def wrapper(*args, **kwargs):
        import matplotlib

        with matplotlib.rc_context(
            {"font.size": 10, "axes.titlesize": 11, "axes.titleweight": "bold"}
        ):
            return func(*args, **kwargs)

    return wrapper


def _style(ax, grid_axis="y"):
    for spine in ("top", "right"):
        ax.spines[spine].set_visible(False)
    for spine in ("left", "bottom"):
        ax.spines[spine].set_color("#c4c3be")
    ax.tick_params(colors=INK_MUTED)
    ax.grid(True, axis=grid_axis, color=GRID_COLOR, linewidth=0.8)
    ax.set_axisbelow(True)


def _symlog_threshold(values: np.ndarray) -> float:
    med = float(np.median(np.abs(values))) if len(values) else 0.0
    return med if med > 0.0 else 0.01


def _symlog_pct_axis(ax, values: np.ndarray) -> None:
    """Symmetric-log y axis with readable percent ticks at powers of ten."""
    thresh = _symlog_threshold(values)
    ax.set_yscale("symlog", linthresh=thresh, linscale=0.6)
    top = max(float(np.max(np.abs(values))) if len(values) else 1.0, thresh)
    decades = [10.0**e for e in range(-3, 4) if thresh <= 10.0**e <= top * 1.0001]
    ticks = [-d for d in decades[::-1]] + [0.0] + decades
    ax.set_yticks(ticks)
    ax.set_ylim(float(np.min(values)) * 1.3 if len(values) else -1.0, top * 1.3)
    ax.set_yticklabels(["0" if t == 0 else f"{t:+g}%" for t in ticks])
    ax.minorticks_off()


def _quantile_curve(ax, pct: np.ndarray, color: str, label: str, fill: bool) -> None:
    srt = np.sort(pct)
    x = (np.arange(len(srt)) + 0.5) / len(srt) * 100.0
    if fill:
        ax.fill_between(x, srt, 0.0, where=srt < 0.0, color=LOSS_COLOR, linewidth=0, step="mid")
        ax.fill_between(x, srt, 0.0, where=srt > 0.0, color=WIN_COLOR, linewidth=0, step="mid")
    else:
        ax.plot(x, srt, color=color, linewidth=2, label=label)


@_scoped_style
def _figures(trades: pd.DataFrame, counts: pd.Series, summary: dict) -> dict[str, str]:
    stats = summary["all"]
    closed = _closed(trades)
    pct = closed["net_pnl_pct"].to_numpy()
    figs = {}

    # 1) Body of the distribution: the central 99% on a linear axis.
    lo, hi = np.percentile(pct, [0.5, 99.5])
    inside = pct[(pct >= lo) & (pct <= hi)]
    fig, ax = _subplots(figsize=(11, 4))
    bins = np.linspace(lo, hi, 90)
    if 0.0 > lo and 0.0 < hi:
        bins = np.unique(np.concatenate([bins, [0.0]]))
    ax.hist(inside[inside <= 0.0], bins=bins, color=LOSS_COLOR, label="losing trades")
    ax.hist(inside[inside > 0.0], bins=bins, color=WIN_COLOR, label="winning trades")
    ax.axvline(stats["mean_win_pct"], color=WIN_COLOR, linestyle="--", linewidth=1.4,
               label=f"mean win {stats['mean_win_pct']:+.3f}%")
    ax.axvline(stats["mean_loss_pct"], color=LOSS_COLOR, linestyle="--", linewidth=1.4,
               label=f"mean loss {stats['mean_loss_pct']:+.3f}%")
    ax.axvline(stats["mean_pct"], color="#1f1f1d", linestyle=":", linewidth=1.4,
               label=f"expectancy {stats['mean_pct']:+.4f}%")
    n_out_lo, n_out_hi = int((pct < lo).sum()), int((pct > hi).sum())
    ax.set_title(
        f"Trade results, central 99% ({n_out_lo} trades below and {n_out_hi} above not shown)",
        loc="left",
    )
    ax.set_xlabel("net PnL per trade, % of balance at open")
    ax.set_ylabel("trades")
    ax.legend(frameon=False, loc="upper left")
    _style(ax)
    figs["histogram"] = _png(fig)

    # 2) Every trade sorted, on a symmetric-log axis so both the body and the tails show.
    fig, ax = _subplots(figsize=(11, 4))
    _quantile_curve(ax, pct, WIN_COLOR, "", fill=True)
    _symlog_pct_axis(ax, pct)
    ax.axhline(0.0, color=NEUTRAL_COLOR, linewidth=0.8)
    ax.set_xlim(0, 100)
    ax.set_xlabel("trade percentile (worst → best)")
    ax.set_ylabel("% of balance (symmetric log)")
    ax.set_title("Every trade sorted, worst to best (red: losing trades)", loc="left")
    _style(ax, "both")
    figs["sorted"] = _png(fig)

    # 3) Result by number of entry fills (DCA depth).
    rows = summary.get("by_entries") or []
    if rows:
        fig, (ax1, ax2) = _subplots(1, 2, figsize=(11, 3.6))
        labels = [r["entries"] for r in rows]
        means = [r["mean_pct"] for r in rows]
        ax1.bar(labels, means, color=[WIN_COLOR if m > 0 else LOSS_COLOR for m in means], width=0.7)
        ax1.axhline(0.0, color=NEUTRAL_COLOR, linewidth=0.8)
        ax1.set_title("Mean result by entry fills per trade", loc="left")
        ax1.set_xlabel("entry fills")
        ax1.set_ylabel("mean % of balance")
        _style(ax1)
        ax2.bar(labels, [r["sum_pct"] for r in rows],
                color=[WIN_COLOR if r["sum_pct"] > 0 else LOSS_COLOR for r in rows], width=0.7)
        ax2.axhline(0.0, color=NEUTRAL_COLOR, linewidth=0.8)
        ax2.set_title("Total contribution (sum of trade %)", loc="left")
        ax2.set_xlabel("entry fills")
        _style(ax2)
        fig.tight_layout()
        figs["entries"] = _png(fig)

    # 4) Duration vs result.
    fig, ax = _subplots(figsize=(11, 3.8))
    hours = closed["duration_hours"].clip(lower=1.0 / 60.0).to_numpy()
    ax.scatter(hours[pct > 0], pct[pct > 0], s=7, c=WIN_COLOR, alpha=0.35, linewidths=0, label="win")
    ax.scatter(hours[pct <= 0], pct[pct <= 0], s=7, c=LOSS_COLOR, alpha=0.5, linewidths=0, label="loss")
    ax.set_xscale("log")
    _symlog_pct_axis(ax, pct)
    ax.axhline(0.0, color=NEUTRAL_COLOR, linewidth=0.8)
    ax.set_xlabel("trade duration, hours (log)")
    ax.set_ylabel("% of balance (symmetric log)")
    ax.set_title("Duration vs result", loc="left")
    ax.legend(frameon=False, loc="upper left", markerscale=2)
    _style(ax, "both")
    figs["duration"] = _png(fig)

    # 5) Trades closed per day.
    if not counts.empty:
        fig, (ax1, ax2) = _subplots(1, 2, figsize=(11, 3.6), gridspec_kw={"width_ratios": [2.4, 1]})
        ax1.plot(counts.index, counts.to_numpy(), color="#c4c3be", linewidth=0.7, label="daily")
        ax1.plot(counts.index, counts.rolling(30, min_periods=1).mean().to_numpy(),
                 color=WIN_COLOR, linewidth=2, label="30-day mean")
        ax1.set_ylabel("closed trades / day")
        ax1.set_title("Trades closed per day", loc="left")
        ax1.legend(frameon=False, loc="upper left")
        _style(ax1)
        ax2.hist(counts.to_numpy(), bins=min(40, max(5, int(counts.max()) + 1)), color=WIN_COLOR)
        ax2.axvline(counts.mean(), color="#1f1f1d", linestyle="--", linewidth=1.2)
        ax2.set_xlabel("closed trades / day")
        ax2.set_ylabel("days")
        ax2.set_title(f"Mean {counts.mean():.1f} per day", loc="left")
        _style(ax2)
        fig.tight_layout()
        figs["daily"] = _png(fig)
    return figs


# ---------------------------------------------------------------------------- HTML


def _fmt(value, kind="pct") -> str:
    if value is None or (isinstance(value, float) and not math.isfinite(value)):
        return "—"
    if kind == "pct":
        return f"{value:+.4f}%"
    if kind == "pct2":
        return f"{value:+.2f}%"
    if kind == "share":
        return f"{value * 100:.2f}%" if 0.0 < abs(value) < 0.01 else f"{value * 100:.1f}%"
    if kind == "abspct":
        return f"{value:.4f}%"
    if kind == "ratio":
        return f"{value:.2f}"
    if kind == "x":
        return f"{value:,.0f}×"
    if kind == "int":
        return f"{int(value):,}"
    if kind == "hours":
        return f"{value:.1f} h"
    if kind == "usd":
        return f"{value:,.0f}"
    return html.escape(str(value))


STAT_ROWS = (
    ("Closed trades", "n_trades", "int"),
    ("Open at end (excluded)", "n_open_at_end", "int"),
    ("Win rate", "win_rate", "share"),
    ("Break-even win rate", "breakeven_win_rate", "share"),
    ("Mean win", "mean_win_pct", "pct"),
    ("Median win", "median_win_pct", "pct"),
    ("Mean loss", "mean_loss_pct", "pct"),
    ("Median loss", "median_loss_pct", "pct"),
    ("Payoff ratio (mean win / |mean loss|)", "payoff_ratio", "ratio"),
    ("Expectancy per trade (mean)", "mean_pct", "pct"),
    ("Median trade", "median_pct", "pct"),
    ("Std dev", "std_pct", "abspct"),
    ("Skewness (moment)", "skew", "ratio"),
    ("Skewness (quantile p10/p50/p90, outlier-robust)", "quantile_skew_p10_p90", "ratio"),
    ("Excess kurtosis", "excess_kurtosis", "ratio"),
    ("Profit factor", "profit_factor", "ratio"),
    ("Largest win", "largest_win_pct", "pct2"),
    ("Largest loss", "largest_loss_pct", "pct2"),
    ("Worst loss in average wins", "worst_loss_in_mean_wins", "x"),
    ("Worst 1% of losses: share of total loss", "loss_share_worst_1pct_of_losses", "share"),
    ("Worst 10% of losses: share of total loss", "loss_share_worst_10pct_of_losses", "share"),
    ("Best 1% of wins: share of total gain", "win_share_best_1pct_of_wins", "share"),
    ("Best 10% of wins: share of total gain", "win_share_best_10pct_of_wins", "share"),
    ("Longest losing streak (trades)", "max_consecutive_losses", "int"),
    ("Median duration", "median_duration_hours", "hours"),
    ("Median duration, wins", "median_duration_hours_wins", "hours"),
    ("Median duration, losses", "median_duration_hours_losses", "hours"),
    ("95th percentile duration", "p95_duration_hours", "hours"),
    ("Mean entry fills per trade", "mean_entries_per_trade", "ratio"),
    ("Sum of trade %", "sum_pct", "pct2"),
    ("Net PnL of closed trades (USD)", "sum_net_pnl_usd", "usd"),
)

DAILY_ROWS = (
    ("Days", "n_days", "int"),
    ("Mean closed trades / day", "mean_trades_per_day", "ratio"),
    ("Median closed trades / day", "median_trades_per_day", "ratio"),
    ("90th percentile / day", "p90_trades_per_day", "ratio"),
    ("Max closed trades / day", "max_trades_per_day", "int"),
    ("Days without a closed trade", "days_without_trades_share", "share"),
)

BRIEF_COLUMNS = (
    ("Trades", "n_trades", "int"),
    ("Win rate", "win_rate", "share"),
    ("Mean win", "mean_win_pct", "pct"),
    ("Mean loss", "mean_loss_pct", "pct"),
    ("Expectancy", "mean_pct", "pct"),
    ("Worst trade", "worst_pct", "pct2"),
    ("Sum of trade %", "sum_pct", "pct2"),
)


def _stat_table(summary: dict, rows, sections: list[tuple[str, str]]) -> str:
    head = "".join(f"<th>{html.escape(label)}</th>" for label, _ in sections)
    body = []
    for label, key, kind in rows:
        cells = "".join(
            f"<td>{_fmt(summary.get(section, {}).get(key), kind)}</td>" for _, section in sections
        )
        body.append(f"<tr><th scope='row'>{html.escape(label)}</th>{cells}</tr>")
    return (
        f"<div class='scroll'><table><thead><tr><th></th>{head}</tr></thead>"
        f"<tbody>{''.join(body)}</tbody></table></div>"
    )


def _rows_table(rows: list[dict], key: str, key_label: str, columns=BRIEF_COLUMNS) -> str:
    head = f"<th>{html.escape(key_label)}</th>" + "".join(
        f"<th>{html.escape(label)}</th>" for label, _, _ in columns
    )
    body = "".join(
        f"<tr><th scope='row'>{html.escape(str(r[key]))}</th>"
        + "".join(f"<td>{_fmt(r.get(k), kind)}</td>" for _, k, kind in columns)
        + "</tr>"
        for r in rows
    )
    return (
        f"<div class='scroll'><table><thead><tr>{head}</tr></thead>"
        f"<tbody>{body}</tbody></table></div>"
    )


def _ladder_table(stats: dict) -> str:
    def part(rows, word):
        return "".join(
            f"<tr><th scope='row'>{word} ≥ {r['threshold_pct']:g}%</th>"
            f"<td>{_fmt(r['n_trades'], 'int')}</td><td>{_fmt(r['share_of_trades'], 'share')}</td>"
            f"<td>{_fmt(r['share_of_gross'], 'share')}</td></tr>"
            for r in rows
        )

    return (
        "<div class='scroll'><table><thead><tr><th></th><th>Trades</th><th>Share of trades</th>"
        "<th>Share of gross loss / gain</th></tr></thead><tbody>"
        f"{part(stats.get('loss_ladder', []), 'Loss')}"
        f"<tr class='sep'><td colspan='4'></td></tr>"
        f"{part(stats.get('win_ladder', []), 'Win')}</tbody></table></div>"
    )


def _robustness_table(stats: dict) -> str:
    total = stats.get("sum_pct") or 0.0
    body = [
        f"<tr><th scope='row'>All trades</th><td>—</td><td>{_fmt(total, 'pct2')}</td><td>100%</td></tr>"
    ]
    for frac, row in stats.get("sum_pct_without_best", {}).items():
        kept = row["sum_pct_after"] / total if total > 0 else None
        body.append(
            f"<tr><th scope='row'>Without best {html.escape(frac)}</th>"
            f"<td>{_fmt(row['n_removed'], 'int')}</td><td>{_fmt(row['sum_pct_after'], 'pct2')}</td>"
            f"<td>{_fmt(kept, 'share')}</td></tr>"
        )
    return (
        "<div class='scroll'><table><thead><tr><th></th><th>Trades removed</th>"
        "<th>Sum of trade %</th><th>Kept</th></tr></thead>"
        f"<tbody>{''.join(body)}</tbody></table></div>"
    )


def _days_table(rows: list[dict]) -> str:
    body = "".join(
        f"<tr><th scope='row'>{r['date']}</th><td>{_fmt(r['sum_pct'], 'pct2')}</td>"
        f"<td>{_fmt(r['n_trades'], 'int')}</td><td>{_fmt(r['n_losses'], 'int')}</td></tr>"
        for r in rows
    )
    return (
        "<div class='scroll'><table><thead><tr><th>Day (UTC)</th><th>Sum of trade %</th>"
        f"<th>Trades closed</th><th>Losing</th></tr></thead><tbody>{body}</tbody></table></div>"
    )


def _tiles(stats: dict, daily: dict) -> str:
    one = stats.get("sum_pct_without_best", {}).get("1%")
    kept = (
        one["sum_pct_after"] / stats["sum_pct"] if one and stats.get("sum_pct", 0) > 0 else None
    )
    tiles = [
        ("Expectancy / trade", _fmt(stats.get("mean_pct")), "mean net result, % of balance"),
        ("Win rate", _fmt(stats.get("win_rate"), "share"),
         f"break-even {_fmt(stats.get('breakeven_win_rate'), 'share')}"),
        ("Mean win", _fmt(stats.get("mean_win_pct")), f"median {_fmt(stats.get('median_win_pct'))}"),
        ("Mean loss", _fmt(stats.get("mean_loss_pct")), f"median {_fmt(stats.get('median_loss_pct'))}"),
        ("Payoff ratio", _fmt(stats.get("payoff_ratio"), "ratio"), "mean win / |mean loss|"),
        ("Worst trade", _fmt(stats.get("largest_loss_pct"), "pct2"),
         f"= {_fmt(stats.get('worst_loss_in_mean_wins'), 'x')} average wins"),
        ("Without best 1%", _fmt(kept, "share"), "of the summed trade result kept"),
        ("Trades / day", _fmt(daily.get("mean_trades_per_day"), "ratio"),
         f"max {_fmt(daily.get('max_trades_per_day'), 'int')}"),
    ]
    return "".join(
        f"<div class='tile'><div class='label'>{label}</div><div class='value'>{value}</div>"
        f"<div class='sub'>{html.escape(sub)}</div></div>"
        for label, value, sub in tiles
    )


CSS = """
:root { --bg:#fcfcfb; --ink:#1f1f1d; --muted:#6b6a65; --line:#e4e3df; --card:#ffffff; }
body { background:var(--bg); color:var(--ink); margin:0;
  font:15px/1.5 -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, sans-serif; }
main { max-width:1180px; margin:0 auto; padding:24px 16px 48px; }
h1 { font-size:22px; margin:0 0 4px; } h2 { font-size:17px; margin:36px 0 6px; }
.note { color:var(--muted); font-size:13px; margin:0 0 12px; max-width:860px; }
.tiles { display:grid; grid-template-columns:repeat(auto-fit,minmax(130px,1fr)); gap:10px; margin-top:16px; }
.tile { background:var(--card); border:1px solid var(--line); border-radius:8px; padding:10px 14px; }
.tile .label { color:var(--muted); font-size:12px; }
.tile .value { font-size:21px; font-variant-numeric:tabular-nums; }
.tile .sub { color:var(--muted); font-size:12px; }
ul.insights { background:var(--card); border:1px solid var(--line); border-radius:8px;
  padding:12px 16px 12px 34px; margin:16px 0 0; max-width:960px; }
ul.insights li { margin:4px 0; }
img { max-width:100%; height:auto; display:block; margin:8px 0; }
.scroll { overflow-x:auto; }
.cols { display:grid; grid-template-columns:repeat(auto-fit,minmax(340px,1fr)); gap:16px; }
table { border-collapse:collapse; font-variant-numeric:tabular-nums; font-size:14px; }
th, td { padding:5px 12px; border-bottom:1px solid var(--line); text-align:right; white-space:nowrap; }
th[scope=row] { text-align:left; font-weight:normal; color:var(--muted); }
thead th { font-weight:600; }
tr.sep td { border:none; height:8px; }
"""


def _page(title: str, body: str) -> str:
    return (
        "<!doctype html>\n<html lang='en'><head><meta charset='utf-8'>"
        "<meta name='viewport' content='width=device-width, initial-scale=1'>"
        f"<title>{html.escape(title)}</title><style>{CSS}</style></head>"
        f"<body><main><h1>{html.escape(title)}</h1>{body}</main></body></html>\n"
    )


DEFINITION_NOTE = (
    "<p class='note'>A <b>trade</b> is one round trip of a (coin, side) position from flat to "
    "flat. Its result is realized PnL plus fees of all its fills, as a percentage of the wallet "
    "balance just before its first fill, so trades early and late in the backtest are "
    "comparable. Positions still open at the end are excluded. “Sum of trade %” adds these "
    "normalized results (a size-neutral measure of contribution).</p>"
)


def render_html(summary: dict, figs: dict[str, str], *, title: str) -> str:
    traded = [side for side in ("long", "short") if side in summary]
    sides = [("All", "all")]
    daily_sides = [("All", "daily")]
    if len(traded) == 2:
        sides += [(side.capitalize(), side) for side in traded]
        daily_sides += [(side.capitalize(), f"daily_{side}") for side in traded]
    pct_rows = tuple((key, key, "pct") for key in summary["all"].get("percentiles_pct", {}))
    pct_summary = {section: summary[section].get("percentiles_pct", {}) for _, section in sides}
    stats = summary["all"]

    def img(key, alt):
        return f"<img src='data:image/png;base64,{figs[key]}' alt='{alt}'>" if key in figs else ""

    coins = summary.get("by_coin", [])
    insights_html = "".join(f"<li>{html.escape(s)}</li>" for s in summary.get("insights", []))
    body = f"""
{DEFINITION_NOTE}
<div class="tiles">{_tiles(stats, summary["daily"])}</div>
<ul class="insights">{insights_html}</ul>

<h2>Shape of the distribution</h2>
<p class="note">The histogram shows the typical trade; the sorted curve below uses a symmetric
log axis so the rare large losses and wins at both ends stay visible.</p>
{img("histogram", "Histogram of trade results")}
{img("sorted", "All trades sorted by result")}

<h2>Tail severity</h2>
<p class="note">How many trades lose (or win) more than a given share of the balance, and how much
of all losses (or gains) they make up. A healthy distribution has few trades in the bottom rows
and most of the gross loss spread over many small losses.</p>
<div class="cols"><div>{_ladder_table(stats)}</div>
<div><p class="note" style="margin-top:6px">Robustness: how much of the summed result remains
after dropping the best trades. Values near 100% mean profit comes from the bulk of trades,
not from a few lucky outliers.</p>{_robustness_table(stats)}</div></div>

<h2>Statistics</h2>
{_stat_table(summary, STAT_ROWS, sides)}
<h2>Percentiles (% of balance)</h2>
{_stat_table(pct_summary, pct_rows, sides)}

<h2>Depth of averaging</h2>
<p class="note">Trades grouped by the number of entry fills. Deep trades are those the strategy had
to average down repeatedly; their losses are usually where the tail comes from.</p>
{img("entries", "Result by number of entry fills")}
{_rows_table(summary.get("by_entries", []), "entries", "Entry fills")}

<h2>Duration</h2>
{img("duration", "Trade duration against result")}

<h2>Stability over time</h2>
{_rows_table(summary.get("by_year", []), "year", "Year (close)")}

<h2>Activity</h2>
{img("daily", "Closed trades per day")}
{_stat_table(summary, DAILY_ROWS, daily_sides)}

<h2>Correlated days</h2>
<p class="note">Days on which closed trades summed to the largest losses and gains. Several losing
coins on the same day reveal exposure to market-wide moves.</p>
<div class="cols"><div><h3>Worst days</h3>{_days_table(summary["extreme_days"]["worst"])}</div>
<div><h3>Best days</h3>{_days_table(summary["extreme_days"]["best"])}</div></div>

<h2>Coins</h2>
<div class="cols"><div><h3>Weakest 10 by sum of trade %</h3>{_rows_table(coins[:10], "coin", "Coin")}</div>
<div><h3>Strongest 10</h3>{_rows_table(coins[::-1][:10], "coin", "Coin")}</div></div>

<h2>By exit type</h2>
<p class="note">Exit type is the order type of the fill that flattened the position.</p>
{_rows_table(summary.get("exit_types", []), "exit_type", "Exit")}
"""
    return _page(title, body)


# ---------------------------------------------------------------------------- comparison


COMPARE_ROWS = (
    # (label, key, format, +1 higher is better / -1 lower is better / 0 neutral)
    ("Closed trades", "n_trades", "int", 0),
    ("Trades / day", "mean_trades_per_day", "ratio", 0),
    ("Win rate", "win_rate", "share", 1),
    ("Break-even win rate", "breakeven_win_rate", "share", -1),
    ("Win-rate margin (points)", "win_rate_margin", "ratio", 1),
    ("Expectancy per trade", "mean_pct", "pct", 1),
    ("Median trade", "median_pct", "pct", 1),
    ("Mean win", "mean_win_pct", "pct", 1),
    ("Mean loss", "mean_loss_pct", "pct", 1),
    ("Median loss", "median_loss_pct", "pct", 1),
    ("Payoff ratio", "payoff_ratio", "ratio", 1),
    ("Profit factor", "profit_factor", "ratio", 1),
    ("p1 trade", "p1", "pct", 1),
    ("p5 trade", "p5", "pct", 1),
    ("Largest loss", "largest_loss_pct", "pct2", 1),
    ("Worst loss in average wins", "worst_loss_in_mean_wins", "x", -1),
    ("Worst 10% of losses: share of total loss", "loss_share_worst_10pct_of_losses", "share", -1),
    ("Losses ≥ 1% of balance", "losses_ge_1pct", "int", -1),
    ("Kept without best 1% of trades", "kept_without_best_1pct", "share", 1),
    ("Skewness (quantile)", "quantile_skew_p10_p90", "ratio", 0),
    ("Mean entry fills per trade", "mean_entries_per_trade", "ratio", 0),
    ("Longest losing streak", "max_consecutive_losses", "int", -1),
    ("Median duration, losses", "median_duration_hours_losses", "hours", -1),
    ("Worst day (sum of trade %)", "worst_day_pct", "pct2", 1),
    ("Sum of trade %", "sum_pct", "pct2", 1),
)


def _compare_metrics(summary: dict) -> dict:
    s = dict(summary["all"])
    s.update(summary.get("daily", {}))
    s.update(s.get("percentiles_pct", {}))
    one = s.get("sum_pct_without_best", {}).get("1%")
    s["kept_without_best_1pct"] = (
        one["sum_pct_after"] / s["sum_pct"] if one and s.get("sum_pct", 0) > 0 else None
    )
    s["losses_ge_1pct"] = next(
        (r["n_trades"] for r in s.get("loss_ladder", []) if r["threshold_pct"] == 1.0), None
    )
    if s.get("breakeven_win_rate") is not None:
        s["win_rate_margin"] = (s["win_rate"] - s["breakeven_win_rate"]) * 100.0
    worst = summary.get("extreme_days", {}).get("worst") or []
    s["worst_day_pct"] = worst[0]["sum_pct"] if worst else None
    return s


@_scoped_style
def render_comparison(runs: list[tuple[str, dict, pd.DataFrame]], *, title: str) -> str:
    """Side-by-side report of several runs: (label, summary, trades) each."""
    if len(runs) > len(SERIES_COLORS):
        raise ValueError(f"compare at most {len(SERIES_COLORS)} runs at once")
    labels = [label for label, _, _ in runs]
    metrics = {label: _compare_metrics(summary) for label, summary, _ in runs}
    head = "".join(
        f"<th><span class='sw' style='background:{SERIES_COLORS[i]}'></span>{html.escape(l)}</th>"
        for i, l in enumerate(labels)
    )
    def cells(key, kind, direction):
        values = [metrics[l].get(key) for l in labels]
        valid = [v for v in values if v is not None and math.isfinite(v)]
        best = None
        if direction and len(valid) > 1 and max(valid) != min(valid):
            best = max(valid) if direction > 0 else min(valid)
        return "".join(
            f"<td class='best'>{_fmt(v, kind)}</td>" if best is not None and v == best
            else f"<td>{_fmt(v, kind)}</td>"
            for v in values
        )

    rows = "".join(
        f"<tr><th scope='row'>{html.escape(name)}</th>{cells(key, kind, direction)}</tr>"
        for name, key, kind, direction in COMPARE_ROWS
    )
    table = (
        f"<div class='scroll'><table><thead><tr><th></th>{head}</tr></thead>"
        f"<tbody>{rows}</tbody></table></div>"
    )

    all_pct = np.concatenate([_closed(t)["net_pnl_pct"].to_numpy() for _, _, t in runs])
    fig, (ax1, ax2) = _subplots(1, 2, figsize=(12, 4.2))
    for i, (label, _, trades) in enumerate(runs):
        pct = _closed(trades)["net_pnl_pct"].to_numpy()
        tail = np.sort(pct)[: max(1, int(math.ceil(len(pct) * 0.05)))]
        ax1.plot((np.arange(len(tail)) + 0.5) / len(pct) * 100.0, tail,
                 color=SERIES_COLORS[i], linewidth=2, label=label)
        losses = np.sort(-pct[pct < 0.0])[::-1]
        if len(losses):
            share = np.cumsum(losses) / losses.sum() * 100.0
            ax2.plot(np.arange(1, len(losses) + 1) / len(losses) * 100.0, share,
                     color=SERIES_COLORS[i], linewidth=2, label=label)
    _symlog_pct_axis(ax1, all_pct[all_pct < 0.0] if (all_pct < 0.0).any() else all_pct)
    ax1.set_ylim(top=0.0)
    ax1.set_xscale("log")
    ax1.set_xlim(right=5.0)
    ax1.set_xlabel("worst x% of all trades (log)")
    ax1.set_ylabel("% of balance (symmetric log)")
    ax1.set_title("Loss tail: the worst 5% of trades, sorted", loc="left")
    ax1.legend(frameon=False, loc="upper left")
    _style(ax1, "both")
    ax2.plot([0, 100], [0, 100], color=NEUTRAL_COLOR, linewidth=0.8, linestyle=":")
    ax2.set_xscale("log")
    ax2.set_xlim(left=0.01)
    ax2.set_xlabel("worst x% of losing trades (log)")
    ax2.set_ylabel("% of total loss")
    ax2.set_title("Loss concentration: higher = losses come from fewer trades", loc="left")
    _style(ax2, "both")
    fig.tight_layout()
    curves = _png(fig)

    years = sorted({r["year"] for _, s, _ in runs for r in s.get("by_year", [])})
    by_year = {
        label: {r["year"]: r for r in summary.get("by_year", [])} for label, summary, _ in runs
    }
    year_rows = "".join(
        f"<tr><th scope='row'>{y}</th>"
        + "".join(
            f"<td>{_fmt(by_year[l].get(y, {}).get('mean_pct'))}</td>"
            f"<td>{_fmt(by_year[l].get(y, {}).get('worst_pct'), 'pct2')}</td>"
            for l in labels
        )
        + "</tr>"
        for y in years
    )
    year_head = "".join(
        f"<th colspan='2'>{html.escape(l)}</th>" for l in labels
    )
    year_sub = "<th>expectancy</th><th>worst</th>" * len(labels)
    year_table = (
        f"<div class='scroll'><table><thead><tr><th></th>{year_head}</tr>"
        f"<tr><th>Year</th>{year_sub}</tr></thead><tbody>{year_rows}</tbody></table></div>"
    )

    body = f"""
{DEFINITION_NOTE}
<style>.sw {{ display:inline-block; width:10px; height:10px; border-radius:2px; margin-right:6px; }}
td.best {{ font-weight:700; background:#eaf2fc; }}</style>
<h2>Key metrics</h2>
{table}
<h2>Distributions</h2>
<p class="note">Left: the worst 5% of trades of each run, sorted; a curve that stays higher has a
milder loss tail, and its left end is the single worst trade. Right: cumulative share of the total
loss produced by the worst losing trades; a curve near the top-left means a few trades cause most
of the damage. In the table, the best value of each row is highlighted (neutral rows are not).</p>
<img src='data:image/png;base64,{curves}' alt='Sorted trade results and loss concentration by run'>
<h2>By year</h2>
{year_table}
"""
    return _page(title, body)


# ---------------------------------------------------------------------------- entry points


def write_trade_distribution_report(
    fdf: pd.DataFrame,
    results_path: str,
    *,
    title: str,
    start=None,
    end=None,
) -> str | None:
    """Write trades.csv, trade_distribution.json and trade_distribution.html.

    Returns the HTML path, or None when there is no closed trade. This is an optional
    artifact: failures are logged and never fail the backtest.
    """
    try:
        trades = reconstruct_trades(fdf)
        trades.to_csv(os.path.join(results_path, TRADES_FILENAME), index=False)
        summary, counts = build_summary(trades, start, end)
        with open(
            os.path.join(results_path, TRADE_DISTRIBUTION_SUMMARY_FILENAME), "w", encoding="utf-8"
        ) as output:
            json.dump(summary, output, indent=2, allow_nan=False)
        if summary["all"]["n_trades"] == 0:
            logging.info("skipping %s: no closed trades", TRADE_DISTRIBUTION_REPORT_FILENAME)
            return None
        figs = _figures(trades, counts, summary)
        path = os.path.join(results_path, TRADE_DISTRIBUTION_REPORT_FILENAME)
        with open(path, "w", encoding="utf-8") as output:
            output.write(render_html(summary, figs, title=title))
    except Exception as exc:
        logging.warning("could not write %s: %s", TRADE_DISTRIBUTION_REPORT_FILENAME, exc)
        return None
    s, d = summary["all"], summary["daily"]
    logging.info(
        "trade distribution: %d trades, win rate %.1f%%, mean win %+.4f%%, mean loss %+.4f%%, "
        "expectancy %+.4f%%, %.1f trades/day -> %s",
        s["n_trades"], s["win_rate"] * 100, s["mean_win_pct"], s["mean_loss_pct"],
        s["mean_pct"], d.get("mean_trades_per_day", 0.0), path,
    )
    return path


def _load_run(result_dir: str) -> tuple[pd.DataFrame, object, object]:
    fdf = pd.read_csv(os.path.join(result_dir, "fills.csv"), index_col=0)
    start = end = None
    bal_eq_path = os.path.join(result_dir, "balance_and_equity.csv.gz")
    if os.path.exists(bal_eq_path):
        index = pd.to_datetime(pd.read_csv(bal_eq_path, usecols=[0]).iloc[:, 0])
        start, end = index.min(), index.max()
    return fdf, start, end


def _run_label(result_dir: str) -> str:
    return os.path.basename(os.path.normpath(result_dir))


def main():
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("result_dirs", nargs="+", help="backtest result directories (with fills.csv)")
    parser.add_argument("--compare", metavar="HTML", help="write a side-by-side comparison here")
    parser.add_argument("--labels", help="comma-separated labels for --compare (default: dir names)")
    args = parser.parse_args()
    logging.basicConfig(level=logging.INFO, format="%(message)s")

    if not args.compare:
        for result_dir in args.result_dirs:
            fdf, start, end = _load_run(result_dir)
            path = write_trade_distribution_report(
                fdf, result_dir, title=f"Trade distribution: {_run_label(result_dir)}",
                start=start, end=end,
            )
            if path is None:
                raise SystemExit(f"no report written for {result_dir}")
            print(path)
        return

    labels = args.labels.split(",") if args.labels else [_run_label(d) for d in args.result_dirs]
    if len(labels) != len(args.result_dirs):
        raise SystemExit("--labels must have one label per result directory")
    runs = []
    for label, result_dir in zip(labels, args.result_dirs):
        fdf, start, end = _load_run(result_dir)
        trades = reconstruct_trades(fdf)
        summary, _ = build_summary(trades, start, end)
        runs.append((label, summary, trades))
    with open(args.compare, "w", encoding="utf-8") as output:
        output.write(render_comparison(runs, title="Trade distribution comparison"))
    print(args.compare)


if __name__ == "__main__":
    main()
