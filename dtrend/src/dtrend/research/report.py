"""Charts in the style of the thread: equity + drawdown per book, forecast deciles, latest forecasts."""

from __future__ import annotations

from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import pandas as pd  # noqa: E402

COLORS = {
    "D-TREND": "#2a6fdb",
    "benchmark": "#e8590c",
    "ts_momentum": "#2b9348",
    "xs_momentum": "#e0a100",
    "xs_carry": "#d6336c",
}


def plot_equity(result, cfg, path: Path) -> None:
    start = result.start
    eq0 = cfg.execution.starting_equity
    fig, (ax, axd) = plt.subplots(2, 1, figsize=(12, 7), sharex=True, gridspec_kw={"height_ratios": [3, 1]})
    for name, book in result.books.items():
        eq = eq0 * (1 + book.returns_net.loc[start:]).cumprod()
        ax.plot(eq.index, eq, label=f"{name} ${eq.iloc[-1]:,.0f}", color=COLORS.get(name), lw=1.6 if name == "D-TREND" else 1)
    bench = eq0 * (1 + result.benchmark).cumprod()
    ax.plot(bench.index, bench, label=f"{result.benchmark_name} buy-and-hold ${bench.iloc[-1]:,.0f}", color=COLORS["benchmark"], lw=1)
    ax.set_yscale("log")
    ax.set_title(f"Equity · compounded · ${eq0:,.0f} start · vol target τ {cfg.risk.target_vol:.0%}")
    ax.legend(loc="upper left", fontsize=8)
    ax.grid(alpha=0.3)
    main = result.books["D-TREND"].returns_net.loc[start:]
    eq = (1 + main).cumprod()
    dd = eq / eq.cummax() - 1
    axd.fill_between(dd.index, dd * 100, 0, color=COLORS["D-TREND"], alpha=0.5)
    axd.set_ylabel("drawdown %")
    axd.grid(alpha=0.3)
    fig.tight_layout()
    fig.savefig(path, dpi=120)
    plt.close(fig)


def plot_deciles(deciles: pd.DataFrame, path: Path) -> None:
    n = len(deciles)
    if n == 0:
        return
    cols = min(4, n)
    rows = (n + cols - 1) // cols
    fig, axes = plt.subplots(rows, cols, figsize=(3.2 * cols, 2.6 * rows), squeeze=False)
    for ax, (h, row) in zip(axes.flat, deciles.iterrows()):
        ax.scatter(row.index, row.values, c=row.index, cmap="Blues", vmin=-3, vmax=9, edgecolor="k", lw=0.3)
        ax.axhline(0, ls="--", color="grey", lw=0.8)
        ax.set_title(str(h), fontsize=9)
    for ax in list(axes.flat)[n:]:
        ax.axis("off")
    fig.suptitle("Forecast deciles: x = decile (0 = most negative), y = fwd vol-adjusted total return", fontsize=9)
    fig.tight_layout()
    fig.savefig(path, dpi=120)
    plt.close(fig)


def plot_latest_forecasts(latest: pd.DataFrame, path: Path) -> None:
    if latest.empty:
        return
    legs = [c for c in latest.columns if c not in ("combined", "target_weight")]
    df = latest.iloc[::-1]
    fig, ax = plt.subplots(figsize=(8, max(4, 0.28 * len(df))))
    y = range(len(df))
    colors = ["#2a6fdb" if v >= 0 else "#d9480f" for v in df["combined"]]
    ax.barh(y, df["combined"], color=colors, height=0.6, label="combined")
    for i, leg in enumerate(legs):
        ax.barh([v + 0.3 + 0.1 * i for v in y], df[leg], height=0.08, color=COLORS.get(leg), label=leg)
    ax.set_yticks(list(y))
    ax.set_yticklabels([s.replace("USDT", "") for s in df.index], fontsize=7)
    ax.axvline(0, color="grey", lw=0.8)
    ax.set_xlim(-20, 20)
    ax.set_xlabel("forecast (scale ±20, blue = long, red = short)")
    ax.set_title(f"D-TREND forecasts · {len(df)} coins")
    ax.legend(fontsize=7, loc="lower right")
    fig.tight_layout()
    fig.savefig(path, dpi=120)
    plt.close(fig)
