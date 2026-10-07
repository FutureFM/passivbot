"""End-to-end research run: model -> combined book + one book per leg -> stats, diagnostics, charts."""

from __future__ import annotations

import copy
import json
import logging
from dataclasses import dataclass
from pathlib import Path

import pandas as pd

from dtrend.config import DTrendConfig, to_dict
from dtrend.data.panel import Panel
from dtrend.model.system import ModelOutput, bars_per_year, run_model
from dtrend.research import analysis as an
from dtrend.research.backtest import SimResult, buy_and_hold, simulate

log = logging.getLogger(__name__)


@dataclass
class ResearchResult:
    model: ModelOutput
    books: dict[str, SimResult]  # "D-TREND" and one per leg
    benchmark: pd.Series
    summary: pd.DataFrame
    yearly: pd.DataFrame
    deciles: pd.DataFrame
    ic: pd.DataFrame
    speed: pd.DataFrame
    start: pd.Timestamp
    benchmark_name: str = "BTCUSDT"


def first_trading_date(model: ModelOutput) -> pd.Timestamp:
    active = model.target.abs().sum(axis=1)
    nz = active[active > 0]
    return nz.index[0] if len(nz) else model.target.index[0]


def run_research(panel: Panel, cfg: DTrendConfig) -> ResearchResult:
    bpy = bars_per_year(cfg)
    if cfg.analysis.apply_speed_limit:
        cfg = apply_speed_limit(panel, cfg)
    model = run_model(panel, cfg)
    ex = cfg.execution
    books = {
        "D-TREND": simulate(model.target, model.buffer(ex.buffer_fraction), panel.close, panel.funding, ex.cost_per_trade)
    }
    for name, leg in model.legs.items():
        books[name] = simulate(
            leg.position, leg.average_position * ex.buffer_fraction, panel.close, panel.funding, ex.cost_per_trade
        )
    start = first_trading_date(model)
    bench_sym = cfg.data.benchmark if cfg.data.benchmark in panel.close.columns else panel.close.columns[0]
    benchmark = buy_and_hold(panel.close[bench_sym].loc[start:])

    summary = pd.DataFrame({k: an.summarize(v, bpy, start) for k, v in books.items()}).T
    summary.loc[f"{bench_sym} buy-and-hold", "sharpe_net"] = an.sharpe(benchmark, bpy)
    summary.loc[f"{bench_sym} buy-and-hold", "max_drawdown_pct"] = 100 * an.max_drawdown(benchmark)

    yearly_in = {}
    for k, v in books.items():
        yearly_in[f"{k} net"] = v.returns_net.loc[start:]
        yearly_in[f"{k} gross"] = v.returns_gross.loc[start:]
    yearly_in[f"{bench_sym} buy-and-hold"] = benchmark
    yearly = an.yearly_sharpe(yearly_in, bpy)

    deciles = an.forecast_deciles(
        model.combined_forecast, panel.close, panel.funding, model.sigma_annual, cfg.analysis.decile_horizons, bpy
    )
    ic_rows = {}
    speed_rows = {}
    for leg_name, leg in model.legs.items():
        ic_rows[f"{leg_name}"] = an.information_coefficient(leg.forecast, panel.close, panel.funding, model.sigma_annual, bpy)
        for rule_name, rule_fc in leg.rule_forecasts.items():
            key = f"{leg_name}.{rule_name}"
            ic_rows[key] = an.information_coefficient(rule_fc, panel.close, panel.funding, model.sigma_annual, bpy)
            speed_rows[key] = {
                "cost_sr_yr": an.rule_cost_sr(
                    rule_fc, model.sigma_annual, ex.cost_per_trade, cfg.forecast.target_abs_forecast, bpy
                )
            }
    ic_rows["combined"] = an.information_coefficient(
        model.combined_forecast, panel.close, panel.funding, model.sigma_annual, bpy
    )
    speed = pd.DataFrame(speed_rows).T
    speed["over_limit"] = speed["cost_sr_yr"] > cfg.analysis.speed_limit_sr
    return ResearchResult(model, books, benchmark, summary, yearly, deciles, pd.DataFrame(ic_rows).T, speed, start, bench_sym)


def apply_speed_limit(panel: Panel, cfg: DTrendConfig) -> DTrendConfig:
    """Drop every rule whose annual cost exceeds the speed limit (in-sample research filter)."""
    model = run_model(panel, cfg)
    bpy = bars_per_year(cfg)
    out = copy.deepcopy(cfg)
    for leg in out.legs:
        if leg.name not in model.legs:
            continue
        keep = []
        for rule in leg.rules:
            cost = an.rule_cost_sr(
                model.legs[leg.name].rule_forecasts[rule.name],
                model.sigma_annual,
                cfg.execution.cost_per_trade,
                cfg.forecast.target_abs_forecast,
                bpy,
            )
            if cost > cfg.analysis.speed_limit_sr:
                log.info("speed limit: drop %s.%s (cost %.3f SR/yr)", leg.name, rule.name, cost)
            else:
                keep.append(rule)
        leg.rules = keep
    return out


def write_outputs(result: ResearchResult, cfg: DTrendConfig, out_dir: str | Path, plots: bool = True) -> Path:
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    result.summary.to_csv(out / "summary.csv")
    result.yearly.to_csv(out / "sharpe_by_year.csv")
    result.deciles.to_csv(out / "forecast_deciles.csv")
    result.ic.to_csv(out / "information_coefficient.csv")
    result.speed.to_csv(out / "speed_limit.csv")
    result.model.target.to_parquet(out / "target_weights.parquet")
    result.model.combined_forecast.to_parquet(out / "combined_forecast.parquet")
    eq = pd.DataFrame({k: v.returns_net for k, v in result.books.items()}).loc[result.start :]
    eq["benchmark"] = result.benchmark
    (cfg.execution.starting_equity * (1 + eq.fillna(0.0)).cumprod()).to_csv(out / "equity.csv")
    latest = latest_forecasts(result.model)
    latest.to_csv(out / "latest_forecasts.csv")
    (out / "config.json").write_text(json.dumps(to_dict(cfg), indent=2, default=str))
    if plots:
        from dtrend.research import report

        report.plot_equity(result, cfg, out / "equity.png")
        report.plot_deciles(result.deciles, out / "forecast_deciles.png")
        report.plot_latest_forecasts(latest, out / "latest_forecasts.png")
    (out / "report.md").write_text(markdown_report(result, cfg))
    return out


def latest_forecasts(model: ModelOutput) -> pd.DataFrame:
    """Per-coin forecasts at the last close: combined and per leg (the thread's bar chart)."""
    ts = model.combined_forecast.index[-1]
    data = {"combined": model.combined_forecast.loc[ts]}
    for name, leg in model.legs.items():
        data[name] = leg.forecast.loc[ts]
    data["target_weight"] = model.target.loc[ts]
    df = pd.DataFrame(data).dropna(subset=["combined"])
    return df.sort_values("combined", ascending=False)


def markdown_report(result: ResearchResult, cfg: DTrendConfig) -> str:
    def fmt(df: pd.DataFrame) -> str:
        return df.round(3).to_markdown() if hasattr(df, "to_markdown") else df.round(3).to_string()

    try:
        import tabulate  # noqa: F401
    except ImportError:
        fmt = lambda df: "```\n" + df.round(3).to_string() + "\n```"  # noqa: E731
    lines = [
        "# D-TREND research report",
        "",
        f"Period: {result.start.date()} -> {result.model.target.index[-1].date()}. "
        f"Vol target tau = {cfg.risk.target_vol:.0%}. Cost {cfg.execution.cost_per_trade * 1e4:.1f} bp per trade. "
        f"Every return includes funding.",
        "",
        "## Books",
        fmt(result.summary),
        "",
        "## Sharpe by year",
        fmt(result.yearly),
        "",
        "## Forecast deciles (mean forward vol-adjusted total return, annualized)",
        fmt(result.deciles),
        "",
        "## Information coefficient",
        fmt(result.ic),
        "",
        f"## Speed limit (rule cost in SR/yr, limit {cfg.analysis.speed_limit_sr})",
        fmt(result.speed),
        "",
    ]
    return "\n".join(lines)
