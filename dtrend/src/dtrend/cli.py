"""Command line: dtrend {download,research,backtest,live,demo}."""

from __future__ import annotations

import argparse
import logging
import sys

from dtrend.config import load_config


def _cmd_download(args) -> int:
    from dtrend.data.binance_vision import BinanceDownloader

    cfg = load_config(args.config)
    dl = BinanceDownloader(cfg.data.data_dir, cfg.data.interval)
    symbols = cfg.data.symbols or dl.list_symbols()
    symbols = [s for s in symbols if s not in set(cfg.data.exclude)]
    if args.limit:
        symbols = symbols[: args.limit]
    logging.info("downloading %d symbols into %s", len(symbols), cfg.data.data_dir)
    dl.update(symbols, args.start or cfg.data.start, cfg.data.end)
    dl.save_exchange_info()
    return 0


def _cmd_research(args) -> int:
    from dtrend.data.panel import load_panel
    from dtrend.research.runner import run_research, write_outputs

    cfg = load_config(args.config)
    if args.speed_limit:
        cfg.analysis.apply_speed_limit = True
    panel = load_panel(cfg.data)
    result = run_research(panel, cfg)
    out = write_outputs(result, cfg, args.out, plots=not args.no_plots)
    print(result.summary.round(3).to_string())
    print(f"\nreport: {out / 'report.md'}")
    return 0


def _cmd_backtest(args) -> int:
    from pathlib import Path

    import pandas as pd

    from dtrend.data.panel import load_panel
    from dtrend.nautilus.backtest import run_nautilus_backtest

    cfg = load_config(args.config)
    panel = load_panel(cfg.data)
    raw_funding = {}
    fdir = Path(cfg.data.data_dir) / "funding"
    for sym in panel.close.columns:
        path = fdir / f"{sym}.parquet"
        if path.exists():
            s = pd.read_parquet(path)["funding_rate"]
            raw_funding[sym] = s.loc[panel.index[0] : panel.index[-1]]
    info = Path(cfg.data.data_dir) / "exchange_info.json"
    res = run_nautilus_backtest(
        panel,
        cfg,
        mode=args.mode,
        raw_funding=raw_funding,
        exchange_info=info if info.exists() else None,
        work_dir=args.out,
        log_level=args.log_level,
    )
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    res.equity.to_csv(out / "nautilus_equity.csv")
    res.orders.to_csv(out / "nautilus_fills.csv")
    res.positions.to_csv(out / "nautilus_positions.csv")
    print(f"rebalances {res.rebalances}, fills {len(res.orders)}, funding paid {res.funding_paid:,.2f} USDT")
    print(f"equity {res.equity.iloc[0]:,.2f} -> {res.equity.iloc[-1]:,.2f} USDT")
    return 0


def _cmd_live(args) -> int:
    from dtrend.nautilus.live import run_live

    cfg = load_config(args.config)
    dry_run = not args.send_orders
    if not dry_run and args.environment.upper() == "LIVE" and not args.i_understand_real_money:
        print("refusing to send real orders without --i-understand-real-money", file=sys.stderr)
        return 2
    run_live(cfg, environment=args.environment, dry_run=dry_run)
    return 0


def _cmd_demo(args) -> int:
    from dtrend.config import config_from_dict
    from dtrend.data.panel import synthetic_panel, with_universe
    from dtrend.research.runner import run_research, write_outputs

    cfg = config_from_dict({})
    panel = with_universe(synthetic_panel(n_symbols=30, n_days=1200), cfg.data)
    result = run_research(panel, cfg)
    out = write_outputs(result, cfg, args.out)
    print(result.summary.round(3).to_string())
    print(f"\nSYNTHETIC data, not a result. report: {out / 'report.md'}")
    return 0


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(prog="dtrend", description="D-TREND multi-leg trend/carry portfolio on Nautilus Trader")
    p.add_argument("-v", "--verbose", action="store_true")
    sub = p.add_subparsers(dest="cmd", required=True)

    d = sub.add_parser("download", help="download Binance USD-M klines + funding (public data, no keys)")
    d.add_argument("config")
    d.add_argument("--start")
    d.add_argument("--limit", type=int, help="only the first N symbols (testing)")
    d.set_defaults(func=_cmd_download)

    r = sub.add_parser("research", help="fast research backtest, diagnostics and charts")
    r.add_argument("config")
    r.add_argument("--out", default="results/research")
    r.add_argument("--speed-limit", action="store_true", help="drop rules over the Carver speed limit")
    r.add_argument("--no-plots", action="store_true")
    r.set_defaults(func=_cmd_research)

    b = sub.add_parser("backtest", help="event-driven Nautilus backtest")
    b.add_argument("config")
    b.add_argument("--out", default="results/nautilus")
    b.add_argument("--mode", choices=["precomputed", "live_equivalent"], default="precomputed")
    b.add_argument("--log-level", default="ERROR")
    b.set_defaults(func=_cmd_backtest)

    lv = sub.add_parser("live", help="Nautilus live node on Binance USDT-M (dry run by default)")
    lv.add_argument("config")
    lv.add_argument("--environment", default="TESTNET", choices=["TESTNET", "DEMO", "LIVE", "testnet", "demo", "live"])
    lv.add_argument("--send-orders", action="store_true", help="send orders (default: log only)")
    lv.add_argument("--i-understand-real-money", action="store_true")
    lv.set_defaults(func=_cmd_live)

    dm = sub.add_parser("demo", help="run the research pipeline on synthetic data (offline)")
    dm.add_argument("--out", default="results/demo")
    dm.set_defaults(func=_cmd_demo)

    args = p.parse_args(argv)
    logging.basicConfig(level=logging.DEBUG if args.verbose else logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    return args.func(args)


if __name__ == "__main__":
    raise SystemExit(main())
