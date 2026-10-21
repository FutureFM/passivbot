import json

import numpy as np
import pandas as pd
import pytest

from trade_distribution_report import (
    TRADE_DISTRIBUTION_REPORT_FILENAME,
    TRADE_DISTRIBUTION_SUMMARY_FILENAME,
    TRADES_FILENAME,
    build_summary,
    daily_trade_counts,
    distribution_stats,
    reconstruct_trades,
    render_comparison,
    write_trade_distribution_report,
)


def fill(ts, coin, typ, pnl, fee, balance, psize, we=0.1):
    return {
        "timestamp": pd.Timestamp(ts),
        "coin": coin,
        "type": typ,
        "pnl": pnl,
        "fee_paid": fee,
        "usd_total_balance": balance,
        "psize": psize,
        "wallet_exposure": we,
    }


ETH_LOSS_PCT = -20.0 / 999.0 * 100.0


def sample_fills():
    rows = [
        # BTC long: 2 entries, partial + full close -> +10 net
        fill("2025-01-01 00:00", "BTC", "entry_ema_anchor_long", 0.0, -1.0, 999.0, 1.0),
        fill("2025-01-01 01:00", "ETH", "entry_ema_anchor_short", 0.0, -1.0, 998.0, -2.0),
        fill("2025-01-01 02:00", "BTC", "entry_ema_anchor_long", 0.0, -1.0, 997.0, 2.0, 0.2),
        fill("2025-01-01 03:00", "BTC", "close_ema_anchor_long", 5.0, -1.0, 1001.0, 1.0),
        fill("2025-01-01 04:00", "BTC", "close_ema_anchor_long", 9.0, -1.0, 1009.0, 0.0),
        # ETH short closed at a loss -> -20 net
        fill("2025-01-03 00:00", "ETH", "close_time_stop_short", -18.0, -1.0, 990.0, 0.0),
        # BTC long again, still open at the end
        fill("2025-01-03 01:00", "BTC", "entry_ema_anchor_long", 0.0, -1.0, 989.0, 1.0),
    ]
    return pd.DataFrame(rows)


def test_reconstruct_trades_splits_round_trips_per_coin_and_side():
    trades = reconstruct_trades(sample_fills())
    assert len(trades) == 3
    btc, eth, open_trade = trades.iloc[0], trades.iloc[1], trades.iloc[2]

    assert (btc.coin, btc.pside, btc.n_entries, btc.n_closes) == ("BTC", "long", 2, 2)
    assert btc.net_pnl == pytest.approx(10.0)
    # Balance before the first BTC fill: 999 - (0 + -1) = 1000.
    assert btc.balance_at_open == pytest.approx(1000.0)
    assert btc.net_pnl_pct == pytest.approx(1.0)
    assert btc.duration_hours == pytest.approx(4.0)
    assert btc.max_wallet_exposure == pytest.approx(0.2)
    assert btc.closed

    assert (eth.coin, eth.pside, eth.exit_type) == ("ETH", "short", "close_time_stop_short")
    assert eth.net_pnl == pytest.approx(-20.0)
    assert eth.balance_at_open == pytest.approx(999.0)
    assert eth.net_pnl_pct == pytest.approx(ETH_LOSS_PCT)

    assert not open_trade.closed
    assert open_trade.exit_type == "open"


def test_trade_net_pnl_reconciles_with_fills():
    fills = sample_fills()
    trades = reconstruct_trades(fills)
    assert trades["net_pnl"].sum() == pytest.approx((fills["pnl"] + fills["fee_paid"]).sum())


def test_distribution_stats_excludes_open_trades_and_derives_break_even():
    stats = distribution_stats(reconstruct_trades(sample_fills()))
    assert stats["n_trades"] == 2
    assert stats["n_open_at_end"] == 1
    assert stats["win_rate"] == pytest.approx(0.5)
    assert stats["mean_win_pct"] == pytest.approx(1.0)
    loss = -ETH_LOSS_PCT
    assert stats["mean_loss_pct"] == pytest.approx(-loss)
    assert stats["payoff_ratio"] == pytest.approx(1.0 / loss)
    # |L| / (W + |L|): the win rate at which mean win and mean loss cancel out.
    assert stats["breakeven_win_rate"] == pytest.approx(loss / (1.0 + loss))
    assert stats["worst_loss_in_mean_wins"] == pytest.approx(loss)
    loss_ge_1 = next(r for r in stats["loss_ladder"] if r["threshold_pct"] == 1.0)
    assert loss_ge_1["n_trades"] == 1
    assert loss_ge_1["share_of_gross"] == pytest.approx(1.0)


def test_daily_counts_include_days_without_closed_trades():
    counts = daily_trade_counts(reconstruct_trades(sample_fills()))
    assert list(counts.to_numpy()) == [1, 0, 1]


def test_empty_fills_produce_no_trades():
    empty = sample_fills().iloc[0:0]
    trades = reconstruct_trades(empty)
    assert trades.empty
    assert distribution_stats(trades) == {"n_trades": 0, "n_open_at_end": 0}


def random_fills(seed, n_trades=300):
    rng = np.random.default_rng(seed)
    rows, balance = [], 10_000.0
    t = pd.Timestamp("2024-01-01")
    for i in range(n_trades):
        coin = f"C{i % 7}"
        n_entries = int(rng.integers(1, 12))
        size = 0.0
        for _ in range(n_entries):
            size += 1.0
            balance -= 0.5
            rows.append(fill(t, coin, "entry_ema_anchor_long", 0.0, -0.5, balance, size))
            t += pd.Timedelta(minutes=37)
        pnl = float(rng.normal(3.0, 2.0)) if rng.random() < 0.8 else float(rng.normal(-8.0, 6.0))
        balance += pnl - 0.5
        rows.append(fill(t, coin, "close_ema_anchor_long", pnl, -0.5, balance, 0.0))
        t += pd.Timedelta(hours=3)
    return pd.DataFrame(rows)


def test_report_files_are_written(tmp_path):
    path = write_trade_distribution_report(random_fills(1), str(tmp_path), title="test run")
    assert path == str(tmp_path / TRADE_DISTRIBUTION_REPORT_FILENAME)
    html = (tmp_path / TRADE_DISTRIBUTION_REPORT_FILENAME).read_text(encoding="utf-8")
    assert "test run" in html and "data:image/png;base64," in html
    summary = json.loads((tmp_path / TRADE_DISTRIBUTION_SUMMARY_FILENAME).read_text())
    assert summary["all"]["n_trades"] == 300
    assert summary["daily"]["mean_trades_per_day"] > 0
    assert summary["by_entries"] and summary["by_year"] and summary["insights"]
    assert len(pd.read_csv(tmp_path / TRADES_FILENAME)) == 300


def test_report_without_closed_trades_is_skipped(tmp_path):
    fills = sample_fills().iloc[[0]]
    assert write_trade_distribution_report(fills, str(tmp_path), title="x") is None
    assert not (tmp_path / TRADE_DISTRIBUTION_REPORT_FILENAME).exists()


def test_report_failure_never_raises(tmp_path, caplog):
    bad = pd.DataFrame([{"timestamp": "2025-01-01", "type": "entry_long"}])
    assert write_trade_distribution_report(bad, str(tmp_path), title="x") is None
    assert "could not write" in caplog.text


def test_comparison_highlights_runs_side_by_side():
    runs = []
    for label, seed in (("a", 1), ("b", 2)):
        trades = reconstruct_trades(random_fills(seed))
        summary, _ = build_summary(trades)
        runs.append((label, summary, trades))
    html = render_comparison(runs, title="cmp")
    assert "Key metrics" in html and "class='best'" in html
    assert html.count("data:image/png;base64,") == 1
