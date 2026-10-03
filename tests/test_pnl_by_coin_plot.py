"""Offline checks for cumulative realized net PnL attribution."""
import os

os.environ.setdefault("MPLBACKEND", "Agg")

import pandas as pd
import pytest

from plotting import create_forager_pnl_figure, create_forager_pnl_by_coin_figure, plt, save_figures


def test_coin_curves_include_fees_both_sides_and_full_time_window(tmp_path):
    fills = pd.DataFrame({
        "coin": ["BTC", "ETH", "BTC", "BTC", "ETH"],
        "timestamp": pd.to_datetime([
            "2026-01-01 00:03", "2026-01-01 00:02", "2026-01-01 00:01",
            "2026-01-01 00:03", "2026-01-01 00:04",
        ]),
        "type": ["close_grid_long", "entry_grid_short", "entry_grid_long",
                 "close_grid_short", "close_grid_short"],
        "pnl": [10.0, 0.0, 0.0, -3.0, 4.0],
        "fee_paid": [-0.2, -0.1, -0.1, -0.3, -0.2],
    })
    original = fills.copy(deep=True)
    bal_eq = pd.DataFrame(
        {"usd_total_equity": [1000.0, 1020.0], "usd_total_balance": [1000.0, 1010.0]},
        index=pd.to_datetime(["2026-01-01 00:00", "2026-01-01 00:05"]),
    )
    figures = create_forager_pnl_figure(fills, bal_eq, autoplot=False, return_figures=True)
    try:
        assert set(figures) == {"pnl_cumsum", "pnl_by_coin"}
        curves = {line.get_label(): line for line in figures["pnl_by_coin"].axes[0].lines
                  if line.get_label() in {"BTC", "ETH"}}
        assert set(curves) == {"BTC", "ETH"}
        assert curves["BTC"].get_ydata().tolist() == pytest.approx([0.0, -0.1, 6.4, 6.4])
        assert curves["ETH"].get_ydata().tolist() == pytest.approx([0.0, -0.1, 3.7, 3.7])
        for line in curves.values():
            assert line.get_drawstyle() == "steps-post"
            assert pd.Timestamp(line.get_xdata()[0]) == bal_eq.index[0]
            assert pd.Timestamp(line.get_xdata()[-1]) == bal_eq.index[-1]
        assert sum(line.get_ydata()[-1] for line in curves.values()) == pytest.approx(
            (fills.pnl + fills.fee_paid).sum()
        )
        pd.testing.assert_frame_equal(fills, original)
        paths = save_figures(figures, str(tmp_path))
        assert (tmp_path / "pnl_by_coin.png").stat().st_size > 0
        assert set(paths) == set(figures)
    finally:
        for fig in figures.values():
            plt.close(fig)


def test_empty_fills_do_not_create_coin_plot():
    assert create_forager_pnl_by_coin_figure(pd.DataFrame(), pd.DataFrame()) == {}
