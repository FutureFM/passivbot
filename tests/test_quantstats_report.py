import builtins

import numpy as np
import pandas as pd
import pytest

from quantstats_report import (
    QUANTSTATS_REPORT_FILENAME,
    daily_strategy_returns,
    write_quantstats_report,
)


def hourly_equity(days=40, start=10_000.0):
    index = pd.date_range("2025-01-01", periods=days * 24, freq="1h")
    growth = np.cumprod(np.full(len(index), 1.0002))
    return pd.DataFrame({"strategy_equity": start * growth, "usd_total_balance": start}, index=index)


def test_daily_returns_use_last_strategy_equity_of_each_utc_day():
    bal_eq = hourly_equity(days=3)
    returns = daily_strategy_returns(bal_eq)
    daily_close = bal_eq["strategy_equity"].resample("1D").last()
    assert len(returns) == 2
    assert returns.iloc[0] == pytest.approx(daily_close.iloc[1] / daily_close.iloc[0] - 1)


def test_report_is_written_offline_next_to_backtest_artifacts(tmp_path):
    path = write_quantstats_report(hourly_equity(), str(tmp_path), title="test backtest")
    assert path == str(tmp_path / QUANTSTATS_REPORT_FILENAME)
    html = (tmp_path / QUANTSTATS_REPORT_FILENAME).read_text(encoding="utf-8")
    assert "test backtest" in html


def test_report_is_skipped_for_less_than_two_daily_returns(tmp_path):
    assert write_quantstats_report(hourly_equity(days=1), str(tmp_path), title="x") is None
    assert not (tmp_path / QUANTSTATS_REPORT_FILENAME).exists()


def test_missing_quantstats_skips_without_failing_the_backtest(tmp_path, monkeypatch, caplog):
    real_import = builtins.__import__

    def fake_import(name, *args, **kwargs):
        if name == "quantstats":
            raise ImportError(name)
        return real_import(name, *args, **kwargs)

    monkeypatch.setattr(builtins, "__import__", fake_import)
    assert write_quantstats_report(hourly_equity(), str(tmp_path), title="x") is None
    assert "quantstats is not installed" in caplog.text


def test_missing_strategy_equity_or_library_error_never_fails(tmp_path, monkeypatch, caplog):
    import quantstats as qs

    assert write_quantstats_report(
        hourly_equity().drop(columns="strategy_equity"), str(tmp_path), title="x"
    ) is None
    monkeypatch.setattr(qs.reports, "html", lambda *a, **k: (_ for _ in ()).throw(RuntimeError("boom")))
    assert write_quantstats_report(hourly_equity(), str(tmp_path), title="x") is None
    assert "failed to write" in caplog.text and "boom" in caplog.text
