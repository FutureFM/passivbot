"""Downloader flow against a fake HTTP session (no network)."""

import io
import zipfile

import pandas as pd
import pytest

from dtrend.data.binance_vision import BinanceDownloader
from dtrend.data.panel import load_panel
from dtrend.config import DataConfig


def _zip(name: str, text: str) -> bytes:
    buf = io.BytesIO()
    with zipfile.ZipFile(buf, "w") as zf:
        zf.writestr(name, text)
    return buf.getvalue()


def _kline_rows(days: pd.DatetimeIndex) -> list[list]:
    rows = []
    for i, d in enumerate(days):
        ms = int(d.timestamp() * 1000)
        px = 100 + i
        rows.append([ms, px, px + 1, px - 1, px + 0.5, 10, ms + 86_399_999, 1000 + i, 5, 1, 1, 0])
    return rows


class FakeResponse:
    def __init__(self, status=200, content=b"", payload=None):
        self.status_code = status
        self.content = content
        self._payload = payload
        self.headers = {}
        self.text = ""

    def json(self):
        return self._payload

    def raise_for_status(self):
        if self.status_code >= 400:
            raise RuntimeError(self.status_code)


class FakeSession:
    """Serves Jan 2024 from the archive; Feb 2024 archive missing (lag); REST serves Feb onward."""

    def __init__(self):
        self.calls = []

    def get(self, url, params=None, timeout=None):
        self.calls.append(url)
        if "monthly/klines/XUSDT/1d/XUSDT-1d-2024-01.zip" in url:
            days = pd.date_range("2024-01-01", "2024-01-31", freq="D", tz="UTC")
            csv = "\n".join(",".join(map(str, r)) for r in _kline_rows(days))
            return FakeResponse(content=_zip("x.csv", csv))
        if "monthly/fundingRate/XUSDT/XUSDT-fundingRate-2024-01.zip" in url:
            csv = "calc_time,funding_interval_hours,last_funding_rate\n" + "\n".join(
                f"{int(t.timestamp() * 1000)},8,0.0001" for t in pd.date_range("2024-01-01", "2024-01-31 16:00", freq="8h", tz="UTC")
            )
            return FakeResponse(content=_zip("f.csv", csv))
        if url.endswith(".zip"):
            return FakeResponse(status=404)
        if url.endswith("/fapi/v1/klines"):
            start = pd.Timestamp(params["startTime"], unit="ms", tz="UTC")
            end = min(pd.Timestamp(params["endTime"], unit="ms", tz="UTC"), pd.Timestamp("2024-02-10", tz="UTC"))
            days = pd.date_range(start.ceil("D"), end, freq="D")
            return FakeResponse(payload=_kline_rows(days) if len(days) else [])
        if url.endswith("/fapi/v1/fundingRate"):
            start = pd.Timestamp(params["startTime"], unit="ms", tz="UTC")
            times = pd.date_range(start.ceil("8h"), pd.Timestamp("2024-02-09 16:00", tz="UTC"), freq="8h")
            return FakeResponse(payload=[{"fundingTime": int(t.timestamp() * 1000), "fundingRate": "0.0002"} for t in times])
        raise AssertionError(url)


def test_update_fills_archive_lag_with_rest(tmp_path):
    dl = BinanceDownloader(tmp_path, "1d", session=FakeSession())
    k = dl.update_klines("XUSDT", "2024-01-01", "2024-02-10")
    assert k.index[0] == pd.Timestamp("2024-01-01", tz="UTC")
    assert pd.Timestamp("2024-02-05", tz="UTC") in k.index  # REST filled the missing Feb archive
    assert k.index.is_unique and k.index.is_monotonic_increasing
    f = dl.update_funding("XUSDT", "2024-01-01", "2024-02-10")
    assert f["funding_rate"].loc["2024-01"].eq(0.0001).all()
    assert f["funding_rate"].loc["2024-02"].eq(0.0002).all()

    panel = load_panel(DataConfig(data_dir=str(tmp_path), symbols=["XUSDT"], start="2024-01-01", universe_size=1, min_history_days=5))
    # Bars are labelled by close time; Jan-2 close holds the 3 settlements of Jan 1 (08:00, 16:00, Jan-2 00:00).
    assert panel.funding["XUSDT"].loc["2024-01-02"] == pytest.approx(0.0003)
