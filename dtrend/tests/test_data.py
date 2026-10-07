import io
import zipfile

import numpy as np
import pandas as pd
import pytest

from dtrend.data.binance_vision import parse_funding_csv, parse_kline_csv, parse_s3_listing, read_zip_csv
from dtrend.data.panel import aggregate_funding, compute_universe

KLINE_ROW = "1704067200000,42000.1,42500,41800,42300.5,1000,1704153599999,42150000,5000,500,21000000,0"


def test_parse_kline_csv_without_header():
    df = parse_kline_csv((KLINE_ROW + "\n").encode())
    assert df.index[0] == pd.Timestamp("2024-01-01", tz="UTC")
    assert df["close"].iloc[0] == 42300.5
    assert df["quote_volume"].iloc[0] == 42150000


def test_parse_kline_csv_with_header():
    header = "open_time,open,high,low,close,volume,close_time,quote_volume,count,taker_buy_volume,taker_buy_quote_volume,ignore"
    df = parse_kline_csv(f"{header}\n{KLINE_ROW}\n".encode())
    assert len(df) == 1 and df["open"].iloc[0] == 42000.1


def test_parse_funding_csv_and_zip():
    csv = "calc_time,funding_interval_hours,last_funding_rate\n1704067200005,8,0.0001\n1704096000003,8,-0.0002\n"
    buf = io.BytesIO()
    with zipfile.ZipFile(buf, "w") as zf:
        zf.writestr("BTCUSDT-fundingRate-2024-01.csv", csv)
    df = parse_funding_csv(read_zip_csv(buf.getvalue()))
    assert list(df["funding_rate"]) == [0.0001, -0.0002]


def test_parse_s3_listing():
    xml = """<?xml version="1.0" encoding="UTF-8"?>
<ListBucketResult xmlns="http://s3.amazonaws.com/doc/2006-03-01/">
  <IsTruncated>true</IsTruncated><NextMarker>data/futures/um/monthly/klines/ETHUSDT/</NextMarker>
  <CommonPrefixes><Prefix>data/futures/um/monthly/klines/BTCUSDT/</Prefix></CommonPrefixes>
  <CommonPrefixes><Prefix>data/futures/um/monthly/klines/ETHUSDT/</Prefix></CommonPrefixes>
</ListBucketResult>"""
    prefixes, marker = parse_s3_listing(xml)
    assert prefixes[0].endswith("BTCUSDT/") and marker.endswith("ETHUSDT/")


def test_funding_goes_to_the_bar_that_holds_the_position():
    closes = pd.date_range("2024-01-01", periods=3, freq="D", tz="UTC")
    events = pd.Series(
        [1.0, 2.0, 4.0, 8.0],
        index=pd.to_datetime(
            ["2024-01-01 00:00:00.004", "2024-01-01 08:00:00.002", "2024-01-01 16:00:00.000", "2024-01-02 00:00:00.009"],
            utc=True,
        ),
    )
    out = aggregate_funding(events, closes, pd.Timedelta(days=1))
    # Event at 00:00 Jan 1 is paid by the position held into the Jan-1 close.
    assert out.tolist() == [1.0, 2.0 + 4.0 + 8.0, 0.0]


def test_universe_is_causal_and_sized():
    idx = pd.date_range("2024-01-01", periods=120, freq="D", tz="UTC")
    close = pd.DataFrame(1.0, index=idx, columns=list("ABCD"))
    close.iloc[:70, 3] = np.nan  # D lists late
    qv = pd.DataFrame({"A": 4.0, "B": 3.0, "C": 2.0, "D": 100.0}, index=idx)
    uni = compute_universe(close, qv, size=2, lookback_days=10, min_history_days=20)
    assert uni.sum(axis=1).max() <= 2
    assert not uni["D"].iloc[:90].any()  # D needs 20 days of history before it can be ranked
    assert uni["D"].iloc[-1]  # and wins by volume at a later month start
    # Changing future volume never changes past membership.
    qv2 = qv.copy()
    qv2.iloc[100:, 2] = 1e9
    uni2 = compute_universe(close, qv2, size=2, lookback_days=10, min_history_days=20)
    pd.testing.assert_frame_equal(uni.iloc[:100], uni2.iloc[:100])


@pytest.mark.parametrize("bad", ["", "\n"])
def test_empty_csv(bad):
    assert parse_kline_csv(bad.encode()).empty
    assert parse_funding_csv(bad.encode()).empty
