import json
import os

import pytest

import procedures
import utils


LUNA_ONBOARD_MS = 1611817200000


def _market(symbol, base, **extra):
    return {
        "id": symbol.split("/")[0] + "USDT",
        "symbol": symbol,
        "base": base,
        "quote": "USDT",
        "settle": "USDT",
        "active": True,
        "swap": True,
        "linear": True,
        "contractSize": 1.0,
        "precision": {"amount": 1.0, "price": 0.0001},
        "info": {},
        **extra,
    }


def _write_synthetic(tmp_path, markets):
    path = tmp_path / "caches" / "binance" / utils.SYNTHETIC_MARKETS_FILENAME
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(markets), encoding="utf-8")
    return path


def _luna():
    return _market("LUNA/USDT:USDT", "LUNA", info={"onboardDate": str(LUNA_ONBOARD_MS)})


def test_load_synthetic_markets_missing_file_is_empty(monkeypatch, tmp_path):
    monkeypatch.chdir(tmp_path)
    assert utils.load_synthetic_markets("binance") == {}


def test_load_synthetic_markets_forces_inactive_and_tags(monkeypatch, tmp_path):
    monkeypatch.chdir(tmp_path)
    _write_synthetic(tmp_path, {"LUNA/USDT:USDT": _luna()})

    synthetic = utils.load_synthetic_markets("binanceusdm")

    market = synthetic["LUNA/USDT:USDT"]
    assert market["active"] is False
    assert utils.is_synthetic_market(market)


@pytest.mark.parametrize(
    "bad",
    [
        {"LUNA/USDT:USDT": _market("LUNA/USDT:USDT", "LUNA")},  # no onboardDate
        {"LUNA/USDT:USDT": _market("XYZ/USDT:USDT", "XYZ", info={"onboardDate": "1"})},
        ["not", "a", "dict"],
    ],
)
def test_load_synthetic_markets_rejects_malformed_file(monkeypatch, tmp_path, bad):
    monkeypatch.chdir(tmp_path)
    _write_synthetic(tmp_path, bad)
    with pytest.raises(ValueError):
        utils.load_synthetic_markets("binance")


@pytest.mark.asyncio
async def test_load_markets_merges_synthetic_without_touching_cache(monkeypatch, tmp_path):
    monkeypatch.chdir(tmp_path)
    btc = _market("BTC/USDT:USDT", "BTC")
    real_luna = _market("LUNA/USDT:USDT", "LUNA", id="REAL")
    markets_path = tmp_path / "caches" / "binance" / "markets.json"
    markets_path.parent.mkdir(parents=True)
    markets_path.write_text(json.dumps({"BTC/USDT:USDT": btc}), encoding="utf-8")
    _write_synthetic(
        tmp_path,
        {"LUNA/USDT:USDT": _luna(), "EOS/USDT:USDT": _market(
            "EOS/USDT:USDT", "EOS", info={"onboardDate": "1578470400000"}
        )},
    )
    maps_seen = []
    monkeypatch.setattr(
        utils, "create_coin_symbol_map_cache", lambda ex, markets, **_: maps_seen.append(markets)
    )

    markets = await utils.load_markets("binance", verbose=False)

    assert set(markets) == {"BTC/USDT:USDT", "LUNA/USDT:USDT", "EOS/USDT:USDT"}
    assert utils.is_synthetic_market(markets["LUNA/USDT:USDT"])
    assert maps_seen == [markets]
    assert json.loads(markets_path.read_text()) == {"BTC/USDT:USDT": btc}

    # A market the exchange lists again always wins over its stub.
    merged = utils.merge_synthetic_markets("binance", {"LUNA/USDT:USDT": real_luna})
    assert merged["LUNA/USDT:USDT"]["id"] == "REAL"
    assert not utils.is_synthetic_market(merged["LUNA/USDT:USDT"])


@pytest.mark.asyncio
async def test_first_timestamps_use_synthetic_onboard_date_without_probe(monkeypatch, tmp_path):
    monkeypatch.chdir(tmp_path)
    _write_synthetic(tmp_path, {"LUNA/USDT:USDT": _luna()})
    (tmp_path / "caches" / "first_ohlcv_timestamps_unified.version").write_text(
        str(procedures.FIRST_OHLCV_TIMESTAMPS_CACHE_VERSION), encoding="utf-8"
    )
    monkeypatch.setattr(procedures, "make_get_filepath", lambda p: str(tmp_path / p))
    monkeypatch.setattr(
        procedures,
        "load_ccxt_instance",
        lambda *_a, **_k: pytest.fail("purged synthetic markets must not be probed"),
    )

    def _no_map(coin, exchange):
        raise utils.UnknownMarketIdentifier(coin)

    monkeypatch.setattr(procedures, "coin_to_symbol", _no_map)

    by_exchange = await procedures.get_first_timestamps_unified(["LUNA"], exchange="binance")
    unified = await procedures.get_first_timestamps_unified(["LUNA"])

    assert by_exchange == {"LUNA": float(LUNA_ONBOARD_MS)}
    assert unified == {"LUNA": float(LUNA_ONBOARD_MS)}
    assert os.path.exists(tmp_path / "caches" / "binance" / utils.SYNTHETIC_MARKETS_FILENAME)
