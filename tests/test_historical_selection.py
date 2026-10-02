"""Daily selection plumbing and real-engine integration, entirely offline."""
from copy import deepcopy
import gzip
import json
import pickle

import numpy as np
import pandas as pd
import pytest

from historical_selection import (
    DAY_MS, PreparedSelection, load_selection, prepare_arrays, prepare_config, prepare_suite,
)
from config.schema import get_template_config
from shared_arrays import SharedArrayManager
from suite_runner import SuiteScenario

START = 1609459200000


def stub_market_resolution(monkeypatch):
    # Keep real PB8 coin-list formatting; replace only network/market-resolution I/O.
    import utils

    async def noop(*args, **kwargs):
        return None

    for name in ('reject_cross_exchange_market_identifier_collisions',
                 '_coalesce_resolved_approved_markets', '_remove_resolved_ignored_markets'):
        monkeypatch.setattr(utils, name, noop)


def config_for(path):
    cfg = get_template_config()
    cfg['backtest'].update(
        organillo_mode=True, organillo_carton_path=str(path),
        start_date='2021-01-01', end_date='2021-01-03', exchanges=['binance'],
        coins={'binance': ['BTC', 'ETH']}, candle_interval_minutes=1,
        maker_fee_override=0.0, taker_fee_override=0.0,
    )
    cfg['live'].update(approved_coins={'long': ['BTC', 'ETH'], 'short': ['BTC', 'ETH']},
                       ignored_coins={'long': [], 'short': []}, warmup_ratio=0.0,
                       max_warmup_minutes=0, minimum_coin_age_days=0, hedge_mode=True)
    for side in ('long', 'short'):
        bot = cfg['bot'][side]
        bot['risk'].update(n_positions=2.0, total_wallet_exposure_limit=1.0,
            entry_cooldown_minutes=0.0, position_exposure_enforcer_enabled=False,
            total_exposure_enforcer_enabled=False, total_exposure_entry_gate_enabled=False)
        bot['hsl']['enabled'] = False
        bot['unstuck']['enabled'] = False
        bot['forager'].update(volume_drop_pct=0.0, volume_ema_span_1m=1.0, volatility_ema_span_1m=1.0)
        strat = bot['strategy']['trailing_martingale']
        strat.update(ema_span_0=1.0, ema_span_1=1.0, volatility_ema_span_1m=1.0,
                     volatility_ema_span_1h=1.0)
        for section in ('entry', 'close'):
            for key in strat[section]:
                if key.endswith('weight') or key.startswith('retracement_'):
                    strat[section][key] = 0.0
        strat['entry'].update(initial_qty_pct=0.1, initial_ema_dist=0.0,
            threshold_base_pct=0.01, double_down_factor=0.5)
        strat['close'].update(qty_pct=1.0, threshold_base_pct=0.01)
    return cfg


def carton(tmp_path, values=((1, 1), (0, 1), (1, 0)), name='selection.csv'):
    path = tmp_path / name
    pd.DataFrame(values, columns=['BTC', 'ETH'],
        index=pd.date_range('2021-01-01', periods=len(values), tz='UTC')).to_csv(path)
    return path


def dataset(n=4320):
    ts = START + np.arange(n, dtype=np.int64) * 60000
    data = np.empty((n, 2, 4), dtype=np.float64)
    data[:, :, :] = [105.0, 95.0, 100.0, 1.0]
    btc = np.full(n, 20000.0)
    mss = {c: {'qty_step': 0.001, 'price_step': 0.01, 'min_qty': 0.001,
               'min_cost': 0.0, 'c_mult': 1.0, 'maker': 0.0, 'taker': 0.0,
               'exchange': 'binance', 'symbol': c} for c in ('BTC', 'ETH')}
    mss['__meta__'] = {'requested_start_ts': START, 'warmup_minutes_requested': 0}
    return data, mss, btc, ts


def run(cfg, *, selection=None, indices=None):
    import passivbot_rust as pbr
    assert not getattr(pbr, '__is_stub__', False), 'requires the rebuilt Rust extension'
    from backtest import build_backtest_payload, execute_backtest
    data, mss, btc, ts = dataset()
    payload = build_backtest_payload(data, mss, cfg, 'binance', btc, ts,
        historical_selection=selection, coin_indices=indices)
    result = execute_backtest(payload, cfg)
    return result, payload


def test_universe_replaces_today_list_without_mutation(tmp_path):
    cfg = config_for(carton(tmp_path))
    cfg['live']['approved_coins'] = {'long': ['SOL'], 'short': []}
    cfg['_coins_sources'] = {'approved_coins': {'long': ['SOL'], 'short': []}}
    before = deepcopy(cfg)
    prepared = prepare_config(cfg)
    assert prepared['live']['approved_coins'] == {'long': ['BTC', 'ETH'], 'short': ['BTC', 'ETH']}
    assert len(prepared['backtest']['organillo_carton_hash']) == 64
    assert cfg == before
    assert prepared['_coins_sources']['approved_coins'] == prepared['live']['approved_coins']


def test_midnight_warmup_and_reordered_master_columns(tmp_path):
    cfg = config_for(carton(tmp_path))
    ts = np.array([START - 60000, START, START + DAY_MS - 60000, START + DAY_MS])
    prepared = prepare_arrays(cfg, ['ETH', 'BTC'], ts, column_indices=[2, 0], n_columns=3)
    np.testing.assert_array_equal(prepared.row_indices, [-1, 0, 0, 1])
    np.testing.assert_array_equal(prepared.eligible[:, 0], [1, 0, 1])
    np.testing.assert_array_equal(prepared.eligible[:, 2], [1, 1, 0])
    assert prepared.eligible.dtype == np.uint8
    assert prepared.row_indices.dtype == np.int32


def test_missing_dates_do_not_silently_carry_forward(tmp_path):
    path = carton(tmp_path)
    frame = pd.read_csv(path, index_col=0).iloc[[0, 2]]
    frame.to_csv(path)
    cfg = config_for(path)
    with pytest.raises(ValueError, match='missing dates'):
        prepare_config(cfg)
    with pytest.raises(ValueError, match='missing dates'):
        prepare_arrays(cfg, ['BTC', 'ETH'], [START + DAY_MS])
    cfg['backtest']['end_date'] = '2021-01-01'
    assert prepare_config(cfg)['backtest']['organillo_mode']


def test_hash_uses_decompressed_content_and_detects_change(tmp_path):
    path = carton(tmp_path)
    zipped = tmp_path/'selection.csv.gz'
    zipped.write_bytes(gzip.compress(path.read_bytes()))
    cfg = config_for(path)
    digest = load_selection(cfg).content_hash
    assert load_selection(config_for(zipped)).content_hash == digest
    prepared = prepare_config(cfg)
    path.write_text(path.read_text().replace('1,1', '0,1', 1))
    with pytest.raises(ValueError, match='changed after preparation'):
        load_selection(prepared)


def test_shared_selection_pickle_and_cleanup(tmp_path):
    cfg = config_for(carton(tmp_path))
    prepared = prepare_arrays(cfg, ['BTC', 'ETH'], [START, START + DAY_MS])
    manager = SharedArrayManager()
    shared = prepared.share(manager)
    worker = pickle.loads(pickle.dumps(shared))
    try:
        np.testing.assert_array_equal(worker.arrays().eligible, prepared.eligible)
        np.testing.assert_array_equal(worker.arrays().row_indices, [0, 1])
        assert not worker.arrays().eligible.flags.writeable
    finally:
        worker.close()
        shared.close()
        manager.cleanup()


def test_suite_cartons_have_independent_universes_and_hashes(tmp_path):
    path = carton(tmp_path, ((1, 0), (1, 0), (1, 0)))
    other = carton(tmp_path, ((0, 1), (0, 1), (0, 1)), name='other.csv')
    cfg = config_for(path)
    scenarios = [SuiteScenario(label='one', start_date=None, end_date=None, coins=None, ignored_coins=None), SuiteScenario(label='two', start_date=None, end_date=None, coins=None, ignored_coins=None,
        overrides={'backtest.organillo_carton_path': str(other)})]
    _, resolved = prepare_suite(cfg, scenarios)
    assert resolved[0].coins == ['BTC'] and resolved[1].coins == ['ETH']
    assert resolved[0].overrides['backtest.organillo_carton_hash'] != resolved[1].overrides['backtest.organillo_carton_hash']


def test_all_ones_parity_with_disabled_feature(tmp_path):
    cfg = config_for(carton(tmp_path, ((1, 1),) * 3))
    enabled, _ = run(cfg)
    disabled_cfg = deepcopy(cfg)
    disabled_cfg['backtest']['organillo_mode'] = False
    disabled, _ = run(disabled_cfg)
    assert len(enabled[0]) > 0
    np.testing.assert_array_equal(enabled[0], disabled[0])
    np.testing.assert_array_equal(enabled[1], disabled[1])
    assert json.dumps(enabled[2], sort_keys=True) == json.dumps(disabled[2], sort_keys=True)


def test_all_zero_blocks_both_sides(tmp_path):
    cfg = config_for(carton(tmp_path, ((0, 0),) * 3))
    result, _ = run(cfg)
    assert len(result[0]) == 0


def test_midnight_rotation_blocks_initial_orders_on_execution_day(tmp_path):
    cfg = config_for(carton(tmp_path, ((1, 0), (0, 1), (1, 0))))
    result, payload = run(cfg)
    assert len(result[0]) > 0
    # Existing held-position DCA is deliberately allowed; initial types identify flat starts.
    types = [str(row[13]) for row in result[0]]
    initial = [row for row in result[0] if 'entry_initial' in str(row[13])]
    assert initial, types[:5]
    for row in initial:
        index, coin = int(row[0]), str(row[2])
        day = index // 1440
        assert coin == ('ETH' if day == 1 else 'BTC'), (index, coin, row[13])
    assert payload.bundle.eligibility.shape == (3, 2)


def test_subset_keeps_physical_column_mapping(tmp_path):
    from backtest import subset_backtest_payload, execute_backtest
    cfg = config_for(carton(tmp_path))
    _, payload = run(cfg)
    subset = subset_backtest_payload(payload, coin_indices=[1, 0])
    np.testing.assert_array_equal(subset.bundle.eligibility, payload.bundle.eligibility[:, [1, 0]])
    np.testing.assert_array_equal(subset.bundle.eligibility_row_indices, payload.bundle.eligibility_row_indices)
    result = execute_backtest(subset, cfg)
    assert len(result[0]) > 0


def test_prepared_arrays_and_shared_arrays_match_direct_backtest(tmp_path):
    cfg = config_for(carton(tmp_path))
    direct, _ = run(cfg)
    _, _, _, ts = dataset()
    prepared = prepare_arrays(cfg, ['BTC', 'ETH'], ts)
    manager = SharedArrayManager()
    shared = prepared.share(manager)
    try:
        actual, payload = run(cfg, selection=shared.arrays())
        np.testing.assert_array_equal(actual[0], direct[0])
        assert json.dumps(actual[2], sort_keys=True) == json.dumps(direct[2], sort_keys=True)
        del payload
    finally:
        shared.close()
        manager.cleanup()


def test_config_roundtrip_and_resume_contract(tmp_path):
    from config import prepare_config as normalize_config, compile_runtime_config, project_config
    from config_utils import clean_config
    from optimize import _resume_config_mismatches
    cfg = config_for(carton(tmp_path))
    cfg = normalize_config(cfg, verbose=False)
    cfg = prepare_config(cfg)
    for actual in (clean_config(cfg), compile_runtime_config(cfg), project_config(cfg, 'optimize')):
        assert actual['backtest']['organillo_mode']
        assert actual['backtest']['organillo_carton_hash'] == cfg['backtest']['organillo_carton_hash']
    changed = deepcopy(cfg)
    changed['backtest']['organillo_carton_hash'] = 'changed'
    assert 'backtest.organillo_carton_hash' in _resume_config_mismatches(cfg, changed)


def _evaluate_worker(serialized, individual):
    evaluator = pickle.loads(serialized)
    try:
        return evaluator.evaluate(individual, [])
    finally:
        if hasattr(evaluator, 'close'):
            evaluator.close()
        else:
            evaluator.__del__()


def make_evaluator(cfg, manager):
    from optimize import Evaluator
    data, mss, btc, ts = dataset()
    hs, _ = manager.create_from(data)
    bs, _ = manager.create_from(btc)
    evaluator = Evaluator({'binance': hs}, {'binance': bs}, {'binance': mss},
        cfg, timestamps={'binance': ts}, shared_array_manager=manager)
    evaluator.use_duplicate_guard = False
    return evaluator


def test_optimizer_spawn_workers_match_serial_and_use_shared_selection(tmp_path):
    import multiprocessing
    from optimize import config_to_individual
    cfg = prepare_config(config_for(carton(tmp_path)))
    manager = SharedArrayManager()
    evaluator = make_evaluator(cfg, manager)
    try:
        individual = config_to_individual(cfg, evaluator.bounds,
            optimization_shape=evaluator.optimization_shape)
        serialized = pickle.dumps(evaluator)
        # The daily arrays and the HLCV cube travel through shared descriptors.
        assert len(serialized) < dataset()[0].nbytes
        expected = evaluator.evaluate(individual.copy(), [])
        with multiprocessing.get_context('spawn').Pool(2) as pool:
            actual = pool.starmap(_evaluate_worker, [(serialized, individual.copy())] * 2)
        assert all(item == expected for item in actual)
    finally:
        evaluator.__del__()
        manager.cleanup()


@pytest.mark.asyncio
async def test_optimizer_suite_contexts_map_cartons_to_master_columns(tmp_path, monkeypatch):
    import optimize_suite
    from optimize import SuiteEvaluator, config_to_individual
    from suite_runner import ExchangeDataset
    path = carton(tmp_path, ((1, 0),) * 3)
    other = carton(tmp_path, ((0, 1),) * 3, name='other.csv')
    cfg = config_for(path)
    cfg['backtest'].update(suite_enabled=True, scenarios=[{'label': 'btc'},
        {'label': 'eth', 'overrides': {'backtest.organillo_carton_path': str(other)}}])
    manager = SharedArrayManager()
    data, mss, btc, ts = dataset()
    hs, _ = manager.create_from(data)
    bs, _ = manager.create_from(btc)
    for settings in mss.values():
        settings.update(first_valid_index=0, last_valid_index=len(ts) - 1)
    master = ExchangeDataset(exchange='binance', coins=['BTC', 'ETH'],
        coin_index={'BTC': 0, 'ETH': 1}, coin_exchange={'BTC': 'binance', 'ETH': 'binance'},
        available_exchanges=['binance'], hlcvs=data, mss=mss, btc_usd_prices=btc,
        timestamps=ts, cache_dir='', hlcvs_spec=hs, btc_spec=bs)

    async def noop(*args, **kwargs):
        return {}

    async def prepared_master(*args, **kwargs):
        return {'binance': master}

    monkeypatch.setattr(optimize_suite, 'load_markets', noop)
    monkeypatch.setattr(optimize_suite, 'format_approved_ignored_coins', noop)
    monkeypatch.setattr(optimize_suite, 'reject_cross_exchange_market_identifier_collisions', noop)
    monkeypatch.setattr(optimize_suite, 'prepare_master_datasets', prepared_master)
    contexts = []
    evaluator = base = None
    try:
        contexts, aggregate = await optimize_suite.prepare_suite_contexts(cfg,
            optimize_suite.extract_suite_config(cfg, None), shared_array_manager=manager)
        assert [ctx.config['backtest']['coins']['binance'] for ctx in contexts] == [['BTC'], ['ETH']]
        for ctx in contexts:
            arrays = ctx.historical_selections['binance'].arrays()
            column = ctx.coin_slice_indices['binance'][0]
            assert np.all(arrays.eligible[:, column] == 1)
            assert np.all(arrays.eligible[:, 1 - column] == 0)
        base_cfg = deepcopy(cfg)
        base_cfg['backtest']['organillo_mode'] = False
        base = make_evaluator(base_cfg, manager)
        evaluator = SuiteEvaluator(base, contexts, aggregate)
        individual = config_to_individual(cfg, base.bounds, optimization_shape=base.optimization_shape)
        serialized = pickle.dumps(evaluator)
        expected = evaluator.evaluate(individual.copy(), [])
        actual = _evaluate_worker(serialized, individual.copy())
        assert actual == expected
        assert 'error' not in expected['metrics']
        assert set(expected['metrics']['suite_metrics']['scenario_labels']) == {'btc', 'eth'}
    finally:
        if evaluator:
            evaluator.close()
        if base:
            base.__del__()
        for ctx in contexts:
            for selection in ctx.historical_selections.values():
                selection.close()
        manager.cleanup()


@pytest.mark.asyncio
@pytest.mark.parametrize('suite', [False, True])
async def test_backtest_cli_entrypoint_writes_hash_and_fills(tmp_path, monkeypatch, suite):
    import backtest
    import sys
    cfg = config_for(carton(tmp_path))
    cfg['live']['approved_coins'] = {'long': ['SOL'], 'short': ['SOL']}
    source = tmp_path / 'input.json'
    output = tmp_path / 'output'
    output.mkdir()
    cfg['backtest']['base_dir'] = str(output)
    if suite:
        other = carton(tmp_path, ((0, 1), (1, 0), (0, 1)), name='other.csv')
        cfg['backtest'].update(suite_enabled=True, scenarios=[{'label': 'one'},
            {'label': 'two', 'overrides': {'backtest.organillo_carton_path': str(other)}}])
    source.write_text(json.dumps(cfg))

    async def noop(*args, **kwargs):
        return {}

    async def local_candles(config, exchange, **kwargs):
        assert config['live']['approved_coins'] == {'long': ['BTC', 'ETH'], 'short': ['BTC', 'ETH']}
        data, mss, btc, ts = dataset()
        return ['BTC', 'ETH'], data, mss, str(output) + '/', str(tmp_path), btc, ts

    monkeypatch.setattr(backtest, 'load_markets', noop)
    stub_market_resolution(monkeypatch)
    monkeypatch.setattr(backtest, 'prepare_hlcvs_mss', local_candles)
    if suite:
        import suite_runner
        from suite_runner import ExchangeDataset

        async def master(*args, **kwargs):
            data, mss, btc, ts = dataset()
            for coin in ('BTC', 'ETH'):
                mss[coin].update(first_valid_index=0, last_valid_index=len(ts) - 1)
            return {'binance': ExchangeDataset(
                exchange='binance', coins=['BTC', 'ETH'], coin_index={'BTC': 0, 'ETH': 1},
                coin_exchange={'BTC': 'binance', 'ETH': 'binance'}, available_exchanges=['binance'],
                hlcvs=data, mss=mss, btc_usd_prices=btc, timestamps=ts, cache_dir='',
            )}

        monkeypatch.setattr(suite_runner, 'load_markets', noop)
        monkeypatch.setattr(suite_runner, 'reject_cross_exchange_market_identifier_collisions', noop)
        monkeypatch.setattr(suite_runner, 'prepare_master_datasets', master)
    monkeypatch.setattr(sys, 'argv', ['backtest.py', str(source), '-dp'])
    await backtest.main()
    results = list(output.rglob('config.json'))
    assert len(results) == (2 if suite else 1)
    if suite:
        digests = {json.loads(p.read_text())['backtest']['organillo_carton_hash'] for p in results}
        assert len(digests) == 2
        results = [p for p in results if '/one/' in str(p)]
    output = results[0].parent
    saved = json.loads((output / 'config.json').read_text())
    metadata = json.loads((output / 'dataset.json').read_text())
    assert saved['backtest']['organillo_carton_hash'] == load_selection(cfg).content_hash
    assert metadata['historical_selection']['content_hash'] == saved['backtest']['organillo_carton_hash']
    assert len(pd.read_csv(output / 'fills.csv')) > 0


@pytest.mark.parametrize('values,dates,error', [
    ([[2, 0]], ['2021-01-01'], '0 or 1'),
    ([[1, 0], [0, 1]], ['2021-01-01', '2021-01-01'], 'distinct'),
    ([[1, 0]], ['2021-01-01T01:00:00Z'], 'midnights'),
    ([[float('nan'), 0]], ['2021-01-01'], '0 or 1'),
])
def test_invalid_cartons_fail_before_simulation(tmp_path, values, dates, error):
    path = tmp_path / 'invalid.csv'
    pd.DataFrame(values, index=dates, columns=['BTC', 'ETH']).to_csv(path)
    with pytest.raises(ValueError, match=error):
        load_selection(config_for(path))


def test_prepared_selection_hash_mismatch_is_rejected(tmp_path):
    cfg = config_for(carton(tmp_path))
    prepared = prepare_arrays(cfg, ['BTC', 'ETH'], dataset()[3])
    cfg['backtest']['organillo_carton_hash'] = 'different'
    with pytest.raises(ValueError, match='does not match'):
        run(cfg, selection=prepared)


@pytest.mark.asyncio
@pytest.mark.parametrize('suite', [False, True])
async def test_optimizer_cli_entrypoint_runs_bounded_candidates(tmp_path, monkeypatch, suite):
    import optimize
    import sys
    cfg = config_for(carton(tmp_path))
    cfg['optimize'].update(backend='deap', population_size=4, iters=4, n_cpus=2, seed=42)
    cfg['live']['approved_coins'] = {'long': ['SOL'], 'short': ['SOL']}
    if suite:
        other = carton(tmp_path, ((0, 1), (1, 0), (0, 1)), name='other.csv')
        cfg['backtest'].update(suite_enabled=True, scenarios=[{},
            {'overrides': {'backtest.organillo_carton_path': str(other)}}])
    source = tmp_path / 'optimize.json'
    source.write_text(json.dumps(cfg))

    async def noop(*args, **kwargs):
        return {}

    async def local_candles(config, exchange, **kwargs):
        assert config['live']['approved_coins'] == {'long': ['BTC', 'ETH'], 'short': ['BTC', 'ETH']}
        data, mss, btc, ts = dataset()
        return ['BTC', 'ETH'], data, mss, str(tmp_path) + '/', str(tmp_path), btc, ts

    stub_market_resolution(monkeypatch)
    monkeypatch.setattr(optimize, 'prepare_hlcvs_mss', local_candles)
    if suite:
        import optimize_suite
        from suite_runner import ExchangeDataset

        async def master(*args, **kwargs):
            data, mss, btc, ts = dataset()
            for coin in ('BTC', 'ETH'):
                mss[coin].update(first_valid_index=0, last_valid_index=len(ts) - 1)
            return {'binance': ExchangeDataset(
                exchange='binance', coins=['BTC', 'ETH'], coin_index={'BTC': 0, 'ETH': 1},
                coin_exchange={'BTC': 'binance', 'ETH': 'binance'}, available_exchanges=['binance'],
                hlcvs=data, mss=mss, btc_usd_prices=btc, timestamps=ts, cache_dir='',
            )}

        monkeypatch.setattr(optimize_suite, 'load_markets', noop)
        monkeypatch.setattr(optimize_suite, 'reject_cross_exchange_market_identifier_collisions', noop)
        monkeypatch.setattr(optimize_suite, 'prepare_master_datasets', master)
    monkeypatch.chdir(tmp_path)
    monkeypatch.setattr(sys, 'argv', ['optimize.py', str(source)])
    with pytest.raises(SystemExit) as exit_info:
        await optimize.main()
    assert exit_info.value.code == 0
    output = next((tmp_path / 'optimize_results').iterdir())
    assert (output / 'checkpoint.pkl').exists()
    assert (output / 'all_results.bin').stat().st_size > 0
    import msgpack
    with (output / 'all_results.bin').open('rb') as stream:
        first = next(msgpack.Unpacker(stream, raw=False))
    if suite:
        scenarios = first['backtest']['scenarios']
        assert len(scenarios) == 2
        for item in scenarios:
            assert item['label']
            assert len(item['overrides']['backtest.organillo_carton_hash']) == 64
        assert scenarios[0]['overrides']['backtest.organillo_carton_hash'] != scenarios[1]['overrides']['backtest.organillo_carton_hash']
    else:
        assert first['backtest']['organillo_carton_hash'] == load_selection(cfg).content_hash


@pytest.mark.parametrize('failure', ['unpaired', 'width', 'row_bounds', 'non_binary'])
def test_rust_bundle_rejects_invalid_selection_arrays(tmp_path, failure):
    import passivbot_rust as pbr
    cfg = config_for(carton(tmp_path))
    _, payload = run(cfg)
    bundle = payload.bundle
    mask = np.array(bundle.eligibility, copy=True)
    rows = np.array(bundle.eligibility_row_indices, copy=True)
    if failure == 'unpaired':
        rows = None
    elif failure == 'width':
        mask = mask[:, :1].copy()
    elif failure == 'row_bounds':
        rows[0] = len(mask)
    else:
        mask[0, 0] = 2
    with pytest.raises(ValueError, match='Organillo'):
        pbr.HlcvsBundle(bundle.hlcvs, bundle.btc_usd, bundle.timestamps, bundle.meta, mask, rows)


@pytest.mark.parametrize('headers', ['BTC,BTC', '1000SHIB,SHIB'])
def test_duplicate_carton_market_identities_are_rejected(tmp_path, headers):
    path = tmp_path / 'duplicate.csv'
    path.write_text(f'date,{headers}\n2021-01-01,1,0\n')
    with pytest.raises(ValueError, match='distinct'):
        load_selection(config_for(path))
