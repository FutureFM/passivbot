import copy
import json
from types import SimpleNamespace

import pytest

from time_stop import encode_target, decode_target, reconstruct_episodes
from test_orchestrator_json_api import make_input, make_symbol, bot_params_pair, bot_params, compute
from live import reconciler
from passivbot_exceptions import FatalBotException

DAY = 86_400_000
SYMBOL = 'BTC/USDT:USDT'


def stop_input(*, pside='long', pct=.25, **params):
    bp = dict(risk_time_stop_max_age_days=1., risk_time_stop_close_pct=pct,
              risk_time_stop_we_trigger_pct=0., risk_time_stop_close_we_min=0.,
              risk_time_stop_close_we_max=1.) | params
    symbol = make_symbol(0, bid=90., ask=91., **{
        pside + '_pos_size': 10. if pside == 'long' else -10.,
        pside + '_pos_price': 100., pside + '_bp': bp})
    symbol[pside]['time_stop'] = dict(anchor_timestamp_ms=1000, pending_target_size=None, grid_ref_price=None)
    global_bp = bot_params_pair()
    global_bp[pside] = bot_params(n_positions=1, total_wallet_exposure_limit=1.)
    inp = make_input(balance=1000., symbols=[symbol], global_bp=global_bp)
    inp['timestamp_ms'] = DAY + 1000
    inp['global'].update(max_realized_loss_pct=0., market_orders_allowed=False)
    return inp


@pytest.mark.parametrize('pside', ['long', 'short'])
def test_temporal_market_is_exclusive_bypasses_loss_budget_and_roundtrips_live_validation(pside):
    import passivbot_rust as pbr
    inp = stop_input(pside=pside)
    out = compute(pbr, inp)
    orders = [o for o in out['orders'] if o['pside'] == pside]
    assert len(orders) == 1
    order = orders[0]
    assert order['order_type'] == 'close_time_stop_' + pside
    assert order['qty'] == (-2.5 if pside == 'long' else 2.5)
    assert order['execution_type'] == 'market'
    assert order['execution_priority'] == 'risk_critical'
    assert order['time_stop_target_size'] == 7.5
    reconciler.parse_and_validate_rust_orchestrator_output(json.dumps(out), {0: SYMBOL}, inp)
    for field, value in [('execution_type', 'limit'), ('time_stop_target_size', 7.6)]:
        invalid = copy.deepcopy(out)
        invalid['orders'][out['orders'].index(order)][field] = value
        with pytest.raises(FatalBotException):
            reconciler.parse_and_validate_rust_orchestrator_output(json.dumps(invalid), {0: SYMBOL}, inp)


@pytest.mark.parametrize('pct,minimum,cap,expected', [(0.,1.,.1,0.), (1.,0.,.1,10.), (.5,0.,.1,1.), (.25,1.,.1,10.)])
def test_pct_min_max_precedence(pct, minimum, cap, expected):
    import passivbot_rust as pbr
    inp = stop_input(pct=pct, risk_time_stop_close_we_min=minimum, risk_time_stop_close_we_max=cap)
    out = compute(pbr, inp)
    stops = [o for o in out['orders'] if o['order_type'].startswith('close_time_stop_')]
    assert sum(abs(o['qty']) for o in stops) == expected


def test_divergence_wel_reduction_does_not_shrink_time_stop_cap():
    import passivbot_rust as pbr
    protection = dict(divergence_filter_enabled=True, divergence_zscore_threshold=1.,
                      divergence_breadth_threshold_pct=40., divergence_breadth_drop_pct=1.,
                      divergence_min_timeframes=2, divergence_we_cap_pct=.5)
    inp = stop_input(pct=.5, risk_time_stop_close_we_max=.1, **protection)
    for i in (1, 2):
        other = make_symbol(i, bid=90., ask=91., long_mode='manual', long_bp=protection)
        for pside in ('long', 'short'):
            other[pside]['strategy_params'] = copy.deepcopy(inp['symbols'][0][pside]['strategy_params'])
        inp['symbols'].append(other)
    for i, symbol in enumerate(inp['symbols']):
        symbol['divergence_roc_pct'] = [-99., -99., None, None] if i == 0 else [0., 0., None, None]
    stops = [o for o in compute(pbr, inp)['orders'] if o['order_type'] == 'close_time_stop_long']
    # Cap = close_we_max * pre-divergence WEL * balance at the position price: 0.1 * 1 * 1000 / 100.
    assert [o['qty'] for o in stops] == [-1.]


def test_pending_reduction_finishes_original_target_without_another_interval_or_trigger():
    import passivbot_rust as pbr
    inp = stop_input(risk_time_stop_we_trigger_pct=1.)
    inp['timestamp_ms'] = 3000  # not aged
    side = inp['symbols'][0]['long']
    side['position']['size'] = 9.
    side['time_stop']['pending_target_size'] = 7.5
    order = compute(pbr, inp)['orders'][0]
    assert order['qty'] == -1.5
    assert order['time_stop_target_size'] == 7.5
    side['time_stop']['pending_target_size'] = 0.
    assert compute(pbr, inp)['orders'][0]['qty'] == -9.


def test_manual_panic_missing_history_and_unexpired_clock():
    import passivbot_rust as pbr
    inp = stop_input()
    for mode in ['manual', 'panic']:
        inp['symbols'][0]['long']['mode'] = mode
        orders = compute(pbr, inp)['orders']
        assert not any(o['order_type'].startswith('close_time_stop_') for o in orders)
    inp['symbols'][0]['long']['mode'] = None
    inp['symbols'][0]['long']['time_stop'] = None
    orders = compute(pbr, inp)['orders']
    assert not any(o['pside'] == 'long' and o['order_type'].startswith(('entry_', 'close_time_stop_')) for o in orders)
    inp['symbols'][0]['long']['time_stop'] = dict(anchor_timestamp_ms=1000, pending_target_size=None, grid_ref_price=None)
    inp['timestamp_ms'] = DAY  # 1000 ms before expiry
    assert not any(o['order_type'].startswith('close_time_stop_') for o in compute(pbr, inp)['orders'])


@pytest.mark.parametrize('prefix,limit', [('',28), ('',34), ('0fe0667832d7BCDE',32), ('x-etgQucuf',36), ('Passivbot#',64)])
@pytest.mark.parametrize('pside', ['long','short'])
def test_client_id_preserves_broker_prefix_exact_target_and_exchange_length(prefix, limit, pside):
    from passivbot import try_decode_type_id_from_custom_id
    from fill_events_manager import _try_decode_type_id_from_custom_id
    type_id = 30 if pside == 'long' else 31
    for target in [0., 7.5, 9.99, 0.00012345, 1234567890123., 9.900000000000002]:
        cid = (prefix + f'0x{type_id:04x}' + 'a'*64)[:limit]
        encoded = encode_target(cid, target)
        assert encoded.startswith(prefix) and len(encoded) <= limit
        assert decode_target(encoded, pside) == target
        assert try_decode_type_id_from_custom_id(encoded) == type_id
        assert _try_decode_type_id_from_custom_id(encoded) == type_id
        if limit == 34 and not prefix:  # Hyperliquid cloid
            int(encoded[2:], 16)


def event(qty, psize, ts, *, pside='long', target=None, c_mult=1., pb_type=None):
    cid = '' if target is None else encode_target(('0x001e' if pside=='long' else '0x001f') + 'a'*30, target)
    return SimpleNamespace(symbol=SYMBOL, position_side=pside, c_mult=c_mult, qty=qty,
                           psize=psize, timestamp=ts, price=90., client_order_id=cid,
                           pb_order_type=pb_type or ('close_time_stop_' + pside if target is not None else 'entry_grid_normal_' + pside))


def facts(events, size, *, coverage=True, c_mult=1.):
    return reconstruct_episodes(events, {SYMBOL:{'long': {'size':size}}}, {SYMBOL:.01},
                                {SYMBOL:c_mult}, lambda **kw: {'ready':coverage},
                                [(SYMBOL,'long')], time_stop_keys=[(SYMBOL,'long')])[(SYMBOL,'long')]


def test_restart_replay_resets_only_after_entire_percentage_and_keeps_entry_clock():
    events = [event(10.,10.,1000), event(-1.,9.,2000,target=7.5)]
    state = facts(events, 9.)['time_stop']
    assert state['anchor_timestamp_ms'] == 1000 and state['pending_target_size'] == 7.5
    events.append(event(-1.5,7.5,3000,target=7.5))
    completed = facts(events,7.5)['time_stop']
    assert completed == dict(anchor_timestamp_ms=3000,pending_target_size=None,grid_ref_price=90.)
    events.append(event(1.,8.5,4000))
    assert facts(events,8.5)['time_stop'] == dict(anchor_timestamp_ms=3000,pending_target_size=None,grid_ref_price=None)
    # Reconstructing in a fresh process depends only on the same exchange evidence.
    assert facts(copy.deepcopy(events),8.5) == facts(events,8.5)
    events += [event(-8.5,0.,5000,pb_type='close_grid_long'), event(2.,2.,6000)]
    assert facts(events,2.)['time_stop']['anchor_timestamp_ms'] == 6000


def test_unproven_history_unknown_reduction_or_unmatched_position_defers():
    opening = event(10.,10.,1000)
    assert facts([opening],10.,coverage=False) is None
    assert facts([opening],11.) is None
    unknown = event(-1.,9.,2000,pb_type='unknown')
    assert facts([opening,unknown],9.)['time_stop'] is None
    ordinary = event(-1.,9.,2000,pb_type='close_grid_long')
    assert facts([opening,ordinary],9.)['time_stop']['anchor_timestamp_ms'] == 1000


def test_contract_multiplier_is_applied_to_encoded_native_target():
    events = [event(10.,1.,1000,c_mult=.1), event(-1.,.9,2000,target=7.5,c_mult=.1),
              event(-1.5,.75,3000,target=7.5,c_mult=.1)]
    assert facts(events,7.5,c_mult=.1)['time_stop']['anchor_timestamp_ms'] == 3000


@pytest.mark.parametrize('field,value', [('time_stop_max_age_days',-1), ('time_stop_close_pct',1.1), ('time_stop_close_pct',True), ('time_stop_we_trigger_pct',float('nan'))])
def test_config_rejects_invalid_global_and_coin_stop_policy(field, value):
    from config_utils import get_template_config, format_config
    for override in [False, True]:
        cfg = get_template_config()
        if override:
            cfg['coin_overrides'] = {'BTC': {'bot': {'long': {'risk': {field:value}}}}}
        else:
            cfg['bot']['long']['risk'][field] = value
        with pytest.raises((ValueError, TypeError, KeyError)):
            format_config(cfg,verbose=False)


def test_config_preserves_stop_in_live_backtest_and_optimizer_overrides():
    from config_utils import get_template_config, format_config
    from test_backtest_maker_fee_override import _base_mss
    from backtest import prep_backtest_args
    cfg = get_template_config()
    cfg['bot']['long']['risk'].update(time_stop_max_age_days=7.,time_stop_close_pct=.25)
    cfg['backtest']['coins'] = {'binance':[SYMBOL]}
    cfg['coin_overrides'] = {SYMBOL: {'bot': {'long': {'risk': {'time_stop_close_pct':.5}}}}}
    cfg['optimize']['bounds']['long']['risk']['time_stop_max_age_days'] = [3,14,.25]
    formatted = format_config(cfg,verbose=False)
    assert formatted['bot']['long']['risk']['time_stop_max_age_days'] == 7.
    assert formatted['coin_overrides'][SYMBOL]['bot']['long']['risk']['time_stop_close_pct'] == .5
    formatted['backtest']['coins'] = {'binance':[SYMBOL]}
    params,_,_,_ = prep_backtest_args(formatted,_base_mss(),'binance')
    assert params[0]['long']['risk_time_stop_max_age_days'] == 7.
    assert params[0]['long']['risk_time_stop_close_pct'] == .5


def test_grid_reference_changes_entry_geometry_without_changing_close_or_exposure_facts():
    import passivbot_rust as pbr
    inp = stop_input()
    inp['timestamp_ms'] = 1001
    inp['global']['max_realized_loss_pct'] = 1.
    without = compute(pbr,inp)['orders']
    inp['symbols'][0]['long']['time_stop']['grid_ref_price'] = 80.
    with_ref = compute(pbr,inp)['orders']
    assert [o for o in without if o['order_type'].startswith('close_')] == [o for o in with_ref if o['order_type'].startswith('close_')]
    entries = [o for o in with_ref if o['order_type'].startswith('entry_') and o['pside']=='long']
    assert all(o['price'] < 80. for o in entries)
    assert inp['symbols'][0]['long']['position']['price'] == 100.


def test_live_rebuild_uses_an_unfilled_open_order_target():
    from passivbot import Passivbot
    e = event(10.,10.,1000)
    order = {'position_side':'long','timestamp':2000,'custom_id':encode_target('0x001e'+'a'*30,7.5)}
    bot = SimpleNamespace(positions={SYMBOL: {'long': {'size':10.},'short':{'size':0.}}},
                          qty_steps={SYMBOL:.01},c_mults={SYMBOL:1.},
                          _pnls_manager=SimpleNamespace(get_events=lambda:[e]),
                          _fill_history_coverage_status=lambda **kw:{'ready':True},
                          bp=lambda *args:1.,open_orders={SYMBOL:[order]},
                          _extract_order_custom_id=lambda o:o['custom_id'])
    state=Passivbot._get_time_stop_states(bot,[SYMBOL])[SYMBOL]['long']
    assert state['pending_target_size']==7.5 and state['anchor_timestamp_ms']==1000


def test_live_rust_payload_includes_fill_factor_cap_and_time_stop_policy():
    from config_utils import get_template_config
    from config.shared_bot import flatten_shared_bot_side
    from passivbot import Passivbot
    side = get_template_config()['bot']['long']
    side['risk'].update(entry_cooldown_factor_per_fill=2.,entry_cooldown_max_minutes=1440.,time_stop_max_age_days=7.,time_stop_close_pct=.25)
    flat=flatten_shared_bot_side(side)
    flat['wallet_exposure_limit']=1.
    def value(pside,key,*args):
        parts=key.split('.')
        result=flat.get(parts[0],0.)
        for part in parts[1:]:
            result=result[part]
        return result
    bot=SimpleNamespace(bp=value,bot_value=value)
    params=Passivbot._bot_params_to_rust_dict(bot,'long',SYMBOL)
    assert params['risk_entry_cooldown_factor_per_fill']==2.
    assert params['risk_entry_cooldown_max_minutes']==1440.
    assert params['risk_time_stop_max_age_days']==7.
    assert params['risk_time_stop_close_pct']==.25


def test_restart_clock_and_pending_target_drive_the_same_real_rust_decision():
    import passivbot_rust as pbr
    events = [event(10.,10.,1000), event(-1.,9.,2000,target=7.5)]
    inp = stop_input()
    inp['timestamp_ms']=3000
    inp['symbols'][0]['long']['position']['size']=9.
    inp['symbols'][0]['long']['time_stop']=facts(events,9.)['time_stop']
    assert compute(pbr,inp)['orders'][0]['qty']==-1.5
    events.append(event(-1.5,7.5,3000,target=7.5))
    inp['symbols'][0]['long']['position']['size']=7.5
    inp['symbols'][0]['long']['time_stop']=facts(events,7.5)['time_stop']
    inp['timestamp_ms']=DAY+2999
    assert not any(o['order_type'].startswith('close_time_stop_') for o in compute(pbr,inp)['orders'])
    inp['timestamp_ms']=DAY+3000
    assert compute(pbr,inp)['orders'][0]['qty']==-1.87  # quantity step is 0.01


def test_short_replay_completes_target_without_losing_signed_quantity_semantics():
    events=[event(-10.,10.,1000,pside='short'),event(1.,9.,2000,pside='short',target=7.5),
            event(1.5,7.5,3000,pside='short',target=7.5)]
    facts=reconstruct_episodes(events,{SYMBOL:{'short':{'size':-7.5}}},{SYMBOL:.01},{SYMBOL:1.},
        lambda **kw:{'ready':True},[(SYMBOL,'short')],time_stop_keys=[(SYMBOL,'short')])
    assert facts[(SYMBOL,'short')]['time_stop']['anchor_timestamp_ms']==3000


@pytest.mark.parametrize("pside", ["long", "short"])
@pytest.mark.parametrize("pct", [.25, 1.])
def test_real_backtest_temporal_losses_use_taker_fees_and_restart_completed_interval(pside, pct):
    import numpy as np
    from backtest import build_backtest_payload, execute_backtest
    from config_utils import get_template_config, format_config

    cfg = get_template_config()
    cfg["live"].update(strategy_kind="ema_anchor", max_realized_loss_pct=0.,
                       market_orders_allowed=False, warmup_ratio=0., minimum_coin_age_days=0.,
                       approved_coins={"long": ["BTC"] if pside == "long" else [],
                                       "short": ["BTC"] if pside == "short" else []})
    for side in ("long", "short"):
        cfg["bot"][side]["risk"].update(n_positions=int(side == pside),
            total_wallet_exposure_limit=float(side == pside), entry_cooldown_minutes=1000.,
            time_stop_max_age_days=10./1440 if side == pside else 0., time_stop_close_pct=pct)
        cfg["bot"][side]["unstuck"]["enabled"] = False
        cfg["bot"][side]["forager"].update(volatility_ema_span_1m=1., volume_ema_span_1m=1.,
            score_weights={"ema_readiness": 0., "volatility": 1., "volume": 0.})
        cfg["bot"][side]["strategy"]["ema_anchor"].update(base_qty_pct=.5, ema_span_0=1.,
            ema_span_1=2., offset=.02, offset_psize_weight=0.,
            offset_volatility_ema_span_1m=1., offset_volatility_ema_span_1h=0.)
    cfg["backtest"].update(start_date="2025-01-01", end_date="2025-01-02", starting_balance=1000.)
    cfg = format_config(cfg, verbose=False)
    cfg["backtest"]["coins"] = {"binance": ["BTC"]}
    count = 80
    prices = np.where(np.arange(count) < 5, 100., 90. if pside == "long" else 110.)
    high = prices + (.2 if pside == "long" else 2.5)
    low = prices - (2.5 if pside == "long" else .2)
    candles = np.stack([high, low, prices, np.ones(count)], axis=1)[:, None, :].astype("float64")
    timestamps = 1735689600000 + np.arange(count, dtype="int64") * 60_000
    markets = {"BTC": dict(maker=.0001, taker=.00055, qty_step=.001, price_step=.1,
                           min_qty=.001, min_cost=1., c_mult=1.)}
    payload = build_backtest_payload(candles, markets, cfg, "binance", np.full(count, 20_000.),
                                     timestamps, skip_btc_analysis=True)
    fills, _, _ = execute_backtest(payload, cfg)
    assert len(fills) >= 2  # no zero-fill smoke can pass
    assert fills[0, 13] == "entry_ema_anchor_" + pside
    stops = fills[1:]
    assert all(row[13] == "close_time_stop_" + pside for row in stops)
    assert all(row[14] == "taker" and float(row[3]) < 0 for row in stops)
    previous_size = abs(float(fills[0, 11]))
    previous_timestamp = int(fills[0, 1])
    for row in stops:
        qty, price, fee = abs(float(row[9])), float(row[10]), float(row[4])
        assert fee == pytest.approx(-qty * price * .00055)
        assert int(row[1]) - previous_timestamp == 11 * 60_000  # decision + next-bar execution
        assert -1e-10 <= previous_size * pct - qty < .001 + 1e-10
        assert abs(float(row[11])) == pytest.approx(previous_size - qty)
        previous_size, previous_timestamp = abs(float(row[11])), int(row[1])
    if pct == 1.:
        assert len(stops) == 1 and previous_size == 0.
    else:
        assert len(stops) >= 3 and previous_size > 0.


def test_replay_rejects_a_target_larger_than_the_preclose_position():
    assert facts([event(10.,10.,1000), event(-1.,9.,2000,target=11.)],9.)['time_stop'] is None


def test_open_retry_can_lower_target_for_minimum_sizing_but_not_reuse_old_episode():
    from passivbot import Passivbot
    events=[event(10.,10.,1000), event(-.1,9.9,2000,target=9.8)]
    order={'position_side':'long','timestamp':3000,'custom_id':encode_target('0x001e'+'a'*30,9.)}
    bot=SimpleNamespace(positions={SYMBOL:{'long':{'size':9.9},'short':{'size':0.}}},
        qty_steps={SYMBOL:.01},c_mults={SYMBOL:1.},
        _pnls_manager=SimpleNamespace(get_events=lambda:events),
        _fill_history_coverage_status=lambda **kw:{'ready':True},
        bp=lambda *args:1.,open_orders={SYMBOL:[order]},_extract_order_custom_id=lambda o:o['custom_id'])
    assert Passivbot._get_time_stop_states(bot,[SYMBOL])[SYMBOL]['long']['pending_target_size']==9.
    events += [event(-9.9,0.,4000,pb_type='close_grid_long'), event(10.,10.,5000)]
    bot.positions[SYMBOL]['long']['size']=10.
    state=Passivbot._get_time_stop_states(bot,[SYMBOL])[SYMBOL]['long']
    assert state['pending_target_size'] is None and state['anchor_timestamp_ms']==5000
