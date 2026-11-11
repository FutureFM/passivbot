use crate::dynamic::{calc_dynamic_distance_multiplier, DynamicDistanceInputs};
use crate::entries::{calc_min_entry_qty, wallet_exposure_limit_with_allowance};
use crate::strategies::{StrategySide, TrailingMartingaleCloseParams};
use crate::types::{
    BotParams, ExchangeParams, Order, OrderType, Position, RuntimeOrderContext, StateParams,
    TrailingPriceBundle,
};
use crate::utils::{
    calc_wallet_exposure, cost_to_qty, quantize_price, quantize_qty, round_, round_dn, round_up,
    tolerant_round_dn_preserve_step, tolerant_round_up_preserve_step, RoundingMode,
};
use serde::Serialize;

#[derive(Debug, Clone, Serialize)]
pub struct TrailingMartingaleCloseThresholdTerms {
    pub base: f64,
    pub wallet_exposure: f64,
    pub volatility_ema_1m: f64,
    pub volatility_ema_1h: f64,
}

#[derive(Debug, Clone, Serialize)]
pub struct TrailingMartingaleCloseDiagnostic {
    pub kind: &'static str,
    pub pside: &'static str,
    pub selected_mode: &'static str,
    pub order_type: String,
    pub qty: f64,
    pub price: f64,
    pub triggered: bool,
    pub status: &'static str,
    pub threshold_pct: f64,
    pub threshold_met: bool,
    pub threshold_terms: TrailingMartingaleCloseThresholdTerms,
    pub retracement_pct: f64,
    pub retracement_met: bool,
    pub wallet_exposure: f64,
    pub effective_wallet_exposure_limit: f64,
    pub wallet_exposure_ratio: f64,
    pub volatility_ema_1m: f64,
    pub volatility_ema_1h: f64,
    pub extrema: TrailingPriceBundle,
}

pub(crate) fn sort_closes_by_price(closes: &mut [Order], descending: bool, context: &str) {
    for order in closes.iter() {
        assert!(
            order.price.is_finite(),
            "{context}: non-finite close price {} for order_type {:?}",
            order.price,
            order.order_type
        );
    }
    if descending {
        closes.sort_by(|a, b| b.price.total_cmp(&a.price));
    } else {
        closes.sort_by(|a, b| a.price.total_cmp(&b.price));
    }
}

pub fn calc_close_qty(
    exchange_params: &ExchangeParams,
    bot_params: &BotParams,
    runtime_context: &RuntimeOrderContext,
    position: &Position,
    close_qty_pct: f64,
    balance: f64,
    close_price: f64,
) -> f64 {
    let full_psize = cost_to_qty(
        balance * wallet_exposure_limit_with_allowance(bot_params, runtime_context),
        position.price,
        exchange_params.c_mult,
    );
    let position_size_abs = position.size.abs();
    let leftover = f64::max(0.0, position_size_abs - full_psize);
    let min_entry_qty = calc_min_entry_qty(close_price, &exchange_params);
    let close_qty = f64::min(
        round_(position_size_abs, exchange_params.qty_step),
        f64::max(
            min_entry_qty,
            round_up(
                full_psize * close_qty_pct + leftover,
                exchange_params.qty_step,
            ),
        ),
    );
    if close_qty > 0.0
        && close_qty < position_size_abs
        && position_size_abs - close_qty < min_entry_qty
    {
        position_size_abs
    } else {
        close_qty
    }
}

pub(crate) fn calc_wel_auto_reduce_long(
    exchange_params: &ExchangeParams,
    state_params: &StateParams,
    bot_params: &BotParams,
    runtime_context: &RuntimeOrderContext,
    position: &Position,
    wallet_exposure: f64,
) -> Option<Order> {
    if !bot_params.risk_wel_enforcer_enabled || bot_params.risk_wel_enforcer_threshold <= 0.0 {
        return None;
    }
    if state_params.balance <= 0.0 || position.price <= 0.0 {
        return None;
    }
    let allowed_limit = wallet_exposure_limit_with_allowance(bot_params, runtime_context);
    if allowed_limit <= 0.0 {
        return None;
    }
    // Strict target: must enforce WE < target_exposure
    let target_exposure = allowed_limit * bot_params.risk_wel_enforcer_threshold;
    if wallet_exposure <= target_exposure {
        return None;
    }
    let position_size_abs = position.size.abs();
    if position_size_abs <= f64::EPSILON {
        return None;
    }
    // Compute strict reduction: ensure WE(new) < target_exposure after rounding/min constraints
    let target_psize_strict =
        (target_exposure * state_params.balance) / (position.price * exchange_params.c_mult);
    let market_price = if state_params.order_book.ask > 0.0 {
        state_params.order_book.ask
    } else {
        position.price
    };
    if market_price <= 0.0 {
        return None;
    }
    let close_price = tolerant_round_up_preserve_step(market_price, exchange_params.price_step)
        .max(exchange_params.price_step);
    let min_qty = calc_min_entry_qty(close_price, exchange_params);
    // Iteratively increase reduction until resulting WE is strictly below target or pos is fully closed
    let mut close_qty;
    let mut steps = 0usize;
    let max_steps = 10_000usize;
    let mut reduce_qty = (position_size_abs - target_psize_strict).max(0.0);
    if reduce_qty <= f64::EPSILON {
        // Already at or below target size: emit dust if exposure still over due to rounding
        reduce_qty = exchange_params.qty_step;
    }
    loop {
        let rq = round_up(reduce_qty, exchange_params.qty_step);
        close_qty = f64::min(
            position_size_abs,
            f64::max(min_qty, rq.min(position_size_abs)),
        );
        if close_qty <= f64::EPSILON {
            return None;
        }
        let new_abs_psize = (position_size_abs - close_qty).max(0.0);
        let new_exposure = calc_wallet_exposure(
            exchange_params.c_mult,
            state_params.balance,
            new_abs_psize,
            position.price,
        );
        if new_exposure < target_exposure - 1e-12 {
            break;
        }
        if new_abs_psize <= f64::EPSILON {
            break;
        }
        reduce_qty += exchange_params.qty_step;
        steps += 1;
        if steps > max_steps {
            // safeguard: break to avoid infinite loop
            break;
        }
    }
    Some(Order {
        qty: -close_qty,
        price: close_price,
        order_type: OrderType::CloseAutoReduceWelLong,
    })
}

pub(crate) fn calc_wel_auto_reduce_short(
    exchange_params: &ExchangeParams,
    state_params: &StateParams,
    bot_params: &BotParams,
    runtime_context: &RuntimeOrderContext,
    position: &Position,
    wallet_exposure: f64,
) -> Option<Order> {
    if !bot_params.risk_wel_enforcer_enabled || bot_params.risk_wel_enforcer_threshold <= 0.0 {
        return None;
    }
    if state_params.balance <= 0.0 || position.price <= 0.0 {
        return None;
    }
    let allowed_limit = wallet_exposure_limit_with_allowance(bot_params, runtime_context);
    if allowed_limit <= 0.0 {
        return None;
    }
    // Strict target: must enforce WE < target_exposure
    let target_exposure = allowed_limit * bot_params.risk_wel_enforcer_threshold;
    if wallet_exposure <= target_exposure {
        return None;
    }
    let position_size_abs = position.size.abs();
    if position_size_abs <= f64::EPSILON {
        return None;
    }
    // Compute strict reduction: ensure WE(new) < target_exposure after rounding/min constraints
    let target_psize_strict =
        (target_exposure * state_params.balance) / (position.price * exchange_params.c_mult);
    let market_price = if state_params.order_book.bid > 0.0 {
        state_params.order_book.bid
    } else {
        position.price
    };
    if market_price <= 0.0 {
        return None;
    }
    let close_price = tolerant_round_dn_preserve_step(market_price, exchange_params.price_step)
        .max(exchange_params.price_step);
    let min_qty = calc_min_entry_qty(close_price, exchange_params);
    // Iteratively increase reduction until resulting WE is strictly below target or pos is fully closed
    let mut close_qty;
    let mut steps = 0usize;
    let max_steps = 10_000usize;
    let mut reduce_qty = (position_size_abs - target_psize_strict).max(0.0);
    if reduce_qty <= f64::EPSILON {
        reduce_qty = exchange_params.qty_step;
    }
    loop {
        let rq = round_up(reduce_qty, exchange_params.qty_step);
        close_qty = f64::min(
            position_size_abs,
            f64::max(min_qty, rq.min(position_size_abs)),
        );
        if close_qty <= f64::EPSILON {
            return None;
        }
        let new_abs_psize = (position_size_abs - close_qty).max(0.0);
        let new_exposure = calc_wallet_exposure(
            exchange_params.c_mult,
            state_params.balance,
            new_abs_psize,
            position.price,
        );
        if new_exposure < target_exposure - 1e-12 {
            break;
        }
        if new_abs_psize <= f64::EPSILON {
            break;
        }
        reduce_qty += exchange_params.qty_step;
        steps += 1;
        if steps > max_steps {
            break;
        }
    }
    Some(Order {
        qty: close_qty,
        price: close_price,
        order_type: OrderType::CloseAutoReduceWelShort,
    })
}

fn calc_close_retracement_multiplier(
    state_params: &StateParams,
    close_params: &TrailingMartingaleCloseParams,
) -> f64 {
    calc_dynamic_distance_multiplier(DynamicDistanceInputs {
        volatility_ema_1m: state_params.volatility_ema_1m,
        volatility_ema_1h: state_params.volatility_ema_1h,
        weight_volatility_1m: close_params.retracement_volatility_1m_weight,
        weight_volatility_1h: close_params.retracement_volatility_1h_weight,
        wallet_exposure_ratio: None,
        weight_wallet_exposure: 0.0,
        min_multiplier: 1.0,
    })
}

fn calc_close_threshold_pct(
    exchange_params: &ExchangeParams,
    state_params: &StateParams,
    bot_params: &BotParams,
    runtime_context: &RuntimeOrderContext,
    close_params: &TrailingMartingaleCloseParams,
    position: &Position,
) -> f64 {
    let wallet_exposure_limit = wallet_exposure_limit_with_allowance(bot_params, runtime_context);
    let wallet_exposure_ratio = if wallet_exposure_limit > 0.0 {
        calc_wallet_exposure(
            exchange_params.c_mult,
            state_params.balance,
            position.size.abs(),
            position.price,
        ) / wallet_exposure_limit
    } else {
        0.0
    };
    let threshold_pct = close_params.threshold_base_pct
        + wallet_exposure_ratio * close_params.threshold_we_weight
        + state_params.volatility_ema_1m * close_params.threshold_volatility_1m_weight
        + state_params.volatility_ema_1h * close_params.threshold_volatility_1h_weight;
    if !tp_adjustment_configured(bot_params) {
        return threshold_pct;
    }
    let wallet_exposure = calc_wallet_exposure(
        exchange_params.c_mult,
        state_params.balance,
        position.size.abs(),
        position.price,
    );
    adjust_close_threshold_pct(
        threshold_pct,
        bot_params,
        &runtime_context.close_decay,
        wallet_exposure,
        wallet_exposure_limit,
    )
}

/// Linear progress in `[0, 1]` of `x` between `start` (0) and `full` (1).
/// `full <= 0` disables the trigger (always 0). NaN input counts as 0.
pub fn tp_decay_linear_progress(x: f64, start: f64, full: f64) -> f64 {
    if full <= 0.0 || x.is_nan() {
        return 0.0;
    }
    if full <= start {
        // degenerate ramp: a step at `start`
        return if x >= start { 1.0 } else { 0.0 };
    }
    ((x - start) / (full - start)).clamp(0.0, 1.0)
}

/// Nominal per-position exposure for the TP-decay exposure trigger:
/// `total_wallet_exposure_limit / n_positions`. Unlike the runtime WEL it does not move with
/// the divergence cap, so the trigger stays fixed across regimes. Falls back to the
/// runtime limit when the nominal base is not configured.
pub fn tp_decay_we_base(bot_params: &BotParams, runtime_wallet_exposure_limit: f64) -> f64 {
    if bot_params.total_wallet_exposure_limit > 0.0 && bot_params.n_positions > 0 {
        bot_params.total_wallet_exposure_limit / bot_params.n_positions as f64
    } else {
        runtime_wallet_exposure_limit
    }
}

/// Strongest of the two TP-decay triggers (time in position, wallet-exposure ratio).
/// An unknown position age contributes nothing: the time trigger never fires on a guess.
pub fn tp_decay_factor(
    bot_params: &BotParams,
    position_age_minutes: Option<f64>,
    wallet_exposure_ratio: f64,
) -> f64 {
    let f_time = position_age_minutes.map_or(0.0, |minutes| {
        tp_decay_linear_progress(
            minutes / 60.0,
            bot_params.tp_decay_time_start_hours,
            bot_params.tp_decay_time_full_hours,
        )
    });
    let f_we = tp_decay_linear_progress(
        wallet_exposure_ratio,
        bot_params.tp_decay_we_start_ratio,
        bot_params.tp_decay_we_full_ratio,
    );
    f_time.max(f_we)
}

/// Moves a close threshold toward `floor_pct` by `factor` (0..1).
/// Never raises it: a threshold already below the floor is left alone.
pub fn apply_tp_decay(threshold_pct: f64, factor: f64, floor_pct: f64) -> f64 {
    if !(factor > 0.0) {
        return threshold_pct;
    }
    let target = threshold_pct.min(floor_pct);
    threshold_pct + factor * (target - threshold_pct)
}

/// True when the close threshold depends on wallet exposure through the TP decay, so a
/// close ladder must be recomputed rung by rung.
pub fn tp_decay_uses_wallet_exposure(bot_params: &BotParams) -> bool {
    bot_params.tp_decay_we_full_ratio > 0.0
}

fn tp_adjustment_configured(bot_params: &BotParams) -> bool {
    bot_params.tp_decay_time_full_hours > 0.0
        || bot_params.tp_decay_we_full_ratio > 0.0
        || bot_params.tp_sma20_above_boost > 0.0
}

/// SMA20 boost first (positive thresholds only, while price is above the SMA20), then the
/// decay toward the floor (gated on price below the SMA20 when `tp_decay_gate_sma20`).
pub fn adjust_close_threshold_pct(
    threshold_pct: f64,
    bot_params: &BotParams,
    decay: &crate::types::CloseDecayInputs,
    wallet_exposure: f64,
    runtime_wallet_exposure_limit: f64,
) -> f64 {
    let boosted = if threshold_pct > 0.0
        && bot_params.tp_sma20_above_boost > 0.0
        && decay.price_above_sma20 == Some(true)
    {
        threshold_pct * (1.0 + bot_params.tp_sma20_above_boost)
    } else {
        threshold_pct
    };
    if bot_params.tp_decay_gate_sma20 && decay.price_above_sma20 != Some(false) {
        return boosted;
    }
    let we_base = tp_decay_we_base(bot_params, runtime_wallet_exposure_limit);
    let wallet_exposure_ratio = if we_base > 0.0 {
        wallet_exposure / we_base
    } else {
        0.0
    };
    let factor = tp_decay_factor(
        bot_params,
        decay.position_age_minutes,
        wallet_exposure_ratio,
    );
    apply_tp_decay(boosted, factor, bot_params.tp_decay_floor_pct)
}

/// Extra reduce-only limit close that offloads part of an over-exposed position at a
/// fraction of the strategy's take-profit distance. `None` when disabled, when exposure is
/// at or below the trigger, when the take-profit distance is not positive (the ordinary
/// close already rests at or through break-even), or when the slice is below the
/// executable minimum (skipped, not enlarged).
pub fn calc_risky_offload_close(
    side: StrategySide,
    exchange_params: &ExchangeParams,
    state_params: &StateParams,
    bot_params: &BotParams,
    runtime_context: &RuntimeOrderContext,
    close_params: &TrailingMartingaleCloseParams,
    position: &Position,
) -> Option<Order> {
    let position_size_abs = position.size.abs();
    let close_pct = bot_params.risky_offload_close_pct;
    if !(bot_params.risky_offload_we_threshold > 0.0)
        || !(close_pct > 0.0)
        || position_size_abs == 0.0
    {
        return None;
    }
    let wallet_exposure_limit = runtime_context.effective_wallet_exposure_limit;
    if !(wallet_exposure_limit > 0.0) {
        return None;
    }
    let wallet_exposure = calc_wallet_exposure(
        exchange_params.c_mult,
        state_params.balance,
        position_size_abs,
        position.price,
    );
    if !(wallet_exposure / wallet_exposure_limit > bot_params.risky_offload_we_threshold) {
        return None;
    }
    let threshold_pct = calc_close_threshold_pct(
        exchange_params,
        state_params,
        bot_params,
        runtime_context,
        close_params,
        position,
    );
    if !(threshold_pct > 0.0) {
        return None;
    }
    let distance = threshold_pct * bot_params.risky_offload_markup_pct.max(0.0);
    let (price, order_type, sign) = match side {
        StrategySide::Long => (
            f64::max(
                state_params.order_book.ask,
                round_up(
                    position.price * (1.0 + distance),
                    exchange_params.price_step,
                ),
            ),
            OrderType::CloseRiskyOffloadLong,
            -1.0,
        ),
        StrategySide::Short => (
            f64::min(
                state_params.order_book.bid,
                round_dn(
                    position.price * (1.0 - distance),
                    exchange_params.price_step,
                ),
            ),
            OrderType::CloseRiskyOffloadShort,
            1.0,
        ),
    };
    let qty = f64::min(
        round_(position_size_abs, exchange_params.qty_step),
        round_(
            position_size_abs * close_pct.min(1.0),
            exchange_params.qty_step,
        ),
    );
    if !(qty > 0.0) || qty < calc_min_entry_qty(price, exchange_params) {
        return None;
    }
    Some(quantize_close_price(
        Order {
            qty: sign * qty,
            price,
            order_type,
        },
        exchange_params,
    ))
}

pub fn calc_grid_close_long(
    exchange_params: &ExchangeParams,
    state_params: &StateParams,
    bot_params: &BotParams,
    runtime_context: &RuntimeOrderContext,
    close_params: &TrailingMartingaleCloseParams,
    position: &Position,
) -> Order {
    if position.size <= 0.0 {
        return Order::default();
    }
    let threshold_pct = calc_close_threshold_pct(
        exchange_params,
        state_params,
        bot_params,
        runtime_context,
        close_params,
        position,
    );
    let close_price = f64::max(
        state_params.order_book.ask,
        round_up(
            position.price * (1.0 + threshold_pct),
            exchange_params.price_step,
        ),
    );
    let qty_pct =
        if close_params.threshold_we_weight == 0.0 && !tp_decay_uses_wallet_exposure(bot_params) {
            1.0
        } else {
            close_params.qty_pct
        };
    let close_qty = -calc_close_qty(
        &exchange_params,
        &bot_params,
        runtime_context,
        &position,
        qty_pct,
        state_params.balance,
        close_price,
    );
    Order {
        qty: close_qty,
        price: close_price,
        order_type: OrderType::CloseGridLong,
    }
}

pub fn calc_trailing_close_long(
    exchange_params: &ExchangeParams,
    state_params: &StateParams,
    bot_params: &BotParams,
    runtime_context: &RuntimeOrderContext,
    close_params: &TrailingMartingaleCloseParams,
    position: &Position,
    trailing_price_bundle: &TrailingPriceBundle,
) -> Order {
    if position.size == 0.0 {
        return Order::default();
    }
    let close_trailing_threshold_pct = calc_close_threshold_pct(
        exchange_params,
        state_params,
        bot_params,
        runtime_context,
        close_params,
        position,
    );
    let close_trailing_retracement_pct = close_params.retracement_base_pct.max(0.0)
        * calc_close_retracement_multiplier(state_params, close_params);
    if close_trailing_threshold_pct <= 0.0 {
        // means trailing close immediately from pos open
        if close_trailing_retracement_pct > 0.0
            && trailing_price_bundle.min_since_max
                < trailing_price_bundle.max_since_open * (1.0 - close_trailing_retracement_pct)
        {
            Order {
                qty: -calc_close_qty(
                    &exchange_params,
                    &bot_params,
                    runtime_context,
                    &position,
                    close_params.qty_pct,
                    state_params.balance,
                    state_params.order_book.ask,
                ),
                price: state_params.order_book.ask,
                order_type: OrderType::CloseTrailingLong,
            }
        } else {
            Order {
                qty: 0.0,
                price: 0.0,
                order_type: OrderType::CloseTrailingLong,
            }
        }
    } else {
        // means trailing close will activate only after a threshold
        if close_trailing_retracement_pct <= 0.0 {
            // close at threshold
            let close_price = f64::max(
                state_params.order_book.ask,
                round_up(
                    position.price * (1.0 + close_trailing_threshold_pct),
                    exchange_params.price_step,
                ),
            );
            Order {
                qty: -calc_close_qty(
                    &exchange_params,
                    &bot_params,
                    runtime_context,
                    &position,
                    close_params.qty_pct,
                    state_params.balance,
                    close_price,
                ),
                price: close_price,
                order_type: OrderType::CloseTrailingLong,
            }
        } else {
            // close if both conditions are met
            if trailing_price_bundle.max_since_open
                > position.price * (1.0 + close_trailing_threshold_pct)
                && trailing_price_bundle.min_since_max
                    < trailing_price_bundle.max_since_open * (1.0 - close_trailing_retracement_pct)
            {
                let close_price = f64::max(
                    state_params.order_book.ask,
                    round_up(
                        position.price
                            * (1.0 + close_trailing_threshold_pct - close_trailing_retracement_pct),
                        exchange_params.price_step,
                    ),
                );
                Order {
                    qty: -calc_close_qty(
                        &exchange_params,
                        &bot_params,
                        runtime_context,
                        &position,
                        close_params.qty_pct,
                        state_params.balance,
                        close_price,
                    ),
                    price: close_price,
                    order_type: OrderType::CloseTrailingLong,
                }
            } else {
                Order {
                    qty: 0.0,
                    price: 0.0,
                    order_type: OrderType::CloseTrailingLong,
                }
            }
        }
    }
}

// Size against the selected raw price first, then normalize executable intent.
// Single-order backtest peeks must use the same price as expanded close bundles.
fn quantize_close_price(mut order: Order, exchange_params: &ExchangeParams) -> Order {
    order.price = quantize_price(
        order.price,
        exchange_params.price_step,
        RoundingMode::Nearest,
        "calc_next_close::price",
    );
    order
}

pub fn calc_next_close_long(
    exchange_params: &ExchangeParams,
    state_params: &StateParams,
    bot_params: &BotParams,
    runtime_context: &RuntimeOrderContext,
    close_params: &TrailingMartingaleCloseParams,
    position: &Position,
    trailing_price_bundle: &TrailingPriceBundle,
) -> Order {
    if position.size == 0.0 {
        // no position
        return Order::default();
    }
    let wallet_exposure = calc_wallet_exposure(
        exchange_params.c_mult,
        state_params.balance,
        position.size,
        position.price,
    );
    if let Some(order) = calc_wel_auto_reduce_long(
        exchange_params,
        state_params,
        bot_params,
        runtime_context,
        position,
        wallet_exposure,
    ) {
        return quantize_close_price(order, exchange_params);
    }
    let order = if close_params.retracement_base_pct > 0.0 {
        calc_trailing_close_long(
            exchange_params,
            state_params,
            bot_params,
            runtime_context,
            close_params,
            position,
            trailing_price_bundle,
        )
    } else {
        calc_grid_close_long(
            exchange_params,
            state_params,
            bot_params,
            runtime_context,
            close_params,
            position,
        )
    };
    quantize_close_price(order, exchange_params)
}

pub fn calc_grid_close_short(
    exchange_params: &ExchangeParams,
    state_params: &StateParams,
    bot_params: &BotParams,
    runtime_context: &RuntimeOrderContext,
    close_params: &TrailingMartingaleCloseParams,
    position: &Position,
) -> Order {
    let position_size_abs = position.size.abs();
    if position_size_abs == 0.0 {
        return Order::default();
    }
    let threshold_pct = calc_close_threshold_pct(
        exchange_params,
        state_params,
        bot_params,
        runtime_context,
        close_params,
        position,
    );
    let close_price = f64::min(
        state_params.order_book.bid,
        round_dn(
            position.price * (1.0 - threshold_pct),
            exchange_params.price_step,
        ),
    );
    let qty_pct =
        if close_params.threshold_we_weight == 0.0 && !tp_decay_uses_wallet_exposure(bot_params) {
            1.0
        } else {
            close_params.qty_pct
        };
    let close_qty = calc_close_qty(
        &exchange_params,
        &bot_params,
        runtime_context,
        &position,
        qty_pct,
        state_params.balance,
        close_price,
    );
    Order {
        qty: close_qty,
        price: close_price,
        order_type: OrderType::CloseGridShort,
    }
}

pub fn calc_trailing_close_short(
    exchange_params: &ExchangeParams,
    state_params: &StateParams,
    bot_params: &BotParams,
    runtime_context: &RuntimeOrderContext,
    close_params: &TrailingMartingaleCloseParams,
    position: &Position,
    trailing_price_bundle: &TrailingPriceBundle,
) -> Order {
    let position_size_abs = position.size.abs();
    if position_size_abs == 0.0 {
        return Order::default();
    }
    let close_trailing_threshold_pct = calc_close_threshold_pct(
        exchange_params,
        state_params,
        bot_params,
        runtime_context,
        close_params,
        position,
    );
    let close_trailing_retracement_pct = close_params.retracement_base_pct.max(0.0)
        * calc_close_retracement_multiplier(state_params, close_params);
    if close_trailing_threshold_pct <= 0.0 {
        // means trailing stop immediately from pos open
        if close_trailing_retracement_pct > 0.0
            && trailing_price_bundle.max_since_min
                > trailing_price_bundle.min_since_open * (1.0 + close_trailing_retracement_pct)
        {
            Order {
                qty: calc_close_qty(
                    &exchange_params,
                    &bot_params,
                    runtime_context,
                    &position,
                    close_params.qty_pct,
                    state_params.balance,
                    state_params.order_book.bid,
                ),
                price: state_params.order_book.bid,
                order_type: OrderType::CloseTrailingShort,
            }
        } else {
            Order {
                qty: 0.0,
                price: 0.0,
                order_type: OrderType::CloseTrailingShort,
            }
        }
    } else {
        // means trailing stop will activate only after a threshold
        if close_trailing_retracement_pct <= 0.0 {
            // close at threshold
            let close_price = f64::min(
                state_params.order_book.bid,
                round_dn(
                    position.price * (1.0 - close_trailing_threshold_pct),
                    exchange_params.price_step,
                ),
            );
            Order {
                qty: calc_close_qty(
                    &exchange_params,
                    &bot_params,
                    runtime_context,
                    &position,
                    close_params.qty_pct,
                    state_params.balance,
                    close_price,
                ),
                price: close_price,
                order_type: OrderType::CloseTrailingShort,
            }
        } else {
            if trailing_price_bundle.min_since_open
                < position.price * (1.0 - close_trailing_threshold_pct)
                && trailing_price_bundle.max_since_min
                    > trailing_price_bundle.min_since_open * (1.0 + close_trailing_retracement_pct)
            {
                let close_price = f64::min(
                    state_params.order_book.bid,
                    round_dn(
                        position.price
                            * (1.0 - close_trailing_threshold_pct + close_trailing_retracement_pct),
                        exchange_params.price_step,
                    ),
                );
                Order {
                    qty: calc_close_qty(
                        &exchange_params,
                        &bot_params,
                        runtime_context,
                        &position,
                        close_params.qty_pct,
                        state_params.balance,
                        close_price,
                    ),
                    price: close_price,
                    order_type: OrderType::CloseTrailingShort,
                }
            } else {
                Order {
                    qty: 0.0,
                    price: 0.0,
                    order_type: OrderType::CloseTrailingShort,
                }
            }
        }
    }
}

pub fn calc_next_close_short(
    exchange_params: &ExchangeParams,
    state_params: &StateParams,
    bot_params: &BotParams,
    runtime_context: &RuntimeOrderContext,
    close_params: &TrailingMartingaleCloseParams,
    position: &Position,
    trailing_price_bundle: &TrailingPriceBundle,
) -> Order {
    let position_size_abs = position.size.abs();
    if position_size_abs == 0.0 {
        // no position
        return Order::default();
    }
    let wallet_exposure = calc_wallet_exposure(
        exchange_params.c_mult,
        state_params.balance,
        position_size_abs,
        position.price,
    );
    if let Some(order) = calc_wel_auto_reduce_short(
        exchange_params,
        state_params,
        bot_params,
        runtime_context,
        position,
        wallet_exposure,
    ) {
        return quantize_close_price(order, exchange_params);
    }
    let order = if close_params.retracement_base_pct > 0.0 {
        calc_trailing_close_short(
            exchange_params,
            state_params,
            bot_params,
            runtime_context,
            close_params,
            position,
            trailing_price_bundle,
        )
    } else {
        calc_grid_close_short(
            exchange_params,
            state_params,
            bot_params,
            runtime_context,
            close_params,
            position,
        )
    };
    quantize_close_price(order, exchange_params)
}

pub fn calc_trailing_martingale_close_diagnostic(
    side: StrategySide,
    exchange_params: &ExchangeParams,
    state_params: &StateParams,
    bot_params: &BotParams,
    runtime_context: &RuntimeOrderContext,
    close_params: &TrailingMartingaleCloseParams,
    position: &Position,
    trailing_price_bundle: &TrailingPriceBundle,
) -> TrailingMartingaleCloseDiagnostic {
    let effective_wallet_exposure_limit =
        wallet_exposure_limit_with_allowance(bot_params, runtime_context);
    let wallet_exposure = calc_wallet_exposure(
        exchange_params.c_mult,
        state_params.balance,
        position.size,
        position.price,
    );
    let wallet_exposure_ratio = if effective_wallet_exposure_limit > 0.0 {
        wallet_exposure / effective_wallet_exposure_limit
    } else {
        0.0
    };
    let threshold_pct = calc_close_threshold_pct(
        exchange_params,
        state_params,
        bot_params,
        runtime_context,
        close_params,
        position,
    );
    let retracement_pct = close_params.retracement_base_pct.max(0.0)
        * calc_close_retracement_multiplier(state_params, close_params);

    let (pside, threshold_met, retracement_met, order) = match side {
        StrategySide::Long => (
            "long",
            threshold_pct <= 0.0
                || trailing_price_bundle.max_since_open > position.price * (1.0 + threshold_pct),
            retracement_pct <= 0.0
                || trailing_price_bundle.min_since_max
                    < trailing_price_bundle.max_since_open * (1.0 - retracement_pct),
            calc_next_close_long(
                exchange_params,
                state_params,
                bot_params,
                runtime_context,
                close_params,
                position,
                trailing_price_bundle,
            ),
        ),
        StrategySide::Short => (
            "short",
            threshold_pct <= 0.0
                || trailing_price_bundle.min_since_open < position.price * (1.0 - threshold_pct),
            retracement_pct <= 0.0
                || trailing_price_bundle.max_since_min
                    > trailing_price_bundle.min_since_open * (1.0 + retracement_pct),
            calc_next_close_short(
                exchange_params,
                state_params,
                bot_params,
                runtime_context,
                close_params,
                position,
                trailing_price_bundle,
            ),
        ),
    };
    let triggered = order.qty != 0.0
        && matches!(
            order.order_type,
            OrderType::CloseTrailingLong | OrderType::CloseTrailingShort
        );
    let selected_mode = match order.order_type {
        OrderType::CloseTrailingLong | OrderType::CloseTrailingShort => "trailing",
        OrderType::CloseGridLong | OrderType::CloseGridShort => "grid",
        OrderType::CloseUnstuckLong | OrderType::CloseUnstuckShort => "unstuck",
        OrderType::CloseAutoReduceWelLong | OrderType::CloseAutoReduceWelShort => "auto_reduce",
        OrderType::ClosePanicLong | OrderType::ClosePanicShort => "panic",
        OrderType::Empty => "none",
        _ => "other",
    };
    let status = if triggered {
        "triggered"
    } else if !threshold_met {
        "waiting_threshold"
    } else if !retracement_met {
        "waiting_retracement"
    } else {
        "armed"
    };

    TrailingMartingaleCloseDiagnostic {
        kind: "close",
        pside,
        selected_mode,
        order_type: order.order_type.to_string(),
        qty: order.qty,
        price: order.price,
        triggered,
        status,
        threshold_pct,
        threshold_met,
        threshold_terms: TrailingMartingaleCloseThresholdTerms {
            base: close_params.threshold_base_pct,
            wallet_exposure: wallet_exposure_ratio * close_params.threshold_we_weight,
            volatility_ema_1m: state_params.volatility_ema_1m
                * close_params.threshold_volatility_1m_weight,
            volatility_ema_1h: state_params.volatility_ema_1h
                * close_params.threshold_volatility_1h_weight,
        },
        retracement_pct,
        retracement_met,
        wallet_exposure,
        effective_wallet_exposure_limit,
        wallet_exposure_ratio,
        volatility_ema_1m: state_params.volatility_ema_1m,
        volatility_ema_1h: state_params.volatility_ema_1h,
        extrema: trailing_price_bundle.clone(),
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn make_runtime_context() -> RuntimeOrderContext {
        RuntimeOrderContext {
            entry_reference_price: None,
            effective_wallet_exposure_limit: 1.0,
            ..Default::default()
        }
    }

    fn make_exchange_params() -> ExchangeParams {
        ExchangeParams {
            qty_step: 0.01,
            price_step: 0.01,
            min_qty: 0.0,
            min_cost: 0.0,
            c_mult: 1.0,
            ..Default::default()
        }
    }

    fn make_close_params() -> TrailingMartingaleCloseParams {
        TrailingMartingaleCloseParams {
            qty_pct: 1.0,
            threshold_base_pct: 0.01,
            threshold_we_weight: 0.0,
            threshold_volatility_1h_weight: 0.0,
            threshold_volatility_1m_weight: 0.0,
            retracement_base_pct: 0.0,
            retracement_volatility_1h_weight: 0.0,
            retracement_volatility_1m_weight: 0.0,
        }
    }

    #[test]
    fn test_wel_strict_reduce_long_minimal() {
        let exchange = make_exchange_params();
        // Balance 1000, price 100, psize 10.001 -> WE = 1.0001; target = 1.0
        let state = StateParams {
            balance: 1000.0,
            order_book: crate::types::OrderBook {
                ask: 100.009,
                bid: 100.0,
                ..Default::default()
            },
            ..Default::default()
        };
        let bot = BotParams {
            wallet_exposure_limit: 1.0, // WEL_base
            risk_we_excess_allowance_pct: 0.0,
            risk_wel_enforcer_threshold: 1.0,
            ..Default::default()
        };
        let pos = Position {
            size: 10.001,
            price: 100.0,
        };
        let we = calc_wallet_exposure(exchange.c_mult, state.balance, pos.size, pos.price);
        assert!(we > 1.0);
        let order = super::calc_wel_auto_reduce_long(
            &exchange,
            &state,
            &bot,
            &make_runtime_context(),
            &pos,
            we,
        )
        .expect("should emit strict reduce order");
        assert!(order.qty < 0.0);
        assert_eq!(order.price, 100.01);
        assert!(order.price >= state.order_book.ask);
        let new_psize = (pos.size - order.qty.abs()).max(0.0);
        let new_we = calc_wallet_exposure(exchange.c_mult, state.balance, new_psize, pos.price);
        assert!(new_we < 1.0, "new_we={} not strictly below target", new_we);
    }

    #[test]
    fn test_wel_enforcer_caps_excess_at_total_limit_for_single_position() {
        let exchange = make_exchange_params();
        let state = StateParams {
            balance: 1000.0,
            order_book: crate::types::OrderBook {
                ask: 100.0,
                bid: 100.0,
                ..Default::default()
            },
            ..Default::default()
        };
        let bot = BotParams {
            wallet_exposure_limit: 1.0,
            total_wallet_exposure_limit: 1.0,
            n_positions: 1,
            risk_we_excess_allowance_pct: 1.5,
            risk_wel_enforcer_threshold: 1.0,
            ..Default::default()
        };
        let pos = Position {
            size: 12.0,
            price: 100.0,
        };
        let we = calc_wallet_exposure(exchange.c_mult, state.balance, pos.size, pos.price);
        assert!(we > 1.0);
        let order = super::calc_wel_auto_reduce_long(
            &exchange,
            &state,
            &bot,
            &make_runtime_context(),
            &pos,
            we,
        )
        .expect("single-position excess must be capped by TWEL");
        let new_psize = (pos.size - order.qty.abs()).max(0.0);
        let new_we = calc_wallet_exposure(exchange.c_mult, state.balance, new_psize, pos.price);
        assert!(new_we < 1.0, "new_we={} not strictly below target", new_we);
    }

    #[test]
    fn test_wel_enforcer_disabled_suppresses_reduce() {
        let exchange = make_exchange_params();
        let state = StateParams {
            balance: 1000.0,
            order_book: crate::types::OrderBook {
                ask: 100.0,
                bid: 100.0,
                ..Default::default()
            },
            ..Default::default()
        };
        let bot = BotParams {
            wallet_exposure_limit: 1.0,
            risk_wel_enforcer_enabled: false,
            risk_wel_enforcer_threshold: 1.0,
            ..Default::default()
        };
        let pos = Position {
            size: 11.0,
            price: 100.0,
        };
        let we = calc_wallet_exposure(exchange.c_mult, state.balance, pos.size, pos.price);
        assert!(we > 1.0);
        assert!(super::calc_wel_auto_reduce_long(
            &exchange,
            &state,
            &bot,
            &make_runtime_context(),
            &pos,
            we,
        )
        .is_none());
    }

    #[test]
    fn test_wel_long_recomputes_minimum_from_quantized_limit_price() {
        let exchange = ExchangeParams {
            qty_step: 0.001,
            price_step: 0.01,
            min_qty: 0.0,
            min_cost: 5.0,
            c_mult: 1.0,
            ..Default::default()
        };
        let state = StateParams {
            balance: 1000.0,
            order_book: crate::types::OrderBook {
                ask: 3.003,
                bid: 3.0,
                ..Default::default()
            },
            ..Default::default()
        };
        let bot = BotParams {
            wallet_exposure_limit: 1.0,
            risk_wel_enforcer_threshold: 1.0,
            ..Default::default()
        };
        let pos = Position {
            size: 10.001,
            price: 100.0,
        };
        let we = calc_wallet_exposure(exchange.c_mult, state.balance, pos.size, pos.price);

        let order = super::calc_wel_auto_reduce_long(
            &exchange,
            &state,
            &bot,
            &make_runtime_context(),
            &pos,
            we,
        )
        .expect("should emit WEL reducer");

        assert_eq!(order.price, 3.01);
        assert!((order.qty - -1.662).abs() < 1e-12);
        assert!(order.price >= state.order_book.ask);
        assert!(order.qty.abs() * order.price * exchange.c_mult >= exchange.min_cost);
    }

    #[test]
    fn test_wel_strict_reduce_short_with_rounding() {
        let exchange = ExchangeParams {
            qty_step: 0.5,
            price_step: 0.5,
            min_qty: 0.0,
            min_cost: 0.0,
            c_mult: 1.0,
            ..Default::default()
        };
        let state = StateParams {
            balance: 500.0,
            order_book: crate::types::OrderBook {
                ask: 50.0,
                bid: 49.991,
                ..Default::default()
            },
            ..Default::default()
        };
        // price 50, psize 10.1 -> WE = (10.1*50)/500 = 1.01; target=1.0
        let bot = BotParams {
            wallet_exposure_limit: 1.0,
            risk_we_excess_allowance_pct: 0.0,
            risk_wel_enforcer_threshold: 1.0,
            ..Default::default()
        };
        let pos = Position {
            size: -10.1,
            price: 50.0,
        };
        let we = calc_wallet_exposure(exchange.c_mult, state.balance, pos.size.abs(), pos.price);
        assert!(we > 1.0);
        let order = super::calc_wel_auto_reduce_short(
            &exchange,
            &state,
            &bot,
            &make_runtime_context(),
            &pos,
            we,
        )
        .expect("should emit strict reduce order");
        assert!(order.qty > 0.0);
        assert_eq!(order.price, 49.5);
        assert!(order.price <= state.order_book.bid);
        let new_psize = (pos.size.abs() - order.qty.abs()).max(0.0);
        let new_we = calc_wallet_exposure(exchange.c_mult, state.balance, new_psize, pos.price);
        assert!(new_we < 1.0, "new_we={} not strictly below target", new_we);
    }

    #[test]
    fn test_grid_close_long_adds_volatility_and_wallet_exposure_threshold() {
        let exchange = make_exchange_params();
        let state = StateParams {
            balance: 10_000.0,
            order_book: crate::types::OrderBook {
                ask: 100.0,
                bid: 100.0,
                ..Default::default()
            },
            volatility_ema_1h: 0.004,
            volatility_ema_1m: 0.002,
            ..Default::default()
        };
        let bot = BotParams {
            wallet_exposure_limit: 1.0,
            risk_we_excess_allowance_pct: 0.0,
            ..Default::default()
        };
        let close = TrailingMartingaleCloseParams {
            threshold_base_pct: 0.008,
            threshold_we_weight: -0.002,
            threshold_volatility_1h_weight: 0.25,
            threshold_volatility_1m_weight: 0.15,
            qty_pct: 0.1,
            ..make_close_params()
        };
        let pos = Position {
            size: 50.0,
            price: 100.0,
        };

        let order = calc_grid_close_long(
            &exchange,
            &state,
            &bot,
            &make_runtime_context(),
            &close,
            &pos,
        );

        assert_eq!(order.order_type, OrderType::CloseGridLong);
        assert_eq!(order.price, 100.83);
        assert_eq!(order.qty, -10.0);
    }

    fn make_recursive_close_state() -> StateParams {
        StateParams {
            balance: 10_000.0,
            order_book: crate::types::OrderBook {
                ask: 100.0,
                bid: 100.0,
                ..Default::default()
            },
            ..Default::default()
        }
    }

    fn make_recursive_close_bot() -> BotParams {
        BotParams {
            wallet_exposure_limit: 1.0,
            risk_we_excess_allowance_pct: 0.0,
            risk_wel_enforcer_enabled: false,
            risk_wel_enforcer_threshold: 0.0,
            ..Default::default()
        }
    }

    #[test]
    fn test_recursive_grid_close_long_negative_we_weight_ascends_as_exposure_falls() {
        let exchange = make_exchange_params();
        let state = make_recursive_close_state();
        let bot = make_recursive_close_bot();
        let close = TrailingMartingaleCloseParams {
            qty_pct: 0.1,
            threshold_base_pct: 0.02,
            threshold_we_weight: -0.01,
            ..make_close_params()
        };
        let pos = Position {
            size: 100.0,
            price: 100.0,
        };

        let closes = calc_closes_long(
            &exchange,
            &state,
            &bot,
            &make_runtime_context(),
            &close,
            &pos,
            &TrailingPriceBundle::default(),
        );

        assert_eq!(closes.len(), 10);
        assert!(closes
            .iter()
            .all(|order| order.order_type == OrderType::CloseGridLong));
        for (idx, order) in closes.iter().enumerate() {
            let expected_price = 101.0 + idx as f64 * 0.1;
            assert!(
                (order.price - expected_price).abs() < 1e-12,
                "idx={} price={} expected={}",
                idx,
                order.price,
                expected_price
            );
            assert!((order.qty + 10.0).abs() < 1e-12);
        }
    }

    #[test]
    fn test_recursive_grid_close_long_positive_we_weight_sorts_returned_orders() {
        let exchange = make_exchange_params();
        let state = make_recursive_close_state();
        let bot = make_recursive_close_bot();
        let close = TrailingMartingaleCloseParams {
            qty_pct: 0.1,
            threshold_base_pct: 0.01,
            threshold_we_weight: 0.01,
            ..make_close_params()
        };
        let pos = Position {
            size: 100.0,
            price: 100.0,
        };

        let closes = calc_closes_long(
            &exchange,
            &state,
            &bot,
            &make_runtime_context(),
            &close,
            &pos,
            &TrailingPriceBundle::default(),
        );

        assert_eq!(closes.len(), 10);
        assert!(closes
            .iter()
            .all(|order| order.order_type == OrderType::CloseGridLong));
        // Recursive generation produces 102.0, 101.9, ..., 101.1 as exposure falls; the public
        // close list is sorted ascending before returning.
        for (idx, order) in closes.iter().enumerate() {
            let expected_price = 101.1 + idx as f64 * 0.1;
            assert!(
                (order.price - expected_price).abs() < 1e-12,
                "idx={} price={} expected={}",
                idx,
                order.price,
                expected_price
            );
            assert!((order.qty + 10.0).abs() < 1e-12);
        }
    }

    #[test]
    fn test_grid_close_long_zero_we_weight_ignores_qty_pct_and_closes_full_position() {
        let exchange = make_exchange_params();
        let state = make_recursive_close_state();
        let bot = make_recursive_close_bot();
        let close = TrailingMartingaleCloseParams {
            qty_pct: 0.1,
            threshold_base_pct: 0.01,
            threshold_we_weight: 0.0,
            ..make_close_params()
        };
        let pos = Position {
            size: 100.0,
            price: 100.0,
        };

        let closes = calc_closes_long(
            &exchange,
            &state,
            &bot,
            &make_runtime_context(),
            &close,
            &pos,
            &TrailingPriceBundle::default(),
        );

        assert_eq!(closes.len(), 1);
        assert_eq!(closes[0].order_type, OrderType::CloseGridLong);
        assert_eq!(closes[0].price, 101.0);
        assert_eq!(closes[0].qty, -100.0);
    }

    #[test]
    fn test_trailing_close_long_returns_single_order_not_recursive_grid() {
        let exchange = make_exchange_params();
        let state = make_recursive_close_state();
        let bot = make_recursive_close_bot();
        let close = TrailingMartingaleCloseParams {
            qty_pct: 0.1,
            threshold_base_pct: 0.01,
            threshold_we_weight: -0.005,
            retracement_base_pct: 0.002,
            ..make_close_params()
        };
        let pos = Position {
            size: 100.0,
            price: 100.0,
        };
        let trailing = TrailingPriceBundle {
            max_since_open: 102.0,
            min_since_max: 101.0,
            ..Default::default()
        };

        let closes = calc_closes_long(
            &exchange,
            &state,
            &bot,
            &make_runtime_context(),
            &close,
            &pos,
            &trailing,
        );

        assert_eq!(closes.len(), 1);
        assert_eq!(closes[0].order_type, OrderType::CloseTrailingLong);
        assert_eq!(closes[0].price, 100.3);
        assert_eq!(closes[0].qty, -10.0);
    }

    #[test]
    #[should_panic(expected = "calc_closes_long: non-finite close price")]
    fn test_sort_closes_by_price_rejects_non_finite_price() {
        let mut closes = vec![Order {
            qty: -1.0,
            price: f64::NAN,
            order_type: OrderType::CloseGridLong,
        }];
        sort_closes_by_price(&mut closes, false, "calc_closes_long");
    }
}

pub fn calc_closes_long(
    exchange_params: &ExchangeParams,
    state_params: &StateParams,
    bot_params: &BotParams,
    runtime_context: &RuntimeOrderContext,
    close_params: &TrailingMartingaleCloseParams,
    position: &Position,
    trailing_price_bundle: &TrailingPriceBundle,
) -> Vec<Order> {
    let mut closes = Vec::<Order>::new();
    let mut psize = position.size;
    for _ in 0..500 {
        let position_mod = Position {
            size: psize,
            price: position.price,
        };
        let mut close = calc_next_close_long(
            exchange_params,
            &state_params,
            bot_params,
            runtime_context,
            close_params,
            &position_mod,
            &trailing_price_bundle,
        );
        close.price = quantize_price(
            close.price,
            exchange_params.price_step,
            RoundingMode::Nearest,
            "calc_closes_long::price",
        );
        close.qty = quantize_qty(
            close.qty,
            exchange_params.qty_step,
            RoundingMode::Nearest,
            "calc_closes_long::qty",
        );
        if close.qty == 0.0 {
            break;
        }
        psize = round_(psize + close.qty, exchange_params.qty_step);
        if !closes.is_empty() {
            if close.order_type == OrderType::CloseTrailingLong {
                break;
            }
            if closes[closes.len() - 1].price == close.price {
                let previous_close = closes.pop();
                let merged_close = Order {
                    qty: round_(
                        previous_close
                            .expect("calc_closes_long merge requires existing close")
                            .qty
                            + close.qty,
                        exchange_params.qty_step,
                    ),
                    price: close.price,
                    order_type: close.order_type,
                };
                let mut merged_close = merged_close;
                merged_close.price = quantize_price(
                    merged_close.price,
                    exchange_params.price_step,
                    RoundingMode::Nearest,
                    "calc_closes_long::merged_price",
                );
                merged_close.qty = quantize_qty(
                    merged_close.qty,
                    exchange_params.qty_step,
                    RoundingMode::Nearest,
                    "calc_closes_long::merged_qty",
                );
                closes.push(merged_close);
                continue;
            }
        }
        closes.push(close);
    }
    sort_closes_by_price(&mut closes, false, "calc_closes_long");
    closes
}

pub fn calc_closes_short(
    exchange_params: &ExchangeParams,
    state_params: &StateParams,
    bot_params: &BotParams,
    runtime_context: &RuntimeOrderContext,
    close_params: &TrailingMartingaleCloseParams,
    position: &Position,
    trailing_price_bundle: &TrailingPriceBundle,
) -> Vec<Order> {
    let mut closes = Vec::<Order>::new();
    let mut psize = position.size;
    for _ in 0..500 {
        let position_mod = Position {
            size: psize,
            price: position.price,
        };
        let mut close = calc_next_close_short(
            exchange_params,
            &state_params,
            bot_params,
            runtime_context,
            close_params,
            &position_mod,
            &trailing_price_bundle,
        );
        close.price = quantize_price(
            close.price,
            exchange_params.price_step,
            RoundingMode::Nearest,
            "calc_closes_short::price",
        )
        .max(exchange_params.price_step);
        close.qty = quantize_qty(
            close.qty,
            exchange_params.qty_step,
            RoundingMode::Nearest,
            "calc_closes_short::qty",
        );
        if close.qty == 0.0 {
            break;
        }
        psize = round_(psize + close.qty, exchange_params.qty_step);
        if !closes.is_empty() {
            if close.order_type == OrderType::CloseTrailingShort {
                break;
            }
            if closes[closes.len() - 1].price == close.price {
                let previous_close = closes.pop();
                let merged_close = Order {
                    qty: round_(
                        previous_close
                            .expect("calc_closes_short merge requires existing close")
                            .qty
                            + close.qty,
                        exchange_params.qty_step,
                    ),
                    price: close.price,
                    order_type: close.order_type,
                };
                let mut merged_close = merged_close;
                merged_close.price = quantize_price(
                    merged_close.price,
                    exchange_params.price_step,
                    RoundingMode::Nearest,
                    "calc_closes_short::merged_price",
                )
                .max(exchange_params.price_step);
                merged_close.qty = quantize_qty(
                    merged_close.qty,
                    exchange_params.qty_step,
                    RoundingMode::Nearest,
                    "calc_closes_short::merged_qty",
                );
                closes.push(merged_close);
                continue;
            }
        }
        closes.push(close);
    }
    sort_closes_by_price(&mut closes, true, "calc_closes_short");
    closes
}

#[cfg(test)]
mod tp_adjustment_tests {
    use super::*;
    use crate::types::{CloseDecayInputs, OrderBook};

    fn exchange() -> ExchangeParams {
        ExchangeParams {
            qty_step: 0.001,
            price_step: 0.01,
            min_qty: 0.0,
            min_cost: 0.0,
            c_mult: 1.0,
            ..Default::default()
        }
    }

    fn state(ask: f64, bid: f64) -> StateParams {
        StateParams {
            balance: 1000.0,
            order_book: OrderBook {
                ask,
                bid,
                ..Default::default()
            },
            ..Default::default()
        }
    }

    /// 5% take profit, full-position close, WEL 1.0 (no TWEL/n_positions -> runtime base).
    fn close_params() -> TrailingMartingaleCloseParams {
        TrailingMartingaleCloseParams {
            qty_pct: 1.0,
            threshold_base_pct: 0.05,
            ..Default::default()
        }
    }

    fn bot() -> BotParams {
        BotParams {
            wallet_exposure_limit: 1.0,
            risk_we_excess_allowance_pct: 0.0,
            risk_wel_enforcer_enabled: false,
            ..Default::default()
        }
    }

    fn ctx(age_minutes: Option<f64>, above_sma20: Option<bool>) -> RuntimeOrderContext {
        RuntimeOrderContext {
            effective_wallet_exposure_limit: 1.0,
            entry_reference_price: None,
            close_decay: CloseDecayInputs {
                position_age_minutes: age_minutes,
                price_above_sma20: above_sma20,
            },
        }
    }

    fn long_tp(b: &BotParams, c: &RuntimeOrderContext, size: f64, ask: f64) -> f64 {
        calc_next_close_long(
            &exchange(),
            &state(ask, ask),
            b,
            c,
            &close_params(),
            &Position { size, price: 100.0 },
            &TrailingPriceBundle::default(),
        )
        .price
    }

    #[test]
    fn linear_progress_bounds_and_disable() {
        assert_eq!(tp_decay_linear_progress(0.5, 1.0, 2.0), 0.0);
        assert_eq!(tp_decay_linear_progress(3.0, 1.0, 2.0), 1.0);
        assert!((tp_decay_linear_progress(1.5, 1.0, 2.0) - 0.5).abs() < 1e-12);
        assert_eq!(tp_decay_linear_progress(100.0, 0.0, 0.0), 0.0);
        assert_eq!(tp_decay_linear_progress(2.0, 2.0, 1.0), 1.0);
        assert_eq!(tp_decay_linear_progress(1.9, 2.0, 1.0), 0.0);
        assert_eq!(tp_decay_linear_progress(f64::NAN, 1.0, 2.0), 0.0);
    }

    #[test]
    fn apply_tp_decay_moves_toward_floor_and_never_raises() {
        assert_eq!(apply_tp_decay(0.05, 0.0, -0.005), 0.05);
        assert!((apply_tp_decay(0.05, 1.0, -0.005) + 0.005).abs() < 1e-12);
        assert!((apply_tp_decay(0.05, 0.5, -0.005) - 0.0225).abs() < 1e-12);
        assert_eq!(apply_tp_decay(-0.01, 1.0, -0.005), -0.01);
        assert_eq!(apply_tp_decay(0.05, f64::NAN, -0.005), 0.05);
    }

    #[test]
    fn default_params_leave_tp_unchanged() {
        let price = long_tp(&bot(), &ctx(Some(100_000.0), Some(true)), 1.0, 50.0);
        assert!((price - 105.0).abs() < 0.011, "price={price}");
    }

    #[test]
    fn time_decay_takes_long_tp_below_entry_at_floor() {
        let mut b = bot();
        b.tp_decay_floor_pct = -0.005;
        b.tp_decay_time_full_hours = 10.0;
        let half = long_tp(&b, &ctx(Some(300.0), None), 1.0, 50.0);
        assert!((half - 102.25).abs() < 0.011, "half={half}");
        let full = long_tp(&b, &ctx(Some(600.0), None), 1.0, 50.0);
        assert!((full - 99.5).abs() < 0.011, "full={full}");
    }

    #[test]
    fn unknown_age_never_decays() {
        let mut b = bot();
        b.tp_decay_floor_pct = -0.005;
        b.tp_decay_time_full_hours = 10.0;
        let price = long_tp(&b, &ctx(None, None), 1.0, 50.0);
        assert!((price - 105.0).abs() < 0.011, "price={price}");
    }

    #[test]
    fn exposure_decay_uses_nominal_twel_over_n_positions() {
        let mut b = bot();
        b.total_wallet_exposure_limit = 2.0;
        b.n_positions = 2;
        b.tp_decay_floor_pct = -0.005;
        b.tp_decay_we_start_ratio = 0.5;
        b.tp_decay_we_full_ratio = 1.0;
        // WE 0.8 over nominal base 1.0 -> factor 0.6 -> 0.05 + 0.6*(-0.055) = 0.017
        let price = long_tp(&b, &ctx(None, None), 8.0, 50.0);
        assert!((price - 101.7).abs() < 0.011, "price={price}");
        // a divergence-capped runtime WEL does not move the trigger
        let mut capped = ctx(None, None);
        capped.effective_wallet_exposure_limit = 0.2;
        let threshold = calc_close_threshold_pct(
            &exchange(),
            &state(50.0, 50.0),
            &b,
            &capped,
            &close_params(),
            &Position {
                size: 8.0,
                price: 100.0,
            },
        );
        assert!((threshold - 0.017).abs() < 1e-9, "threshold={threshold}");
    }

    #[test]
    fn decayed_tp_never_goes_below_the_ask() {
        let mut b = bot();
        b.tp_decay_floor_pct = -0.5;
        b.tp_decay_time_full_hours = 1.0;
        let price = long_tp(&b, &ctx(Some(600.0), None), 1.0, 99.9);
        assert!((price - 99.9).abs() < 0.011, "price={price}");
    }

    #[test]
    fn short_tp_decays_above_entry() {
        let mut b = bot();
        b.tp_decay_floor_pct = -0.005;
        b.tp_decay_time_full_hours = 10.0;
        let order = calc_next_close_short(
            &exchange(),
            &state(200.0, 200.0),
            &b,
            &ctx(Some(600.0), None),
            &close_params(),
            &Position {
                size: -1.0,
                price: 100.0,
            },
            &TrailingPriceBundle::default(),
        );
        assert!((order.price - 100.5).abs() < 0.011, "price={}", order.price);
    }

    #[test]
    fn sma20_gate_decays_only_below_and_unknown_stays_closed() {
        let mut b = bot();
        b.tp_decay_floor_pct = -0.005;
        b.tp_decay_time_full_hours = 1.0;
        b.tp_decay_gate_sma20 = true;
        assert!((long_tp(&b, &ctx(Some(600.0), Some(false)), 1.0, 50.0) - 99.5).abs() < 0.011);
        assert!((long_tp(&b, &ctx(Some(600.0), Some(true)), 1.0, 50.0) - 105.0).abs() < 0.011);
        assert!((long_tp(&b, &ctx(Some(600.0), None), 1.0, 50.0) - 105.0).abs() < 0.011);
    }

    #[test]
    fn sma20_boost_scales_positive_tp_only_above() {
        let mut b = bot();
        b.tp_sma20_above_boost = 0.33;
        let above = long_tp(&b, &ctx(None, Some(true)), 1.0, 50.0);
        assert!(above > 106.6 && above < 106.7, "above={above}");
        assert!((long_tp(&b, &ctx(None, Some(false)), 1.0, 50.0) - 105.0).abs() < 0.011);
        assert!((long_tp(&b, &ctx(None, None), 1.0, 50.0) - 105.0).abs() < 0.011);
        assert_eq!(
            adjust_close_threshold_pct(-0.01, &b, &ctx(None, Some(true)).close_decay, 0.0, 1.0),
            -0.01
        );
    }

    #[test]
    fn exposure_decay_turns_the_close_into_a_ladder() {
        // With a WE-driven decay the threshold changes as the position shrinks, so the
        // full-position shortcut must not apply.
        let mut b = bot();
        b.tp_decay_floor_pct = 0.0;
        b.tp_decay_we_start_ratio = 0.0;
        b.tp_decay_we_full_ratio = 1.0;
        let mut params = close_params();
        params.qty_pct = 0.25;
        let closes = calc_closes_long(
            &exchange(),
            &state(50.0, 50.0),
            &b,
            &ctx(None, None),
            &params,
            &Position {
                size: 8.0,
                price: 100.0,
            },
            &TrailingPriceBundle::default(),
        );
        assert!(closes.len() > 1, "closes={closes:?}");
        assert!(closes.windows(2).all(|w| w[0].price <= w[1].price));
        let total: f64 = closes.iter().map(|o| -o.qty).sum();
        assert!((total - 8.0).abs() < 1e-9, "total={total}");
    }

    fn offload_bot() -> BotParams {
        let mut b = bot();
        b.risky_offload_we_threshold = 0.5;
        b.risky_offload_close_pct = 0.3;
        b.risky_offload_markup_pct = 0.4;
        b
    }

    fn offload(b: &BotParams, size: f64, ask: f64) -> Option<Order> {
        calc_risky_offload_close(
            StrategySide::Long,
            &exchange(),
            &state(ask, ask),
            b,
            &ctx(None, None),
            &close_params(),
            &Position { size, price: 100.0 },
        )
    }

    #[test]
    fn offload_needs_exposure_above_the_trigger() {
        assert!(offload(&offload_bot(), 5.0, 50.0).is_none()); // WE ratio 0.5, not above
        assert!(offload(&bot(), 8.0, 50.0).is_none()); // disabled
        let order = offload(&offload_bot(), 8.0, 50.0).unwrap();
        assert_eq!(order.order_type, OrderType::CloseRiskyOffloadLong);
        assert!((order.qty + 2.4).abs() < 1e-9, "qty={}", order.qty);
        // 40% of the 5% take-profit distance
        assert!((order.price - 102.0).abs() < 0.011, "price={}", order.price);
    }

    #[test]
    fn offload_never_below_ask_and_short_mirrors() {
        let order = offload(&offload_bot(), 8.0, 103.0).unwrap();
        assert!((order.price - 103.0).abs() < 1e-9);
        let short = calc_risky_offload_close(
            StrategySide::Short,
            &exchange(),
            &state(200.0, 200.0),
            &offload_bot(),
            &ctx(None, None),
            &close_params(),
            &Position {
                size: -8.0,
                price: 100.0,
            },
        )
        .unwrap();
        assert_eq!(short.order_type, OrderType::CloseRiskyOffloadShort);
        assert!((short.qty - 2.4).abs() < 1e-9 && (short.price - 98.0).abs() < 0.011);
    }

    #[test]
    fn offload_skips_non_positive_tp_and_sub_minimum_slices() {
        let mut b = offload_bot();
        b.tp_decay_floor_pct = -0.01;
        b.tp_decay_time_full_hours = 1.0;
        let decayed = calc_risky_offload_close(
            StrategySide::Long,
            &exchange(),
            &state(50.0, 50.0),
            &b,
            &ctx(Some(600.0), None),
            &close_params(),
            &Position {
                size: 8.0,
                price: 100.0,
            },
        );
        assert!(decayed.is_none());
        let mut ex = exchange();
        ex.min_cost = 1_000.0;
        let small = calc_risky_offload_close(
            StrategySide::Long,
            &ex,
            &state(50.0, 50.0),
            &offload_bot(),
            &ctx(None, None),
            &close_params(),
            &Position {
                size: 8.0,
                price: 100.0,
            },
        );
        assert!(small.is_none());
    }
}
