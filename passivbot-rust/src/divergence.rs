//! Cross-asset outlier protection shared by live and backtest.

use std::collections::VecDeque;

use crate::types::BotParams;

pub const HORIZONS_MINUTES: [usize; 6] = [5, 15, 60, 240, 1440, 4320];
/// The first horizons are always evaluated; the remaining 1d/3d horizons only count for
/// params with `divergence_extended_horizons` (they keep a multi-day collapse flagged after
/// short horizons normalize).
pub const BASE_HORIZON_COUNT: usize = 4;
pub type DivergenceRocs = [Option<f64>; 6];
/// Breadth thresholds are calibrated for horizons up to this length. Longer horizons scale
/// the adverse-move threshold by sqrt(horizon / this), the random-walk growth of a typical
/// move, so an ordinary multi-day bear market is not mistaken for a market-wide shock.
const BREADTH_REFERENCE_MINUTES: f64 = 240.0;
/// Trailing window over which a flag must persist before `divergence_hold_minutes` latches it.
pub const PERSISTENCE_WINDOW_MINUTES: usize = 60;

fn breadth_drop_scale(horizon_minutes: usize) -> f64 {
    (horizon_minutes as f64 / BREADTH_REFERENCE_MINUTES)
        .sqrt()
        .max(1.0)
}

#[derive(Clone, Copy, Debug, PartialEq)]
pub struct DivergenceEffect {
    pub wallet_exposure_factor: f64,
    pub delay_multiplier: f64,
    pub flagged_timeframes: usize,
    /// Severity in [0, 1] driving both factors; 0 when the coin is not protected.
    pub intensity: f64,
    /// Share in [0, 1] of this coin's evaluated horizons suppressed by market-wide breadth
    /// (a broad adverse move for this side).
    pub market_stress: f64,
    /// Extra entry-cooldown minutes for held positions:
    /// `delay_add_minutes * intensity + market_delay_add_minutes * market_stress`.
    pub delay_add_minutes: f64,
}

impl Default for DivergenceEffect {
    fn default() -> Self {
        Self {
            wallet_exposure_factor: 1.0,
            delay_multiplier: 1.0,
            flagged_timeframes: 0,
            intensity: 0.0,
            market_stress: 0.0,
            delay_add_minutes: 0.0,
        }
    }
}

impl DivergenceEffect {
    /// True when the instantaneous detector protects this coin (enough horizons flagged).
    pub fn is_flagged(&self) -> bool {
        self.flagged_timeframes > 0
    }
}

fn effect_for(
    p: &BotParams,
    intensity: f64,
    flagged_timeframes: usize,
    market_stress: f64,
) -> DivergenceEffect {
    let mut effect = DivergenceEffect {
        flagged_timeframes,
        intensity,
        market_stress,
        ..Default::default()
    };
    effect.delay_add_minutes = p.divergence_delay_add_minutes.max(0.0) * intensity
        + p.divergence_market_delay_add_minutes.max(0.0) * market_stress;
    if p.divergence_we_cap_pct > 0.0 && p.divergence_we_cap_pct < 1.0 {
        effect.wallet_exposure_factor = 1.0 + intensity * (p.divergence_we_cap_pct - 1.0);
    }
    if p.divergence_delay_multiplier > 1.0 {
        effect.delay_multiplier = 1.0 + intensity * (p.divergence_delay_multiplier - 1.0);
    }
    effect
}

/// Combine the instantaneous effect with a held (latched) intensity: the stronger one wins.
/// A zero held intensity leaves the instantaneous effect unchanged.
pub fn with_held_intensity(raw: DivergenceEffect, held: f64, p: &BotParams) -> DivergenceEffect {
    if !p.divergence_filter_enabled || !(held > raw.intensity) {
        return raw;
    }
    effect_for(p, held.min(1.0), raw.flagged_timeframes, raw.market_stress)
}

/// `roc_by_coin` contains percentage changes aligned to HORIZONS_MINUTES.
/// A horizon with fewer than three valid coins cannot activate protection.
pub fn detect(
    roc_by_coin: &[DivergenceRocs],
    params: &[&BotParams],
    short: bool,
) -> Vec<DivergenceEffect> {
    let mut effects = vec![DivergenceEffect::default(); roc_by_coin.len()];
    if roc_by_coin.len() != params.len() || !params.iter().any(|p| p.divergence_filter_enabled) {
        return effects;
    }
    let mut counts = vec![0usize; roc_by_coin.len()];
    let mut worst_z = vec![0.0f64; roc_by_coin.len()];
    let mut evaluated = vec![0usize; roc_by_coin.len()];
    let mut suppressed = vec![0usize; roc_by_coin.len()];
    // Breadth only depends on the (usually shared) per-coin adverse threshold; compute it once
    // per distinct threshold instead of once per coin.
    let mut breadth_cache: Vec<(f64, f64)> = Vec::new();
    for tf in 0..HORIZONS_MINUTES.len() {
        let values: Vec<(usize, f64)> = roc_by_coin
            .iter()
            .enumerate()
            .filter_map(|(idx, row)| row[tf].filter(|v| v.is_finite()).map(|v| (idx, v)))
            .collect();
        if values.len() < 3 {
            continue;
        }
        breadth_cache.clear();
        let n = values.len() as f64;
        let drop_scale = breadth_drop_scale(HORIZONS_MINUTES[tf]);
        let mean = values.iter().map(|(_, roc)| roc).sum::<f64>() / n;
        let std = (values
            .iter()
            .map(|(_, roc)| (roc - mean).powi(2))
            .sum::<f64>()
            / n)
            .sqrt()
            .max(1e-10);
        for &(idx, roc) in &values {
            let p = params[idx];
            if !p.divergence_filter_enabled
                || (tf >= BASE_HORIZON_COUNT && !p.divergence_extended_horizons)
            {
                continue;
            }
            let breadth_drop_pct = p.divergence_breadth_drop_pct * drop_scale;
            let breadth_pct = match breadth_cache
                .iter()
                .find(|(threshold, _)| *threshold == breadth_drop_pct)
            {
                Some(&(_, pct)) => pct,
                None => {
                    let pct = values
                        .iter()
                        .filter(|(_, other)| {
                            if short {
                                *other > breadth_drop_pct
                            } else {
                                *other < -breadth_drop_pct
                            }
                        })
                        .count() as f64
                        / n
                        * 100.0;
                    breadth_cache.push((breadth_drop_pct, pct));
                    pct
                }
            };
            evaluated[idx] += 1;
            if breadth_pct >= p.divergence_breadth_threshold_pct {
                suppressed[idx] += 1;
                continue;
            }
            let z = (roc - mean) / std;
            if (short && z > p.divergence_zscore_threshold)
                || (!short && z < -p.divergence_zscore_threshold)
            {
                counts[idx] += 1;
                worst_z[idx] = worst_z[idx].max(z.abs());
            }
        }
    }
    for (idx, effect) in effects.iter_mut().enumerate() {
        let p = params[idx];
        if !p.divergence_filter_enabled {
            continue;
        }
        let market_stress = if evaluated[idx] > 0 {
            suppressed[idx] as f64 / evaluated[idx] as f64
        } else {
            0.0
        };
        if counts[idx] < p.divergence_min_timeframes {
            *effect = effect_for(p, 0.0, 0, market_stress);
            continue;
        }
        let intensity = if p.divergence_zscore_threshold > 0.0 {
            (worst_z[idx] / p.divergence_zscore_threshold - 1.0).clamp(0.0, 1.0)
        } else {
            1.0
        };
        *effect = effect_for(p, intensity, counts[idx], market_stress);
    }
    effects
}

/// Convert a duration in minutes to whole bars of `interval_minutes`, rounding up.
pub fn minutes_to_bars(minutes: f64, interval_minutes: f64) -> usize {
    if !(minutes > 0.0) || !(interval_minutes > 0.0) {
        return 0;
    }
    (minutes / interval_minutes).ceil() as usize
}

/// Per-coin, per-side latch for `divergence_hold_minutes`.
///
/// A flag that is present on at least `divergence_persistence_pct` of the trailing
/// `PERSISTENCE_WINDOW_MINUTES` latches the window's strongest intensity for
/// `divergence_hold_minutes`. The held value at step `t` is the maximum latched value
/// over `(t - hold, t]`. State depends only on the detector history inside
/// `window + hold`, so live recomputes it from candles after any restart.
#[derive(Clone, Debug, Default)]
pub struct HoldState {
    flags: VecDeque<bool>,
    flagged_count: usize,
    /// Monotonic (decreasing) intensities of flagged steps inside the persistence window.
    window_max: VecDeque<(u64, f64)>,
    /// Monotonic (decreasing) latched intensities inside the hold window.
    held: VecDeque<(u64, f64)>,
}

impl HoldState {
    /// Record step `step` (consecutive steps of one bar each) and return the held intensity.
    pub fn update(
        &mut self,
        step: u64,
        raw: &DivergenceEffect,
        persistence_pct: f64,
        window_bars: usize,
        hold_bars: usize,
    ) -> f64 {
        if window_bars == 0 || hold_bars == 0 {
            return 0.0;
        }
        let flagged = raw.is_flagged();
        self.flags.push_back(flagged);
        if flagged {
            self.flagged_count += 1;
            while self
                .window_max
                .back()
                .is_some_and(|&(_, v)| v <= raw.intensity)
            {
                self.window_max.pop_back();
            }
            self.window_max.push_back((step, raw.intensity));
        }
        while self.flags.len() > window_bars {
            if self.flags.pop_front() == Some(true) {
                self.flagged_count -= 1;
            }
        }
        while self
            .window_max
            .front()
            .is_some_and(|&(t, _)| t + window_bars as u64 <= step)
        {
            self.window_max.pop_front();
        }
        let persistent = self.flagged_count > 0
            && self.flagged_count as f64 * 100.0 >= persistence_pct * window_bars as f64;
        if persistent {
            if let Some(&(_, value)) = self.window_max.front() {
                while self.held.back().is_some_and(|&(_, v)| v <= value) {
                    self.held.pop_back();
                }
                self.held.push_back((step, value));
            }
        }
        while self
            .held
            .front()
            .is_some_and(|&(t, _)| t + hold_bars as u64 <= step)
        {
            self.held.pop_front();
        }
        self.held.front().map_or(0.0, |&(_, v)| v)
    }
}

/// True when any coin of this side has a hold configured.
pub fn hold_enabled(params: &[&BotParams]) -> bool {
    params
        .iter()
        .any(|p| p.divergence_filter_enabled && p.divergence_hold_minutes > 0.0)
}

/// Advance every coin's hold state by one step and return the held intensities.
pub fn update_hold_states(
    states: &mut [HoldState],
    step: u64,
    raw: &[DivergenceEffect],
    params: &[&BotParams],
    interval_minutes: f64,
) -> Vec<f64> {
    states
        .iter_mut()
        .zip(raw.iter().zip(params.iter()))
        .map(|(state, (effect, p))| {
            if !p.divergence_filter_enabled {
                return 0.0;
            }
            state.update(
                step,
                effect,
                p.divergence_persistence_pct,
                minutes_to_bars(PERSISTENCE_WINDOW_MINUTES as f64, interval_minutes),
                minutes_to_bars(p.divergence_hold_minutes, interval_minutes),
            )
        })
        .collect()
}

/// Replay a contiguous 1m history (oldest first) and return each coin's held intensity at the
/// last step. Live uses this to rebuild the latch from candles; it matches the backtest state
/// whenever the history covers `PERSISTENCE_WINDOW_MINUTES + divergence_hold_minutes`.
pub fn held_intensity_from_history(
    history: &[Vec<DivergenceRocs>],
    params: &[&BotParams],
    short: bool,
) -> Vec<f64> {
    let mut states = vec![HoldState::default(); params.len()];
    let mut held = vec![0.0; params.len()];
    for (step, rocs) in history.iter().enumerate() {
        if rocs.len() != params.len() {
            continue;
        }
        let raw = detect(rocs, params, short);
        held = update_hold_states(&mut states, step as u64, &raw, params, 1.0);
    }
    held
}

#[derive(serde::Deserialize)]
#[serde(deny_unknown_fields)]
struct HeldIntensityRequest {
    /// Contiguous 1m steps, oldest first; each step has one ROC row per coin.
    history: Vec<Vec<DivergenceRocs>>,
    long: Vec<BotParams>,
    short: Vec<BotParams>,
}

#[derive(serde::Serialize)]
struct HeldIntensityResponse {
    long: Vec<f64>,
    short: Vec<f64>,
}

/// JSON transport for live: `{history, long, short}` -> `{long, short}` held intensities.
pub fn held_intensity_json(raw: &str) -> Result<String, String> {
    let request: HeldIntensityRequest =
        serde_json::from_str(raw).map_err(|e| format!("invalid divergence hold input: {e}"))?;
    let n = request.long.len();
    if request.short.len() != n || request.history.iter().any(|step| step.len() != n) {
        return Err("divergence hold input rows must match the number of coins".into());
    }
    let long: Vec<&BotParams> = request.long.iter().collect();
    let short: Vec<&BotParams> = request.short.iter().collect();
    let response = HeldIntensityResponse {
        long: if hold_enabled(&long) {
            held_intensity_from_history(&request.history, &long, false)
        } else {
            vec![0.0; n]
        },
        short: if hold_enabled(&short) {
            held_intensity_from_history(&request.history, &short, true)
        } else {
            vec![0.0; n]
        },
    };
    serde_json::to_string(&response).map_err(|e| e.to_string())
}

#[pyo3::prelude::pyfunction]
pub fn divergence_held_intensity_json(raw: &str) -> pyo3::PyResult<String> {
    held_intensity_json(raw).map_err(pyo3::exceptions::PyValueError::new_err)
}

#[cfg(test)]
mod tests {
    use super::*;

    fn assert_unprotected(effect: DivergenceEffect) {
        assert_eq!(effect.flagged_timeframes, 0);
        assert_eq!(effect.wallet_exposure_factor, 1.0);
        assert_eq!(effect.delay_multiplier, 1.0);
        assert_eq!(effect.delay_add_minutes, 0.0);
    }

    fn params() -> BotParams {
        BotParams {
            divergence_filter_enabled: true,
            divergence_zscore_threshold: 1.0,
            divergence_breadth_threshold_pct: 40.0,
            divergence_breadth_drop_pct: 1.0,
            divergence_delay_multiplier: 3.0,
            divergence_we_cap_pct: 0.5,
            divergence_min_timeframes: 2,
            ..Default::default()
        }
    }

    fn flagged(intensity: f64) -> DivergenceEffect {
        DivergenceEffect {
            flagged_timeframes: 2,
            intensity,
            ..Default::default()
        }
    }

    #[test]
    fn hold_latches_only_persistent_flags() {
        let off = DivergenceEffect::default();
        // A lone flag in a 10-bar window never reaches 70% persistence.
        let mut state = HoldState::default();
        let mut held = 0.0;
        for step in 0..30u64 {
            let raw = if step % 10 == 0 { flagged(0.9) } else { off };
            held = state.update(step, &raw, 70.0, 10, 50);
        }
        assert_eq!(held, 0.0);

        // Seven of the last ten bars flagged latches the window maximum for the hold period.
        let mut state = HoldState::default();
        for step in 0..10u64 {
            let raw = if step < 3 {
                off
            } else {
                flagged(if step == 5 { 0.8 } else { 0.4 })
            };
            held = state.update(step, &raw, 70.0, 10, 50);
        }
        assert_eq!(held, 0.8);
        // The flag disappears but the latch holds until `hold` bars after the last latch.
        for step in 10..59u64 {
            held = state.update(step, &off, 70.0, 10, 50);
            assert!(held > 0.0, "released early at {step}");
        }
        // Last latching step was 12 (bars 3..=12 still >= 7 flags); released at 12 + 50.
        for step in 59..62u64 {
            held = state.update(step, &off, 70.0, 10, 50);
        }
        assert!(held > 0.0);
        held = state.update(62, &off, 70.0, 10, 50);
        assert_eq!(held, 0.0);
    }

    #[test]
    fn held_intensity_only_strengthens_protection() {
        let mut p = params();
        p.divergence_we_cap_pct = 0.2;
        p.divergence_delay_multiplier = 11.0;
        let raw = DivergenceEffect::default();
        let effect = with_held_intensity(raw, 0.5, &p);
        assert!((effect.wallet_exposure_factor - 0.6).abs() < 1e-12);
        assert!((effect.delay_multiplier - 6.0).abs() < 1e-12);
        // A stronger instantaneous signal is not weakened by a smaller held value.
        let strong = effect_for(&p, 0.9, 3, 0.0);
        assert_eq!(with_held_intensity(strong, 0.5, &p), strong);
        // Disabled sides ignore held values.
        p.divergence_filter_enabled = false;
        assert_eq!(with_held_intensity(raw, 1.0, &p), raw);
    }

    #[test]
    fn history_replay_matches_incremental_updates() {
        let mut p = params();
        p.divergence_hold_minutes = 30.0;
        p.divergence_persistence_pct = 50.0;
        let refs = vec![&p; 4];
        let crash = [Some(-50.0), Some(-50.0), None, None, None, None];
        let flat = [Some(0.0), Some(0.0), None, None, None, None];
        let history: Vec<Vec<DivergenceRocs>> = (0..120)
            .map(|t| {
                let first = if (20..80).contains(&t) { crash } else { flat };
                vec![first, flat, flat, flat]
            })
            .collect();
        let mut states = vec![HoldState::default(); 4];
        let mut incremental = vec![0.0; 4];
        for (step, rocs) in history.iter().enumerate() {
            let raw = detect(rocs, &refs, false);
            incremental = update_hold_states(&mut states, step as u64, &raw, &refs, 1.0);
        }
        let replay = held_intensity_from_history(&history, &refs, false);
        assert_eq!(incremental, replay);
        // The crash ended at step 79; a 30-minute hold keeps protection 40 steps later only
        // through the latch (the instantaneous detector is quiet).
        let short_history = &history[..100];
        let held = held_intensity_from_history(short_history, &refs, false);
        assert!(held[0] > 0.0);
        assert_eq!(held[1], 0.0);
        // Replaying only the trailing window + hold reproduces the full-history value.
        let tail = &history[100 - 90..100];
        assert_eq!(held_intensity_from_history(tail, &refs, false), held);
    }

    #[test]
    fn additive_delay_scales_with_intensity_and_market_stress() {
        let mut p = params();
        p.divergence_delay_add_minutes = 10.0;
        p.divergence_market_delay_add_minutes = 20.0;
        let refs = vec![&p; 3];
        // Isolated outlier: full intensity (|z| far above 2x threshold), no market stress.
        let isolated = [
            [Some(-100.0), Some(-100.0), None, None, None, None],
            [Some(0.0), Some(0.0), None, None, None, None],
            [Some(0.0), Some(0.0), None, None, None, None],
        ];
        let effect = detect(&isolated, &refs, false)[0];
        assert_eq!(effect.market_stress, 0.0);
        assert!((effect.delay_add_minutes - 10.0 * effect.intensity).abs() < 1e-12);
        // Market-wide drop: every evaluated horizon is suppressed by breadth for every coin.
        let broad = [[Some(-100.0); 6], [Some(-10.0); 6], [Some(-10.0); 6]];
        for effect in detect(&broad, &refs, false) {
            assert_eq!(effect.flagged_timeframes, 0);
            assert_eq!(effect.market_stress, 1.0);
            assert_eq!(effect.delay_add_minutes, 20.0);
            assert_eq!(effect.wallet_exposure_factor, 1.0);
        }
        // Long-side stress does not apply to the short side.
        assert_eq!(detect(&broad, &refs, true)[1].market_stress, 0.0);
        // Defaults add nothing.
        let base = params();
        let refs = vec![&base; 3];
        assert_eq!(detect(&broad, &refs, false)[1].delay_add_minutes, 0.0);
    }

    #[test]
    fn isolated_drop_and_pump_are_directional() {
        let p = params();
        let refs = vec![&p; 3];
        let drops = [
            [Some(-100.0), Some(-100.0), None, None, None, None],
            [Some(0.0), Some(0.0), None, None, None, None],
            [Some(0.0), Some(0.0), None, None, None, None],
        ];
        let long = detect(&drops, &refs, false);
        assert_eq!(long[0].flagged_timeframes, 2);
        assert!(long[0].wallet_exposure_factor < 1.0);
        assert!(long[0].delay_multiplier > 1.0);
        assert_eq!(detect(&drops, &refs, true)[0].flagged_timeframes, 0);
        let pumps = drops.map(|row| row.map(|v| v.map(|x| -x)));
        assert_eq!(detect(&pumps, &refs, true)[0].flagged_timeframes, 2);
    }

    #[test]
    fn broad_move_and_insufficient_coverage_do_not_trigger() {
        let p = params();
        let refs = vec![&p; 3];
        let broad = [[Some(-100.0); 6], [Some(-10.0); 6], [Some(-10.0); 6]];
        assert_unprotected(detect(&broad, &refs, false)[0]);
        let missing = [[Some(-100.0); 6], [Some(0.0); 6], [None; 6]];
        assert_unprotected(detect(&missing, &refs, false)[0]);
    }

    #[test]
    fn default_two_sigma_threshold_cannot_flag_a_lone_outlier_among_three() {
        let mut p = params();
        p.divergence_zscore_threshold = 2.0;
        let refs = vec![&p; 3];
        let rocs = [[Some(-99.0); 6], [Some(0.0); 6], [Some(0.0); 6]];
        assert_unprotected(detect(&rocs, &refs, false)[0]);
    }

    #[test]
    fn extended_horizons_count_only_when_enabled() {
        // A multi-day collapse whose short horizons have normalized (a small rebound).
        let collapse = [
            Some(1.0),
            Some(2.0),
            Some(0.5),
            Some(0.1),
            Some(-78.0),
            Some(-85.0),
        ];
        let flat = [Some(0.0); 6];
        let rocs = [collapse, flat, flat];
        let base = params();
        let refs = vec![&base; 3];
        assert_unprotected(detect(&rocs, &refs, false)[0]);

        let mut extended = params();
        extended.divergence_extended_horizons = true;
        let refs = vec![&extended; 3];
        let effect = detect(&rocs, &refs, false)[0];
        assert_eq!(effect.flagged_timeframes, 2);
        assert!(effect.wallet_exposure_factor < 1.0);
        // A single extended horizon does not satisfy a two-horizon minimum.
        let one_day_only = [
            [None, None, None, None, Some(-78.0), None],
            [None, None, None, None, Some(0.0), None],
            [None, None, None, None, Some(0.0), None],
        ];
        assert_eq!(detect(&one_day_only, &refs, false)[0].flagged_timeframes, 0);
    }

    #[test]
    fn breadth_threshold_scales_only_beyond_four_hours() {
        for h in [5, 15, 60, 240] {
            assert_eq!(breadth_drop_scale(h), 1.0);
        }
        assert!((breadth_drop_scale(1440) - 6.0f64.sqrt()).abs() < 1e-12);
        assert!((breadth_drop_scale(4320) - 18.0f64.sqrt()).abs() < 1e-12);
    }

    #[test]
    fn ordinary_multi_day_bear_market_does_not_suppress_extended_horizons() {
        // Market down ~8% over 1d and ~10% over 3d (typical bear week); one coin collapses.
        let mut p = params();
        p.divergence_breadth_drop_pct = 5.0;
        p.divergence_extended_horizons = true;
        p.divergence_zscore_threshold = 2.5;
        let mut rocs = vec![[None, None, None, None, Some(-8.0), Some(-10.0)]; 20];
        rocs[0] = [None, None, None, None, Some(-60.0), Some(-65.0)];
        let refs = vec![&p; 20];
        assert_eq!(detect(&rocs, &refs, false)[0].flagged_timeframes, 2);
        // A genuine market-wide shock beyond the scaled thresholds still suppresses them.
        let mut shock = vec![[None, None, None, None, Some(-25.0), Some(-35.0)]; 20];
        shock[0] = [None, None, None, None, Some(-90.0), Some(-95.0)];
        assert_eq!(detect(&shock, &refs, false)[0].flagged_timeframes, 0);
    }
}
