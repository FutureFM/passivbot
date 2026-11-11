//! Market-wide adverse-move ("crash") protection shared by live and backtest.
//!
//! The detector is cross-sectional and stateless per step: the share of coins whose ROC over
//! one horizon moved against the position side beyond a threshold. The only memory is a
//! linear cooldown from the last active step, which depends on detector history inside
//! `market_crash_cooldown_minutes`; live rebuilds it from completed candles each cycle.

use crate::divergence::{DivergenceRocs, BASE_HORIZON_COUNT, HORIZONS_MINUTES};
use crate::types::BotParams;

/// Severities at or below this are treated as an expired cooldown.
const EXPIRED_SEVERITY: f64 = 0.001;

/// ROC slot of `market_crash_timeframe_minutes` (5, 15, 60 or 240), if enabled.
pub fn timeframe_slot(p: &BotParams) -> Option<usize> {
    if !(p.market_crash_timeframe_minutes > 0.0) {
        return None;
    }
    HORIZONS_MINUTES
        .iter()
        .take(BASE_HORIZON_COUNT)
        .position(|&h| h as f64 == p.market_crash_timeframe_minutes)
}

/// True when the detector runs for these (global) side params.
pub fn enabled(p: &BotParams) -> bool {
    timeframe_slot(p).is_some()
}

/// Instantaneous breadth test. `None` when it cannot be evaluated (disabled, or fewer than
/// three coins with a valid ROC). Otherwise `(active, severity)`: active when the adverse
/// share reaches the breadth threshold; severity rescales the share between the threshold
/// and 100% to `[0, 1]`, so a share exactly at the threshold is active with severity 0.
pub fn raw_severity(rocs: &[DivergenceRocs], p: &BotParams, short: bool) -> Option<(bool, f64)> {
    let slot = timeframe_slot(p)?;
    let values: Vec<f64> = rocs
        .iter()
        .filter_map(|row| row[slot].filter(|v| v.is_finite()))
        .collect();
    if values.len() < 3 {
        return None;
    }
    let min_drop = p.market_crash_min_drop_pct;
    let adverse = values
        .iter()
        .filter(|&&roc| {
            if short {
                roc > min_drop
            } else {
                roc < -min_drop
            }
        })
        .count();
    let adverse_pct = adverse as f64 / values.len() as f64 * 100.0;
    let threshold = p.market_crash_breadth_threshold_pct;
    if adverse_pct < threshold {
        return Some((false, 0.0));
    }
    let severity = (adverse_pct - threshold) / (100.0 - threshold).max(1e-10);
    Some((true, severity.clamp(0.0, 1.0)))
}

/// Cooldown memory: the last active severity and the minutes elapsed since it stopped.
#[derive(Clone, Copy, Debug, Default, PartialEq)]
pub struct CrashState {
    last_active_severity: f64,
    inactive_minutes: Option<f64>,
}

impl CrashState {
    /// Advance one step of `step_minutes` and return the severity in force.
    ///
    /// An active step sets the severity and re-arms the cooldown. Otherwise the last active
    /// severity decays linearly over `cooldown_minutes`, starting undecayed on the first
    /// inactive step. A step that cannot be evaluated counts as inactive, so missing data
    /// lets a running cooldown finish instead of releasing it early.
    pub fn update(
        &mut self,
        raw: Option<(bool, f64)>,
        cooldown_minutes: f64,
        step_minutes: f64,
    ) -> f64 {
        if let Some((true, severity)) = raw {
            self.last_active_severity = severity;
            self.inactive_minutes = None;
            return severity;
        }
        if !(cooldown_minutes > 0.0) || !(self.last_active_severity > 0.0) {
            *self = Self::default();
            return 0.0;
        }
        let elapsed = match self.inactive_minutes {
            None => 0.0,
            Some(minutes) => minutes + step_minutes.max(0.0),
        };
        self.inactive_minutes = Some(elapsed);
        let decayed = self.last_active_severity * (1.0 - elapsed / cooldown_minutes).max(0.0);
        if decayed > EXPIRED_SEVERITY {
            decayed
        } else {
            *self = Self::default();
            0.0
        }
    }
}

/// Minutes of contiguous 1m detector history that reproduce the state exactly.
pub fn required_history_minutes(p: &BotParams) -> usize {
    if !enabled(p) {
        return 0;
    }
    p.market_crash_cooldown_minutes.max(0.0).ceil() as usize + 2
}

/// Replay a contiguous 1m history (oldest first) and return the severity at the last step.
pub fn severity_from_history(history: &[Vec<DivergenceRocs>], p: &BotParams, short: bool) -> f64 {
    if !enabled(p) {
        return 0.0;
    }
    let mut state = CrashState::default();
    let mut severity = 0.0;
    for rocs in history {
        severity = state.update(
            raw_severity(rocs, p, short),
            p.market_crash_cooldown_minutes,
            1.0,
        );
    }
    severity
}

/// Entry-only wallet-exposure-limit factor: `1 + severity * (cap - 1)` for `0 < cap < 1`.
pub fn entry_wel_factor(p: &BotParams, severity: f64) -> f64 {
    let cap = p.market_crash_we_cap_pct;
    if severity > 0.0 && cap > 0.0 && cap < 1.0 {
        1.0 + severity.min(1.0) * (cap - 1.0)
    } else {
        1.0
    }
}

/// True when flat coins on this side must not open a position.
pub fn freezes_initial_entries(p: &BotParams, severity: f64) -> bool {
    severity > 0.0 && p.market_crash_freeze_initial_entries
}

#[derive(serde::Deserialize)]
#[serde(deny_unknown_fields)]
struct SeverityRequest {
    /// Contiguous 1m steps, oldest first; each step has one ROC row per coin.
    history: Vec<Vec<DivergenceRocs>>,
    long: BotParams,
    short: BotParams,
}

#[derive(serde::Serialize)]
struct SeverityResponse {
    long: f64,
    short: f64,
}

/// JSON transport for live: `{history, long, short}` (global side params) -> severities.
pub fn severity_json(raw: &str) -> Result<String, String> {
    let request: SeverityRequest =
        serde_json::from_str(raw).map_err(|e| format!("invalid market crash input: {e}"))?;
    let response = SeverityResponse {
        long: severity_from_history(&request.history, &request.long, false),
        short: severity_from_history(&request.history, &request.short, true),
    };
    serde_json::to_string(&response).map_err(|e| e.to_string())
}

#[pyo3::prelude::pyfunction]
pub fn market_crash_severity_json(raw: &str) -> pyo3::PyResult<String> {
    severity_json(raw).map_err(pyo3::exceptions::PyValueError::new_err)
}

#[cfg(test)]
mod tests {
    use super::*;

    fn params() -> BotParams {
        BotParams {
            market_crash_timeframe_minutes: 5.0,
            market_crash_min_drop_pct: 3.0,
            market_crash_breadth_threshold_pct: 50.0,
            market_crash_cooldown_minutes: 10.0,
            ..Default::default()
        }
    }

    fn step(five_minute_rocs: &[f64]) -> Vec<DivergenceRocs> {
        five_minute_rocs
            .iter()
            .map(|&roc| {
                let mut row: DivergenceRocs = [None; 6];
                row[0] = Some(roc);
                row
            })
            .collect()
    }

    #[test]
    fn disabled_or_thin_universe_is_not_evaluated() {
        let mut p = params();
        assert_eq!(raw_severity(&step(&[-5.0, -5.0]), &p, false), None);
        p.market_crash_timeframe_minutes = 0.0;
        assert_eq!(raw_severity(&step(&[-5.0, -5.0, -5.0]), &p, false), None);
        // only the four short horizons are supported
        p.market_crash_timeframe_minutes = 1440.0;
        assert!(!enabled(&p));
    }

    #[test]
    fn breadth_scales_severity_between_threshold_and_full() {
        let p = params();
        // 1 of 4 dropping: below the 50% threshold
        assert_eq!(
            raw_severity(&step(&[-5.0, 0.0, 0.0, 0.0]), &p, false),
            Some((false, 0.0))
        );
        // exactly at the threshold: active, severity 0 (no effect)
        assert_eq!(
            raw_severity(&step(&[-5.0, -5.0, 0.0, 0.0]), &p, false),
            Some((true, 0.0))
        );
        let (active, severity) = raw_severity(&step(&[-5.0, -5.0, -5.0, 0.0]), &p, false).unwrap();
        assert!(active && (severity - 0.5).abs() < 1e-12);
        assert_eq!(
            raw_severity(&step(&[-5.0, -5.0, -5.0, -5.0]), &p, false),
            Some((true, 1.0))
        );
    }

    #[test]
    fn the_adverse_direction_follows_the_position_side() {
        let p = params();
        let pump = step(&[5.0, 5.0, 5.0, 5.0]);
        assert_eq!(raw_severity(&pump, &p, false), Some((false, 0.0)));
        assert_eq!(raw_severity(&pump, &p, true), Some((true, 1.0)));
    }

    #[test]
    fn cooldown_decays_linearly_from_the_last_active_severity() {
        let p = params();
        let mut state = CrashState::default();
        assert_eq!(state.update(Some((true, 1.0)), 10.0, 1.0), 1.0);
        // a weaker active step replaces the severity the cooldown decays from
        assert_eq!(state.update(Some((true, 0.5)), 10.0, 1.0), 0.5);
        // first inactive step is undecayed, then 10% of the cooldown per minute
        assert_eq!(state.update(Some((false, 0.0)), 10.0, 1.0), 0.5);
        assert!((state.update(Some((false, 0.0)), 10.0, 1.0) - 0.45).abs() < 1e-12);
        // an unevaluated step keeps the cooldown running
        assert!((state.update(None, 10.0, 1.0) - 0.40).abs() < 1e-12);
        for _ in 0..7 {
            state.update(Some((false, 0.0)), 10.0, 1.0);
        }
        assert_eq!(state.update(Some((false, 0.0)), 10.0, 1.0), 0.0);
        assert_eq!(state, CrashState::default());
        assert_eq!(p.market_crash_cooldown_minutes, 10.0);
    }

    #[test]
    fn no_cooldown_releases_on_the_first_inactive_step() {
        let mut state = CrashState::default();
        assert_eq!(state.update(Some((true, 1.0)), 0.0, 1.0), 1.0);
        assert_eq!(state.update(Some((false, 0.0)), 0.0, 1.0), 0.0);
    }

    #[test]
    fn coarser_candles_advance_the_cooldown_in_minutes() {
        let mut state = CrashState::default();
        state.update(Some((true, 1.0)), 60.0, 15.0);
        assert_eq!(state.update(Some((false, 0.0)), 60.0, 15.0), 1.0);
        assert!((state.update(Some((false, 0.0)), 60.0, 15.0) - 0.75).abs() < 1e-12);
    }

    #[test]
    fn history_replay_matches_incremental_updates_from_any_covering_window() {
        let p = params();
        let calm = step(&[0.0, 0.0, 0.0, 0.0]);
        let crash = step(&[-5.0, -5.0, -5.0, -5.0]);
        let mut history = vec![calm.clone(); 40];
        history[20] = crash.clone();
        history[21] = crash;
        let window = required_history_minutes(&p);
        let mut state = CrashState::default();
        for end in 1..=history.len() {
            let incremental = state.update(
                raw_severity(&history[end - 1], &p, false),
                p.market_crash_cooldown_minutes,
                1.0,
            );
            let start = end.saturating_sub(window);
            let replayed = severity_from_history(&history[start..end], &p, false);
            assert!(
                (incremental - replayed).abs() < 1e-12,
                "step {end}: incremental {incremental} != replayed {replayed}"
            );
        }
    }

    #[test]
    fn effects_need_positive_severity() {
        let mut p = params();
        p.market_crash_freeze_initial_entries = true;
        p.market_crash_we_cap_pct = 0.5;
        assert!(!freezes_initial_entries(&p, 0.0));
        assert!(freezes_initial_entries(&p, 0.01));
        assert_eq!(entry_wel_factor(&p, 0.0), 1.0);
        assert!((entry_wel_factor(&p, 0.5) - 0.75).abs() < 1e-12);
        p.market_crash_we_cap_pct = 1.0;
        assert_eq!(entry_wel_factor(&p, 1.0), 1.0);
    }

    #[test]
    fn json_transport_returns_both_sides() {
        let p = params();
        let history = vec![step(&[-5.0, -5.0, -5.0, -5.0])];
        let request = serde_json::json!({ "history": history, "long": p, "short": p });
        let response: serde_json::Value =
            serde_json::from_str(&severity_json(&request.to_string()).unwrap()).unwrap();
        assert_eq!(response["long"], 1.0);
        assert_eq!(response["short"], 0.0);
        assert!(severity_json("{}").is_err());
    }
}
