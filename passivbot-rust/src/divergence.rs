//! Cross-asset outlier protection shared by live and backtest.

use crate::types::BotParams;

pub const HORIZONS_MINUTES: [usize; 6] = [5, 15, 60, 240, 1440, 4320];
/// The first horizons are always evaluated; the remaining 1d/3d horizons only count for
/// params with `divergence_extended_horizons` (they keep a multi-day collapse flagged after
/// short horizons normalize).
pub const BASE_HORIZON_COUNT: usize = 4;
pub type DivergenceRocs = [Option<f64>; 6];

#[derive(Clone, Copy, Debug, PartialEq)]
pub struct DivergenceEffect {
    pub wallet_exposure_factor: f64,
    pub delay_multiplier: f64,
    pub flagged_timeframes: usize,
}

impl Default for DivergenceEffect {
    fn default() -> Self {
        Self {
            wallet_exposure_factor: 1.0,
            delay_multiplier: 1.0,
            flagged_timeframes: 0,
        }
    }
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
    for tf in 0..HORIZONS_MINUTES.len() {
        let values: Vec<(usize, f64)> = roc_by_coin
            .iter()
            .enumerate()
            .filter_map(|(idx, row)| row[tf].filter(|v| v.is_finite()).map(|v| (idx, v)))
            .collect();
        if values.len() < 3 {
            continue;
        }
        let n = values.len() as f64;
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
            let breadth_pct = values
                .iter()
                .filter(|(_, other)| {
                    if short {
                        *other > p.divergence_breadth_drop_pct
                    } else {
                        *other < -p.divergence_breadth_drop_pct
                    }
                })
                .count() as f64
                / n
                * 100.0;
            if breadth_pct >= p.divergence_breadth_threshold_pct {
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
        if !p.divergence_filter_enabled || counts[idx] < p.divergence_min_timeframes {
            continue;
        }
        let intensity = if p.divergence_zscore_threshold > 0.0 {
            (worst_z[idx] / p.divergence_zscore_threshold - 1.0).clamp(0.0, 1.0)
        } else {
            1.0
        };
        effect.flagged_timeframes = counts[idx];
        if p.divergence_we_cap_pct > 0.0 && p.divergence_we_cap_pct < 1.0 {
            effect.wallet_exposure_factor = 1.0 + intensity * (p.divergence_we_cap_pct - 1.0);
        }
        if p.divergence_delay_multiplier > 1.0 {
            effect.delay_multiplier = 1.0 + intensity * (p.divergence_delay_multiplier - 1.0);
        }
    }
    effects
}

#[cfg(test)]
mod tests {
    use super::*;

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
        assert_eq!(detect(&broad, &refs, false)[0], DivergenceEffect::default());
        let missing = [[Some(-100.0); 6], [Some(0.0); 6], [None; 6]];
        assert_eq!(
            detect(&missing, &refs, false)[0],
            DivergenceEffect::default()
        );
    }

    #[test]
    fn default_two_sigma_threshold_cannot_flag_a_lone_outlier_among_three() {
        let mut p = params();
        p.divergence_zscore_threshold = 2.0;
        let refs = vec![&p; 3];
        let rocs = [[Some(-99.0); 6], [Some(0.0); 6], [Some(0.0); 6]];
        assert_eq!(detect(&rocs, &refs, false)[0], DivergenceEffect::default());
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
        assert_eq!(detect(&rocs, &refs, false)[0], DivergenceEffect::default());

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
}
