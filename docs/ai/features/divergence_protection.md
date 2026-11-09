# Cross-Asset Divergence Protection

## Contract

- The detector compares each coin's 5, 15, 60, and 240 minute close-to-close ROC, expressed in percent, against the population mean and population standard deviation for that horizon. With `divergence_extended_horizons`, 1440 and 4320 minute horizons are added for that side; they keep a multi-day collapse flagged after the short horizons normalize. Without it, those horizons are neither computed nor counted. For horizons longer than 240 minutes the `breadth_drop_pct` threshold is multiplied by `sqrt(horizon / 240)` (×2.45 at 1d, ×4.24 at 3d), the random-walk growth of a typical move, so an ordinary multi-day bear market does not suppress them; the 5–240 minute thresholds are unchanged.
- Each horizon requires at least three coins with valid closes at both aligned endpoints. Missing coverage yields no signal for that horizon. Live uses the latest completed 1m candle at a common timestamp; backtest uses the current historical decision candle.
- Long protection flags a negative outlier (`z < -threshold`); short protection flags a positive outlier (`z > threshold`). A horizon is excluded for a coin when at least `breadth_threshold_pct` of the valid universe moves in that side's adverse direction by more than `breadth_drop_pct`.
- `divergence_long_horizon_minutes` (0 = off; `1440` or `4320` only, reusing the 1d/3d ROCs) adds one robust market-relative horizon: `z = (ln(1 + roc/100) - median) / (1.4826 * MAD)` across coins with a valid ROC (at least three). The outlier cannot inflate its own denominator and a market-wide move cancels out; it is never muted by breadth and does not count toward `market_stress`. Long flags `z < -divergence_long_horizon_zscore`, short `z > divergence_long_horizon_zscore`; a flag counts as one more horizon toward `min_timeframes` and its `|z|` enters `worst_z`. Enabling it computes the 1d/3d ROCs (live fetches three days of 1m candles) without adding the mean/std 1d/3d tests unless `divergence_extended_horizons` is also set. Motivation: LUNA's 2022 slide coincided with a market-wide fall, so breadth muted the mean/std horizons.
- A coin is protected after at least `min_timeframes` horizons flag it. Severity is `clamp(abs(worst_z) / zscore_threshold - 1, 0, 1)`, or 1 when the configured threshold is zero.
- The effective per-symbol WEL becomes `base_WEL * (1 + severity * (we_cap_pct - 1))`. The base is recomputed each decision; the cap does not accumulate. Only `0 < we_cap_pct < 1` changes WEL. The reduced WEL governs entries (including the `ema_anchor` per-position entry cap) and time-stop triggers, but not the time-stop reduction cap. With `divergence_cap_entries_only` the reduced WEL governs only entries (initial and re-entries, min-cost eligibility, entry cooldown exposure ratio); the held position keeps the base WEL for strategy closes, the position-exposure (WEL) enforcer, unstuck and time-stop triggers, so the cap never forces exits at the bottom of a collapse. Entries and closes are then generated as separate strategy requests.
- For an existing position, the adaptive entry cooldown (`bot.<side>.entry_cooldown`) is multiplied by `1 + severity * (delay_multiplier - 1)` and capped by `entry_cooldown.max_duration_minutes` (1440 when unset), never below the adaptive duration. A zero cooldown stays zero. The divergence multiplier does not extend initial-entry cooldown when the position is flat. Closes and independent risk reducers remain available. GPU screening does not model divergence: it warns, pins divergence bounds to the configured values, and exact CPU validation of accepted results applies it.
- `divergence_hold_minutes` (default 0, off) latches persistent flags. Each decision candle, a coin's instantaneous protection counts as one flagged minute; when at least `divergence_persistence_pct` (default 70) of the trailing 60 minutes are flagged, the window's highest severity is latched. The held severity is the maximum latched value over the trailing `divergence_hold_minutes`, and the applied severity is `max(instantaneous, held)` — the latch only strengthens protection. The state depends only on detector history inside `60 + hold` minutes: the backtest updates it every candle (including warmup) and live replays that window from completed 1m candles each cycle through `divergence_held_intensity_json`, so restarts reproduce it without local state. Motivation: when a coin collapses during a market-wide sell-off, brief rebounds reset the short horizons and breadth suppresses the 1d/3d horizons, so the instantaneous flag flickers and entries refill the full WEL in the gaps.
- Additive delays (defaults 0): a held position's cooldown becomes `adaptive * delay_multiplier_effect + divergence_delay_add_minutes * severity + divergence_market_delay_add_minutes * market_stress`, capped by `entry_cooldown.max_duration_minutes` (1440 when unset) and never below the adaptive duration. `market_stress` is the share of the coin's evaluated horizons suppressed by the breadth rule (a market-wide adverse move for that side); it applies to all held positions on the side, including non-outliers. Unlike the multiplier, the additive terms act when the adaptive cooldown is zero. Flat positions are unaffected. Live fill lookbacks extend to the cooldown ceiling whenever an additive term is configured.
- Detection is disabled by default on both sides. A missing population never produces a protective signal, matching the PB7 backtest behavior. Live reports missing candle coverage. Failure-only exception in live: a symbol whose latest completed candle is unavailable (no ROC at all for the decision minute, typically a slow or failing candle API during a crash) keeps its last good ROC row and held intensity for up to 5 minutes (`keep_last_good_state`), then falls back to no signal. That fallback lives in memory only; a restart starts from candles, so it never changes decisions reproducible from candles when data is available. This does not predict a sudden crash or prevent orders already filled before the completed candle is available.

## Configuration

The fields live under `bot.<long|short>.risk`:

| Field | Default | Meaning |
|---|---:|---|
| `divergence_filter_enabled` | `false` | Enable the detector for this position side. |
| `divergence_zscore_threshold` | `2.0` | Outlier threshold. |
| `divergence_breadth_threshold_pct` | `40.0` | Market-wide adverse-move share that suppresses a horizon. |
| `divergence_breadth_drop_pct` | `1.0` | Adverse ROC threshold in percentage points. |
| `divergence_min_timeframes` | `2` | Required flagged horizons (1–4, or 1–6 with extended horizons). |
| `divergence_extended_horizons` | `false` | Add the 1440 and 4320 minute horizons. Live then fetches three days of 1m candles per symbol. |
| `divergence_delay_multiplier` | `1.0` | Full-severity entry-cooldown multiplier. |
| `divergence_we_cap_pct` | `1.0` | Full-severity WEL fraction. |
| `divergence_hold_minutes` | `0.0` | Minutes a persistent flag stays latched; 0 disables. |
| `divergence_persistence_pct` | `70.0` | Share of the trailing 60 minutes that must be flagged to latch. |
| `divergence_delay_add_minutes` | `0.0` | Held-position cooldown minutes added at full severity. |
| `divergence_market_delay_add_minutes` | `0.0` | Held-position cooldown minutes added at full market-wide stress. |
| `divergence_long_horizon_minutes` | `0.0` | Robust market-relative horizon: `0` (off), `1440` or `4320`. Adds one countable horizon (`min_timeframes` up to 5, or 7 with extended horizons). |
| `divergence_long_horizon_zscore` | `3.0` | Robust z threshold of the long horizon. |
| `divergence_cap_entries_only` | `false` | Apply `divergence_we_cap_pct` to new entries only; held positions keep the base WEL. |

With only three coins, the largest possible absolute population z-score for one isolated outlier is `sqrt(2) ≈ 1.414`. A threshold of 2.0 therefore needs at least six valid coins to flag a lone outlier; use a lower threshold if operating with only three to five coins.

To tune the six numeric fields, add only the desired keys under `optimize.bounds.<side>.risk`. Bounds are optional: fields without bounds retain their `bot.<side>.risk` values during optimization. `divergence_long_horizon_zscore` is also searchable. Set the booleans `divergence_filter_enabled`, `divergence_extended_horizons`, `divergence_cap_entries_only` and the discrete `divergence_long_horizon_minutes` in `bot.<side>.risk`; they are not optimizer bounds.

## Validation

- Rust unit tests cover isolated long drops, isolated short pumps, market-wide moves, fewer than three valid coins, extended horizons only counting when enabled, and the robust long horizon flagging a lone slide inside a falling market (directional, supported slots only).
- Real-extension orchestrator tests cover WEL capping, cooldown extension, short-side direction, the long horizon reducing entries, and `divergence_cap_entries_only` removing the WEL-enforcer reduction of a held position while still blocking entries above the capped WEL.
- Python tests cover common completed-candle alignment, missing coverage, and the last-good-state fallback (TTL, partial rows of recent listings).

## Code

- `passivbot-rust/src/divergence.rs`: shared detector and severity.
- `passivbot-rust/src/orchestrator.rs`: WEL and cooldown effects.
- `passivbot-rust/src/backtest.rs`: historical ROC preparation. Horizons are converted to backtest candles; a horizon that is not a whole number of candles (e.g. 5 minutes with 15 minute candles) yields no signal.
- `src/divergence_inputs.py`: live completed-candle ROC preparation and hold-latch ROC history.
