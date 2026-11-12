# Market Crash, Risky Offload and TP Decay Contract

Three opt-in risk features for `trailing_martingale` (market crash also applies to every
strategy's entries). All parameters live under `bot.<side>.risk`; defaults disable them. Rust
owns the behavior (`market_crash.rs`, `closes.rs`, `orchestrator.rs`, `backtest.rs`); Python
only supplies reconstructed facts.

## Market crash

1. `market_crash_timeframe_minutes` (`0` = off; `5`, `15`, `60` or `240`) selects one ROC horizon.
   A step is active when at least `market_crash_breadth_threshold_pct` percent of the coins with
   a valid ROC moved against the side by more than `market_crash_min_drop_pct` percent. Fewer than
   three valid coins means "cannot evaluate", which counts as inactive.
2. Severity is `(share - threshold) / (100 - threshold)` in `[0, 1]`. After the last active step it
   decays linearly to zero over `market_crash_cooldown_minutes`.
3. Effects while severity is above zero: `market_crash_freeze_initial_entries` blocks initial
   entries of flat coins; `market_crash_we_cap_pct` (`0 < cap < 1`) shrinks the WEL used by entries
   only, by `1 + severity * (cap - 1)`. A held position's budget, closes, unstuck and exposure
   enforcers keep the base WEL, so the feature never forces exits at the bottom.
4. Statelessness: live replays `market_crash_cooldown_minutes + 2` minutes of completed 1m ROC
   history through `market_crash_severity_json` each cycle (shared with the divergence history).
   Missing history gives severity `0` with a warning. The backtest advances the same state machine
   per candle.

## Risky offload

1. While `WE / WEL > risky_offload_we_threshold` (`0` = off) and `risky_offload_close_pct > 0`, one
   extra reduce-only limit close of that fraction of the position rests at
   `risky_offload_markup_pct` times the strategy take-profit distance (order type
   `close_risky_offload_<side>`).
2. It is emitted every decision in live and backtest, not only when the ordinary close is about to
   fill. It is skipped, never enlarged, when below the executable minimum or when the take-profit
   distance is not positive.
3. Closes are trimmed to the position after generation (`trim_closes_to_position`): the furthest
   close, normally the offload, is reduced first, so offload plus take-profit never exceed the
   position.

## TP decay

1. The close threshold moves toward `tp_decay_floor_pct` (may be negative; never raises a
   threshold already below it) by `factor = max(time, exposure)`. Time ramps between
   `tp_decay_time_start_hours` and `tp_decay_time_full_hours` of position age; exposure ramps
   between `tp_decay_we_start_ratio` and `tp_decay_we_full_ratio` of `WE / (TWEL / n_positions)`,
   which does not move with the divergence cap. A `full` of `0` disables that trigger; `full <=
   start` is a step.
2. `tp_decay_gate_sma20` applies decay only while price is below the SMA of the last 20 completed
   UTC daily closes. `tp_sma20_above_boost` multiplies positive thresholds by `1 + boost` while
   price is above it.
3. Age is the timestamp of the fill that opened the current episode, from
   `time_stop.reconstruct_episodes` over the canonical fill stream; no local clock exists.
   Unproven coverage gives age unknown. An unknown age contributes no time decay, an unknown SMA20
   keeps the gate closed and disables the boost. Nothing is replaced by a neutral guess.
4. The daily close is the close of the last 1m candle at or before 23:59 UTC of that day, carried
   forward only within the day (`close_decay_inputs.daily_sma20_from_candles`, same as the
   backtest). The current day never enters the average. The per-day memo only avoids refetching.

## Optimization

GPU screening models none of these features: it warns, ranks without them and pins their bounds
(`market_crash_`, `risky_offload_`, `tp_decay_`, `tp_sma20_` prefixes) to configured values; exact
CPU validation applies them. Their bounds are optional and have no template defaults.

## Code and tests

Rust: `market_crash.rs`, `closes.rs`, `orchestrator.rs` (`close_decay_inputs`), `backtest.rs`.
Python: `src/close_decay_inputs.py`, `Passivbot._get_position_open_timestamps`, the crash block
in `Passivbot` orchestrator-input building. Tests: `tests/test_close_decay_inputs.py`,
`tests/test_orchestrator_json_api.py`, Rust unit tests in the modules above.
