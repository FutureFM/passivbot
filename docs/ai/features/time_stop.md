# Time Stop Contract

## Clock and episode

1. `bot.<side>.risk.time_stop_max_age_days > 0` and `time_stop_close_pct > 0`
   enable the feature per coin and side. Defaults disable it.
2. The clock starts at a fill-proven flat-to-position boundary, or the fill that
   completes the most recent temporal reduction in that episode. Position increases,
   ordinary closes, balance changes, and WEL/divergence changes do not reset it.
3. The WE/WEL trigger is evaluated at the closing decision, using the effective
   runtime WEL and actual average entry price. It does not start or erase the clock.
   There is no negative-PnL prerequisite.
4. A submitted, rejected, cancelled, or incompletely executed temporal close does
   not grant a new interval. While enabled, an unfinished target continues independent of age,
   balance, or trigger changes. A flat position ends the episode.

## Quantity and execution

1. Zero percentage means no temporal action, including the minimum-exposure rule.
   Percentage one closes the full position and ignores the maximum-exposure cap.
   A positive minimum-exposure threshold can likewise request full flattening.
2. Other percentages reduce the current position fraction, capped when
   `0 < time_stop_close_we_max < 1` at that fraction of the runtime WEL before the
   divergence reduction, so divergence protection cannot shrink a temporal close.
   The trigger and minimum-exposure rule use the divergence-adjusted WEL.
   Zero and one disable the cap. Quantity is quantized to exchange steps.
3. A new sub-minimum partial request is skipped, not enlarged. An unfinished
   market fragment is completed with the executable minimum if necessary. Shared
   close sizing may absorb untradeable dust. These minimum rules can close more
   than the nominal percentage or exposure cap; aggregate quantity never exceeds
   the position.
4. Temporal closes are explicit reduce-only market orders even when
   `live.market_orders_allowed=false`. They bypass `max_realized_loss_pct`.
   HSL/panic has priority. A temporal close is otherwise exclusive for its pair,
   ahead of exposure/unstuck reducers and ordinary orders. Entries remain blocked
   until the target is complete. Manual mode is respected.
5. Partial fills reduce toward one fixed remaining-position target. The target is
   emitted by Rust after final executable sizing and encoded in the client order
   identifier, retaining broker attribution and exchange length/character limits.
   Retry orders carry the same target unless exchange minimum/dust sizing requires
   a further reduction. Ordinary/manual reductions can complete an active target;
   increases never reset the clock and must still be reduced to its fixed target.

## Reconstruction and unavailable evidence

Python replays the canonical manager-owned fill stream for the current episode,
confirms its after-state against the exchange position, and proves coverage from
before the opening. It shares this replay with entry fill counts. Rust consumes
an explicitly optional factual clock/target bundle and owns all trading decisions.

There is no authoritative local timer or pending-close latch. Current open orders
and historical temporal fills carry the target. Open-order creation timestamps
prove episode membership; stale orders cannot supply a new episode’s target. Full history, type/target
attribution, and position confirmation are required after restart; downtime counts.
Missing or ambiguous evidence defers temporal closes and entries for the affected
pair while independent closes remain available. Never use process start time as an
opening or relabel an unknown reducing fill as temporal. History repair retries with
bounded backoff. Exchange retention or unavailable client IDs can prevent recovery.

The usual client ID stores a binary64 target. Short broker-prefixed identifiers
use an exact decimal mantissa/exponent encoding; supported compact exponents are
-30 through 31. An unrepresentable target/identifier fails before submission rather
than truncating evidence. Existing broker prefixes are retained in both formats.

After a completed temporal partial close, grid entry geometry may reference its
last execution price, until a subsequent increasing fill. Real position average
price, PnL, exposure, close prices, and EMA-anchor behavior remain unchanged.

## Validation and code

- Rust calculator and backtest: `passivbot-rust/src/orchestrator.rs`, `backtest.rs`.
- Exchange evidence and identifiers: `src/time_stop.py`, `src/passivbot.py`,
  `src/fill_events_manager.py`, `src/live/reconciler.py`.
- Coverage, restart, percentage completion, long/short, priority, and producer
  contract regressions: `tests/test_time_stop.py`.
- Config and optimizer paths: `src/config/`; user reference: `../../time_stop.md`.
