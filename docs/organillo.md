# Historical coin selection (Organillo)

Backtesting and optimization can import a daily coin eligibility matrix produced by an external
selection tool. PB8 prepares candles for the union of coins selected during the requested period.
Forager then ranks the eligible coins using its existing parameters and position limits. The
full data union does not make future selections available to current trading decisions.

Enable these settings in an existing strategy config:

```json
{
  "backtest": {
    "organillo_mode": true,
    "organillo_carton_path": "configs/examples/historical_selection.csv",
    "start_date": "2025-01-01",
    "end_date": "2025-01-03"
  }
}
```

This is a config fragment; retain the strategy, exchanges, fees and other settings from your config.
The intentional example `configs/examples/backtest_organillo.json` is also runnable:

```bash
passivbot backtest configs/examples/backtest_organillo.json -dp
passivbot optimize configs/examples/backtest_organillo.json
```

Historical selection is CPU-only: use `optimize.backend` `pymoo` or `deap`; the GPU backend rejects it. These commands may download public candles and market metadata. The example illustrates input
format and wiring; it is not an optimized strategy. Relative carton paths resolve from the current
working directory. Both `.csv` and `.csv.gz` are supported.

## Matrix format and clock

```csv
date,BTC,ETH
2025-01-01,1,0
2025-01-02,0,1
2025-01-03,1,1
```

The first column contains dates at **00:00 UTC**. Its header may be blank. Dates must be unique and
in increasing order; values must be `0` or `1`. Remaining headers identify markets using PB8's
normal backtest coin identities. Every candle-series coin passed to the engine needs a matching
carton column. Extra carton columns are allowed.

A row is effective from its timestamp until the next UTC midnight. Selection for January 2 must
have been computable at January 2 00:00 using only information already available then. A tool that
uses January 2's completed daily candle must date its selection January 3. PB8 cannot verify the
provenance of external selections.

Orders computed from the completed 23:59 candle can execute in the 00:00 candle. PB8 uses the
selection effective at that execution boundary. Candle intervals must divide 24 hours and align
to UTC midnight. EMA warmup candles before carton coverage remain available for indicators and
cannot start positions.

The carton must cover every day in the requested period and every operational candle timestamp.
Missing days inside that period raise an error; selections are never silently carried forward.
Gaps outside the requested period do not prevent a run. A period with no selected coins raises an
error during universe preparation because no candle universe can be built.

## Position handling

`1` makes a coin eligible for new positions, subject to normal Forager, strategy and risk checks.
`0` applies PB8's existing **graceful stop** mode to both position sides. Flat coins cannot start new
positions. Held positions retain PB8's existing management, including permitted reentries/DCA,
take profit, unstuck and risk actions. No forced liquidation or new fill policy is added.

The carton replaces the simulation's current `live.approved_coins` with the historical union.
Disabled sides, ignored coins, market availability, per-coin overrides and configured position
limits still apply. A market enters the decision and exposure-counting universe only at its first
selection within the requested simulation period. Selections before that period are warmup input,
not prior admission. Once admitted, a market remains subject to ordinary PB8 tradability and
exposure-counting rules even on a later `0` day; its entry mode becomes graceful stop. This keeps
held-position management and the existing non-shrinking dynamic exposure denominator intact.
Future-only markets do not consume Forager slots, change current exposure counts, or receive
coin-mode HSL updates before admission. When PB8 derives global warmup from strategy spans,
its maximum includes only markets admitted by that decision time. Future-only EMA overrides cannot
delay earlier trading; an explicitly configured global warmup retains its normal behavior.
The configured `n_positions` and exposure formulas are unchanged. This mechanism does not enable
live historical selection.

A selected coin may be unavailable on an exchange for part of the period; ordinary PB8 candle
tradability rules continue to govern that interval.

## Suites and reproducibility

Each suite scenario may override `backtest.organillo_mode` and `backtest.organillo_carton_path`
using its normal dotted `overrides` keys. Its date range and explicit `coins` list restrict that
scenario's historical universe. Different cartons share the master candle dataset while retaining
independent daily eligibility matrices.

PB8 computes `backtest.organillo_carton_hash` from decompressed CSV bytes. Leave this field unset
in a new input config. Prepared result configs and backtest `dataset.json` record the digest;
optimizer suite scenarios record their individual digests. Changing a carton invalidates an exact
optimizer resume. Clear a previously saved hash only when intentionally starting a new experiment.
Identical CSV bytes produce the same digest whether compressed or uncompressed.

The optimizer prepares the daily matrix once per dataset/scenario and transports shared-memory
descriptors to workers. It does not expand the matrix to one row per minute or read the CSV for
every candidate.
