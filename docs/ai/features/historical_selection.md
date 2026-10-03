# Historical Coin Selection Contract

Simulation-only settings live in `config.backtest`: `organillo_mode` (default false),
`organillo_carton_path` (default null), and derived `organillo_carton_hash` (default null).

Python owns CSV parsing, coverage validation, historical universe preparation, canonical market
identity mapping and shared-array transport. Rust owns the resulting trading mode and all strategy,
Forager, position and risk behavior.

## Invariants

- Daily rows are unique, increasing UTC midnights. Binary values are required; no missing-day
  carry-forward is allowed inside the requested/operational period.
- A simulation copy replaces today's approved list with the selected union for its date range.
  Explicit scenario coin lists intersect that union. Existing ignored lists and disabled sides apply.
  This is a data-loading universe: first selection within the requested simulation period admits
  a market to decision/risk inputs. Future-only selections and warmup rows cannot admit it early.
  Admission is derived from the immutable mask and physical-column mapping, not mutable RAM state.
  Retired markets remain admitted so normal tradability, graceful-stop management and the existing
  grow-only dynamic exposure-counting rules continue to apply. Both orchestrator paths, exposure
  counts and coin-mode HSL updates use this causal universe. If no explicit global warmup was
  provided, the Rust span-based warmup fallback uses a prefix maximum over admitted markets;
  future-only overrides cannot delay past trading. Explicit global warmup is unchanged.
- The compact mask is `uint8[day, physical_hlcv_column]`; `int32[timestep]` maps actual candle
  timestamps to day rows. `-1` represents warmup before coverage. Logical coin order may differ
  from physical master columns, and time/coin subsetting must preserve both mappings.
- Backtest orders computed at candle `k` close execute at `k+1`; eligibility uses the selection
  effective at that next candle's opening timestamp, including midnight transitions.
- Ineligible markets map to existing `GracefulStop` in both fresh and cached orchestrator inputs.
  Market-data validity, held positions, DCA, exits, risk formulas and fills retain existing behavior.
- Eligible markets retain the existing forced-normal/default mode. No independent fill gate or
  position-close policy is introduced.
- Feature-disabled payloads have no selection arrays and preserve prior results.
- Both bundle arrays must be provided together. Rust validates shapes, binary mask values, and
  row bounds before consuming them. Daily width is the physical HLCV width, including inactive
  master columns. Supplied shared selections must match the prepared config digest.
- Worker transport uses immutable shared arrays; parent-owned blocks are cleaned up after workers
  finish. The CSV is loaded during preparation, not on every optimizer candidate evaluation.
- SHA256 of decompressed bytes is saved in prepared configs and dataset metadata. Each optimizer
  suite scenario has its own hash; exact resume detects changed selection content.

## Failures and scope

Missing files, malformed matrices, missing operational dates, unknown dataset columns, incompatible
interval boundaries, hash mismatches and an empty selected universe propagate before simulation.
The exporter remains responsible for historical point-in-time correctness. PB8 does not derive
market cap, categories or selection rules and does not enable this mechanism in live trading.

## Locations and validation

- `src/historical_selection.py`: input and transport
- `src/backtest.py`, `src/suite_runner.py`: backtest integration
- `src/optimize.py`, `src/optimize_suite.py`: worker/suite integration and resume
- `src/backtest_dataset.py`: result metadata
- `passivbot-rust/src/{types,python,backtest}.rs`: bundle boundary and mode application
- `tests/test_historical_selection.py`: malformed input, UTC boundaries, physical-column mapping,
  parity with disabled mode, initial entries, future-selection/candle perturbation invariance,
  shared worker evaluation and CLI entrypoint artifacts
- Rust `organillo_uses_graceful_stop_in_both_orchestrator_paths`: fresh/cached mode and held positions
- Rust `organillo_admits_risk_universe_causally_with_physical_column_mapping`: admission boundary,
  warmup exclusion, retirement, exposure counts on both sides and cache rebuild. Python causal
  regressions also cover automatic warmup with future-only EMA overrides and enabled HSL.

Run affected config, backtest, suite and optimizer tests, Rust tests, extension rebuild/fingerprint
verification, and a bounded CLI integration with prepared candle data. See `../../organillo.md`.
