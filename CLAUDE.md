# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

**Follow [AGENTS.md](./AGENTS.md) first.** It holds the canonical agent rules (authority tiers for
live/authenticated actions, public-repo data boundary, instruction precedence) and is maintained for
cross-platform portability. Then read `docs/ai/principles.md` and use `docs/ai/README.md` as a router
to load only the task-relevant contracts under `docs/ai/` and `docs/ai/features/`.

## Commands

Python 3.12 or 3.14 only (3.13 is unsupported). A local `venv/` exists at the repo root.

```bash
pytest                                          # default suite; pytest.ini sets pythonpath=src, excludes fake_live_slow
pytest tests/test_time_stop.py::test_name       # single test
pytest tests/test_run_fake_live.py -m fake_live # offline fake-exchange harness tests
MPLBACKEND=Agg pytest                           # as in CI (plotting tests)

bash rustbuild.sh                               # cargo fmt + maturin develop --release + sync/stamp extension into src/
cd passivbot-rust && cargo test --no-default-features && cd ..
cd passivbot-rust && cargo check --tests && cd ..

# AI-docs checks (CI runs these; required when touching docs/ai or live events)
PYTHONPATH=src python src/tools/check_ai_docs.py
PYTHONPATH=src python src/tools/generate_live_event_registry.py --check

passivbot backtest path/to/config.json          # may download public candles
passivbot optimize path/to/config.json [-t anchors -ft long.risk]
passivbot backtest --suite / passivbot optimize --suite
PYTHONPATH=src python src/tools/run_fake_live.py configs/fake_live_hsl_btc.hjson scenarios/fake_live/hsl_long_red_restart.hjson --user fake_hsl_restart_test
```

Never run `passivbot live` (or any authenticated exchange call) as a smoke test — see AGENTS.md.

**Stale extension gotcha:** if Rust changes appear ignored, the loaded `passivbot_rust` is stale.
Rebuild with `rustbuild.sh` and verify with
`PYTHONPATH=src python -c "import passivbot_rust; from rust_utils import verify_loaded_runtime_extension; print(verify_loaded_runtime_extension())"`
(details in `docs/ai/runbooks/rust_extension.md`). Rust changes must be followed by a rebuild before
running Python tests that exercise them.

## Architecture (big picture)

- **Rust (`passivbot-rust/src/`) owns all trading behavior**: `orchestrator.rs` computes the
  complete ideal-order set per tick (entries, closes, unstuck, HSL panic, time stops, cooldown
  gates) and is shared by live and backtest; `backtest.rs` is the simulator; `strategies/`
  (`trailing_martingale`, `ema_anchor`, deprecated `trailing_grid_v7`) generate per-side orders;
  `types.rs` holds `BotParams`; `python.rs` is the PyO3 boundary (JSON API + backtest entry).
- **Python (`src/`) owns orchestration and I/O**: `passivbot.py` (live bot, builds orchestrator
  input, gates and submits orders), `src/live/` (state refresh, reconciler, executor, event bus),
  `exchanges/` (CCXT adapters, broker codes), `candlestick_manager.py`, `fill_events_manager.py`
  (canonical fill/PnL stream), `backtest.py` / `optimize.py` / `suite_runner.py`.
- **Live loop**: refresh state → build canonical inputs → Rust ideal orders → validate the whole
  batch (malformed = fatal, no fallback to prior ideals) → reconcile vs exchange orders → gate →
  execute. Trading must be reproducible after restart from exchange state + config only (no
  decision-changing local state).
- **Adding a bot parameter touches many surfaces in lockstep**: `src/config/schema.py` (defaults),
  `validate.py`, `shared_bot.py` (Python→Rust param mapping), `optimize_bounds.py`, `overrides.py`
  (if per-coin overridable), `hydrate.py`, Rust `types.rs` + `python.rs` extraction, the
  `bot_params` key list in `passivbot.py`, example configs, `docs/configuration.md`, and
  `CHANGELOG.md` under `Unreleased`. Config ownership: `config.live` (shared live+backtest),
  `config.backtest` (simulation-only), `config.optimize` (optimizer-only).

## Fork features on top of v8.1.0

This branch line (fork of `enarjord/passivbot`, synced with upstream `master` / schema v8.6.0) adds
the features below. Each has a contract doc — read it before changing the feature. Divergence,
time stops and historical selection are not modeled by GPU screening: the GPU optimizer warns,
ranks without them and pins their searchable bounds to the configured values
(`pin_cpu_only_feature_bounds`); exact CPU validation of results applies them. The `ema_anchor`
exposure cap IS modeled on GPU (`mps_ema_anchor_directional.metal` `entry_cap`, multicoin
`position_cap`); keep both in sync with `strategies/ema_anchor.rs`.

| Feature | Config | Contract / user doc | Main code |
|---|---|---|---|
| Historical coin selection ("Organillo"): daily 0/1 CSV(.gz) "carton" restricts the backtest/optimizer universe; unselected coins get `GracefulStop`; causal admission (future selections never affect current decisions); content-hashed for resume | `backtest.organillo_mode`, `organillo_carton_path` (`organillo_carton_hash` is derived — leave unset) | `docs/ai/features/historical_selection.md`, `docs/organillo.md` | `src/historical_selection.py`, Rust `backtest.rs` |
| Cross-asset divergence protection: z-score of 5/15/60/240m ROC vs population; outliers shrink WEL and lengthen held-position entry cooldown | `bot.<side>.risk.divergence_*` | `docs/ai/features/divergence_protection.md` | Rust `divergence.rs`, `orchestrator.rs`; `src/divergence_inputs.py` (live ROC) |
| Divergence × upstream adaptive cooldown: held positions multiply `bot.<side>.entry_cooldown` duration by the divergence delay, capped by `max_duration_minutes` (or 1440) | `entry_cooldown.*`, `risk.divergence_delay_multiplier` | `docs/ai/features/strategy_runtime.md` | Rust `divergence_scaled_cooldown` in `orchestrator.rs`; `Passivbot._entry_cooldown_horizon` |
| `ema_anchor` per-position exposure cap (entries cropped at WEL × (1+excess allowance)) | always on | CHANGELOG | `passivbot-rust/src/strategies/ema_anchor.rs` |
| Time-based stops: reduce-only market closes (`close_time_stop_<side>`) after N days; clock and partial-close target are reconstructed from fills + client order IDs (target encoded in the ID), no local timer | `risk.time_stop_*` | `docs/ai/features/time_stop.md`, `docs/time_stop.md` | Rust `calc_time_stop_close` in `orchestrator.rs`; `src/time_stop.py` (`reconstruct_episodes`, ID encode/decode), `src/passivbot.py`, `src/live/reconciler.py` |
| `pnl_by_coin.png` backtest chart | — | `docs/backtesting.md` | `src/plotting.py` |
| `quantstats_report.html` tearsheet of daily strategy-equity returns (optional dep, never fails the backtest) | `-dp quantstats` disables | `docs/backtesting.md` | `src/quantstats_report.py`, called from `src/backtest.py` |

Shared mechanics worth knowing: `time_stop.reconstruct_episodes` replays the fill stream to rebuild
the time-stop clock and target; missing/ambiguous evidence yields `None`, which Rust treats as
"defer entries/temporal closes for that pair" while ordinary closes continue. Optional divergence
optimizer bounds must survive both config loading (`config/hydrate.py`) and cleanup/export
(`preserve_optional_adaptive_bounds` in `config/optimize_bounds.py`). Integer-valued bot params
coming from the optimizer arrive as floats (see `divergence_min_timeframes` in `python.rs`).
Tests: `tests/test_historical_selection.py`, `test_divergence_inputs.py`, `test_time_stop.py`,
`test_orchestrator_json_api.py`, `test_pnl_by_coin_plot.py`,
`tests/optimization/test_gpu_cpu_only_features.py`.
