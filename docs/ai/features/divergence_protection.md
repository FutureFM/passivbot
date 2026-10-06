# Cross-Asset Divergence Protection

## Contract

- The detector compares each coin's 5, 15, 60, and 240 minute close-to-close ROC, expressed in percent, against the population mean and population standard deviation for that horizon.
- Each horizon requires at least three coins with valid closes at both aligned endpoints. Missing coverage yields no signal for that horizon. Live uses the latest completed 1m candle at a common timestamp; backtest uses the current historical decision candle.
- Long protection flags a negative outlier (`z < -threshold`); short protection flags a positive outlier (`z > threshold`). A horizon is excluded for a coin when at least `breadth_threshold_pct` of the valid universe moves in that side's adverse direction by more than `breadth_drop_pct`.
- A coin is protected after at least `min_timeframes` horizons flag it. Severity is `clamp(abs(worst_z) / zscore_threshold - 1, 0, 1)`, or 1 when the configured threshold is zero.
- The effective per-symbol WEL becomes `base_WEL * (1 + severity * (we_cap_pct - 1))`. The base is recomputed each decision; the cap does not accumulate. Only `0 < we_cap_pct < 1` changes WEL. The reduced WEL governs entries (including the `ema_anchor` per-position entry cap) and time-stop triggers, but not the time-stop reduction cap.
- For an existing position, the entry cooldown becomes `risk.entry_cooldown_minutes * (1 + severity * (delay_multiplier - 1))`. A zero base cooldown stays zero. The divergence multiplier does not extend initial-entry cooldown when the position is flat. Closes and independent risk reducers remain available.
- Detection is disabled by default on both sides. A missing population never produces a protective signal, matching the PB7 backtest behavior. Live reports missing candle coverage. This does not predict a sudden crash or prevent orders already filled before the completed candle is available.

## Configuration

The fields live under `bot.<long|short>.risk`:

| Field | Default | Meaning |
|---|---:|---|
| `divergence_filter_enabled` | `false` | Enable the detector for this position side. |
| `divergence_zscore_threshold` | `2.0` | Outlier threshold. |
| `divergence_breadth_threshold_pct` | `40.0` | Market-wide adverse-move share that suppresses a horizon. |
| `divergence_breadth_drop_pct` | `1.0` | Adverse ROC threshold in percentage points. |
| `divergence_min_timeframes` | `2` | Required flagged horizons (1–4). |
| `divergence_delay_multiplier` | `1.0` | Full-severity entry-cooldown multiplier. |
| `divergence_we_cap_pct` | `1.0` | Full-severity WEL fraction. |

With only three coins, the largest possible absolute population z-score for one isolated outlier is `sqrt(2) ≈ 1.414`. A threshold of 2.0 therefore needs at least six valid coins to flag a lone outlier; use a lower threshold if operating with only three to five coins.

To tune the six numeric fields, add only the desired keys under `optimize.bounds.<side>.risk`. Bounds are optional: fields without bounds retain their `bot.<side>.risk` values during optimization. Set the boolean `divergence_filter_enabled` in `bot.<side>.risk`; it is not an optimizer bound.

## Validation

- Rust unit tests cover isolated long drops, isolated short pumps, market-wide moves, and fewer than three valid coins.
- Real-extension orchestrator tests cover WEL capping, cooldown extension, and short-side direction.
- Python tests cover common completed-candle alignment and missing coverage.

## Code

- `passivbot-rust/src/divergence.rs`: shared detector and severity.
- `passivbot-rust/src/orchestrator.rs`: WEL and cooldown effects.
- `passivbot-rust/src/backtest.rs`: historical ROC preparation.
- `src/divergence_inputs.py`: live completed-candle ROC preparation.
