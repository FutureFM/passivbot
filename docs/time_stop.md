# Time-based position reductions

Configure each side under `bot.<side>.risk`; the same fields support coin overrides
and optimizer bounds. The feature is disabled by default and applies to every strategy.

| Field | Default | Meaning |
|---|---:|---|
| `time_stop_max_age_days` | 0 | Interval in days; zero disables. Fractional days are supported. |
| `time_stop_close_pct` | 1 | Position fraction to close, 0–1. Zero disables; one flattens. |
| `time_stop_we_trigger_pct` | 0 | Require current WE to reach this fraction of effective WEL before starting a reduction. |
| `time_stop_close_we_min` | 0 | When positive and current WE is at or below this fraction of WEL, close the full position. |
| `time_stop_close_we_max` | 1 | Cap a partial reduction at this fraction of WEL. Divergence protection does not lower this cap. Zero and one disable the cap. |

For example, these long-side settings request a 25% reduction every seven days,
provided exposure reaches half its effective limit:

```json
{
  "time_stop_max_age_days": 7.0,
  "time_stop_close_pct": 0.25,
  "time_stop_we_trigger_pct": 0.5,
  "time_stop_close_we_min": 0.05,
  "time_stop_close_we_max": 0.25
}
```

The interval starts at the opening fill of the current position or completion of
its last temporal reduction. Reentries, take profits, and exposure changes do not
reset it. The exposure trigger is checked when closing; it does not time a continuous
period above the threshold. A position already old enough may close immediately
when the trigger becomes true. Positions can close at a profit or a loss.

Temporal closes use **reduce-only market orders**, independently of
`live.market_orders_allowed`, and may realize losses beyond `max_realized_loss_pct`.
Market execution is subject to exchange availability and slippage. HSL/panic retains
priority; manual mode disables temporal management. Other orders for the same pair
are suppressed while its reduction is active.

For partial reductions, the remaining-position target is fixed at the closing
decision. If a market order executes incompletely, the bot recovers its target
from exchange order/fill identifiers and retries the remainder. The clock restarts
only when that target is reached. At 100%, it continues until flat. Exchange minimums
and dust handling can require closing slightly more than the nominal fraction; a
new partial amount below the executable minimum is skipped.

Restarting preserves the original interval, including downtime, when the required
exchange history and order attribution are recoverable. No local timer file is
required. Missing evidence defers affected temporal closes and entries visibly while
other independently evaluable closes continue. Do not strip client order identifiers
from fill history. Exchange history retention can limit recovery.

For grid strategies, the completed reduction's execution price becomes the entry
reference until the next entry fill. This does not change the real average entry
price or PnL. EMA-anchor quotes retain their existing geometry.

The existing entry cooldown still applies; this feature adds no separate reentry
cooldown after flattening. Stop duration has no 24-hour strategy cap.

Optimizer bounds use the same nested paths under `optimize.bounds.<side>.risk`.
Default stop bounds are fixed at disabled settings. Set explicit bounds when tuning
or fixing an enabled stop, for example `time_stop_max_age_days: [3, 14, 0.25]` and
`time_stop_close_pct: [0.1, 0.5, 0.05]`. Change related fixed bounds to your selected
trigger/min/max values as well. These are illustrative ranges, not optimized settings.
