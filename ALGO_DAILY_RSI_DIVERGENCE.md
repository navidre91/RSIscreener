# Daily RSI Divergence — Algorithm Only

This document precisely describes the algorithm implemented in `screener_daily_rsi_divergence.py` to detect price/RSI divergences on daily bars. It covers only the signal logic and ranking, independent of data sources or notifications.

## Inputs and Notation
- Time series on daily bars for a symbol:
  - `c[t]` — close price at bar index `t` (0-based, increasing in time)
  - `RSI[t]` — RSI(14) at `t` computed by Wilder’s smoothing on closes
- Tunable parameters (defaults in the script):
  - `RSI_PERIOD = 14`
  - `PIVOT_WINDOW = 3` (left/right span for swing points)
  - `RECENT_BARS = 20` (max age, in bars, of the latest pivot)
- Latest bar index is `T = len(c) - 1`.

## RSI Calculation (Wilder)
Given closes `c[t]`, compute Wilder’s RSI with period `RSI_PERIOD`:
- `delta[t] = c[t] - c[t-1]`
- `gain[t] = max(delta[t], 0)`; `loss[t] = max(-delta[t], 0)`
- Exponentially smoothed averages (Wilder smoothing):
  - `avg_gain = EMA(gain, alpha = 1/RSI_PERIOD, min_periods = RSI_PERIOD)`
  - `avg_loss = EMA(loss, alpha = 1/RSI_PERIOD, min_periods = RSI_PERIOD)`
- `RS = avg_gain / avg_loss` (treat division by 0 as NaN)
- `RSI = 100 - (100 / (1 + RS))`

## Pivot Detection
The algorithm detects local swing lows and highs using a symmetric window `PIVOT_WINDOW`:
- A pivot low at index `i` satisfies: `c[i] < c[i-k] for all k=1..PIVOT_WINDOW` and `c[i] <= c[i+k] for all k=1..PIVOT_WINDOW`.
- A pivot high at index `i` satisfies: `c[i] > c[i-k] for all k=1..PIVOT_WINDOW` and `c[i] >= c[i+k] for all k=1..PIVOT_WINDOW`.
- Only indices `i` where both left and right windows fit (`PIVOT_WINDOW <= i <= T - PIVOT_WINDOW`) are considered.

The most recent two pivots of the relevant type are used:
- For bullish divergence: the last two pivot lows `(i1, p1)`, `(i2, p2)` with `i1 < i2`.
- For bearish divergence: the last two pivot highs `(i1, p1)`, `(i2, p2)` with `i1 < i2`.

A recency constraint is applied to the latest pivot: `T - i2 <= RECENT_BARS`.

## Divergence Definitions
Let `r1 = RSI[i1]` and `r2 = RSI[i2]` (RSI sampled at the pivot indices). If either RSI value is NaN, the symbol is skipped.

- Bullish divergence (higher RSI low vs. lower price low):
  - Price makes a lower low: `p2 < p1`
  - RSI makes a higher low: `r2 > r1`

- Bearish divergence (lower RSI high vs. higher price high):
  - Price makes a higher high: `p2 > p1`
  - RSI makes a lower high: `r2 < r1`

## Strength Scoring and Ranking
A unitless, positive score is assigned to allow cross-symbol ranking.

- Bullish strength:
  - `price_drop = max(p1 - p2, 0)`; `price_drop_pct = price_drop / p1` (0 if `p1` is 0)
  - `rsi_gain = max(r2 - r1, 0)`
  - `strength = rsi_gain * price_drop_pct`

- Bearish strength:
  - `price_rise = max(p2 - p1, 0)`; `price_rise_pct = price_rise / p1` (0 if `p1` is 0)
  - `rsi_drop = max(r1 - r2, 0)`
  - `strength = rsi_drop * price_rise_pct`

Symbols are sorted by `strength` descending. If there is a tie, the later pivot time (`i2`) sorts first (i.e., more recent signals rank higher).

## Output Per Match
For each detected divergence, the algorithm produces:
- `symbol`, `type` (`bullish` or `bearish`)
- `last_price = c[T]`, `last_rsi = RSI[T]`
- `pivot_start_dt` (time at `i1`), `pivot_dt` (time at `i2`)
- `p1`, `p2`, `r1`, `r2`
- `price_drop_pct` and `rsi_gain` (bullish), or `price_rise_pct` and `rsi_drop` (bearish)
- `strength`

## Pseudocode
```
for each symbol:
  if len(c) < RSI_PERIOD + safety_margin: continue
  RSI = WilderRSI(c, RSI_PERIOD)
  if all NaN(RSI): continue

  # Bullish path
  lows = pivot_lows(c, PIVOT_WINDOW)
  if count(lows) >= 2:
    (i1, p1), (i2, p2) = last_two(lows)
    if i2 > i1 and (T - i2) <= RECENT_BARS:
      r1, r2 = RSI[i1], RSI[i2]
      if p2 < p1 and r2 > r1 and not NaN(r1, r2):
        price_drop_pct = max(p1 - p2, 0) / max(p1, tiny)
        rsi_gain = max(r2 - r1, 0)
        strength = rsi_gain * price_drop_pct
        emit(bullish, fields…)

  # Bearish path
  highs = pivot_highs(c, PIVOT_WINDOW)
  if count(highs) >= 2:
    (i1, p1), (i2, p2) = last_two(highs)
    if i2 > i1 and (T - i2) <= RECENT_BARS:
      r1, r2 = RSI[i1], RSI[i2]
      if p2 > p1 and r2 < r1 and not NaN(r1, r2):
        price_rise_pct = max(p2 - p1, 0) / max(p1, tiny)
        rsi_drop = max(r1 - r2, 0)
        strength = rsi_drop * price_rise_pct
        emit(bearish, fields…)

rank all emitted rows by strength desc, then by pivot_dt desc
```

## Figures (Conceptual)

Bullish divergence: price makes a lower low while RSI makes a higher low.

```
Price (c)
  p
  |            .
  |          .   .
  |        .       .     (lower low)
  |_____.__           .
         ^i1  p1       ^i2  p2

RSI(14)
  r
  |       .__
  |     .    .
  |   .        .__    (higher low)
  |__.             .
      ^i1  r1         ^i2  r2

Condition: p2 < p1  AND  r2 > r1
```

Bearish divergence: price makes a higher high while RSI makes a lower high.

```
Price (c)
  p
  |            .
  |          .   .  (higher high)
  |        .       .
  |_____.__           .
         ^i1  p1       ^i2  p2

RSI(14)
  r
  |       .__
  |     .    .__  (lower high)
  |   .         .
  |__.             .
      ^i1  r1         ^i2  r2

Condition: p2 > p1  AND  r2 < r1
```

## Edge Handling and Safeguards
- Requires sufficient bars: at least `RSI_PERIOD + 10` samples to ensure RSI stabilization and pivot eligibility.
- RSI values at pivot indices must be non-NaN; otherwise the symbol is skipped for that divergence type.
- Recency filter prevents stale signals: the latest pivot `i2` must lie within `RECENT_BARS` of `T`.
- Pivot logic is strict (all-left strictly lower/greater, all-right non-strict) to avoid flat runs producing spurious pivots.

## Complexity
- Per symbol, pivot scans are linear in the number of bars, `O(N)`, with a small constant factor controlled by `PIVOT_WINDOW`.
- RSI computation is `O(N)` via a single pass of exponential smoothing.

## Notes
- Both bullish and bearish divergences are supported by the algorithm; ranking is comparable through the same multiplicative structure of an RSI delta and a price percentage change.
- The use of percentage change on price and absolute change on RSI creates a unitless score that emphasizes concurrent magnitude in both dimensions.

