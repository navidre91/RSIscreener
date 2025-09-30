
# Robust Intraday Swing Trading Playbook (Rare but High-Conviction Setups)

*Version:* 1.0  
*Scope:* Intraday **swing** trades held ~30–240 minutes across liquid equities/ETFs/futures.  
*Philosophy:* Signals should be **rare by construction** (strict confluence) but **decisive** when they appear.  
*Disclaimer:* This document is for educational purposes only and is **not** financial advice. Backtest thoroughly on your own data, understand venue-specific rules/slippage/fees, and risk only what you can afford to lose.

---

## Table of Contents

1. [Global Assumptions & Feature Definitions](#global-assumptions--feature-definitions)  
2. [Strength Score (Make Signals Rare on Purpose)](#strength-score-make-signals-rare-on-purpose)  
3. [Algorithms](#algorithms)  
   - [A1. Volatility Squeeze (NR7) → Opening Range Breakout (ORB)](#a1-volatility-squeeze-nr7--opening-range-breakout-orb)  
   - [A2. Event-Driven Opening Drive (Earnings/Macro) with VWAP Hold](#a2-event-driven-opening-drive-earningsmacro-with-vwap-hold)  
   - [A3. Order-Flow Imbalance Trend Day](#a3-order-flow-imbalance-trend-day)  
   - [A4. Liquidity Sweep → Reclaim at Higher-Timeframe Level](#a4-liquidity-sweep--reclaim-at-higher-timeframe-level)  
   - [A5. VWAP Extreme: Mean-Revert or Trend-Hold Continuation](#a5-vwap-extreme-mean-revert-or-trend-hold-continuation)  
   - [A6. Halt/Resume Continuation (Volatility or News Halts)](#a6-haltresume-continuation-volatility-or-news-halts)  
4. [Backtesting & Evaluation Protocol](#backtesting--evaluation-protocol)  
5. [Reference Implementation Notes for an LLM](#reference-implementation-notes-for-an-llm)  
6. [Glossary](#glossary)

---

## Global Assumptions & Feature Definitions

**Markets.** Liquid stocks/ETFs or index futures (e.g., ES, NQ).  
**Timeframes.** 1‑minute bars for signals/execution; daily bars for context.  
**Session.** Regular trading hours (RTH). If you include premarket, compute VWAP and RVOL separately for premarket vs RTH, or reset at RTH open.

### Core Features (to precompute each day)

- **Daily ATR(14):** Average True Range on daily bars for gap/NR7 context.  
- **NR7 flag:** `true` if yesterday’s (D-1) daily range is the smallest of the last 7 sessions.  
- **Opening Range (OR):** Fixed window from session open. Use **15 minutes** by default (`OR15H`, `OR15L`). Some variants use 5‑minute OR.  
- **Initial Balance (IB):** First 60 minutes high/low (optional for filters/targets).  
- **VWAP:** Intraday cumulative VWAP from RTH open. If you support premarket, track `VWAP_RTH` and `VWAP_full`.  
- **RVOL:** Relative volume at time *t*. Practical definition:  
  `RVOL(t) = cumVolume_today(t) / median(cumVolume_past_N_days(t))`, with N=20–40.  
- **Intraday ATR(15):** Rolling ATR over 15 one‑minute bars, used for stop sizing.  
- **Round Numbers (RN):** Price grid (e.g., multiples of 1.00/0.50/0.25 depending on instrument price).  
- **Prior Day H/L/C:** Previous RTH high/low/close.  
- **Distance-to-VWAP z-score:** `z = (price - VWAP) / stdev(price - VWAP over last M mins)`, default M=60.  
- **Order Imbalance (proxy):** Prefer tick-by-tick signed volume (`uptick_volume - downtick_volume`). If only OHLCV is available, approximate with *cumulative delta proxy*:  
  `imbalance(t) = sign(close_t - close_{t-1}) * volume_t` and accumulate over a window.

**Universe.** Top ~300 names by dollar volume + target index future/ETF.  
**Data quality.** Use consolidated trades; ensure time alignment and handle halts (no bars) explicitly.

---

## Strength Score (Make Signals Rare on Purpose)

Compute a **10-point strength score** per setup. Only trade a signal if `score >= 8`:

1. Fresh **catalyst** present (earnings, guidance, macro print, halt) — 1  
2. **Gap context** (beyond part of prior day’s range or following NR7) — 1  
3. **RVOL ≥ 2** during OR window — 1  
4. Direction aligned with **daily/4‑hour** trend (e.g., above 20D/50D MAs for longs) — 1  
5. **Sector/Index confirmation** aligned — 1  
6. **Location** at prior day’s H/L, RN, or multi‑day balance edge — 1  
7. **Order‑flow persistence** (sustained net aggression) — 1  
8. **VWAP behavior** (hold for trends; exhaustion beyond for mean-revert) — 1  
9. **Clean structure** (no immediate overhead/underfoot supply; no known binary risk) — 1  
10. **Asymmetric R:R ex‑ante** (≥ 2.5:1 to first target with placed stop) — 1

---

## Algorithms

### A1. Volatility Squeeze (NR7) → Opening Range Breakout (ORB)

**Intuition.** After volatility contraction (NR7), the next day often expands. A strong break of the opening range with RVOL confirmation tends to carry for hours, especially when aligned with higher‑timeframe trend and supported by the market/sector.

**Default Parameters**
```yaml
A1_params:
  use_OR_minutes: 15          # 5 or 15
  require_NR7_yesterday: true
  min_gap_ATR_ratio: 0.75     # today's open vs prior close relative to ATR(D1)
  min_RVOL_open: 2.0
  trend_filter: daily_close_above_20D_for_longs  # customizable
  r_multiple_first_target: 1.0  # of OR height
  stop_buffer_ticks: 0.1 * ATR15
  max_time_in_trade_minutes: 240
```

**Scanner Conditions (Long)**
- `NR7_yesterday == true`
- `|open_today - prior_close| >= min_gap_ATR_ratio * ATR14_daily`
- After OR window closes, `close > OR_high` and `RVOL_OR >= min_RVOL_open`
- Index/sector trending up (configurable)  
*(Short: symmetric conditions.)*

**Entry & Risk**
- **Entry:** Buy first higher‑low retest that **holds** `OR_high` *or* buy continuation on break/hold above `OR_high + small buffer` if retest not offered.
- **Stop:** Below `OR_high` reclaim low or `VWAP` (whichever is tighter per plan).
- **Targets:** Scale at **1× OR height**; trail remainder under `9–20 EMA` or `VWAP`.

**Don’t-Trade Filters**
- News risk later in the day (FOMC, earnings after close), or RVOL collapses <1 into mid‑day.

#### Pseudocode — Scanner
```pseudocode
function scan_A1_NR7_ORB(bars_1m, daily_bars, sector_series, params):
    signals = []
    y = yesterday(daily_bars)
    if not params.require_NR7_yesterday or is_NR7(y, daily_bars, lookback=7):
        gap = abs(session_open(bars_1m) - y.close)
        if gap >= params.min_gap_ATR_ratio * ATR(daily_bars, 14).last:
            ORH, ORL = opening_range(bars_1m, minutes=params.use_OR_minutes)
            RVOL_OR = rvol_until(bars_1m, t=OR_end_time, lookback_days=20)
            if close_at(OR_end_time, bars_1m) > ORH and RVOL_OR >= params.min_RVOL_open:
                if trend_agrees(daily_bars, direction="up", rule=params.trend_filter) and sector_confirms(sector_series):
                    signals.append(make_signal("A1_LONG", OR_end_time, ORH, ORL))
            if close_at(OR_end_time, bars_1m) < ORL and RVOL_OR >= params.min_RVOL_open:
                if trend_agrees(daily_bars, direction="down", rule=params.trend_filter) and sector_confirms(sector_series):
                    signals.append(make_signal("A1_SHORT", OR_end_time, ORH, ORL))
    return signals
```

#### Pseudocode — Execution
```pseudocode
function execute_A1(signal, bars_1m, params):
    if signal.type == "A1_LONG":
        entry = wait_for_retest_hold(bars_1m, level=signal.ORH) or break_of(signal.ORH, buffer=params.stop_buffer_ticks)
        stop  = last_reclaim_low_below(signal.ORH) or vwap(bars_1m).last
        risk  = entry - stop
        target1 = entry + (signal.ORH - signal.ORL) * params.r_multiple_first_target
        submit_order(entry, stop, size=position_size(risk))
        set_scale_target(target1)
        trail_on_MA_or_VWAP(direction="long")
        exit_if_time_exceeds(params.max_time_in_trade_minutes)
    else:
        # symmetric for short
        ...
```

---

### A2. Event-Driven Opening Drive (Earnings/Macro) with VWAP Hold

**Intuition.** A fresh catalyst creates one-sided inventory imbalance at the open. Early RVOL and a **hold above VWAP** (for longs) often sustain a directional move.

**Default Parameters**
```yaml
A2_params:
  event_required: true         # earnings, guidance, macro print
  min_gap_pct: 2.0             # %
  min_RVOL_open: 2.0
  vwap_hold_lookback_min: 10
  entry_trigger: "premarket_high_break_and_hold"  # or "first_consolidation_break"
  stop_method: "last_higher_low_or_VWAP"
  partial_scale_rr: 2.0
  max_time_in_trade_minutes: 240
```

**Scanner Conditions (Long)**
- Tagged **event day** with **positive surprise** (rule of thumb: large gap up).  
- During OR window, **price stays above VWAP** and RVOL ≥ threshold.  
- Break/hold above **premarket high** or **opening consolidation high**.

**Entry & Risk**
- **Entry:** Break/hold above trigger (premarket high or first flag).  
- **Stop:** Below last higher‑low or VWAP.  
- **Targets:** Partial at **2R**; trail remainder if sector/market breadth supports.

#### Pseudocode — Scanner
```pseudocode
function scan_A2_event_open_drive(bars_1m, events, premarket_levels, params):
    if not events.today.has_event and params.event_required:
        return []
    signals = []
    gap_pct = percent_change(session_open(bars_1m), prior_close(bars_1m))
    if abs(gap_pct) >= params.min_gap_pct:
        ORH, ORL = opening_range(bars_1m, minutes=15)
        VW = vwap_series(bars_1m)
        vwap_hold = holds_above(VW, window=params.vwap_hold_lookback_min)  # or below for shorts
        RVOL_OR = rvol_until(bars_1m, t=OR_end_time, lookback_days=20)
        if vwap_hold and RVOL_OR >= params.min_RVOL_open:
            trig = premarket_levels.high if params.entry_trigger == "premarket_high_break_and_hold" else ORH
            if close_after_break(bars_1m, level=trig, direction="up"):
                signals.append(make_signal("A2_LONG", trig, ORH, ORL))
        # symmetric short if gap down with VWAP hold below
    return signals
```

#### Pseudocode — Execution
```pseudocode
function execute_A2(signal, bars_1m, params):
    if signal.type == "A2_LONG":
        entry = confirm_break_and_hold(bars_1m, level=signal.trigger_level)
        stop  = last_higher_low_or_vwap(bars_1m)
        risk  = entry - stop
        submit_order(entry, stop, size=position_size(risk))
        set_scale_target(entry + 2 * risk)   # partial at 2R
        trail_on_VWAP_or_EMA(direction="long")
        exit_if_time_exceeds(params.max_time_in_trade_minutes)
    else:
        ...
```

---

### A3. Order-Flow Imbalance Trend Day

**Intuition.** Persistent net aggression (market orders lifting offers or hitting bids) early in the session can drive a trend day. If VWAP reclaims against the move **fail**, the path of least resistance is continuation.

**Default Parameters**
```yaml
A3_params:
  imbalance_window_min: 15
  min_imbalance_sigma: 2.0      # z-score of net signed volume vs 60-min lookback
  min_RVOL_open: 1.5
  vwap_fail_retests: 2          # number of failed reclaims allowed
  stop_method: "last_retest_swing"
  trail_method: "ema20"         # or "vwap"
  max_time_in_trade_minutes: 240
```

**Scanner Conditions (Long)**
- Net signed volume (uptick‑downtick) > `min_imbalance_sigma` for at least `imbalance_window_min`.  
- At least one **failed test** back to VWAP (price holds above).  
- RVOL ≥ threshold; optional: index/sector in agreement.

**Entry & Risk**
- **Entry:** On break of prior swing high after a failed VWAP reclaim.  
- **Stop:** Below last failed-reclaim swing low.  
- **Targets:** Trail via EMA(20) or VWAP; optional partial at 2R.

#### Pseudocode — Scanner
```pseudocode
function scan_A3_order_imbalance(bars_1m, tick_data_or_proxy, params):
    imbal_series = signed_volume_series(tick_data_or_proxy)  # or proxy from 1m bars
    z = zscore(rolling_sum(imbal_series, window=params.imbalance_window_min), lookback=60)
    if z.last >= params.min_imbalance_sigma:
        RVOL = rvol_until(bars_1m, t=now(), lookback_days=20)
        if RVOL >= params.min_RVOL_open and failed_vwap_reclaims(bars_1m, direction="up") >= 1:
            trig = break_of_recent_swing_high(bars_1m)
            return [make_signal("A3_LONG", trig)]
    elif z.last <= -params.min_imbalance_sigma:
        # symmetric short
        ...
    return []
```

#### Pseudocode — Execution
```pseudocode
function execute_A3(signal, bars_1m, params):
    entry = confirm_break(bars_1m, level=signal.trigger_level)
    stop  = last_failed_reclaim_swing_low(bars_1m) if signal.is_long else last_failed_reclaim_swing_high(bars_1m)
    risk  = abs(entry - stop)
    submit_order(entry, stop, size=position_size(risk))
    enable_trailing(method=params.trail_method, direction=signal.direction)
    time_exit(params.max_time_in_trade_minutes)
```

---

### A4. Liquidity Sweep → Reclaim at Higher-Timeframe Level

**Intuition.** Stops cluster at obvious levels (prior day H/L, RN). A **sweep** through such a level that **fails** (fast reclaim) traps late participants and can drive a sharp reversal back through the range.

**Default Parameters**
```yaml
A4_params:
  sweep_levels: ["prior_day_high","prior_day_low","round_numbers"]
  max_sweep_close_time_min: 5        # reclaim must occur within this many minutes
  min_RVOL_spike_sigma: 2.0
  entry_on_reclaim_close: true
  stop_buffer_ticks: 0.1 * ATR15
  first_target: "VWAP"
  second_target: "opposite_side_of_day_range"
```

**Signal (Long)**
- Price **wicks below** prior day low or RN, then **closes back above** within `max_sweep_close_time_min`.  
- **Volume spike** during sweep (≥ `min_RVOL_spike_sigma` vs 60‑min).  
- Optional: rejection candle body closes inside prior range.

**Entry & Risk**
- **Entry:** On the **reclaim close** or first pullback that holds the reclaimed level.  
- **Stop:** Tight — below the swept level (with small buffer).  
- **Targets:** VWAP first, then opposite side of the day range if momentum persists.

#### Pseudocode — Scanner
```pseudocode
function scan_A4_sweep_reclaim(bars_1m, prior_levels, params):
    signals = []
    for bar in recent_bars(bars_1m, lookback=120):
        swept_L = bar.low < prior_levels.low or is_below_round_number(bar.low)
        swept_H = bar.high > prior_levels.high or is_above_round_number(bar.high)
        if swept_L and closes_back_above_level(bar, prior_levels.low, within=params.max_sweep_close_time_min) and volume_spike(bar, sigma=params.min_RVOL_spike_sigma):
            signals.append(make_signal("A4_LONG", level=prior_levels.low))
        if swept_H and closes_back_below_level(bar, prior_levels.high, within=params.max_sweep_close_time_min) and volume_spike(bar, sigma=params.min_RVOL_spike_sigma):
            signals.append(make_signal("A4_SHORT", level=prior_levels.high))
    return deduplicate(signals)
```

#### Pseudocode — Execution
```pseudocode
function execute_A4(signal, bars_1m, params):
    if signal.type == "A4_LONG":
        entry = close_of_reclaim_bar(bars_1m) if params.entry_on_reclaim_close else first_retest_hold(signal.level)
        stop  = signal.level - params.stop_buffer_ticks
        submit_order(entry, stop, size=position_size(entry - stop))
        set_scale_target(vwap(bars_1m).last)
        set_secondary_target(opposite_side_of_day_range(bars_1m))
    else:
        ...
```

---

### A5. VWAP Extreme: Mean-Revert or Trend-Hold Continuation

**Intuition.** VWAP is a widely used intraday anchor. **Large deviations** from VWAP with **exhaustion** often snap back; on true trend days, repeated **holds above/below VWAP** offer high-quality continuation entries.

**Default Parameters**
```yaml
A5_params:
  zscore_window_min: 60
  min_abs_zscore: 2.5
  exhaustion_bar_rules:
    - "range > 1.5 * ATR15"
    - "volume > 2.0 * vol_60m_avg"
    - "long_tail_against_move"     # optional
  choose_mode: "auto"              # "mean_revert" or "trend" or "auto" (via breadth/imbalance)
  stop_method_MR: "beyond_extreme_highlow"
  stop_method_trend: "below_VWAP"  # or above for shorts
  target_MR: "VWAP"
  trail_trend: "EMA20_or_VWAP"
```

**Mean-Revert Mode (Long example)**
- `z = (price - VWAP)/stdev(price - VWAP, 60)` less than `-min_abs_zscore`.  
- Exhaustion bar forms (range/volume spike, tail).  
- Broad market not in clear trend (breadth mixed; low order-imbalance persistence).

**Trend Mode (Long example)**
- Price repeatedly **holds above VWAP**, pullbacks shallow (~1–2× ATR15), breadth/imbalance supportive.

#### Pseudocode — Scanner
```pseudocode
function scan_A5_vwap_extreme(bars_1m, breadth, imbalance, params):
    VW = vwap_series(bars_1m)
    z = zscore(price_series(bars_1m) - VW, window=params.zscore_window_min)
    mode = decide_mode(breadth, imbalance, params.choose_mode)
    signals = []

    if mode in ["mean_revert","auto"] and z.last <= -params.min_abs_zscore and is_exhaustion_bar(last_bar(bars_1m), params.exhaustion_bar_rules):
        signals.append(make_signal("A5_LONG_MR", level=last_bar(bars_1m).low))

    if mode in ["trend","auto"] and holds_above(VW, window=15) and shallow_pullbacks(bars_1m, ATR15):
        trig = break_of_recent_consolidation(bars_1m)
        signals.append(make_signal("A5_LONG_TREND", trigger=trig))

    # symmetric for shorts if z >= +threshold or holding below VWAP
    return signals
```

#### Pseudocode — Execution
```pseudocode
function execute_A5(signal, bars_1m, params):
    if signal.type == "A5_LONG_MR":
        entry = confirm_exhaustion_reversal(bars_1m)
        stop  = beyond_extreme_low(bars_1m)  # per stop_method_MR
        submit_order(entry, stop, size=position_size(entry - stop))
        set_scale_target(vwap(bars_1m).last)  # target_MR
    elif signal.type == "A5_LONG_TREND":
        entry = confirm_break(bars_1m, level=signal.trigger)
        stop  = below_vwap_or_last_higher_low(bars_1m)
        submit_order(entry, stop, size=position_size(entry - stop))
        trail_on_VWAP_or_EMA(direction="long")
```

---

### A6. Halt/Resume Continuation (Volatility or News Halts)

**Intuition.** Reopen auctions concentrate information/liquidity. A decisive break and **hold** of the **reopen range** (or initial post‑reopen flag) in the direction of the news often carries for a sustained intraday swing.

**Default Parameters**
```yaml
A6_params:
  watch_for_halts: true
  reopen_range_minutes: 3
  min_RVOL_post_reopen: 3.0
  entry_mode: "reopen_range_break_and_hold"
  stop_method: "below_reopen_range"   # or above for shorts
  scale_rr: 2.0
  trail_method: "EMA20_or_VWAP"
  max_time_in_trade_minutes: 240
```

**Signal (Long)**
- LULD/news **halt detected**; on reopen, compute `RRH/RRL` (reopen range).  
- Break **and hold** above RRH with RVOL ≥ threshold and supportive tape.

**Entry & Risk**
- **Entry:** Confirmed break/hold above RRH.  
- **Stop:** Below RRL (or last higher‑low if tighter).  
- **Targets:** Partial at `2R`; trail remainder with EMA/VWAP.

#### Pseudocode — Scanner
```pseudocode
function scan_A6_halt_resume(bars_1m, halt_events, params):
    if not params.watch_for_halts: 
        return []
    signals = []
    for halt in halt_events.today:
        post = bars_after(halt.reopen_time, bars_1m, minutes=30)
        RRH, RRL = range_high_low(post, first_minutes=params.reopen_range_minutes)
        if break_and_hold(post, level=RRH, direction="up") and rvol_since(post, halt.reopen_time) >= params.min_RVOL_post_reopen:
            signals.append(make_signal("A6_LONG", RRH, RRL, time=first_break_time(post, RRH)))
        if break_and_hold(post, level=RRL, direction="down") and rvol_since(post, halt.reopen_time) >= params.min_RVOL_post_reopen:
            signals.append(make_signal("A6_SHORT", RRH, RRL, time=first_break_time(post, RRL)))
    return signals
```

#### Pseudocode — Execution
```pseudocode
function execute_A6(signal, bars_1m, params):
    entry = confirm_break_and_hold(bars_1m, level=(signal.RRH if signal.is_long else signal.RRL))
    stop  = signal.RRL if signal.is_long else signal.RRH
    risk  = abs(entry - stop)
    submit_order(entry, stop, size=position_size(risk))
    set_scale_target(entry + params.scale_rr * risk * (1 if signal.is_long else -1))
    enable_trailing(method=params.trail_method, direction=signal.direction)
    time_exit(params.max_time_in_trade_minutes)
```

---

## Backtesting & Evaluation Protocol

**Walk-Forward Validation**
1. Split history into consecutive train/test windows (e.g., 6–12 months each).  
2. Tune only on train; record test performance; **never** peek ahead.

**Metrics to Record (per setup & overall)**
- Signal **frequency** (per day/week) and **coverage** (% of universe days)
- **Hit rate**, **avg R multiple**, **median R**, **expectancy**
- **Max adverse excursion (MAE)**, **max favorable excursion (MFE)**
- **Time in trade**, **slippage**, **fees**
- **Drawdown** (per setup and blended)

**Risk Controls**
- Daily loss cap (e.g., **−1R**).  
- **One‑shot rule:** If the A+ version fails, don’t re‑enter the B version.  
- Reduce size when RVOL falls; avoid trading low‑RVOL mid‑day chop.  
- Limit simultaneous correlated exposures (e.g., 1 per sector).

**Event Tagging**
- Maintain an **events calendar** (earnings, macro prints, scheduled speeches).  
- Mark **halts** explicitly in your bar series; skip bars with no prints.

---

## Reference Implementation Notes for an LLM

Use the following standard interfaces to keep each algorithm modular.

```yaml
data_interfaces:
  MinuteBar:
    fields: [timestamp, open, high, low, close, volume]
  DailyBar:
    fields: [date, open, high, low, close, volume]
  Events:
    fields: [timestamp, type, symbol, payload]  # e.g., earnings, macro, halt
  SectorSeries:
    fields: [timestamp, sector_etf_price, breadth]
  Signal:
    fields: [symbol, type, time, trigger_level, metadata]
  Trade:
    fields: [entry_time, entry_price, stop_price, direction, size, exits[]]

utility_functions_expected:
  - opening_range(bars_1m, minutes)
  - vwap_series(bars_1m)  # cumulative from RTH open
  - rvol_until(bars_1m, t, lookback_days)
  - ATR(series, period)
  - is_NR7(yesterday_bar, daily_bars, lookback)
  - holds_above(series_or_level, window) / holds_below(...)
  - break_and_hold(bars, level, direction)
  - signed_volume_series(tick_or_proxy)
  - zscore(series, window)
  - shallow_pullbacks(bars_1m, atr_series)
  - volume_spike(bar, sigma)
  - trend_agrees(daily_bars, direction, rule)
  - sector_confirms(sector_series)
  - position_size(risk_per_share)  # converts to shares/contracts for desired R
```

**Backtest Engine Expectations**
- Bar-by-bar or event-driven **replay** with **no look-ahead bias**.  
- **Order models** (market/limit/stop), realistic **slippage**, and **fees**.  
- Support **partial exits** and **trailers**.  
- Output a tidy **trade log** with MAE/MFE and timestamps for analysis.

**Trade Log Schema (CSV)**
```csv
symbol,setup,entry_time,entry,stop,exit1_time,exit1,exit2_time,exit2,shares,side,mae,mfe
```

---

## Glossary

- **NR7:** Day whose high‑low range is the smallest of the last 7 sessions.  
- **ORB (Opening Range Breakout):** Trade on break of the first N minutes’ range.  
- **RVOL:** Relative volume vs typical volume at the same time of day.  
- **VWAP:** Volume‑weighted average price.  
- **Reclaim:** Price moves back above/below a breached level and **holds**.  
- **Imbalance:** Net aggressive buying/selling pressure over a window.  
- **R:** Risk unit = `entry - stop` (abs). Targets often specified in R multiples.

---

### Quick Usage Notes
- Start with **A1** and **A2** for the cleanest intraday contexts (squeeze or fresh news).  
- Use the **strength score** to keep signals rare.  
- Prefer **liquid names in play**; track **RVOL** and **VWAP** behavior religiously.  
- Keep risk tight; **one setup, one shot**.

*End of document.*
