# Swing RSI (Daily) Scanner — Algorithm and Usage

This document explains in detail how `swingRSI.py` works, what the signal means, how option suggestions are produced, and how to run or schedule the scanner.

File: `swingRSI.py`

## Overview
- Purpose: Detect a daily swing‑low RSI setup and propose simple call option structures (vertical or butterfly) to enter on the next trading day.
- Data: Alpaca Market Data (IEX feed). Universe: the current S&P 500 constituents from Wikipedia (with a CSV fallback).
- Output: A concise Telegram message summarizing qualifying symbols and suggested strikes.
- Run window: Near the US market open (09:35–09:45 ET) so yesterday’s daily bar is complete and today is the “next bar”. A `--force-run` override is available for manual backfills.

## Signal Definition
For the most recent completed daily bar `t` (i.e., the last bar at the time of running):

```
RSI = RSI(close, length=14)  # Wilder’s RSI
if low[t] == min(low[t-4 : t+1]):         # 5-day swing low using only past
    A_day = t - 2
    if RSI[A_day] < 35 and RSI[t] > RSI[A_day] + 3:
        P_A = close[A_day]                # target
        B   = low[t]                      # swing low reference
        # Trade is placed on the next bar (t+1)
```

Interpretation:
- Identify a fresh 5‑day swing low at `t` using only information available at `t` (no future bars).
- Require RSI to be depressed two bars earlier (`A_day = t-2`), and for RSI to have improved by at least 3 points by `t`.
- Use `P_A` (the close at `A_day`) as a reference target and `B` (the low at `t`) as the swing‑low anchor.

## Step‑by‑Step Logic
1. Pull daily OHLCV for the S&P 500 universe with enough lookback (default 400 bars).
2. Compute RSI(14) using Wilder’s smoothing on closing prices.
3. Let `t = last index` (the most recent completed daily bar at run time).
4. Check 5‑day swing low condition: `low[t] == min(low[t-4 : t])` (inclusive window of size 5). A tiny floating tolerance is applied in code.
5. Set `A_day = t - 2`. If `A_day < 0`, skip.
6. Evaluate RSI conditions: `RSI[A_day] < 35` and `RSI[t] > RSI[A_day] + 3`.
7. If satisfied, compute:
   - `P_A = close[A_day]`  (target reference)
   - `B   = low[t]`        (swing‑low)
   - `current_price = close[t]` (used for strike rounding; you can substitute a real‑time price if desired.)
8. Build option suggestions (vertical and/or butterfly) for “next bar” entry.
9. Aggregate all hits, sort by strongest RSI improvement `(RSI[t] - RSI[A_day])`, and send a Telegram summary.

## RSI Calculation (Wilder’s)
Let `delta = close.diff()`.
- `gains = max(delta, 0)`; `losses = max(-delta, 0)`.
- Wilder’s smoothing via EWM: `avg_gain = EWM(alpha=1/period, min_periods=period)`, similarly for `avg_loss`.
- `RS = avg_gain / avg_loss`; `RSI = 100 - (100 / (1 + RS))`.

This formulation yields the commonly used Wilder RSI and is stable for streaming daily bars.

## Swing Low Definition
- A 5‑bar swing low at `t` means `low[t]` is the minimum of `low[t-4]..low[t]`. This is a past‑only decision; no future data leakage.
- A small absolute tolerance is used in code to mitigate floating‑point comparison sensitivity.

## Why `A_day = t‑2`?
- The setup looks for depressed RSI values two days before the pivot (`A_day`) and a subsequent improvement at the pivot day `t`.
- Using `t-2` avoids anchoring to the immediate bar before `t`, which can be noisy; it often approximates the “impulse preceding the pivot”.

## Derived Levels
- `P_A = close[A_day]`: a natural reference/target derived from the earlier context.
- `B = low[t]`: the swing‑low reference that defined the pivot.

## Option Structure Suggestions
The scanner provides non‑binding suggestions, not orders. Position management and risk/exit rules are out of scope by design.

- Vertical call spread:
  - `K1 = round_to_liquid_strike(current_price)`
  - `K2 = round_to_liquid_strike(P_A)`
  - DTE guidance: 21–45 days
- Symmetric call butterfly:
  - `K2 = round_to_liquid_strike(P_A)`
  - `K1 = round_to_liquid_strike(current_price)`
  - `K3 = K2 + (K2 - K1)` (symmetry about `K2`)
  - DTE guidance: 14–35 days

Strike rounding heuristic (`round_to_liquid_strike`):
- Price < $25: increments of $0.50
- $25–$200: increments of $1.00
- $200–$500: increments of $5.00
- ≥ $500: increments of $10.00

These buckets are a pragmatic approximation of common US option strike grids and can be adjusted in code if your venue differs.

## Data & Universe
- Universe: S&P 500 list scraped from Wikipedia (fallback to a maintained public CSV if scraping fails).
- Market data: Alpaca Market Data (IEX feed), timeframe `1Day`, with `adjustment=all`, ordered ascending.
- Default lookback: 400 trading days.

## Run Window and “Next Bar”
- The script enforces a narrow open window (09:35–09:45 ET) on trading days to ensure:
  - The most recent daily bar is “yesterday” and fully formed.
  - Entries (if you place them) correspond to “the next bar” (today’s session).
- Override: `--force-run` bypasses the time gate for manual tests or backfills.

## Output Format (Telegram)
For each hit, the message shows:
- Symbol; dates for `t` and `A_day` (ET)
- `RSI_t`, `RSI_A`, latest close `Px`, `P_A`, and `B(low)`
- Suggested structure(s) with strikes and DTE guidance

Example (wrapped for clarity):
```
📅 Swing RSI scan — 2025-09-25 09:37 ET
Signals: 3  |  Options: both

AAPL    t=2025-09-24 A=2025-09-22  RSI_t=36.5 RSI_A=31.9  Px=190.35  P_A=186.90  B(low)=188.20
    butterfly: K1=190.00, K2=187.00, K3=184.00, DTE 14–35
    vertical:  K1=190.00, K2=187.00, DTE 21–45
```

Note: The script does not submit orders; entries, sizing, and exits are your responsibility.

## Configuration
Environment variables (required):
- `ALPACA_API_KEY_ID`, `ALPACA_API_SECRET_KEY` — for Alpaca data.
- `TELEGRAM_BOT_TOKEN`, `TELEGRAM_CHAT_ID` — for notifications.

Environment variables (optional):
- `RSI_PERIOD` (default `14`)
- `LOOKBACK_DAYS` (default `400`)
- `SWING_LEFT_WINDOW` (default `4`) → 5‑bar swing low window
- `SWING_OPTIONS_STRUCTURE` — `butterfly` | `vertical` | `both` (default `butterfly`)
- `BUTTERFLY_DTE_RANGE` (default `14–35`)
- `VERTICAL_DTE_RANGE` (default `21–45`)
- `LOGLEVEL` (default `INFO`)

CLI flags:
- `--force-run` — bypass the open‑window time gate.
- `--structure {butterfly,vertical,both}` — per‑run override for suggestion type (overrides `SWING_OPTIONS_STRUCTURE`).

## Usage
Manual runs:
- Default (enforces time gate):
  - `python swingRSI.py`
- Force run now:
  - `python swingRSI.py --force-run`
- Force run with vertical suggestions:
  - `python swingRSI.py --force-run --structure vertical`

GitHub Actions (scheduled and manual):
- Scheduled workflow: `.github/workflows/rsi-swing.yml` runs on weekdays around 09:37 ET.
- Manual one‑off force run: use the “Run workflow” button and set the `force_run` input to true. The workflow will append `--force-run` only for that manual invocation.

## Implementation Notes (for developers)
Key functions in `swingRSI.py`:
- `fetch_daily_bars(symbols)` — batches requests to Alpaca, returns per‑symbol DataFrames sorted ascending with columns `[t,o,h,l,c,v]`.
- `wilder_rsi(close, period)` — Wilder’s RSI using EWM smoothing.
- `detect_latest_swing_signal(df)` — checks only the latest bar `t` for the setup; returns a `SwingSignal` or `None`.
- `round_to_liquid_strike(price)` — strike rounding heuristic by price tier.
- `scan_symbols(symbols)` — applies detection to all symbols and builds suggestion strings for the configured structure(s).
- `is_open_window_daily()` — 09:35–09:45 ET gate using Alpaca’s clock API.

Sorting: results are ordered by `(RSI[t] - RSI[A_day])` descending, favoring the strongest RSI improvements.

Numerical details:
- Float comparisons for the swing low use an absolute tolerance to avoid rejecting equal minima due to representation noise.

## Limitations & Extensions
- “current_price” uses the latest daily close. For live entries, you might replace it with an intraday mid or pre‑open print.
- Strike grid buckets are heuristic; adjust per your brokerage/venue if needed.
- No order placement or risk/exit logic — only signal detection and suggestions.
- If you need to detect the setup historically (not just the last bar), extend `detect_latest_swing_signal` to iterate all bars and gate with “past‑only” windows.

## Disclaimer
This tool is for informational and educational purposes only and does not constitute financial advice or an offer to buy/sell securities or derivatives. Options involve significant risk and are not suitable for all investors. Always perform your own due diligence.

