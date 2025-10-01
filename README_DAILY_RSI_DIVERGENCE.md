# Daily RSI Divergence Screener

This document explains how `screener_daily_rsi_divergence.py` scans the S&P 500 for daily bullish RSI(14) divergences and broadcasts ranked alerts to Telegram near the market open.

## High-Level Flow
- Verifies required Alpaca and Telegram credentials are present.
- Confirms the current time is within the 9:35-9:45 ET execution window on an Alpaca trading day.
- Retrieves the current S&P 500 constituents from Wikipedia with a GitHub CSV fallback.
- Downloads the last `LOOKBACK_DAYS` of daily bars for the universe via Alpaca Market Data (IEX feed).
- Calculates RSI(14) (configurable) and looks for bullish price/RSI divergence over the most recent `RECENT_BARS` candles using pivot windows of size `PIVOT_WINDOW`.
- Ranks bullish signals by a composite "strength" score and sends a formatted Telegram report.

## Configuration
The script is configured exclusively through environment variables:

| Variable | Purpose | Default |
| --- | --- | --- |
| `ALPACA_API_KEY_ID` / `ALPACA_API_SECRET_KEY` | Alpaca credentials for data access | required |
| `ALPACA_TRADE_BASE` | Alpaca trading API base (used for clock) | `https://paper-api.alpaca.markets` |
| `TELEGRAM_BOT_TOKEN` / `TELEGRAM_CHAT_ID` | Telegram bot credentials used for notifications | required |
| `RSI_PERIOD` | RSI calculation period | `14` |
| `TIMEFRAME` | Alpaca bar timeframe | `1Day` |
| `MAX_SYMBOLS_PER_BATCH` | Max symbols per Alpaca request | `300` |
| `LOOKBACK_DAYS` | Historical bar depth | `400` |
| `DIVERGENCE_TYPES` | Divergence types to search (`bullish` respected) | `bullish,bearish` |
| `DIVERGENCE_PIVOT_WINDOW` | Pivot span in bars for swing detection | `3` |
| `DIVERGENCE_RECENT_BARS` | Maximum age (bars) of the latest pivot | `20` |
| `LOGLEVEL` | Logging verbosity | `INFO` |

> Note: The current implementation filters to bullish divergences even if `DIVERGENCE_TYPES` includes `bearish`.

## Divergence Detection
- **Pivot identification**: Uses `PIVOT_WINDOW` to locate local lows (price) and evaluates pairs of the two most recent pivots.
- **Bullish criteria**: Requires the second price low to be lower than the first while RSI creates a higher low.
- **Recency filter**: Signals older than `RECENT_BARS` bars are discarded.
- **Strength scoring**: Signals are sorted by `strength = max(delta_RSI, 0) * max(delta_price_pct, 0)` with recent ties broken by pivot time.

## Telegram Output
- Header includes timestamp (ET) and the count of bullish matches.
- Each line contains:
  - Rank (`1` = strongest)
  - Symbol and fixed "Bullish" label
  - Strength score, RSI delta, percentage price drop between pivots
  - Latest RSI and price
  - Date of the divergence pivot
- Messages are split into multiple Telegram sends if the payload exceeds 4,000 characters.

## Running the Screener
```bash
python3 screener_daily_rsi_divergence.py
```

Run the script during market hours with valid Alpaca data entitlements. Many deployments schedule it twice around 9:35 ET to guarantee execution within the allowed window.

## Dependencies
- Python 3.9+
- Libraries: `requests`, `pandas`, `numpy`, and the standard library modules imported by the script.

Install dependencies using the project `requirements.txt` if not already satisfied.
