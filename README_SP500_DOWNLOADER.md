# S&P 500 Daily Downloader

This script fetches daily OHLCV (+ dividends & splits) for all *current* S&P 500 constituents and stores them as tidy Parquet files. It uses the widely-used `yfinance` package (Yahoo Finance).

> **Note**: The list of S&P 500 tickers and the price data are retrieved live at runtime. For production use, review Yahoo's TOS/licensing yourself.

## Quick Start

```bash
# 1) Create a virtual env (recommended)
python -m venv .venv && . .venv/bin/activate  # (Windows: .venv\Scripts\activate)

# 2) Install dependencies
pip install -r requirements.txt

# 3) Run
python sp500_daily_downloader.py
```

By default, data are saved under `data/sp500_daily/{TICKER}.parquet`. A metadata file with the constituents is written to `data/metadata/sp500_constituents.parquet` and a small `run_metadata.json` is also created.

## Options

```bash
python sp500_daily_downloader.py   --start 1980-01-01   --end 2025-09-25   --out data   --combined   --workers 12
```

- `--start` / `--end`: date range (inclusive). If `--end` is omitted, today's UTC date is used.
- `--out`: base output folder (default: `data`). Per‑ticker files in `data/sp500_daily/`.
- `--combined`: also builds a single `sp500_daily_all.parquet` by concatenating all per‑ticker files.
- `--auto-adjust` / `--no-auto-adjust`: choose between adjusted or unadjusted OHLC. With `--auto-adjust` (default), `close` equals `adj_close`. With `--no-auto-adjust`, both `close` (unadjusted) and `adj_close` are present.
- `--resume` (default) / `--fresh`: resuming will append only missing dates per ticker when files already exist; `--fresh` refetches and overwrites.
- `--workers`: parallelism (default: 12).
- `--tickers-file`: path to a newline-separated list of Yahoo-format tickers if you want to override the S&P 500 list.

## Output Schema

Each per‑ticker Parquet has the following columns:

- `date` (date)
- `open`, `high`, `low`, `close`, `adj_close` (float)
- `volume` (int)
- `dividends` (float)
- `stock_splits` (float)
- `ticker` (string)

All files are sorted by `date` and deduplicated by date.

## Tips

- **First run:** use a longer start date (e.g., 1980-01-01). Subsequent runs can be incremental with `--resume`.
- **Tickers with dots:** (e.g., `BRK.B`) are automatically converted to Yahoo's `BRK-B` format.
- **Speed:** Increase `--workers` to parallelize more, but be respectful of remote rate limits. The script employs simple retries with exponential backoff.
- **Alternative data sources:** For guaranteed SLAs/licensing, consider commercial APIs (Polygon.io, Tiingo, Intrinio, etc.). You can adapt `_retryable_history` to call your provider instead of `yfinance`.