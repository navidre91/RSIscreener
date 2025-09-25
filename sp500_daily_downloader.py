#!/usr/bin/env python3
"""
S&P 500 Daily Data Downloader
-----------------------------
Downloads and organizes daily OHLCV (+ dividends & splits) for all current S&P 500 constituents.

Features
- Fetch current S&P 500 tickers (Yahoo format) + metadata
- Per‑ticker Parquet files: data/sp500_daily/{TICKER}.parquet
- Optional combined Parquet: data/sp500_daily_all.parquet
- Incremental resume: if a per‑ticker file exists, append only missing dates
- Robust retries + progress bar

Usage
-----
# 1) Install deps:
#    pip install -r requirements.txt
#
# 2) Run with defaults (starts in 1990):
#    python sp500_daily_downloader.py
#
# 3) Custom range and options:
#    python sp500_daily_downloader.py --start 1980-01-01 --end 2025-09-25 --out data --combined
#
# Notes
# - Requires Internet access at runtime to fetch tickers/quotes.
# - yfinance pulls from Yahoo Finance. For commercial use, check data licensing/TOS yourself.

"""

import argparse
import json
import math
import os
import sys
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime, timedelta, date
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import pandas as pd
import yfinance as yf
from tqdm import tqdm

# ---------- Config defaults ----------

DEFAULT_START = "1990-01-01"
# End is inclusive date. We'll internally add +1 day because yfinance `end` is exclusive.
DEFAULT_OUTDIR = "data"
DEFAULT_MAX_WORKERS = 12
DEFAULT_RETRIES = 3
DEFAULT_BACKOFF = 2.0  # exponential backoff base seconds
DEFAULT_TIMEOUT = 60   # per request soft timeout, seconds (best-effort)
DEFAULT_AUTO_ADJUST = True

# ---------- Helpers ----------

def _parse_date(d: Optional[str]) -> Optional[date]:
    if d is None:
        return None
    return datetime.strptime(d, "%Y-%m-%d").date()

def _today() -> date:
    # Use UTC "today" to avoid timezone surprises
    return datetime.utcnow().date()

def _yahoo_symbol(symbol: str) -> str:
    # Wikipedia uses '.' for class shares; Yahoo uses '-'
    # Example: BRK.B -> BRK-B ; BF.B -> BF-B
    return symbol.replace(".", "-").strip().upper()

def _ensure_dir(p: Path) -> None:
    p.mkdir(parents=True, exist_ok=True)

def _normalize_history_df(df: pd.DataFrame, ticker: str, auto_adjust: bool) -> pd.DataFrame:
    """
    Normalize yfinance Ticker.history() dataframe into a tidy table.
    Columns mapped to: date, open, high, low, close, adj_close, volume, dividends, stock_splits, ticker
    """
    if df is None or df.empty:
        return pd.DataFrame(columns=[
            "date","open","high","low","close","adj_close","volume","dividends","stock_splits","ticker"
        ])

    # Index is DatetimeIndex; ensure it's tz-naive (yfinance daily typically is already)
    df = df.copy()
    df.index.name = "date"
    df.reset_index(inplace=True)

    colmap = {}
    if "Open" in df.columns: colmap["Open"] = "open"
    if "High" in df.columns: colmap["High"] = "high"
    if "Low" in df.columns: colmap["Low"] = "low"
    if "Close" in df.columns: colmap["Close"] = "close"
    if "Adj Close" in df.columns: colmap["Adj Close"] = "adj_close"
    if "Volume" in df.columns: colmap["Volume"] = "volume"
    if "Dividends" in df.columns: colmap["Dividends"] = "dividends"
    if "Stock Splits" in df.columns: colmap["Stock Splits"] = "stock_splits"
    df.rename(columns=colmap, inplace=True)

    # Ensure adj_close exists even when auto_adjust=True (Adj Close absent)
    if "adj_close" not in df.columns and "close" in df.columns and auto_adjust:
        df["adj_close"] = df["close"]

    # Fill optional columns if missing
    for c in ["dividends", "stock_splits"]:
        if c not in df.columns:
            df[c] = 0.0

    # Enforce minimal schema
    needed = ["date","open","high","low","close","adj_close","volume","dividends","stock_splits"]
    for c in needed:
        if c not in df.columns:
            df[c] = pd.NA

    df["ticker"] = ticker
    # Keep only expected columns, in order
    df = df[["date","open","high","low","close","adj_close","volume","dividends","stock_splits","ticker"]]
    # Sort and drop duplicate dates
    df["date"] = pd.to_datetime(df["date"]).dt.date
    df = df.drop_duplicates(subset=["date"]).sort_values("date")

    return df

def _save_parquet(df: pd.DataFrame, path: Path) -> None:
    if df is None or df.empty:
        return
    try:
        import pyarrow  # noqa: F401
        df.to_parquet(path, index=False)
    except Exception as e:
        # Fallback to CSV if parquet unavailable
        csv_path = path.with_suffix(".csv")
        df.to_csv(csv_path, index=False)
        print(f"[warn] pyarrow not available or failed to write parquet; wrote CSV instead: {csv_path} ({e})")

def _read_parquet_or_csv(path: Path) -> pd.DataFrame:
    if not path.exists():
        return pd.DataFrame()
    if path.suffix.lower() == ".parquet" or (path.with_suffix(".parquet").exists() and path.suffix.lower() != ".csv"):
        # If asked to read .parquet but the file is CSV, we will try parquet first, then CSV
        try:
            return pd.read_parquet(path)
        except Exception:
            # maybe CSV exists instead
            csv_path = path.with_suffix(".csv")
            if csv_path.exists():
                return pd.read_csv(csv_path, parse_dates=["date"])
            raise
    if path.suffix.lower() == ".csv":
        return pd.read_csv(path, parse_dates=["date"])
    # default parquet
    try:
        return pd.read_parquet(path)
    except Exception:
        csv_path = path.with_suffix(".csv")
        if csv_path.exists():
            return pd.read_csv(csv_path, parse_dates=["date"])
        raise

def fetch_sp500_metadata() -> Tuple[pd.DataFrame, List[str]]:
    """
    Return (metadata_df, tickers_yahoo)
    Attempts yfinance's helpers first; falls back to parsing Wikipedia via pandas.read_html.
    """
    # yfinance has built-in helpers (requiring internet)
    try:
        tickers = yf.tickers_sp500()  # already in Yahoo format (e.g., BRK-B)
        # Metadata via pandas.read_html to get names/sectors
        tables = pd.read_html("https://en.wikipedia.org/wiki/List_of_S%26P_500_companies")
        meta = tables[0]
        meta.columns = [c.lower().replace(" ", "_") for c in meta.columns]
        # Expected columns: symbol, security, gics_sector, gics_sub-industry, headquarters_location, date_added, cik, founded
        meta["symbol_yahoo"] = meta["symbol"].map(_yahoo_symbol)
        meta.rename(columns={
            "symbol": "symbol_wiki",
            "security": "company",
            "gics_sector": "sector",
            "gics_sub-industry": "sub_industry",
        }, inplace=True)
        # Ensure we only keep the tickers present in yfinance list
        meta = meta[meta["symbol_yahoo"].isin(set(tickers))].reset_index(drop=True)
        # Reorder a bit
        cols = ["symbol_wiki","symbol_yahoo","company","sector","sub_industry","headquarters_location","date_added","cik","founded"]
        meta = meta[[c for c in cols if c in meta.columns]]
        return meta, tickers
    except Exception as e1:
        # Fallback: everything from Wikipedia table
        tables = pd.read_html("https://en.wikipedia.org/wiki/List_of_S%26P_500_companies")
        meta = tables[0]
        meta.columns = [c.lower().replace(" ", "_") for c in meta.columns]
        meta["symbol_yahoo"] = meta["symbol"].map(_yahoo_symbol)
        meta.rename(columns={
            "symbol": "symbol_wiki",
            "security": "company",
            "gics_sector": "sector",
            "gics_sub-industry": "sub_industry",
        }, inplace=True)
        tickers = meta["symbol_yahoo"].dropna().astype(str).str.upper().tolist()
        cols = ["symbol_wiki","symbol_yahoo","company","sector","sub_industry","headquarters_location","date_added","cik","founded"]
        meta = meta[[c for c in cols if c in meta.columns]]
        return meta, tickers

def _retryable_history(ticker: str, start_date: date, end_date: date, auto_adjust: bool, retries: int, backoff: float) -> pd.DataFrame:
    """
    Call yf.Ticker(ticker).history with retries and backoff.
    """
    last_exc = None
    for attempt in range(1, retries + 2):  # retries + final attempt
        try:
            tk = yf.Ticker(ticker)
            # yfinance end is exclusive for daily; add +1 day to be inclusive
            df = tk.history(
                start=start_date.strftime("%Y-%m-%d"),
                end=(end_date + timedelta(days=1)).strftime("%Y-%m-%d"),
                interval="1d",
                auto_adjust=auto_adjust,
                actions=True,
            )
            return df
        except Exception as e:
            last_exc = e
            sleep_s = backoff ** attempt
            time.sleep(sleep_s)
    raise RuntimeError(f"Failed to fetch history for {ticker} after retries: {last_exc}")

def _download_one(ticker: str, outdir: Path, start_dt: date, end_dt: date, auto_adjust: bool, retries: int, backoff: float, resume: bool) -> Tuple[str, int, int]:
    """
    Download one ticker and write/append its parquet file.
    Returns: (ticker, n_existing_rows, n_new_rows)
    """
    tpath = outdir / f"{ticker}.parquet"
    existing = pd.DataFrame()
    n_existing = 0

    effective_start = start_dt
    if resume and tpath.exists():
        try:
            existing = _read_parquet_or_csv(tpath)
            if not existing.empty:
                n_existing = len(existing)
                # Compute next date to fetch
                max_date = pd.to_datetime(existing["date"]).dt.date.max()
                next_day = max_date + timedelta(days=1)
                # Only move start forward
                if next_day > effective_start:
                    effective_start = next_day
        except Exception as e:
            print(f"[warn] Failed reading existing file for {ticker}: {e}. Will attempt full refresh.")

    if effective_start > end_dt:
        # Already up to date
        return ticker, n_existing, 0

    raw = _retryable_history(ticker, effective_start, end_dt, auto_adjust=auto_adjust, retries=retries, backoff=backoff)
    tidy = _normalize_history_df(raw, ticker, auto_adjust=auto_adjust)

    if not existing.empty:
        # Append and drop duplicates by date
        combined = pd.concat([existing, tidy], ignore_index=True)
        combined = combined.drop_duplicates(subset=["date"]).sort_values("date").reset_index(drop=True)
        _save_parquet(combined, tpath)
        n_new = len(combined) - n_existing
    else:
        _save_parquet(tidy, tpath)
        n_new = len(tidy)

    return ticker, n_existing, n_new

def build_combined(outdir: Path, combined_path: Path) -> Tuple[int, int]:
    """
    Combine all per-ticker parquet/csv files into a single Parquet.
    Returns (n_rows, n_tickers).
    """
    files = sorted(list(outdir.glob("*.parquet"))) + sorted(list(outdir.glob("*.csv")))
    frames = []
    tickers = 0
    for fp in tqdm(files, desc="Combining"):
        try:
            df = _read_parquet_or_csv(fp)
            if df is not None and not df.empty:
                # Ensure schema
                keep = ["date","open","high","low","close","adj_close","volume","dividends","stock_splits","ticker"]
                for c in keep:
                    if c not in df.columns:
                        df[c] = pd.NA
                df = df[keep]
                frames.append(df)
                tickers += 1
        except Exception as e:
            print(f"[warn] Skipping {fp.name}: {e}")
    if not frames:
        print("[warn] No per-ticker files found to combine.")
        return 0, 0
    big = pd.concat(frames, ignore_index=True)
    big["date"] = pd.to_datetime(big["date"]).dt.date
    big.sort_values(["ticker","date"], inplace=True)
    _save_parquet(big, combined_path)
    return len(big), tickers

def main():
    parser = argparse.ArgumentParser(description="Download daily OHLCV for S&P 500 constituents.")
    parser.add_argument("--start", type=str, default=DEFAULT_START, help="Start date (YYYY-MM-DD). Default: %(default)s")
    parser.add_argument("--end", type=str, default=None, help="End date (YYYY-MM-DD). Default: today (UTC)")
    parser.add_argument("--out", type=str, default=DEFAULT_OUTDIR, help="Output base directory. Default: %(default)s")
    parser.add_argument("--auto-adjust", dest="auto_adjust", action="store_true", default=DEFAULT_AUTO_ADJUST, help="Return adjusted OHLC (yfinance auto_adjust=True). Default: %(default)s")
    parser.add_argument("--no-auto-adjust", dest="auto_adjust", action="store_false", help="Disable auto-adjust (keeps unadjusted OHLC and Adj Close).")
    parser.add_argument("--workers", type=int, default=DEFAULT_MAX_WORKERS, help="Max concurrent downloads. Default: %(default)s")
    parser.add_argument("--retries", type=int, default=DEFAULT_RETRIES, help="Retries per ticker on failure. Default: %(default)s")
    parser.add_argument("--backoff", type=float, default=DEFAULT_BACKOFF, help="Exponential backoff base seconds. Default: %(default)s")
    parser.add_argument("--resume", dest="resume", action="store_true", default=True, help="Resume from existing per-ticker files if present. Default: %(default)s")
    parser.add_argument("--fresh", dest="resume", action="store_false", help="Ignore existing files and refetch (will overwrite).")
    parser.add_argument("--combined", action="store_true", help="Also build a combined Parquet at the end.")
    parser.add_argument("--tickers-file", type=str, default=None, help="Optional path to newline-separated custom tickers (Yahoo format).")
    args = parser.parse_args()

    start_dt = _parse_date(args.start)
    if start_dt is None:
        print("ERROR: --start must be YYYY-MM-DD", file=sys.stderr)
        sys.exit(2)
    end_dt = _parse_date(args.end) if args.end else _today()
    if end_dt < start_dt:
        print("ERROR: --end cannot be before --start", file=sys.stderr)
        sys.exit(2)

    base = Path(args.out).expanduser().resolve()
    data_dir = base / "sp500_daily"
    meta_dir = base / "metadata"
    _ensure_dir(data_dir)
    _ensure_dir(meta_dir)

    # --- Get S&P 500 tickers & metadata ---
    if args.tickers_file:
        with open(args.tickers_file, "r") as f:
            tickers = [line.strip().upper() for line in f if line.strip()]
        meta_df = pd.DataFrame({"symbol_yahoo": tickers})
    else:
        meta_df, tickers = fetch_sp500_metadata()
        # Save metadata
        meta_path = meta_dir / "sp500_constituents.parquet"
        _save_parquet(meta_df, meta_path)

    if not tickers:
        print("ERROR: No tickers found.", file=sys.stderr)
        sys.exit(1)

    print(f"Found {len(tickers)} S&P 500 tickers.")
    # --- Download per ticker in parallel ---
    results = []
    with ThreadPoolExecutor(max_workers=args.workers) as ex:
        futures = []
        for t in tickers:
            t_out = data_dir / f"{t}.parquet"
            # If not resuming (fresh), remove existing
            if not args.resume and t_out.exists():
                try:
                    t_out.unlink()
                except Exception:
                    pass
            futures.append(ex.submit(
                _download_one, t, data_dir, start_dt, end_dt, args.auto_adjust, args.retries, args.backoff, args.resume
            ))
        for fut in tqdm(as_completed(futures), total=len(futures), desc="Downloading"):
            try:
                ticker, n_existing, n_new = fut.result()
                results.append((ticker, n_existing, n_new))
            except Exception as e:
                print(f"[error] {e}")

    # --- Summary & combined ---
    total_new = sum(n for _,_,n in results)
    up_to_date = sum(1 for _,_,n in results if n == 0)
    print(f"Completed. Added {total_new:,} new rows across {len(results)} tickers; {up_to_date} already up-to-date.")

    meta_json = {
        "run_completed_at_utc": datetime.utcnow().isoformat(timespec="seconds"),
        "start_date": start_dt.isoformat(),
        "end_date": end_dt.isoformat(),
        "tickers_count": len(tickers),
        "auto_adjust": bool(args.auto_adjust),
        "resume": bool(args.resume),
        "workers": int(args.workers),
    }
    with open(meta_dir / "run_metadata.json", "w") as f:
        json.dump(meta_json, f, indent=2)

    if args.combined:
        combined_path = base / "sp500_daily_all.parquet"
        n_rows, n_tickers = build_combined(data_dir, combined_path)
        print(f"Built combined file: {combined_path} ({n_rows:,} rows from {n_tickers} tickers)")

if __name__ == "__main__":
    main()