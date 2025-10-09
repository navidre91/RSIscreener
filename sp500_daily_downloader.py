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
import os
import sys
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime, timedelta, date, timezone
from pathlib import Path
from typing import Dict, List, Optional, Tuple
from collections import defaultdict

import pandas as pd
import yfinance as yf
from tqdm import tqdm
import requests
from requests.adapters import HTTPAdapter

# ---------- Config defaults ----------

DEFAULT_START = "1970-01-01"
# End is inclusive date. We'll internally add +1 day because yfinance `end` is exclusive.
DEFAULT_OUTDIR = "data"
DEFAULT_MAX_WORKERS = 8
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
    return datetime.now(timezone.utc).date()

# Reduce yfinance logging verbosity
import logging as _logging
_logging.getLogger("yfinance").setLevel(_logging.ERROR)
_logging.getLogger("yfinance.data").setLevel(_logging.ERROR)

# Shared requests Session for yfinance with enlarged connection pools.
_YF_SESSION = None  # type: ignore

def _configure_yf_session(pool_maxsize: int = 32) -> None:
    global _YF_SESSION
    s = requests.Session()
    adapter = HTTPAdapter(pool_connections=pool_maxsize, pool_maxsize=pool_maxsize)
    s.mount("https://", adapter)
    s.mount("http://", adapter)
    _YF_SESSION = s
    
def _is_rate_limited_error(exc: Exception) -> bool:
    m = str(exc).lower()
    return ("too many requests" in m) or ("rate limit" in m) or ("429" in m)

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


def _last_stored_date(ticker: str, data_dir: Path) -> Optional[date]:
    """
    Return the most recent date stored for ticker in data_dir, or None if missing/empty.
    """
    candidates = [
        data_dir / f"{ticker}.parquet",
        data_dir / f"{ticker}.csv",
    ]
    for path in candidates:
        if not path.exists():
            continue
        try:
            if path.suffix.lower() == ".parquet":
                df = pd.read_parquet(path, columns=["date"])
            else:
                df = pd.read_csv(path, usecols=["date"], parse_dates=["date"])
        except Exception:
            try:
                df = _read_parquet_or_csv(path)
            except Exception:
                return None
        if df is None or df.empty or "date" not in df.columns:
            return None
        col = pd.to_datetime(df["date"], errors="coerce")
        if col.isna().all():
            return None
        last_ts = col.max()
        if pd.isna(last_ts):
            return None
        if isinstance(last_ts, pd.Timestamp):
            return last_ts.date()
        if hasattr(last_ts, "to_pydatetime"):
            return last_ts.to_pydatetime().date()
        return last_ts
    return None


def _summarize_gap_status(
    tickers: List[str],
    data_dir: Path,
    target_end: date,
) -> Tuple[List[str], Dict[int, List[Tuple[str, date]]]]:
    """
    Inspect existing per-ticker files and group tickers by gap length (days).
    Returns (new_tickers, gap_map) where gap_map[gap] = list of (ticker, last_date).
    """
    gap_map: Dict[int, List[Tuple[str, date]]] = defaultdict(list)
    new_tickers: List[str] = []
    unique = list(dict.fromkeys(tickers))
    for sym in unique:
        last_date = _last_stored_date(sym, data_dir)
        if last_date is None:
            new_tickers.append(sym)
            continue
        gap = (target_end - last_date).days
        if gap < 0:
            gap = 0
        gap_map[gap].append((sym, last_date))
    new_tickers.sort()
    for entries in gap_map.values():
        entries.sort(key=lambda item: item[0])
    return new_tickers, gap_map


def _print_gap_summary(new_tickers: List[str], gap_map: Dict[int, List[Tuple[str, date]]], target_end: date) -> None:
    """
    Print a human-readable summary of tickers grouped by gap length.
    """
    print(f"Pre-update status vs target end {target_end.isoformat()}:")
    if new_tickers:
        sample = ", ".join(new_tickers[:10])
        if len(new_tickers) > 10:
            sample += ", ..."
        print(f"  New tickers (no local data): {len(new_tickers)} [{sample}]")
    else:
        print("  New tickers (no local data): 0")

    if not gap_map:
        return

    for gap in sorted(gap_map.keys()):
        entries = gap_map[gap]
        if gap == 0:
            print(f"  Up-to-date (gap 0 days): {len(entries)} tickers")
            continue
        sample_entries = ", ".join(f"{sym} (last {dt.isoformat()})" for sym, dt in entries[:5])
        if len(entries) > 5:
            sample_entries += ", ..."
        print(f"  Gap {gap} day{'s' if gap != 1 else ''}: {len(entries)} tickers [{sample_entries}]")

def fetch_sp500_metadata(meta_dir: Optional[Path] = None) -> Tuple[pd.DataFrame, List[str]]:
    """
    Return (metadata_df, tickers_yahoo)
    Strategy:
    1) If a cached metadata file exists in meta_dir, load from cache.
    2) Otherwise fetch Wikipedia with requests (custom User-Agent) and parse via pandas.read_html.
    Notes:
    - yfinance.tickers_sp500() is deprecated/removed in some versions, so we don't rely on it.
    """
    # 1) Try cache if provided
    if meta_dir is not None:
        cache_parquet = meta_dir / "sp500_constituents.parquet"
        cache_csv = meta_dir / "sp500_constituents.csv"
        for fp in (cache_parquet, cache_csv):
            if fp.exists():
                try:
                    cached = _read_parquet_or_csv(fp)
                    if cached is not None and not cached.empty:
                        # Expect a column named symbol_yahoo in cache; if absent, derive from symbol_wiki
                        if "symbol_yahoo" not in cached.columns and "symbol_wiki" in cached.columns:
                            cached = cached.copy()
                            cached["symbol_yahoo"] = cached["symbol_wiki"].map(_yahoo_symbol)
                        tickers = (
                            cached.get("symbol_yahoo")
                            .dropna()
                            .astype(str)
                            .str.upper()
                            .tolist()
                        )
                        if tickers:
                            return cached, tickers
                except Exception:
                    pass  # ignore cache read issues and proceed to network fetch

    # 2) Fetch from Wikipedia with requests (to avoid 403 without UA)
    url = "https://en.wikipedia.org/wiki/List_of_S%26P_500_companies"
    try:
        headers = {
            "User-Agent": (
                "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) "
                "AppleWebKit/605.1.15 (KHTML, like Gecko) Version/17.0 Safari/605.1.15"
            )
        }
        resp = requests.get(url, headers=headers, timeout=30)
        resp.raise_for_status()
        tables = pd.read_html(resp.text)
        meta = tables[0]
        meta.columns = [c.lower().replace(" ", "_") for c in meta.columns]
        # Expected columns: symbol, security, gics_sector, gics_sub-industry, headquarters_location, date_added, cik, founded
        if "symbol" not in meta.columns:
            raise RuntimeError("Wikipedia table format unexpected: missing 'Symbol' column")
        meta["symbol_yahoo"] = meta["symbol"].map(_yahoo_symbol)
        meta.rename(columns={
            "symbol": "symbol_wiki",
            "security": "company",
            "gics_sector": "sector",
            "gics_sub-industry": "sub_industry",
        }, inplace=True)
        tickers = meta["symbol_yahoo"].dropna().astype(str).str.upper().tolist()
        cols = [
            "symbol_wiki","symbol_yahoo","company","sector","sub_industry",
            "headquarters_location","date_added","cik","founded"
        ]
        meta = meta[[c for c in cols if c in meta.columns]]
        return meta, tickers
    except Exception as e:
        msg = (
            f"Failed to retrieve S&P 500 constituents from Wikipedia ({e}). "
            "If you are offline or blocked, either: "
            "1) pass --tickers-file with a newline-separated Yahoo tickers list, or "
            "2) place a cached file at <out>/metadata/sp500_constituents.parquet(csv)."
        )
        raise RuntimeError(msg)

def _retryable_history(ticker: str, start_date: date, end_date: date, auto_adjust: bool, retries: int, backoff: float) -> pd.DataFrame:
    """
    Call yf.Ticker(ticker).history with retries and backoff.
    """
    last_exc = None
    delay = 1.0
    for attempt in range(1, retries + 2):  # retries + final attempt
        try:
            tk = yf.Ticker(ticker, session=_YF_SESSION)
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
            # Increase wait on rate limits
            if _is_rate_limited_error(e):
                delay *= 1.5
            sleep_s = max(delay, (backoff ** attempt))
            time.sleep(sleep_s)
    raise RuntimeError(f"Failed to fetch history for {ticker} after retries: {last_exc}")

def _retryable_history_max(ticker: str, auto_adjust: bool, retries: int, backoff: float) -> pd.DataFrame:
    """
    Retrieve full available history using period="max".
    """
    last_exc = None
    delay = 1.0
    for attempt in range(1, retries + 2):
        try:
            tk = yf.Ticker(ticker, session=_YF_SESSION)
            df = tk.history(
                period="max",
                interval="1d",
                auto_adjust=auto_adjust,
                actions=True,
            )
            return df
        except Exception as e:
            last_exc = e
            if _is_rate_limited_error(e):
                delay *= 1.5
            sleep_s = max(delay, (backoff ** attempt))
            time.sleep(sleep_s)
    raise RuntimeError(f"Failed to fetch full history for {ticker} after retries: {last_exc}")

def _download_one(ticker: str, outdir: Path, start_dt: date, end_dt: date, auto_adjust: bool, retries: int, backoff: float, resume: bool, use_max_on_new: bool) -> Tuple[str, int, int]:
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

    if n_existing == 0 and use_max_on_new:
        raw = _retryable_history_max(ticker, auto_adjust=auto_adjust, retries=retries, backoff=backoff)
    else:
        raw = _retryable_history(ticker, effective_start, end_dt, auto_adjust=auto_adjust, retries=retries, backoff=backoff)
    tidy = _normalize_history_df(raw, ticker, auto_adjust=auto_adjust)

    if tidy is None or tidy.empty:
        # Nothing new downloaded; keep existing stats
        return ticker, n_existing, 0

    if not existing.empty:
        # Append and drop duplicates by date
        pieces = [existing]
        if not tidy.empty:
            pieces.append(tidy)
        combined = pd.concat(pieces, ignore_index=True)
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


def update_sp500_daily(
    start: Optional[str] = DEFAULT_START,
    end: Optional[str] = None,
    out: str = DEFAULT_OUTDIR,
    auto_adjust: bool = DEFAULT_AUTO_ADJUST,
    workers: int = DEFAULT_MAX_WORKERS,
    retries: int = DEFAULT_RETRIES,
    backoff: float = DEFAULT_BACKOFF,
    resume: bool = True,
    combined: bool = False,
    tickers_file: Optional[str] = None,
    use_max_on_new: bool = True,
    show_summary: bool = True,
) -> Dict[str, object]:
    """
    Programmatic API to download/update S&P 500 daily OHLCV.

    Returns a summary dict with counts, optional combined file info, and gap diagnostics.
    """
    if start is None:
        start_dt = _parse_date(DEFAULT_START)
    else:
        start_dt = _parse_date(start)
    if start_dt is None:
        raise ValueError("start must be YYYY-MM-DD")

    if end is None:
        end_dt = _today()
    else:
        end_dt = _parse_date(end)
    if end_dt is None:
        raise ValueError("end must be YYYY-MM-DD")

    if end_dt < start_dt:
        raise ValueError("end cannot be before start")

    base = Path(out).expanduser().resolve()
    data_dir = base / "sp500_daily"
    meta_dir = base / "metadata"
    _ensure_dir(data_dir)
    _ensure_dir(meta_dir)

    try:
        _configure_yf_session(pool_maxsize=max(20, int(workers) * 2))
    except Exception:
        pass

    if tickers_file:
        with open(tickers_file, "r") as f:
            tickers = [line.strip().upper() for line in f if line.strip()]
        meta_df = pd.DataFrame({"symbol_yahoo": tickers})
        _save_parquet(meta_df, meta_dir / "sp500_constituents.parquet")
    else:
        meta_df, tickers = fetch_sp500_metadata(meta_dir)
        _save_parquet(meta_df, meta_dir / "sp500_constituents.parquet")

    if not tickers:
        raise RuntimeError("No tickers found.")

    tickers = [t.strip().upper() for t in tickers if t.strip()]

    if show_summary:
        print(f"Found {len(tickers)} S&P 500 tickers.")

    new_tickers, gap_map = _summarize_gap_status(tickers, data_dir, end_dt)
    if show_summary:
        _print_gap_summary(new_tickers, gap_map, end_dt)

    results = []
    with ThreadPoolExecutor(max_workers=int(workers)) as ex:
        futures = []
        for t in tickers:
            t_out = data_dir / f"{t}.parquet"
            if not resume and t_out.exists():
                try:
                    t_out.unlink()
                except Exception:
                    pass
            futures.append(ex.submit(
                _download_one, t, data_dir, start_dt, end_dt, auto_adjust, retries, backoff, resume, use_max_on_new
            ))
        for fut in tqdm(as_completed(futures), total=len(futures), desc="Downloading"):
            try:
                ticker, n_existing, n_new = fut.result()
                results.append((ticker, n_existing, n_new))
            except Exception as e:
                print(f"[error] {e}")

    total_new = sum(n for _,_,n in results)
    up_to_date = sum(1 for _,_,n in results if n == 0)
    if show_summary:
        print(f"Completed. Added {total_new:,} new rows across {len(results)} tickers; {up_to_date} already up-to-date.")

    meta_json = {
        "run_completed_at_utc": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "start_date": start_dt.isoformat(),
        "end_date": end_dt.isoformat(),
        "tickers_count": len(tickers),
        "auto_adjust": bool(auto_adjust),
        "resume": bool(resume),
        "workers": int(workers),
        "use_max_on_new": bool(use_max_on_new),
    }
    meta_path = meta_dir / "run_metadata.json"
    with open(meta_path, "w") as f:
        json.dump(meta_json, f, indent=2)

    if combined:
        combined_path = base / "sp500_daily_all.parquet"
        n_rows, n_tickers = build_combined(data_dir, combined_path)
        if show_summary:
            print(f"Built combined file: {combined_path} ({n_rows:,} rows from {n_tickers} tickers)")
    else:
        combined_path = None
        n_rows = n_tickers = None

    gap_details = {
        int(gap): {
            "count": len(entries),
            "examples": [(sym, dt.isoformat()) for sym, dt in entries[:5]],
        }
        for gap, entries in gap_map.items()
    }

    summary: Dict[str, object] = {
        "n_tickers": len(tickers),
        "total_new_rows": total_new,
        "up_to_date": up_to_date,
        "out_dir": str(data_dir),
        "start_date": start_dt.isoformat(),
        "end_date": end_dt.isoformat(),
        "meta_path": str(meta_path),
        "gap_details": gap_details,
        "new_tickers": new_tickers,
        "new_tickers_count": len(new_tickers),
        "new_ticker_examples": new_tickers[:10],
    }
    if combined_path is not None:
        summary["combined_file"] = str(combined_path)
        summary["combined_rows"] = n_rows
        summary["combined_from_tickers"] = n_tickers

    return summary

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
    parser.add_argument("--use-max-on-new", dest="use_max_on_new", action="store_true", default=True, help="For tickers with no existing file, fetch full history (period=max). Default: %(default)s")
    parser.add_argument("--no-use-max-on-new", dest="use_max_on_new", action="store_false", help="Disable period=max for new tickers; start from --start instead.")
    parser.add_argument("--tickers-file", type=str, default=None, help="Optional path to newline-separated custom tickers (Yahoo format).")
    args = parser.parse_args()

    try:
        update_sp500_daily(
            start=args.start,
            end=args.end,
            out=args.out,
            auto_adjust=args.auto_adjust,
            workers=args.workers,
            retries=args.retries,
            backoff=args.backoff,
            resume=args.resume,
            combined=args.combined,
            tickers_file=args.tickers_file,
            use_max_on_new=args.use_max_on_new,
            show_summary=True,
        )
    except ValueError as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        sys.exit(2)
    except RuntimeError as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        sys.exit(1)

if __name__ == "__main__":
    main()
