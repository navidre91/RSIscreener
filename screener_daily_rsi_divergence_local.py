#!/usr/bin/env python3
#%%
"""
Daily RSI(14) Divergence Screener (Local / Jupyter style)
---------------------------------------------------------
- Uses Yahoo Finance via yfinance (no API key required)
- Scans S&P 500 (from Wikipedia) or a custom ticker list
- No Telegram notifications; results are shown in DataFrames
- Structured with #%% cells for step-by-step exploration

Edit the Config cell to tweak parameters or limit the universe for quick runs.
"""

#%% Config
import logging
import os
import subprocess
import sys
import time
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Dict, List, Optional, Tuple
from zoneinfo import ZoneInfo

import numpy as np
import pandas as pd
import requests
import yfinance as yf
from requests.adapters import HTTPAdapter


# General config (tweak here)
RSI_PERIOD: int = int(os.getenv("RSI_PERIOD", "14"))
LOOKBACK_DAYS: int = int(os.getenv("LOOKBACK_DAYS", "400"))
MAX_SYMBOLS_PER_BATCH: int = int(os.getenv("MAX_SYMBOLS_PER_BATCH", "60"))

# yfinance concurrency/session tuning
YF_THREADS: int = int(os.getenv("YF_THREADS", "4"))  # keep modest to avoid rate limits
YF_POOL_MAXSIZE: int = int(os.getenv("YF_POOL_MAXSIZE", str(max(20, YF_THREADS * 2))))
YF_RETRIES: int = int(os.getenv("YF_RETRIES", "3"))
YF_BACKOFF: float = float(os.getenv("YF_BACKOFF", "1.8"))
YF_BATCH_PAUSE: float = float(os.getenv("YF_BATCH_PAUSE", "0.35"))

# Divergence detection tuning
DIVERGENCE_TYPES = {t.strip().lower() for t in os.getenv("DIVERGENCE_TYPES", "bullish").split(",") if t.strip()}
PIVOT_WINDOW: int = int(os.getenv("DIVERGENCE_PIVOT_WINDOW", "3"))
RECENT_BARS: int = int(os.getenv("DIVERGENCE_RECENT_BARS", "20"))

# Universe control
# - Set to None to fetch current S&P 500 from Wikipedia
# - Or set to a custom list like: ["AAPL", "MSFT", "NVDA", "AMZN", "GOOGL"]
UNIVERSE: Optional[List[str]] = None

# For quick experimentation, limit scan to the first N symbols (after sorting)
DEBUG_LIMIT: Optional[int] = None  # e.g., 50; set to None for full universe

NY = ZoneInfo("America/New_York")
LOGLEVEL = os.getenv("LOGLEVEL", "INFO").upper()
logging.basicConfig(level=LOGLEVEL, format="%(asctime)s %(levelname)s %(message)s")

# Local data preference (use pre-downloaded daily data if available)
DATA_BASE = Path(os.getenv("DATA_BASE", "data")).expanduser()
LOCAL_DAILY_DIR = DATA_BASE / "sp500_daily"
USE_LOCAL_DATA = os.getenv("USE_LOCAL_DATA", None)
if USE_LOCAL_DATA is None:
    USE_LOCAL_DATA = LOCAL_DAILY_DIR.exists()
else:
    USE_LOCAL_DATA = USE_LOCAL_DATA.strip() not in ("0", "false", "False")

# Reduce yfinance logging noise
logging.getLogger("yfinance").setLevel(logging.ERROR)
logging.getLogger("yfinance.data").setLevel(logging.ERROR)


#%% Local data refresh
def refresh_local_daily_cache() -> None:
    """Invoke the downloader so local daily data stays up to date."""
    if not USE_LOCAL_DATA:
        logging.info("Skipping local data refresh; USE_LOCAL_DATA disabled.")
        return

    downloader_script = Path(__file__).resolve().with_name("sp500_daily_downloader.py")
    if not downloader_script.exists():
        logging.warning("Downloader script not found at %s", downloader_script)
        return

    cmd = [
        sys.executable,
        str(downloader_script),
        "--out",
        str(DATA_BASE),
        "--workers",
        "3",
        "--retries",
        "3",
        "--backoff",
        "2.0",
    ]
    logging.info("Refreshing local cache via sp500_daily_downloader.py…")
    try:
        subprocess.run(cmd, check=True)
    except subprocess.CalledProcessError as exc:
        logging.error(
            "sp500_daily_downloader.py failed (exit code %s); continuing with existing data.",
            exc.returncode,
        )


#%% Helpers: symbols and batching
def _yahoo_symbol(symbol: str) -> str:
    """Map Wikipedia/Alpaca-style tickers to Yahoo format.

    Example: BRK.B -> BRK-B ; BF.B -> BF-B
    """
    return symbol.replace(".", "-").strip().upper()


def _make_yf_session(pool_maxsize: int) -> requests.Session:
    s = requests.Session()
    adapter = HTTPAdapter(pool_connections=pool_maxsize, pool_maxsize=pool_maxsize)
    s.mount("https://", adapter)
    s.mount("http://", adapter)
    return s


_YF_SESSION = _make_yf_session(YF_POOL_MAXSIZE)


#%% Data helpers: robust yfinance downloads
def _is_rate_limited_error(exc: Exception) -> bool:
    msg = str(exc).lower()
    return ("rate limit" in msg) or ("too many requests" in msg) or ("429" in msg)


def _download_batch(yahoo_syms: List[str], start_date) -> pd.DataFrame:
    delay = 1.0
    threads = max(1, int(YF_THREADS))
    last_exc = None
    for attempt in range(1, YF_RETRIES + 2):
        try:
            return yf.download(
                tickers=yahoo_syms,
                start=start_date.strftime("%Y-%m-%d"),
                interval="1d",
                auto_adjust=False,
                progress=False,
                threads=threads,
                session=_YF_SESSION,
            )
        except Exception as e:
            last_exc = e
            if _is_rate_limited_error(e):
                new_threads = max(1, threads // 2)
                if new_threads != threads:
                    logging.warning(f"Rate limited; reducing threads {threads} -> {new_threads}")
                    threads = new_threads
                logging.warning(f"Rate limited; sleeping {delay:.2f}s before retry (attempt {attempt})")
                time.sleep(delay)
                delay *= YF_BACKOFF
                continue
            logging.warning(f"Batch download error (attempt {attempt}): {e}; sleeping {delay:.2f}s")
            time.sleep(delay)
            delay *= YF_BACKOFF
    logging.error(f"Failed to download batch after retries: {last_exc}")
    return pd.DataFrame()


def _read_local_ticker(ticker_yahoo: str) -> pd.DataFrame:
    """Read local per-ticker file and normalize to t,o,h,l,c,v. Empty if missing."""
    fp_parquet = LOCAL_DAILY_DIR / f"{ticker_yahoo}.parquet"
    fp_csv = LOCAL_DAILY_DIR / f"{ticker_yahoo}.csv"
    if fp_parquet.exists():
        try:
            df = pd.read_parquet(fp_parquet)
        except Exception:
            df = pd.DataFrame()
    elif fp_csv.exists():
        try:
            df = pd.read_csv(fp_csv, parse_dates=["date"])
        except Exception:
            df = pd.DataFrame()
    else:
        return pd.DataFrame()

    if df is None or df.empty:
        return pd.DataFrame()
    # Expect columns: date, open, high, low, close, volume
    keep_map = {
        "open": "o", "high": "h", "low": "l", "close": "c", "volume": "v"
    }
    # Normalize columns to lowercase
    df = df.copy()
    df.columns = [c.lower() for c in df.columns]
    for col in ["date", "open", "high", "low", "close", "volume"]:
        if col not in df.columns:
            return pd.DataFrame()
    # Date -> tz-aware UTC
    t = pd.to_datetime(df["date"]).dt.tz_localize("UTC")
    out = pd.DataFrame({
        "t": t,
        "o": pd.to_numeric(df["open"], errors="coerce"),
        "h": pd.to_numeric(df["high"], errors="coerce"),
        "l": pd.to_numeric(df["low"], errors="coerce"),
        "c": pd.to_numeric(df["close"], errors="coerce"),
        "v": pd.to_numeric(df["volume"], errors="coerce"),
    }).dropna(subset=["c"]).sort_values("t")
    # Trim by LOOKBACK_DAYS
    min_t = (datetime.now(timezone.utc) - timedelta(days=LOOKBACK_DAYS))
    out = out[out["t"] >= min_t]
    return out


def get_sp500_symbols() -> List[str]:
    """Fetch current S&P 500 symbols (Wikipedia) and return uppercase list.

    Falls back to a public CSV if Wikipedia can't be parsed.
    """
    wiki_url = "https://en.wikipedia.org/wiki/List_of_S%26P_500_companies"
    hdrs = {
        "User-Agent": "Mozilla/5.0 (X11; Linux x86_64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/125.0 Safari/537.36",
        "Accept-Language": "en-US,en;q=0.9",
        "Referer": "https://en.wikipedia.org/",
        "Cache-Control": "no-cache",
    }
    try:
        r = requests.get(wiki_url, headers=hdrs, timeout=30)
        r.raise_for_status()
        import io
        tables = pd.read_html(io.StringIO(r.text))
        df = next((t for t in tables if "Symbol" in t.columns), None)
        if df is None:
            raise RuntimeError("Couldn't find 'Symbol' column on Wikipedia page.")
        syms = df["Symbol"].astype(str).str.strip().str.upper().tolist()
        return sorted({s for s in syms if s})
    except Exception as e:
        logging.warning(f"Wikipedia fetch failed ({e}); falling back to public CSV.")
        csv_url = "https://raw.githubusercontent.com/datasets/s-and-p-500-companies/main/data/constituents.csv"
        r = requests.get(csv_url, headers={"User-Agent": hdrs["User-Agent"]}, timeout=30)
        r.raise_for_status()
        import io
        df = pd.read_csv(io.StringIO(r.text))
        col = "Symbol" if "Symbol" in df.columns else "symbol"
        syms = df[col].astype(str).str.strip().str.upper().tolist()
        return sorted({s for s in syms if s})


def chunk_symbols(symbols: List[str], max_per_batch: int = MAX_SYMBOLS_PER_BATCH) -> List[List[str]]:
    out: List[List[str]] = []
    cur: List[str] = []
    budget = 15000
    for s in symbols:
        added = len(s) + (1 if cur else 0)
        if len(",".join(cur)) + added > budget or len(cur) >= max_per_batch:
            out.append(cur)
            cur = [s]
        else:
            cur.append(s)
    if cur:
        out.append(cur)
    return out


#%% Data: fetch daily bars from Yahoo Finance (yfinance)
def fetch_daily_bars(symbols: List[str]) -> Dict[str, pd.DataFrame]:
    """Fetch recent daily OHLCV bars for given symbols from Yahoo Finance.

    Returns a dict of DataFrames with columns: t (UTC), o, h, l, c, v
    """
    start_date = (datetime.now(timezone.utc) - timedelta(days=LOOKBACK_DAYS)).date()
    out: Dict[str, pd.DataFrame] = {}

    # 1) Try local cache if enabled
    missing_syms: List[str] = []
    if USE_LOCAL_DATA and LOCAL_DAILY_DIR.exists():
        for s in symbols:
            ysym = _yahoo_symbol(s)
            df_local = _read_local_ticker(ysym)
            if df_local is not None and not df_local.empty:
                out[s] = df_local[["t", "o", "h", "l", "c", "v"]]
            else:
                missing_syms.append(s)
    else:
        missing_syms = list(symbols)

    # 2) Fetch remaining from Yahoo
    for batch in chunk_symbols(missing_syms):
        yahoo_syms = [_yahoo_symbol(s) for s in batch]
        back_map = {ys: orig for ys, orig in zip(yahoo_syms, batch)}

        df = _download_batch(yahoo_syms, start_date)

        if df is None or df.empty:
            # fallback: try downloading one-by-one to salvage data
            for ysym in yahoo_syms:
                sdf = _download_batch([ysym], start_date)
                if sdf is None or sdf.empty:
                    continue
                try:
                    o = pd.to_numeric(sdf["Open"], errors="coerce")
                    h = pd.to_numeric(sdf["High"], errors="coerce")
                    l = pd.to_numeric(sdf["Low"], errors="coerce")
                    c = pd.to_numeric(sdf["Close"], errors="coerce")
                    v = pd.to_numeric(sdf["Volume"], errors="coerce")
                    idx = pd.to_datetime(sdf.index)
                    idx = idx.tz_localize("UTC") if getattr(idx, "tz", None) is None else idx.tz_convert("UTC")
                    tidy = pd.DataFrame({"t": idx, "o": o, "h": h, "l": l, "c": c, "v": v}).dropna(subset=["c"]).sort_values("t")
                    orig = back_map.get(ysym, ysym)
                    out[orig] = tidy[["t", "o", "h", "l", "c", "v"]]
                except Exception:
                    continue
            if YF_BATCH_PAUSE > 0:
                time.sleep(YF_BATCH_PAUSE)
            continue

        if isinstance(df.columns, pd.MultiIndex):
            # Multi-ticker: columns like ('Open','AAPL'), ('High','AAPL'), ...
            fields = ["Open", "High", "Low", "Close", "Volume"]
            available_syms = sorted({sym for (fld, sym) in df.columns if fld in fields})
            for ysym in available_syms:
                try:
                    o = pd.to_numeric(df[("Open", ysym)], errors="coerce")
                    h = pd.to_numeric(df[("High", ysym)], errors="coerce")
                    l = pd.to_numeric(df[("Low", ysym)], errors="coerce")
                    c = pd.to_numeric(df[("Close", ysym)], errors="coerce")
                    v = pd.to_numeric(df[("Volume", ysym)], errors="coerce")
                except Exception:
                    continue
                idx = pd.to_datetime(df.index)
                if getattr(idx, "tz", None) is None:
                    idx = idx.tz_localize("UTC")
                else:
                    idx = idx.tz_convert("UTC")
                tidy = pd.DataFrame({
                    "t": idx,
                    "o": o,
                    "h": h,
                    "l": l,
                    "c": c,
                    "v": v,
                }).dropna(subset=["c"]).sort_values("t")
                orig = back_map.get(ysym, ysym)
                out[orig] = tidy[["t", "o", "h", "l", "c", "v"]]
        else:
            # Single symbol result
            try:
                o = pd.to_numeric(df["Open"], errors="coerce")
                h = pd.to_numeric(df["High"], errors="coerce")
                l = pd.to_numeric(df["Low"], errors="coerce")
                c = pd.to_numeric(df["Close"], errors="coerce")
                v = pd.to_numeric(df["Volume"], errors="coerce")
            except Exception:
                continue
            idx = pd.to_datetime(df.index)
            if getattr(idx, "tz", None) is None:
                idx = idx.tz_localize("UTC")
            else:
                idx = idx.tz_convert("UTC")
            tidy = pd.DataFrame({
                "t": idx,
                "o": o,
                "h": h,
                "l": l,
                "c": c,
                "v": v,
            }).dropna(subset=["c"]).sort_values("t")
            ysym = yahoo_syms[0] if yahoo_syms else None
            orig = back_map.get(ysym, ysym or "")
            if orig:
                out[orig] = tidy[["t", "o", "h", "l", "c", "v"]]
        if YF_BATCH_PAUSE > 0:
            time.sleep(YF_BATCH_PAUSE)
    return out


#%% Technicals: RSI and pivots
def wilder_rsi(close: pd.Series, period: int = 14) -> pd.Series:
    delta = close.diff()
    gains = (delta.clip(lower=0)).abs()
    losses = (-delta.clip(upper=0)).abs()
    avg_gain = gains.ewm(alpha=1 / period, min_periods=period, adjust=False).mean()
    avg_loss = losses.ewm(alpha=1 / period, min_periods=period, adjust=False).mean()
    rs = avg_gain / avg_loss.replace(0, np.nan)
    return 100 - (100 / (1 + rs))


def find_pivot_lows(series: pd.Series, window: int) -> List[Tuple[int, float]]:
    arr = series.values
    n = len(arr)
    pivots: List[Tuple[int, float]] = []
    for i in range(window, n - window):
        left = arr[i - window : i]
        right = arr[i + 1 : i + 1 + window]
        cur = arr[i]
        if np.all(cur < left) and np.all(cur <= right):
            pivots.append((i, float(cur)))
    return pivots


def find_pivot_highs(series: pd.Series, window: int) -> List[Tuple[int, float]]:
    arr = series.values
    n = len(arr)
    pivots: List[Tuple[int, float]] = []
    for i in range(window, n - window):
        left = arr[i - window : i]
        right = arr[i + 1 : i + 1 + window]
        cur = arr[i]
        if np.all(cur > left) and np.all(cur >= right):
            pivots.append((i, float(cur)))
    return pivots


def detect_bullish_divergence(price: pd.Series, rsi: pd.Series, pivot_window: int, recent_bars: int) -> Optional[Dict]:
    lows = find_pivot_lows(price, pivot_window)
    if len(lows) < 2:
        return None
    i1, p1 = lows[-2]
    i2, p2 = lows[-1]
    if i2 <= i1:
        return None
    if (len(price) - 1) - i2 > recent_bars:
        return None
    r1 = float(rsi.iloc[i1]) if i1 < len(rsi) else np.nan
    r2 = float(rsi.iloc[i2]) if i2 < len(rsi) else np.nan
    if np.isnan(r1) or np.isnan(r2):
        return None
    if p2 < p1 and r2 > r1:
        return {"type": "bullish", "i1": i1, "i2": i2, "p1": p1, "p2": p2, "r1": r1, "r2": r2}
    return None


def detect_bearish_divergence(price: pd.Series, rsi: pd.Series, pivot_window: int, recent_bars: int) -> Optional[Dict]:
    highs = find_pivot_highs(price, pivot_window)
    if len(highs) < 2:
        return None
    i1, p1 = highs[-2]
    i2, p2 = highs[-1]
    if i2 <= i1:
        return None
    if (len(price) - 1) - i2 > recent_bars:
        return None
    r1 = float(rsi.iloc[i1]) if i1 < len(rsi) else np.nan
    r2 = float(rsi.iloc[i2]) if i2 < len(rsi) else np.nan
    if np.isnan(r1) or np.isnan(r2):
        return None
    if p2 > p1 and r2 < r1:
        return {"type": "bearish", "i1": i1, "i2": i2, "p1": p1, "p2": p2, "r1": r1, "r2": r2}
    return None


#%% Scan: detect divergences and summarize
def scan_for_divergences(symbols: List[str]) -> pd.DataFrame:
    bars_by_sym = fetch_daily_bars(symbols)
    rows = []
    for sym, df in bars_by_sym.items():
        if df.shape[0] < (RSI_PERIOD + 10):
            continue
        close = df["c"].reset_index(drop=True)
        rsi = wilder_rsi(close, RSI_PERIOD)
        if rsi.isna().all():
            continue

        # Bullish
        bull = detect_bullish_divergence(close, rsi, PIVOT_WINDOW, RECENT_BARS) if "bullish" in DIVERGENCE_TYPES else None
        if bull:
            price_drop = max(bull["p1"] - bull["p2"], 0.0)
            price_drop_pct = (price_drop / bull["p1"]) if bull["p1"] else 0.0
            rsi_gain = max(bull["r2"] - bull["r1"], 0.0)
            strength = rsi_gain * price_drop_pct
            rows.append({
                "symbol": sym,
                "type": "bullish",
                "last_price": float(close.iloc[-1]),
                "last_rsi": float(rsi.dropna().iloc[-1]),
                "pivot_dt": df["t"].iloc[int(bull["i2"])].to_pydatetime(),
                "pivot_start_dt": df["t"].iloc[int(bull["i1"])].to_pydatetime(),
                "strength": strength,
                "price_drop_pct": price_drop_pct,
                "rsi_gain": rsi_gain,
                "p1": float(bull["p1"]),
                "p2": float(bull["p2"]),
                "r1": float(bull["r1"]),
                "r2": float(bull["r2"]),
            })

        # Bearish
        bear = detect_bearish_divergence(close, rsi, PIVOT_WINDOW, RECENT_BARS) if "bearish" in DIVERGENCE_TYPES else None
        if bear:
            price_rise = max(bear["p2"] - bear["p1"], 0.0)
            price_rise_pct = (price_rise / bear["p1"]) if bear["p1"] else 0.0
            rsi_drop = max(bear["r1"] - bear["r2"], 0.0)
            strength = rsi_drop * price_rise_pct
            rows.append({
                "symbol": sym,
                "type": "bearish",
                "last_price": float(close.iloc[-1]),
                "last_rsi": float(rsi.dropna().iloc[-1]),
                "pivot_dt": df["t"].iloc[int(bear["i2"])].to_pydatetime(),
                "pivot_start_dt": df["t"].iloc[int(bear["i1"])].to_pydatetime(),
                "strength": strength,
                "price_rise_pct": price_rise_pct,
                "rsi_drop": rsi_drop,
                "p1": float(bear["p1"]),
                "p2": float(bear["p2"]),
                "r1": float(bear["r1"]),
                "r2": float(bear["r2"]),
            })

    if not rows:
        return pd.DataFrame(columns=[
            "symbol","type","last_price","last_rsi","pivot_dt","pivot_start_dt","strength",
            "price_drop_pct","rsi_gain","price_rise_pct","rsi_drop","p1","p2","r1","r2"
        ])

    hits = pd.DataFrame(rows)
    hits.sort_values(by=["strength", "pivot_dt"], ascending=[False, False], inplace=True)
    hits.reset_index(drop=True, inplace=True)
    return hits


#%% Run: build universe and scan
def build_universe() -> List[str]:
    if UNIVERSE is not None:
        syms = [s.strip().upper() for s in UNIVERSE if s.strip()]
    else:
        syms = get_sp500_symbols()
    # Note: we intentionally do NOT translate to Yahoo here; fetch function does that
    syms = sorted({s for s in syms if s})
    if DEBUG_LIMIT is not None:
        syms = syms[: int(DEBUG_LIMIT)]
    return syms


if __name__ == "__main__":
    refresh_local_daily_cache()
    # Typical local run (no open window gating)
    syms = build_universe()
    print(f"Universe size: {len(syms)} symbols")
    results = scan_for_divergences(syms)
    now_ny = datetime.now(timezone.utc).astimezone(NY).strftime("%Y-%m-%d %H:%M")
    print(f"Completed daily RSI divergence scan — {now_ny} ET")
    if results.empty:
        print("No divergence signals found.")
    else:
        # Show top 30 by strength
        display_cols = [
            "symbol","type","strength","pivot_start_dt","pivot_dt","last_rsi","last_price",
            "price_drop_pct","rsi_gain","price_rise_pct","rsi_drop"
        ]
        display_cols = [c for c in display_cols if c in results.columns]
        print(results[display_cols].head(30))


#%% Optional: visualize a specific symbol's price and RSI with pivots
def plot_symbol_with_rsi(symbol: str, lookback_days: int = LOOKBACK_DAYS):
    """Quick visualization helper. Requires matplotlib."""
    try:
        import matplotlib.pyplot as plt
    except Exception as e:
        print(f"matplotlib not available: {e}")
        return

    data = fetch_daily_bars([symbol]).get(symbol)
    if data is None or data.empty:
        print(f"No data for {symbol}")
        return

    c = data["c"].reset_index(drop=True)
    r = wilder_rsi(c, RSI_PERIOD)
    t = data["t"].to_numpy()

    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 7), sharex=True, gridspec_kw={"height_ratios": [3, 1]})
    ax1.plot(t, data["c"], label="Close", color="black")
    ax1.set_title(f"{symbol} price")
    ax1.grid(True, alpha=0.3)

    ax2.plot(t, r, label="RSI(14)", color="blue")
    ax2.axhline(30, color="red", linestyle="--", alpha=0.5)
    ax2.axhline(70, color="green", linestyle="--", alpha=0.5)
    ax2.set_title("RSI(14)")
    ax2.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.show()

# %%
