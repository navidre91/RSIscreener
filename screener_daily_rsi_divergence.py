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


# Telegram / notification config
TELEGRAM_TOKEN = os.getenv("TELEGRAM_BOT_TOKEN", "")
TELEGRAM_CHAT_ID = os.getenv("TELEGRAM_CHAT_ID", "")
SKIP_OPEN_WINDOW_CHECK = os.getenv("SKIP_OPEN_WINDOW_CHECK", "1").strip().lower() in {"1", "true", "yes"}

# Large-cap filtering for Telegram notifications
# - If enabled, only send alerts for symbols with market cap >= LARGECAP_MIN
NOTIFY_ONLY_LARGECAP: bool = os.getenv("NOTIFY_ONLY_LARGECAP", "0").strip().lower() in {"1", "true", "yes"}
LARGECAP_MIN: float = float(os.getenv("LARGECAP_MIN", str(10_000_000_000)))  # $10B default

# Market cap computation mode for enrichment
# - shares_only: compute using shares outstanding × last_price (local shares preferred)
# - hybrid: use local shares × price, then try Yahoo direct caps, then Yahoo shares × price
CAP_MODE: str = os.getenv("CAP_MODE", "hybrid").strip().lower()

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
DIVERGENCE_TYPES = {
    t.strip().lower()
    for t in os.getenv("DIVERGENCE_TYPES", "bullish").split(",")
    if t.strip()
}
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
LOCAL_META_DIR = DATA_BASE / "metadata"
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

    # Decide whether to refresh shares snapshot based on file age
    skip_shares = False
    try:
        max_age_days = 7
        now = datetime.now(timezone.utc)
        candidates = [
            LOCAL_META_DIR / "sp500_shares.parquet",
            LOCAL_META_DIR / "sp500_shares.csv",
        ]
        # Choose the newest existing candidate
        existing = [p for p in candidates if p.exists()]
        if existing:
            latest = max(existing, key=lambda p: p.stat().st_mtime)
            mtime = datetime.fromtimestamp(latest.stat().st_mtime, tz=timezone.utc)
            if (now - mtime) <= timedelta(days=max_age_days):
                skip_shares = True
                logging.info(
                    "Skipping shares snapshot (last updated %s; <= %d days old)",
                    mtime.isoformat(timespec="seconds"),
                    max_age_days,
                )
    except Exception:
        skip_shares = False

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
    if skip_shares:
        cmd.append("--no-with-shares")
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


#%% Notification helpers
def fail_if_missing_env() -> None:
    missing = []
    for key, value in {
        "TELEGRAM_BOT_TOKEN": TELEGRAM_TOKEN,
        "TELEGRAM_CHAT_ID": TELEGRAM_CHAT_ID,
    }.items():
        if not value:
            missing.append(key)
    if missing:
        raise SystemExit(f"Missing required env vars: {', '.join(missing)}")


def is_open_window_daily() -> bool:
    """Limit runs to 9:35–9:45 ET on weekdays unless overridden."""
    if SKIP_OPEN_WINDOW_CHECK:
        return True
    now_ny = datetime.now(timezone.utc).astimezone(NY)
    if now_ny.weekday() >= 5:
        return False
    start = now_ny.replace(hour=9, minute=35, second=0, microsecond=0)
    end = now_ny.replace(hour=9, minute=45, second=0, microsecond=0)
    return start <= now_ny <= end


def send_telegram(text: str) -> None:
    url = f"https://api.telegram.org/bot{TELEGRAM_TOKEN}/sendMessage"
    payload = {
        "chat_id": TELEGRAM_CHAT_ID,
        "text": text,
        "parse_mode": "HTML",
        "disable_web_page_preview": True,
    }
    try:
        r = requests.post(url, data=payload, timeout=30)
        if not r.ok:
            logging.error("Telegram send failed: %s %s", r.status_code, r.text)
    except Exception as exc:
        logging.error("Telegram send encountered error: %s", exc)


#%% Market cap helpers (for large-cap filtering)
def _safe_get_market_cap_from_fast_info(fast_info) -> Optional[float]:
    try:
        # yfinance fast_info may be a dict-like or object with attributes
        if fast_info is None:
            return None
        if isinstance(fast_info, dict):
            cap = fast_info.get("market_cap")
            if cap is None:
                cap = fast_info.get("marketCap")
            return float(cap) if cap is not None else None
        # attribute-style
        cap = getattr(fast_info, "market_cap", None)
        if cap is None:
            cap = getattr(fast_info, "marketCap", None)
        return float(cap) if cap is not None else None
    except Exception:
        return None


def fetch_market_caps(symbols: List[str]) -> Dict[str, Optional[float]]:
    """Return mapping of original symbols -> market cap (float in USD) or None if unavailable.

    Uses yfinance fast_info with a fallback to info when needed. Symbols are assumed US equities.
    Network calls are kept minimal since we only fetch for the result set before notifying.
    """
    caps: Dict[str, Optional[float]] = {}
    if not symbols:
        return caps
    uniq = list(dict.fromkeys([s.strip().upper() for s in symbols if s and s.strip()]))
    def _try_get_cap(tk) -> Optional[float]:
        # 1) fast_info variants
        cap = _safe_get_market_cap_from_fast_info(getattr(tk, "fast_info", None))
        if cap is not None:
            return float(cap)
        # 2) info dict
        try:
            info = tk.get_info() if hasattr(tk, "get_info") else getattr(tk, "info", {})
            if isinstance(info, dict):
                for key in ("marketCap", "market_cap", "enterpriseValue"):
                    cap_raw = info.get(key)
                    if cap_raw is not None:
                        return float(cap_raw)
        except Exception:
            pass
        return None

    for sym in uniq:
        ysym = _yahoo_symbol(sym)
        cap_val: Optional[float] = None
        # Attempt with shared session
        try:
            tk = yf.Ticker(ysym, session=_YF_SESSION)
            cap_val = _try_get_cap(tk)
        except Exception:
            cap_val = None

        # Retry without shared session if still missing
        if cap_val is None:
            try:
                tk2 = yf.Ticker(ysym)
                cap_val = _try_get_cap(tk2)
            except Exception:
                cap_val = None

        caps[sym] = cap_val
        time.sleep(0.03)
    return caps


def fetch_shares_outstanding(symbols: List[str]) -> Dict[str, Optional[float]]:
    """Return mapping of symbol -> shares outstanding (float) when available.

    Attempts, in order: fast_info (shares), info['sharesOutstanding'], get_shares_full()/get_shares().
    """
    out: Dict[str, Optional[float]] = {}
    if not symbols:
        return out
    uniq = list(dict.fromkeys([s.strip().upper() for s in symbols if s and s.strip()]))
    def _try_get_shares(tk) -> Optional[float]:
        # 1) fast_info derived
        try:
            fi = getattr(tk, "fast_info", None)
            if fi is not None:
                if isinstance(fi, dict):
                    cand = (
                        fi.get("shares_outstanding")
                        or fi.get("sharesOutstanding")
                        or fi.get("shares")
                        or fi.get("implied_shares_outstanding")
                        or fi.get("impliedSharesOutstanding")
                    )
                else:
                    cand = (
                        getattr(fi, "shares_outstanding", None)
                        or getattr(fi, "sharesOutstanding", None)
                        or getattr(fi, "shares", None)
                        or getattr(fi, "implied_shares_outstanding", None)
                        or getattr(fi, "impliedSharesOutstanding", None)
                    )
                if cand is not None:
                    return float(cand)
        except Exception:
            pass

        # 2) info dict keys
        try:
            info = tk.get_info() if hasattr(tk, "get_info") else getattr(tk, "info", {})
            if isinstance(info, dict):
                for key in (
                    "sharesOutstanding",
                    "shares_outstanding",
                    "impliedSharesOutstanding",
                    "floatShares",
                    "shares",
                ):
                    val = info.get(key)
                    if val is not None:
                        return float(val)
        except Exception:
            pass

        # 3) historical shares endpoints
        try:
            if hasattr(tk, "get_shares_full"):
                sdf = tk.get_shares_full()
                if sdf is not None and not sdf.empty:
                    series = pd.to_numeric(sdf.iloc[:, -1], errors="coerce").dropna()
                    if not series.empty:
                        return float(series.iloc[-1])
        except Exception:
            pass
        try:
            if hasattr(tk, "get_shares"):
                s = tk.get_shares()
                if s is not None:
                    if isinstance(s, (float, int)):
                        return float(s)
                    ser = pd.Series(s)
                    val = pd.to_numeric(ser, errors="coerce").dropna()
                    if not val.empty:
                        return float(val.iloc[-1])
        except Exception:
            pass
        return None

    for sym in uniq:
        ysym = _yahoo_symbol(sym)
        shares: Optional[float] = None
        try:
            tk = yf.Ticker(ysym, session=_YF_SESSION)
            shares = _try_get_shares(tk)
        except Exception:
            shares = None
        if shares is None:
            try:
                tk2 = yf.Ticker(ysym)
                shares = _try_get_shares(tk2)
            except Exception:
                shares = None
        out[sym] = shares
        time.sleep(0.03)
    return out


def enrich_with_market_caps(df: pd.DataFrame) -> pd.DataFrame:
    """Ensure a 'market_cap' column exists and is populated where possible.

    Strategy:
    - If 'market_cap' missing or has nulls, fetch direct caps via yfinance.
    - For any remaining nulls, try computing cap = last_price * shares_outstanding.
    - Returns a copy; original DataFrame is unchanged.
    """
    if df is None or df.empty:
        return df

    enriched = df.copy()
    # Determine which symbols need cap values
    if "market_cap" not in enriched.columns:
        need_mask = pd.Series([True] * len(enriched), index=enriched.index)
        enriched["market_cap"] = pd.NA
    else:
        need_mask = enriched["market_cap"].isna()

    if need_mask.any():
        # First try local offline metadata for shares
        if "last_price" in enriched.columns:
            local_shares = _load_local_shares_map()
            if local_shares:
                for idx in enriched.index[need_mask]:
                    sym = str(enriched.at[idx, "symbol"]).upper()
                    px = enriched.at[idx, "last_price"]
                    so = local_shares.get(sym)
                    try:
                        if px is not None and not pd.isna(px) and so is not None and not pd.isna(so):
                            enriched.at[idx, "market_cap"] = float(px) * float(so)
                    except Exception:
                        pass

        # Direct caps next (network) unless CAP_MODE enforces shares-only
        remaining = enriched["market_cap"].isna()
        if CAP_MODE != "shares_only" and remaining.any():
            syms = (
                enriched.loc[remaining, "symbol"].astype(str).str.upper().drop_duplicates().tolist()
            )
            caps_map = fetch_market_caps(syms)
            for idx in enriched.index[remaining]:
                sym = str(enriched.at[idx, "symbol"]).upper()
                cap = caps_map.get(sym)
                try:
                    if cap is not None and not pd.isna(cap):
                        enriched.at[idx, "market_cap"] = float(cap)
                except Exception:
                    pass

        # Fallback where still missing: compute from fetched shares * last_price (network)
        still_missing = enriched["market_cap"].isna()
        if still_missing.any() and "last_price" in enriched.columns:
            syms_miss = (
                enriched.loc[still_missing, "symbol"].astype(str).str.upper().drop_duplicates().tolist()
            )
            shares_map = fetch_shares_outstanding(syms_miss)
            for idx in enriched.index[still_missing]:
                sym = str(enriched.at[idx, "symbol"]).upper()
                px = enriched.at[idx, "last_price"]
                shares = shares_map.get(sym)
                try:
                    if px is not None and not pd.isna(px) and shares is not None and not pd.isna(shares):
                        enriched.at[idx, "market_cap"] = float(px) * float(shares)
                except Exception:
                    pass

    return enriched


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


def _load_local_shares_map() -> Dict[str, float]:
    """Load a mapping of Yahoo symbol -> shares_outstanding from metadata if available.

    Looks for metadata/sp500_shares.parquet or metadata/sp500_constituents.parquet(csv).
    Returns empty dict if nothing is found.
    """
    shares_map: Dict[str, float] = {}
    try:
        if not LOCAL_META_DIR.exists():
            return shares_map
        candidates = [
            LOCAL_META_DIR / "sp500_shares.parquet",
            LOCAL_META_DIR / "sp500_shares.csv",
            LOCAL_META_DIR / "sp500_constituents.parquet",
            LOCAL_META_DIR / "sp500_constituents.csv",
        ]
        for fp in candidates:
            if not fp.exists():
                continue
            try:
                if fp.suffix.lower() == ".parquet":
                    df = pd.read_parquet(fp)
                else:
                    df = pd.read_csv(fp)
            except Exception:
                continue
            if df is None or df.empty:
                continue
            cols = {c.lower(): c for c in df.columns}
            sym_col = cols.get("symbol_yahoo") or cols.get("symbol")
            so_col = cols.get("shares_outstanding") or cols.get("sharesoutstanding")
            if not sym_col or not so_col:
                continue
            sub = df[[sym_col, so_col]].dropna()
            if sub.empty:
                continue
            for _, row in sub.iterrows():
                sym = str(row[sym_col]).strip().upper()
                try:
                    shares_map[sym] = float(row[so_col])
                except Exception:
                    continue
            if shares_map:
                break
    except Exception:
        return {}
    return shares_map


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


#%% Formatting helpers for output / Telegram
def _fmt_number(value: Optional[float], fmt: str = ".2f") -> str:
    if value is None or pd.isna(value):
        return "n/a"
    return f"{value:{fmt}}"


def _fmt_pct(value: Optional[float]) -> str:
    if value is None or pd.isna(value):
        return "n/a"
    return f"{value * 100:.1f}%"


def _fmt_date(dt_val: Optional[datetime]) -> str:
    if dt_val is None or pd.isna(dt_val):
        return "n/a"
    return dt_val.astimezone(NY).strftime("%Y-%m-%d")


def _fmt_cap(value: Optional[float]) -> str:
    if value is None or pd.isna(value):
        return "n/a"
    try:
        v = float(value)
    except Exception:
        return "n/a"
    if v >= 1e12:
        return f"{v/1e12:.1f}T"
    if v >= 1e9:
        return f"{v/1e9:.1f}B"
    if v >= 1e6:
        return f"{v/1e6:.1f}M"
    if v >= 1e3:
        return f"{v/1e3:.1f}K"
    return f"{v:.0f}"


def _format_hits_for_telegram(results: pd.DataFrame, limit: int = 40) -> List[str]:
    lines: List[str] = []
    for rank, (_, row) in enumerate(results.head(limit).iterrows(), start=1):
        typ = str(row.get("type", "")).strip().lower()
        type_label = typ.capitalize() if typ else "N/A"
        price_pct = row.get("price_drop_pct") if typ == "bullish" else row.get("price_rise_pct")
        rsi_delta = row.get("rsi_gain") if typ == "bullish" else row.get("rsi_drop")
        price_label = "dPx"
        rsi_label = "dRSI"
        cap_str = _fmt_cap(row.get("market_cap"))

        pivot_dt = row.get("pivot_dt")
        pivot_start_dt = row.get("pivot_start_dt")
        pivot_str = _fmt_date(pivot_dt)
        pivot_start_str = _fmt_date(pivot_start_dt)

        lines.append(
            (
                f"{rank:>2}. {row.get('symbol', '???'):<6}  {type_label:<7}  "
                f"Strength={_fmt_number(row.get('strength'), '.3f')}  "
                f"{rsi_label}={_fmt_number(rsi_delta, '.1f')}  "
                f"{price_label}={_fmt_pct(price_pct)}  "
                f"RSI={_fmt_number(row.get('last_rsi'), '.1f')}  "
                f"Px={_fmt_number(row.get('last_price'), '.2f')}  "
                f"Cap={cap_str}  "
                f"pivot={pivot_str}"
            )
        )
        lines.append(
            (
                f"     pivots: price {_fmt_number(row.get('p1'), '.2f')}->{_fmt_number(row.get('p2'), '.2f')} "
                f"({pivot_start_str}->{pivot_str}); "
                f"RSI {_fmt_number(row.get('r1'), '.1f')}->{_fmt_number(row.get('r2'), '.1f')}"
            )
        )
    return lines


def send_telegram_report(results: pd.DataFrame, now_ny: str) -> None:
    algo_line = (
        "Algorithm: price/RSI divergence on daily bars "
        f"(pivot_window={PIVOT_WINDOW}, recent_bars={RECENT_BARS}, rsi_period={RSI_PERIOD})."
    )

    # Always enrich with market cap for display, with robust fallbacks
    # Enrich with market cap (no-op if already present and filled)
    enriched = enrich_with_market_caps(results)

    # Optionally filter to large caps for notifications only
    if NOTIFY_ONLY_LARGECAP and not enriched.empty:
        before_n = len(enriched)
        filtered = enriched[(~enriched["market_cap"].isna()) & (enriched["market_cap"] >= float(LARGECAP_MIN))]
        after_n = len(filtered)
        logging.info(
            "Large-cap filter applied for Telegram: kept %d of %d (>= %.0fB)",
            after_n,
            before_n,
            LARGECAP_MIN / 1e9,
        )
    else:
        filtered = enriched

    header = f"📅 Daily RSI divergence scan — {now_ny} ET\nMatches: {len(filtered)}"

    if filtered.empty:
        send_telegram(header + "\n" + algo_line + "\n(no divergence signals)")
        return

    lines = _format_hits_for_telegram(filtered)
    full = header + "\n" + algo_line + "\n\n" + "\n".join(lines)
    if len(full) <= 4000:
        send_telegram(full)
        return

    send_telegram(header)
    block: List[str] = []
    cur_len = 0
    for line in lines:
        line_len = len(line) + 1  # include newline
        if cur_len + line_len > 4000 and block:
            send_telegram("\n".join(block))
            block = []
            cur_len = 0
        block.append(line)
        cur_len += line_len
    if block:
        send_telegram("\n".join(block))


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


def main() -> None:
    fail_if_missing_env()

    if not is_open_window_daily():
        logging.info("Not in the open window; exiting.")
        return

    refresh_local_daily_cache()
    syms = build_universe()
    logging.info("Universe size: %d symbols", len(syms))
    print(f"Universe size: {len(syms)} symbols")

    results = scan_for_divergences(syms)
    # Enrich with market caps so terminal print includes Cap
    results = enrich_with_market_caps(results)
    # Add human-friendly market cap in billions for terminal display
    if "market_cap" in results.columns:
        try:
            results["market_cap_b"] = (results["market_cap"].astype(float) / 1e9)
        except Exception:
            # Fallback without casting
            results["market_cap_b"] = results["market_cap"] / 1e9
    now_ny_str = datetime.now(timezone.utc).astimezone(NY).strftime("%Y-%m-%d %H:%M")
    print(f"Completed daily RSI divergence scan — {now_ny_str} ET")

    if results.empty:
        print("No divergence signals found.")
    else:
        display_cols = [
            "symbol","type","strength","pivot_start_dt","pivot_dt","last_rsi","last_price",
            "market_cap_b","market_cap","price_drop_pct","rsi_gain","price_rise_pct","rsi_drop"
        ]
        display_cols = [c for c in display_cols if c in results.columns]
        display_df = results[display_cols].head(30).copy()
        if "market_cap_b" in display_df.columns:
            display_df["market_cap_b"] = pd.to_numeric(display_df["market_cap_b"], errors="coerce").round(1)
        print(display_df)

    # Save per-symbol annotated charts to results/<YYYY-MM-DD> as <SYMBOL>.png
    try:
        save_all_signal_charts(results)
    except Exception as e:
        logging.error("Chart generation failed: %s", e)

    send_telegram_report(results, now_ny_str)


if __name__ == "__main__":
    main()


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


#%% Report charts: save annotated divergence PNGs per ticker
def _to_utc_ts(dt_val) -> pd.Timestamp:
    ts = pd.Timestamp(dt_val)
    if ts.tz is None:
        ts = ts.tz_localize("UTC")
    else:
        ts = ts.tz_convert("UTC")
    return ts


def _find_index_for_dt(t_series: pd.Series, target_dt: datetime) -> Optional[int]:
    """Find exact index for target_dt in t_series (both tz-aware UTC). Fallback to nearest within 2 days."""
    if t_series is None or len(t_series) == 0 or target_dt is None:
        return None
    try:
        ts_target = _to_utc_ts(target_dt)
        idx = pd.Index(t_series)
        pos = idx.get_indexer([ts_target])
        if pos is not None and len(pos) and pos[0] != -1:
            return int(pos[0])
        # Fallback to nearest within 2 days
        ts_index = pd.to_datetime(t_series)
        diffs = (ts_index.view("int64") - ts_target.value).astype("int64").abs()
        near = int(diffs.argmin())
        # Check tolerance (~2 days)
        near_dt = ts_index.iloc[near]
        tol_days = abs((near_dt - ts_target).total_seconds()) / 86400.0
        return near if tol_days <= 2.5 else None
    except Exception:
        return None


def _save_divergence_chart(symbol: str, data: pd.DataFrame, row: pd.Series, out_path: Path) -> None:
    """Save a PNG showing price + RSI and the two divergence pivots.

    - symbol: ticker string
    - data: DataFrame with columns t (UTC), o,h,l,c,v
    - row: a result row containing fields: type, pivot_start_dt, pivot_dt, p1,p2,r1,r2, strength
    - out_path: full file path to write (e.g., results/2025-10-09/AAPL.png)
    """
    if data is None or data.empty:
        return
    try:
        import matplotlib
        matplotlib.use("Agg", force=True)
        import matplotlib.pyplot as plt
    except Exception as e:
        logging.error("matplotlib not available: %s", e)
        return

    c = pd.to_numeric(data["c"], errors="coerce").reset_index(drop=True)
    t = pd.to_datetime(data["t"])  # tz-aware UTC
    r = wilder_rsi(c, RSI_PERIOD)

    typ = str(row.get("type", "")).strip().lower()
    color = "green" if typ == "bullish" else "red" if typ == "bearish" else "blue"

    dt1 = row.get("pivot_start_dt")
    dt2 = row.get("pivot_dt")
    i1 = _find_index_for_dt(t, dt1)
    i2 = _find_index_for_dt(t, dt2)
    if i1 is None or i2 is None:
        # if indices cannot be located, skip chart for this row
        return

    # Y values: prefer stored pivot values for fidelity
    p1 = float(row.get("p1")) if row.get("p1") is not None else float(c.iloc[i1])
    p2 = float(row.get("p2")) if row.get("p2") is not None else float(c.iloc[i2])
    r1 = float(row.get("r1")) if row.get("r1") is not None else float(r.iloc[i1])
    r2 = float(row.get("r2")) if row.get("r2") is not None else float(r.iloc[i2])

    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(11.5, 6.5), sharex=True, gridspec_kw={"height_ratios": [3, 1]})

    # Price panel
    ax1.plot(t, c, color="black", linewidth=1.1, label="Close")
    ax1.scatter([t.iloc[i1], t.iloc[i2]], [p1, p2], color=color, s=36, zorder=3)
    ax1.plot([t.iloc[i1], t.iloc[i2]], [p1, p2], color=color, linestyle="--", linewidth=1.6)
    ax1.set_ylabel("Price")
    ax1.grid(True, alpha=0.3)

    # RSI panel
    ax2.plot(t, r, color="tab:blue", linewidth=1.0, label=f"RSI({RSI_PERIOD})")
    ax2.scatter([t.iloc[i1], t.iloc[i2]], [r1, r2], color=color, s=30, zorder=3)
    ax2.plot([t.iloc[i1], t.iloc[i2]], [r1, r2], color=color, linestyle="--", linewidth=1.4)
    ax2.axhline(30, color="red", linestyle=":", alpha=0.6)
    ax2.axhline(70, color="green", linestyle=":", alpha=0.6)
    ax2.set_ylabel("RSI")
    ax2.grid(True, alpha=0.3)

    # Titles/annotations
    pivot_str = _fmt_date(row.get("pivot_dt"))
    strength_str = _fmt_number(row.get("strength"), ".3f")
    info_line = (
        f"{typ.capitalize()} divergence  •  strength={strength_str}  •  pivots={_fmt_date(row.get('pivot_start_dt'))}->{pivot_str}"
    )
    ax1.set_title(f"{symbol} — Price", loc="left")
    ax2.set_title(info_line, loc="left")

    plt.tight_layout()
    try:
        out_path.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(out_path, dpi=160, bbox_inches="tight")
    finally:
        plt.close(fig)


def save_all_signal_charts(results: pd.DataFrame, out_base: str = "results") -> Optional[Path]:
    """Save PNG charts for all detected signals into results/<YYYY-MM-DD>/<SYMBOL>.png.

    Returns the path to the dated output directory, or None if no charts were generated.
    """
    if results is None or results.empty:
        return None

    # Build output directory based on ET date
    now_ny = datetime.now(timezone.utc).astimezone(NY)
    date_dir = Path(out_base) / now_ny.strftime("%Y-%m-%d")

    symbols = (
        results.get("symbol").astype(str).dropna().str.upper().drop_duplicates().tolist()
        if "symbol" in results.columns else []
    )
    if not symbols:
        return None

    # Fetch all bar data once for efficiency
    bars_by_sym = fetch_daily_bars(symbols)

    n_saved = 0
    for _, row in results.iterrows():
        sym = str(row.get("symbol", "")).upper()
        if not sym:
            continue
        data = bars_by_sym.get(sym)
        if data is None or data.empty:
            continue
        out_path = date_dir / f"{sym}.png"
        try:
            _save_divergence_chart(sym, data, row, out_path)
            n_saved += 1
        except Exception as e:
            logging.debug("Failed to save chart for %s: %s", sym, e)
            continue

    if n_saved == 0:
        return None
    return date_dir
