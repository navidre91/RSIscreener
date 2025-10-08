#!/usr/bin/env python3
"""
Daily RSI(14) divergence screener: scans S&P 500 on daily candles and
notifies via Telegram once near market open.

Data: Yahoo Finance via yfinance (no API key required)
Universe: Wikipedia S&P 500 constituents
"""

import os
import logging
from datetime import datetime, timedelta, timezone
import time
from zoneinfo import ZoneInfo
from typing import List, Tuple, Dict, Optional

import numpy as np
import pandas as pd
import requests
import yfinance as yf
from requests.adapters import HTTPAdapter

# ========= CONFIG =========
TELEGRAM_TOKEN = os.getenv("TELEGRAM_BOT_TOKEN", "")
TELEGRAM_CHAT_ID = os.getenv("TELEGRAM_CHAT_ID", "")

RSI_PERIOD = int(os.getenv("RSI_PERIOD", "14"))
MAX_SYMBOLS_PER_BATCH = int(os.getenv("MAX_SYMBOLS_PER_BATCH", "60"))
LOOKBACK_DAYS = int(os.getenv("LOOKBACK_DAYS", "400"))

# yfinance concurrency/session tuning
YF_THREADS = int(os.getenv("YF_THREADS", "4"))  # set 1 to disable concurrency
YF_POOL_MAXSIZE = int(os.getenv("YF_POOL_MAXSIZE", str(max(20, YF_THREADS * 2))))
YF_RETRIES = int(os.getenv("YF_RETRIES", "3"))
YF_BACKOFF = float(os.getenv("YF_BACKOFF", "1.8"))
YF_BATCH_PAUSE = float(os.getenv("YF_BATCH_PAUSE", "0.35"))

# Divergence detection tuning (shared names with intraday script)
DIVERGENCE_TYPES = {t.strip().lower() for t in os.getenv("DIVERGENCE_TYPES", "bullish,bearish").split(",") if t.strip()}
PIVOT_WINDOW = int(os.getenv("DIVERGENCE_PIVOT_WINDOW", "3"))
RECENT_BARS = int(os.getenv("DIVERGENCE_RECENT_BARS", "20"))

NY = ZoneInfo("America/New_York")
LOGLEVEL = os.getenv("LOGLEVEL", "INFO").upper()
# ==========================

logging.basicConfig(level=LOGLEVEL, format="%(asctime)s %(levelname)s %(message)s")


def _yahoo_symbol(symbol: str) -> str:
    """Map Wikipedia/Alpaca-style tickers to Yahoo format.

    Example: BRK.B -> BRK-B ; BF.B -> BF-B
    """
    return symbol.replace(".", "-").strip().upper()


def _make_yf_session(pool_maxsize: int) -> requests.Session:
    """Create a requests.Session with enlarged connection pool for yfinance."""
    s = requests.Session()
    adapter = HTTPAdapter(pool_connections=pool_maxsize, pool_maxsize=pool_maxsize)
    s.mount("https://", adapter)
    s.mount("http://", adapter)
    return s


_YF_SESSION = _make_yf_session(YF_POOL_MAXSIZE)


def fail_if_missing_env():
    missing = []
    for k, v in {
        "TELEGRAM_BOT_TOKEN": TELEGRAM_TOKEN,
        "TELEGRAM_CHAT_ID": TELEGRAM_CHAT_ID,
    }.items():
        if not v:
            missing.append(k)
    if missing:
        raise SystemExit(f"Missing required env vars: {', '.join(missing)}")


def is_open_window_daily() -> bool:
    """Return True only during a narrow open window (9:35–9:45 ET) on weekdays.

    Note: We do not call an external market clock. This avoids API
    dependencies; it may run on market holidays if scheduled.
    """
    now_ny = datetime.now(timezone.utc).astimezone(NY)
    # Monday=0 ... Sunday=6
    if now_ny.weekday() >= 5:
        return False
    start = now_ny.replace(hour=9, minute=35, second=0, microsecond=0)
    end = now_ny.replace(hour=9, minute=45, second=0, microsecond=0)
    return start <= now_ny <= end


def get_sp500_symbols() -> List[str]:
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


def chunk_symbols(symbols: List[str], max_per_batch=MAX_SYMBOLS_PER_BATCH) -> List[List[str]]:
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


def _is_rate_limited_error(exc: Exception) -> bool:
    msg = str(exc).lower()
    return ("rate limit" in msg) or ("too many requests" in msg) or ("429" in msg)


def _download_batch(yahoo_syms: List[str], start_date: datetime.date) -> pd.DataFrame:
    delay = 1.0
    threads = max(1, int(YF_THREADS))
    last_exc = None
    for attempt in range(1, YF_RETRIES + 2):  # retries + final
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


def fetch_daily_bars(symbols: List[str]) -> Dict[str, pd.DataFrame]:
    """Fetch recent daily OHLCV bars for given symbols from Yahoo Finance.

    Returns a dict of DataFrames with columns: t (UTC), o, h, l, c, v
    """
    start_date = (datetime.now(timezone.utc) - timedelta(days=LOOKBACK_DAYS)).date()
    out: Dict[str, pd.DataFrame] = {}

    for batch in chunk_symbols(symbols):
        # Map original -> Yahoo and back
        yahoo_syms = [_yahoo_symbol(s) for s in batch]
        back_map = {ys: orig for ys, orig in zip(yahoo_syms, batch)}

        df = _download_batch(yahoo_syms, start_date)

        if df is None or df.empty:
            # fallback: try one-by-one to salvage some symbols
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

        # Normalize into per-symbol frames
        if isinstance(df.columns, pd.MultiIndex):
            # Columns like ('Open', 'AAPL'), ('High', 'AAPL'), ...
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
                # Ensure timezone-aware UTC
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
                }).dropna(subset=["c"])  # drop rows without close
                tidy = tidy.sort_values("t")
                orig = back_map.get(ysym, ysym)
                out[orig] = tidy[["t", "o", "h", "l", "c", "v"]]
        else:
            # Single symbol DataFrame: columns like 'Open','High','Low','Close','Volume'
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
            # Attempt to find which symbol this corresponds to
            # If only one ticker requested, map it back; otherwise use first mapping
            ysym = yahoo_syms[0] if yahoo_syms else None
            orig = back_map.get(ysym, ysym or "")
            if orig:
                out[orig] = tidy[["t", "o", "h", "l", "c", "v"]]
        if YF_BATCH_PAUSE > 0:
            time.sleep(YF_BATCH_PAUSE)
    return out


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


def scan_for_divergences(symbols: List[str]) -> List[Dict[str, object]]:
    bars_by_sym = fetch_daily_bars(symbols)
    hits: List[Dict[str, object]] = []
    for sym, df in bars_by_sym.items():
        if df.shape[0] < (RSI_PERIOD + 10):
            continue
        close = df["c"].reset_index(drop=True)
        rsi = wilder_rsi(close, RSI_PERIOD)
        if rsi.isna().all():
            continue

        bull = detect_bullish_divergence(close, rsi, PIVOT_WINDOW, RECENT_BARS) if "bullish" in DIVERGENCE_TYPES else None
        if not bull:
            continue

        price_drop = max(bull["p1"] - bull["p2"], 0.0)
        price_drop_pct = (price_drop / bull["p1"]) if bull["p1"] else 0.0
        rsi_gain = max(bull["r2"] - bull["r1"], 0.0)
        strength = rsi_gain * price_drop_pct

        last_px = float(close.iloc[-1])
        last_rsi = float(rsi.dropna().iloc[-1])
        pivot_dt = df["t"].iloc[int(bull["i2"])].to_pydatetime()
        pivot_start_dt = df["t"].iloc[int(bull["i1"])].to_pydatetime()
        hits.append(
            {
                "symbol": sym,
                "type": "bullish",
                "last_price": last_px,
                "last_rsi": last_rsi,
                "pivot_dt": pivot_dt,
                "pivot_start_dt": pivot_start_dt,
                "strength": strength,
                "price_drop_pct": price_drop_pct,
                "rsi_gain": rsi_gain,
                "p1": float(bull["p1"]),
                "p2": float(bull["p2"]),
                "r1": float(bull["r1"]),
                "r2": float(bull["r2"]),
            }
        )

    hits.sort(key=lambda h: (h["strength"], h["pivot_dt"]), reverse=True)
    return hits


def send_telegram(text: str):
    url = f"https://api.telegram.org/bot{TELEGRAM_TOKEN}/sendMessage"
    payload = {
        "chat_id": TELEGRAM_CHAT_ID,
        "text": text,
        "parse_mode": "HTML",
        "disable_web_page_preview": True,
    }
    r = requests.post(url, data=payload, timeout=30)
    if not r.ok:
        logging.error(f"Telegram send failed: {r.status_code} {r.text}")


def main():
    fail_if_missing_env()

    if not is_open_window_daily():
        logging.info("Not in the open window; exiting.")
        return

    syms = get_sp500_symbols()
    logging.info(f"Universe size: {len(syms)} symbols")

    matches = scan_for_divergences(syms)
    now_ny = datetime.now(timezone.utc).astimezone(NY).strftime("%Y-%m-%d %H:%M")
    header = f"📅 Daily RSI divergence scan — {now_ny} ET\nBullish matches: {len(matches)}"
    algo_line = (
        "Algorithm: price makes a lower low while RSI makes a higher low within the last "
        f"{RECENT_BARS} bars (pivot window={PIVOT_WINDOW}, RSI period={RSI_PERIOD}); "
        "strength = delta_RSI * delta_price_pct."
    )

    if not matches:
        send_telegram(header + "\n" + algo_line + "\n(no divergence signals)")
        return

    lines: List[str] = []
    for rank, hit in enumerate(matches, start=1):
        d_str = hit["pivot_dt"].astimezone(NY).strftime("%Y-%m-%d")
        d_start = hit["pivot_start_dt"].astimezone(NY).strftime("%Y-%m-%d")
        lines.append(
            (
                f"{rank:>2}. {hit['symbol']:<6}  Bullish  Strength={hit['strength']:.3f}  "
                f"dRSI={hit['rsi_gain']:.1f}  dPx={hit['price_drop_pct']*100:.1f}%  "
                f"RSI={hit['last_rsi']:5.1f}  Px={hit['last_price']:.2f}  pivot={d_str}"
            )
        )
        lines.append(
            (
                f"     pivots: price {hit['p1']:.2f}->{hit['p2']:.2f} ({d_start}->{d_str}); "
                f"RSI {hit['r1']:.1f}->{hit['r2']:.1f}"
            )
        )

    full = header + "\n" + algo_line + "\n\n" + "\n".join(lines)
    if len(full) <= 4000:
        send_telegram(full)
    else:
        send_telegram(header)
        block: List[str] = []
        cur = 0
        for ln in lines:
            if cur + len(ln) + 1 > 4000:
                send_telegram("\n".join(block))
                block = []
                cur = 0
            block.append(ln)
            cur += len(ln) + 1
        if block:
            send_telegram("\n".join(block))


if __name__ == "__main__":
    main()
