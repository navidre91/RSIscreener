#!/usr/bin/env python3
"""
Daily RSI(14) divergence screener: scans S&P 500 on daily candles and
notifies via Telegram once near market open.

Data: Alpaca Market Data (IEX feed)
Universe: Wikipedia S&P 500 constituents
"""

import os
import logging
from datetime import datetime, timedelta, timezone
from zoneinfo import ZoneInfo
from typing import List, Tuple, Dict, Optional

import numpy as np
import pandas as pd
import requests

# ========= CONFIG =========
ALPACA_KEY = os.getenv("ALPACA_API_KEY_ID", "")
ALPACA_SECRET = os.getenv("ALPACA_API_SECRET_KEY", "")
ALPACA_DATA_BASE = "https://data.alpaca.markets"
ALPACA_TRADE_BASE = os.getenv("ALPACA_TRADE_BASE", "https://paper-api.alpaca.markets")

TELEGRAM_TOKEN = os.getenv("TELEGRAM_BOT_TOKEN", "")
TELEGRAM_CHAT_ID = os.getenv("TELEGRAM_CHAT_ID", "")

RSI_PERIOD = int(os.getenv("RSI_PERIOD", "14"))
TIMEFRAME = "1Day"
MAX_SYMBOLS_PER_BATCH = int(os.getenv("MAX_SYMBOLS_PER_BATCH", "300"))
LOOKBACK_DAYS = int(os.getenv("LOOKBACK_DAYS", "400"))

# Divergence detection tuning (shared names with intraday script)
DIVERGENCE_TYPES = {t.strip().lower() for t in os.getenv("DIVERGENCE_TYPES", "bullish,bearish").split(",") if t.strip()}
PIVOT_WINDOW = int(os.getenv("DIVERGENCE_PIVOT_WINDOW", "3"))
RECENT_BARS = int(os.getenv("DIVERGENCE_RECENT_BARS", "20"))

NY = ZoneInfo("America/New_York")
LOGLEVEL = os.getenv("LOGLEVEL", "INFO").upper()
# ==========================

logging.basicConfig(level=LOGLEVEL, format="%(asctime)s %(levelname)s %(message)s")
HEADERS = {"APCA-API-KEY-ID": ALPACA_KEY, "APCA-API-SECRET-KEY": ALPACA_SECRET}


def fail_if_missing_env():
    missing = []
    for k, v in {
        "ALPACA_API_KEY_ID": ALPACA_KEY,
        "ALPACA_API_SECRET_KEY": ALPACA_SECRET,
        "TELEGRAM_BOT_TOKEN": TELEGRAM_TOKEN,
        "TELEGRAM_CHAT_ID": TELEGRAM_CHAT_ID,
    }.items():
        if not v:
            missing.append(k)
    if missing:
        raise SystemExit(f"Missing required env vars: {', '.join(missing)}")


def is_open_window_daily() -> bool:
    """Return True only during a narrow open window (9:35–9:45 ET) on an open trading day.

    This avoids duplicate runs when scheduling two UTC times to cover DST.
    """
    try:
        r = requests.get(f"{ALPACA_TRADE_BASE}/v2/clock", headers=HEADERS, timeout=10)
        r.raise_for_status()
        clk = r.json()
        if not clk.get("is_open", False):
            return False

        now_utc = datetime.fromisoformat(clk["timestamp"].replace("Z", "+00:00"))
        now_ny = now_utc.astimezone(NY)
        # Allow only during 9:35–9:45 ET
        start = now_ny.replace(hour=9, minute=35, second=0, microsecond=0)
        end = now_ny.replace(hour=9, minute=45, second=0, microsecond=0)
        return start <= now_ny <= end
    except Exception as e:
        logging.warning(f"Clock check failed, defaulting to run: {e}")
        return True


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


def fetch_daily_bars(symbols: List[str]) -> Dict[str, pd.DataFrame]:
    since_utc = datetime.now(timezone.utc) - timedelta(days=LOOKBACK_DAYS)
    start_iso = since_utc.isoformat(timespec="seconds").replace("+00:00", "Z")

    result: Dict[str, List[dict]] = {}
    for batch in chunk_symbols(symbols):
        next_token = None
        for _ in range(5):
            params = {
                "timeframe": TIMEFRAME,
                "symbols": ",".join(batch),
                "start": start_iso,
                "adjustment": "all",
                "feed": "iex",
                "sort": "asc",
            }
            if next_token:
                params["page_token"] = next_token

            r = requests.get(f"{ALPACA_DATA_BASE}/v2/stocks/bars", headers=HEADERS, params=params, timeout=30)
            r.raise_for_status()
            data = r.json()
            bars_map = data.get("bars", {})
            if not isinstance(bars_map, dict):
                logging.warning(f"Expected 'bars' to be a dict, got {type(bars_map)}")
                continue
            for sym, b_list in bars_map.items():
                result.setdefault(sym, []).extend(b_list)
            next_token = data.get("next_page_token")
            if not next_token:
                break

    out: Dict[str, pd.DataFrame] = {}
    for sym, blist in result.items():
        if not blist:
            continue
        df = pd.DataFrame(blist)
        df["t"] = pd.to_datetime(df["t"], utc=True)
        for k in ["o", "h", "l", "c"]:
            df[k] = pd.to_numeric(df[k], errors="coerce")
        df["v"] = pd.to_numeric(df["v"], errors="coerce")
        df = df.sort_values("t").dropna(subset=["c"])  # ensure ascending / clean
        out[sym] = df[["t", "o", "h", "l", "c", "v"]]
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


def scan_for_divergences(symbols: List[str]) -> List[Tuple[str, str, float, float, datetime]]:
    bars_by_sym = fetch_daily_bars(symbols)
    hits: List[Tuple[str, str, float, float, datetime]] = []
    for sym, df in bars_by_sym.items():
        if df.shape[0] < (RSI_PERIOD + 10):
            continue
        close = df["c"].reset_index(drop=True)
        rsi = wilder_rsi(close, RSI_PERIOD)
        if rsi.isna().all():
            continue

        candidates: List[Dict] = []
        if "bullish" in DIVERGENCE_TYPES:
            bull = detect_bullish_divergence(close, rsi, PIVOT_WINDOW, RECENT_BARS)
            if bull:
                candidates.append(bull)
        if "bearish" in DIVERGENCE_TYPES:
            bear = detect_bearish_divergence(close, rsi, PIVOT_WINDOW, RECENT_BARS)
            if bear:
                candidates.append(bear)

        if not candidates:
            continue

        div = max(candidates, key=lambda d: d["i2"])  # most recent pivot
        last_px = float(close.iloc[-1])
        last_rsi = float(rsi.dropna().iloc[-1])
        pivot_dt = df["t"].iloc[int(div["i2"])].to_pydatetime()
        hits.append((sym, div["type"], last_px, last_rsi, pivot_dt))

    hits.sort(key=lambda x: x[4], reverse=True)  # most recent pivot first
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
    header = f"📅 Daily RSI divergence scan — {now_ny} ET\nMatches: {len(matches)}"

    if not matches:
        send_telegram(header + "\n(no divergence signals)")
        return

    lines: List[str] = []
    for sym, div_type, px, rsi_val, ptime in matches:
        d_str = ptime.astimezone(NY).strftime("%Y-%m-%d")
        pretty = "Bullish" if div_type == "bullish" else "Bearish"
        lines.append(f"{sym:<6}  {pretty:<7}  RSI={rsi_val:5.1f}  Px={px:.2f}  pivot={d_str}")

    full = header + "\n\n" + "\n".join(lines)
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

