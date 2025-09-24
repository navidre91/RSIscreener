#!/usr/bin/env python3
"""
Scan the S&P 500 every 5 minutes for RSI(14) bullish/bearish divergences and notify via Telegram.

Data: Alpaca Market Data (Basic/free, IEX feed)
Universe: Wikipedia S&P 500 constituents
"""

import os, logging
from datetime import datetime, timedelta, timezone
from zoneinfo import ZoneInfo
from typing import List, Tuple, Dict, Optional

import requests
import pandas as pd
import numpy as np

# ========= CONFIG =========
ALPACA_KEY = os.getenv("ALPACA_API_KEY_ID", "")
ALPACA_SECRET = os.getenv("ALPACA_API_SECRET_KEY", "")
ALPACA_DATA_BASE = "https://data.alpaca.markets"  # data REST base URL
ALPACA_TRADE_BASE = os.getenv("ALPACA_TRADE_BASE", "https://paper-api.alpaca.markets")

TELEGRAM_TOKEN = os.getenv("TELEGRAM_BOT_TOKEN", "")
TELEGRAM_CHAT_ID = os.getenv("TELEGRAM_CHAT_ID", "")

RSI_PERIOD = int(os.getenv("RSI_PERIOD", "14"))
TIMEFRAME = "5Min"
MAX_SYMBOLS_PER_BATCH = int(os.getenv("MAX_SYMBOLS_PER_BATCH", "300"))
LOOKBACK_MINUTES = int(os.getenv("LOOKBACK_MINUTES", "150"))

# Divergence detection tuning
DIVERGENCE_TYPES = {t.strip().lower() for t in os.getenv("DIVERGENCE_TYPES", "bullish,bearish").split(",") if t.strip()}
PIVOT_WINDOW = int(os.getenv("DIVERGENCE_PIVOT_WINDOW", "3"))  # bars on each side of pivot
RECENT_BARS = int(os.getenv("DIVERGENCE_RECENT_BARS", "15"))   # last pivot must be within this many bars

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


def is_regular_hours_now() -> bool:
    """Use Alpaca clock + NY hours gate to avoid pre/post market alerts."""
    try:
        r = requests.get(f"{ALPACA_TRADE_BASE}/v2/clock", headers=HEADERS, timeout=10)
        r.raise_for_status()
        clk = r.json()
        if not clk.get("is_open", False):
            return False

        now_utc = datetime.fromisoformat(clk["timestamp"].replace("Z", "+00:00"))
        now_ny = now_utc.astimezone(NY)
        # 9:35–16:00 ET to ensure at least one 5m bar formed
        start = now_ny.replace(hour=9, minute=35, second=0, microsecond=0)
        end = now_ny.replace(hour=16, minute=0, second=0, microsecond=0)
        return start <= now_ny <= end
    except Exception as e:
        logging.warning(f"Clock check failed, defaulting to run: {e}")
        return True


def get_sp500_symbols() -> List[str]:
    """Fetch S&P 500 symbols with a Wikipedia scrape; fallback to a maintained CSV."""
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
        syms = sorted({s for s in syms if s})
        return syms
    except Exception as e:
        logging.warning(f"Wikipedia fetch failed ({e}); falling back to public CSV.")
        csv_url = "https://raw.githubusercontent.com/datasets/s-and-p-500-companies/main/data/constituents.csv"
        r = requests.get(csv_url, headers={"User-Agent": hdrs["User-Agent"]}, timeout=30)
        r.raise_for_status()
        import io
        df = pd.read_csv(io.StringIO(r.text))
        col = "Symbol" if "Symbol" in df.columns else "symbol"
        syms = df[col].astype(str).str.strip().str.upper().tolist()
        syms = sorted({s for s in syms if s})
        return syms


def chunk_symbols(symbols: List[str], max_per_batch=MAX_SYMBOLS_PER_BATCH) -> List[List[str]]:
    out, cur = [], []
    budget = 15000  # keep URL under ~16k
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


def fetch_5m_bars(symbols: List[str]) -> Dict[str, pd.DataFrame]:
    """Fetch recent 5m bars for many symbols via Alpaca. Returns dict[symbol] -> DataFrame with columns: t, o, h, l, c, v"""
    since_utc = datetime.now(timezone.utc) - timedelta(minutes=LOOKBACK_MINUTES)
    start_iso = since_utc.isoformat(timespec="seconds").replace("+00:00", "Z")

    result: Dict[str, List[dict]] = {}
    for batch in chunk_symbols(symbols):
        next_token = None
        bars_collected = 0
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

            bars_collected += sum(len(v) for v in bars_map.values())
            next_token = data.get("next_page_token")
            if not next_token:
                break

        logging.info(f"Fetched ~{bars_collected} bars for batch size {len(batch)}")

    out: Dict[str, pd.DataFrame] = {}
    for sym, blist in result.items():
        if not blist:
            continue
        df = pd.DataFrame(blist)
        df["t"] = pd.to_datetime(df["t"], utc=True)
        for k in ["o", "h", "l", "c"]:
            df[k] = pd.to_numeric(df[k], errors="coerce")
        df["v"] = pd.to_numeric(df["v"], errors="coerce")
        df = df.sort_values("t").dropna(subset=["c"])
        out[sym] = df[["t", "o", "h", "l", "c", "v"]]
    return out


def wilder_rsi(close: pd.Series, period: int = 14) -> pd.Series:
    """RSI (Wilder's smoothing). Keeps full length with NaNs for early values."""
    delta = close.diff()
    gains = (delta.clip(lower=0)).abs()
    losses = (-delta.clip(upper=0)).abs()
    avg_gain = gains.ewm(alpha=1 / period, min_periods=period, adjust=False).mean()
    avg_loss = losses.ewm(alpha=1 / period, min_periods=period, adjust=False).mean()
    rs = avg_gain / avg_loss.replace(0, np.nan)
    rsi = 100 - (100 / (1 + rs))
    return rsi


def find_pivot_lows(series: pd.Series, window: int) -> List[Tuple[int, float]]:
    """Return list of (index_position, value) for pivot lows using a simple fractal rule."""
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
    """Return list of (index_position, value) for pivot highs using a simple fractal rule."""
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
    """Bullish divergence: price makes lower low, RSI makes higher low (using last two pivot lows)."""
    lows = find_pivot_lows(price, pivot_window)
    if len(lows) < 2:
        return None
    i1, p1 = lows[-2]
    i2, p2 = lows[-1]
    if i2 <= i1:
        return None
    # recency gate
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
    """Bearish divergence: price makes higher high, RSI makes lower high (using last two pivot highs)."""
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
    """Return list of (symbol, divergence_type, last_price, last_rsi, pivot_time) for recent divergences."""
    bars_by_sym = fetch_5m_bars(symbols)

    hits: List[Tuple[str, str, float, float, datetime]] = []
    for sym, df in bars_by_sym.items():
        if df.shape[0] < (RSI_PERIOD + 10):  # need enough bars for pivots and RSI
            continue
        close = df["c"].reset_index(drop=True)
        rsi = wilder_rsi(close, RSI_PERIOD)
        if rsi.isna().all():
            continue

        found: List[Dict] = []
        if "bullish" in DIVERGENCE_TYPES:
            bull = detect_bullish_divergence(close, rsi, PIVOT_WINDOW, RECENT_BARS)
            if bull:
                found.append(bull)
        if "bearish" in DIVERGENCE_TYPES:
            bear = detect_bearish_divergence(close, rsi, PIVOT_WINDOW, RECENT_BARS)
            if bear:
                found.append(bear)

        if not found:
            continue

        # Use the most recent divergence (by i2) if both present
        div = max(found, key=lambda d: d["i2"])  # type: ignore[index]
        last_px = float(close.iloc[-1])
        last_rsi = float(rsi.iloc[-1]) if not np.isnan(rsi.iloc[-1]) else float(rsi.dropna().iloc[-1])
        pivot_time = df["t"].iloc[int(div["i2"])]
        hits.append((sym, div["type"], last_px, last_rsi, pivot_time.to_pydatetime()))

    # Sort by pivot_time descending (most recent first)
    hits.sort(key=lambda x: x[4], reverse=True)
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

    if not is_regular_hours_now():
        logging.info("Market not in regular hours window; exiting.")
        return

    syms = get_sp500_symbols()
    logging.info(f"Universe size: {len(syms)} symbols")

    hits = scan_for_divergences(syms)
    now_ny = datetime.now(timezone.utc).astimezone(NY).strftime("%Y-%m-%d %H:%M")
    header = f"⏱️ 5m RSI divergence scan — {now_ny} ET\nMatches: {len(hits)}"

    if not hits:
        send_telegram(header + "\n(no divergence signals this interval)")
        return

    # Compose lines
    lines: List[str] = []
    for sym, div_type, px, rsi_val, ptime in hits:
        t_str = ptime.astimezone(NY).strftime("%H:%M")
        pretty = "Bullish" if div_type == "bullish" else "Bearish"
        lines.append(f"{sym:<6}  {pretty:<7}  RSI={rsi_val:5.1f}  Px={px:.2f}  @{t_str} ET")

    full = header + "\n\n" + "\n".join(lines)

    max_len = 4000
    if len(full) <= max_len:
        send_telegram(full)
    else:
        send_telegram(header)
        block: List[str] = []
        cur = 0
        for ln in lines:
            if cur + len(ln) + 1 > max_len:
                send_telegram("\n".join(block))
                block = []
                cur = 0
            block.append(ln)
            cur += len(ln) + 1
        if block:
            send_telegram("\n".join(block))


if __name__ == "__main__":
    main()

