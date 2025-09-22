#!/usr/bin/env python3
"""
Scan the S&P 500 every 5 minutes for RSI(14) conditions and notify via Telegram.

Data: Alpaca Market Data (Basic/free, IEX feed)
Universe: Wikipedia S&P 500 constituents
Author: you
"""

import os, sys, math, time, json, logging
import requests
import pandas as pd
import numpy as np
from datetime import datetime, timedelta, timezone
from zoneinfo import ZoneInfo

# ========= CONFIG =========
ALPACA_KEY = os.getenv("ALPACA_API_KEY_ID", "")
ALPACA_SECRET = os.getenv("ALPACA_API_SECRET_KEY", "")
# Alpaca data (REST) base URL:
ALPACA_DATA_BASE = "https://data.alpaca.markets"
# Alpaca trading API (clock)
ALPACA_TRADE_BASE = os.getenv("ALPACA_TRADE_BASE", "https://paper-api.alpaca.markets")

TELEGRAM_TOKEN = os.getenv("TELEGRAM_BOT_TOKEN", "")
TELEGRAM_CHAT_ID = os.getenv("TELEGRAM_CHAT_ID", "")  # e.g., "-1001234567890" for a channel/group

RSI_PERIOD = int(os.getenv("RSI_PERIOD", "14"))
TIMEFRAME = "5Min"
ALERT_MODE = os.getenv("ALERT_MODE", "cross")  # 'cross' or 'absolute'
MAX_SYMBOLS_PER_BATCH = int(os.getenv("MAX_SYMBOLS_PER_BATCH", "300"))  # keep URL under ~16k chars
LOOKBACK_MINUTES = int(os.getenv("LOOKBACK_MINUTES", "150"))  # ~30 bars of 5m; enough for RSI(14)
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
        # A simple regular-hours window gate: 9:35–16:00 ET to ensure at least one 5m bar formed
        start = now_ny.replace(hour=9, minute=35, second=0, microsecond=0)
        end   = now_ny.replace(hour=16, minute=0, second=0, microsecond=0)
        return start <= now_ny <= end
    except Exception as e:
        logging.warning(f"Clock check failed, defaulting to run: {e}")
        return True  # be permissive if clock call fails

def get_sp500_symbols() -> list[str]:
    """Robust S&P 500 list fetcher: Wikipedia with headers; fallback to a maintained CSV."""
    wiki_url = "https://en.wikipedia.org/wiki/List_of_S%26P_500_companies"
    hdrs = {
        "User-Agent": "Mozilla/5.0 (X11; Linux x86_64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/125.0 Safari/537.36",
        "Accept-Language": "en-US,en;q=0.9",
        "Referer": "https://en.wikipedia.org/",
        "Cache-Control": "no-cache",
    }
    try:
        # Fetch the HTML ourselves (avoids 403) then let pandas parse it.
        r = requests.get(wiki_url, headers=hdrs, timeout=30)
        r.raise_for_status()
        tables = pd.read_html(r.text)  # uses lxml if installed
        df = next((t for t in tables if "Symbol" in t.columns), None)
        if df is None:
            raise RuntimeError("Couldn't find 'Symbol' column on Wikipedia page.")
        syms = (
            df["Symbol"].astype(str).str.strip().str.upper().tolist()
        )
        # Deduplicate & sort
        syms = sorted({s for s in syms if s})
        return syms
    except Exception as e:
        logging.warning(f"Wikipedia fetch failed ({e}); falling back to public CSV.")
        # Fallback: maintained CSV of S&P 500 constituents
        csv_url = "https://raw.githubusercontent.com/datasets/s-and-p-500-companies/main/data/constituents.csv"
        r = requests.get(csv_url, headers={"User-Agent": hdrs["User-Agent"]}, timeout=30)
        r.raise_for_status()
        import io
        df = pd.read_csv(io.StringIO(r.text))
        col = "Symbol" if "Symbol" in df.columns else "symbol"
        syms = df[col].astype(str).str.strip().str.upper().tolist()
        syms = sorted({s for s in syms if s})
        return syms


def chunk_symbols(symbols: list[str], max_per_batch=MAX_SYMBOLS_PER_BATCH) -> list[list[str]]:
    out, cur, cur_len = [], [], 0
    # keep URL under ~16k; rough budget per symbol incl comma ~8 chars average
    budget = 15000
    for s in symbols:
        # conservative length check
        added = len(s) + (1 if cur else 0)
        if len(",".join(cur)) + added > budget or len(cur) >= max_per_batch:
            out.append(cur); cur = [s]
        else:
            cur.append(s)
    if cur:
        out.append(cur)
    return out

def fetch_5m_bars(symbols: list[str]) -> dict[str, pd.DataFrame]:
    """Fetch recent 5m bars for many symbols via Alpaca. Returns dict[symbol] -> DataFrame with columns: t, o, h, l, c, v"""
    since_utc = datetime.now(timezone.utc) - timedelta(minutes=LOOKBACK_MINUTES)
    start_iso = since_utc.isoformat(timespec="seconds").replace("+00:00", "Z")

    result: dict[str, list[dict]] = {}
    for batch in chunk_symbols(symbols):
        # Page through if needed to ensure every symbol has at least RSI_PERIOD+1 bars
        next_token = None
        bars_collected = 0
        # Fetch until we have enough (or a couple pages to be safe)
        for _ in range(5):
            params = {
                "timeframe": TIMEFRAME,
                "symbols": ",".join(batch),
                "start": start_iso,
                "adjustment": "all",
                "feed": "iex",  # real-time IEX on Basic plan
                "sort": "asc",
            }
            if next_token:
                params["page_token"] = next_token

            r = requests.get(f"{ALPACA_DATA_BASE}/v2/stocks/bars", headers=HEADERS, params=params, timeout=30)
            r.raise_for_status()
            data = r.json()

            bars_map = data.get("bars", {})
            # The multi-symbol response is a dict mapping symbol to its bars
            if not isinstance(bars_map, dict):
                logging.warning(f"Expected 'bars' to be a dict, but got {type(bars_map)}. Skipping.")
                continue

            for sym, b_list in bars_map.items():
                result.setdefault(sym, []).extend(b_list)
            
            bars_collected += sum(len(v) for v in bars_map.values())

            next_token = data.get("next_page_token")
            if not next_token:
                break

        logging.info(f"Fetched ~{bars_collected} bars for batch size {len(batch)}")

    # Convert to DataFrames
    out: dict[str, pd.DataFrame] = {}
    for sym, blist in result.items():
        if not blist:
            continue
        df = pd.DataFrame(blist)
        # Normalize columns
        cols = { "t":"t", "o":"o", "h":"h", "l":"l", "c":"c", "v":"v" }
        # ensure correct types
        df["t"] = pd.to_datetime(df["t"], utc=True)
        for k in ["o","h","l","c"]:
            df[k] = pd.to_numeric(df[k], errors="coerce")
        df["v"] = pd.to_numeric(df["v"], errors="coerce")
        df = df.sort_values("t").dropna(subset=["c"])
        out[sym] = df[["t","o","h","l","c","v"]]
    return out

def wilder_rsi(close: pd.Series, period: int = 14) -> pd.Series:
    """RSI (Wilder's smoothing)"""
    delta = close.diff()
    gains = (delta.clip(lower=0)).abs()
    losses = (-delta.clip(upper=0)).abs()
    avg_gain = gains.ewm(alpha=1/period, min_periods=period, adjust=False).mean()
    avg_loss = losses.ewm(alpha=1/period, min_periods=period, adjust=False).mean()
    rs = avg_gain / avg_loss.replace(0, np.nan)
    rsi = 100 - (100 / (1 + rs))
    return rsi

def scan(symbols: list[str]) -> list[tuple[str,float,float]]:
    """Return [(symbol, rsi_last, price_last)] that meet the alert condition."""
    # Pull data
    bars_by_sym = fetch_5m_bars(symbols)

    hits = []
    for sym, df in bars_by_sym.items():
        if df.shape[0] < (RSI_PERIOD + 1):
            continue
        rsi = wilder_rsi(df["c"], RSI_PERIOD).dropna()
        if rsi.empty:
            continue
        last = float(rsi.iloc[-1])
        last_close = float(df["c"].iloc[-1])
        if ALERT_MODE == "cross":
            prev = float(rsi.iloc[-2]) if len(rsi) >= 2 else np.nan
            if not np.isnan(prev) and prev <= 30.0 and last > 30.0:
                hits.append((sym, last, last_close))
        else:  # absolute
            if last > 30.0:
                hits.append((sym, last, last_close))

    hits.sort(key=lambda x: x[1], reverse=True)  # highest RSI first
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

    hits = scan(syms)
    now_ny = datetime.now(timezone.utc).astimezone(NY).strftime("%Y-%m-%d %H:%M")
    mode_desc = "crossed up > 30" if ALERT_MODE == "cross" else "RSI > 30"
    header = f"⏱️ 5m RSI scan ({mode_desc}) — {now_ny} ET\nMatches: {len(hits)}"

    if not hits:
        send_telegram(header + "\n(no new signals this interval)")
        return

    # Telegram messages are limited to 4096 chars; chunk if needed
    lines = []
    for sym, rsi_val, px in hits:
        lines.append(f"{sym: <6}  RSI={rsi_val:5.1f}  Px={px:.2f}")
    full = header + "\n\n" + "\n".join(lines)

    # chunk safely
    max_len = 4000
    if len(full) <= max_len:
        send_telegram(full)
    else:
        # send header + first chunk, then continue
        send_telegram(header)
        block = []
        cur = 0
        for ln in lines:
            if cur + len(ln) + 1 > max_len:
                send_telegram("\n".join(block)); block = []; cur = 0
            block.append(ln); cur += len(ln) + 1
        if block:
            send_telegram("\n".join(block))

if __name__ == "__main__":
    main()
