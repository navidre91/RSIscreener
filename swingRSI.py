#!/usr/bin/env python3
"""
Swing RSI Scanner (Daily)
------------------------
Implements the pseudo-code you described for a daily swing-low RSI setup
and generates option structure suggestions (vertical or butterfly) for
the next trading day.

Signal (for last completed daily bar t):
- Compute RSI(14)
- If low[t] == min(low[t-4:t+1])   # 5-day swing low using only past
- Let A_day = t-2
- If RSI[A_day] < 35 and RSI[t] > RSI[A_day] + 3:
    - P_A = close[A_day]
    - B   = low[t]
    - Suggest options structure for next bar (t+1):
        - butterfly: K2 = round(P_A), K1 = round(current_price), K3 = K2 + (K2 - K1), DTE 14–35
        - vertical:  K1 = round(current_price), K2 = round(P_A), DTE 21–45

Data: Alpaca Market Data (IEX feed)
Universe: S&P 500 (Wikipedia)
Notify: Telegram

Notes
- We detect only on the most recent completed daily bar (yesterday when run near today open).
- "current_price" is taken as the last close (c[t]). You may substitute a real-time price if desired.
"""

import os
import argparse
import logging
from dataclasses import dataclass
from datetime import datetime, timedelta, timezone
from typing import Dict, List, Optional, Tuple
from zoneinfo import ZoneInfo

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
LOOKBACK_DAYS = int(os.getenv("LOOKBACK_DAYS", "400"))

# Swing detection params
SWING_LEFT_WINDOW = int(os.getenv("SWING_LEFT_WINDOW", "4"))  # t is a 5-day swing low if l[t] == min(l[t-4:t+1])
RSI_A_MAX = float(os.getenv("RSI_A_MAX", "35"))                # RSI[A_day] must be below this
RSI_DELTA_MIN = float(os.getenv("RSI_DELTA_MIN", "3"))          # RSI[t] - RSI[A_day] must exceed this

# Option suggestion config
SWING_OPTIONS_STRUCTURE = os.getenv("SWING_OPTIONS_STRUCTURE", "butterfly").strip().lower()  # 'butterfly' | 'vertical' | 'both'

# DTE guidance strings (informational)
BUTTERFLY_DTE_RANGE = os.getenv("BUTTERFLY_DTE_RANGE", "14–35")
VERTICAL_DTE_RANGE = os.getenv("VERTICAL_DTE_RANGE", "21–45")

NY = ZoneInfo("America/New_York")
LOGLEVEL = os.getenv("LOGLEVEL", "INFO").upper()

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
    """Return True during a narrow open window (9:35–9:45 ET) on an open trading day.

    Running in this window lets us use yesterday's completed daily bar and
    place trades on today's session ("next bar").
    """
    try:
        r = requests.get(f"{ALPACA_TRADE_BASE}/v2/clock", headers=HEADERS, timeout=10)
        r.raise_for_status()
        clk = r.json()
        if not clk.get("is_open", False):
            return False

        now_utc = datetime.fromisoformat(clk["timestamp"].replace("Z", "+00:00"))
        now_ny = now_utc.astimezone(NY)
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


def chunk_symbols(symbols: List[str], max_per_batch: int = 300) -> List[List[str]]:
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


def fetch_daily_bars(symbols: List[str], lookback_days: int = LOOKBACK_DAYS) -> Dict[str, pd.DataFrame]:
    since_utc = datetime.now(timezone.utc) - timedelta(days=lookback_days)
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


def _round_to_increment(x: float, inc: float) -> float:
    return round(x / inc) * inc


def round_to_liquid_strike(price: float) -> float:
    """Round to a "liquid" strike increment based on price level.

    Heuristic grid:
    - < $25:   $0.50
    - $25–$200: $1.00
    - $200–$500: $5.00
    - >= $500: $10.00
    """
    p = float(price)
    if p < 25:
        inc = 0.5
    elif p < 200:
        inc = 1.0
    elif p < 500:
        inc = 5.0
    else:
        inc = 10.0
    val = _round_to_increment(p, inc)
    # Normalize minor FP artifacts
    return float(f"{val:.2f}")


@dataclass
class SwingSignal:
    symbol: str
    t_index: int
    a_index: int
    date_t: datetime
    date_a: datetime
    last_close: float
    rsi_t: float
    rsi_a: float
    p_a: float  # target close at A_day
    b_low: float  # swing low


def detect_latest_swing_signal(df: pd.DataFrame) -> Optional[SwingSignal]:
    """Detect the swing RSI signal only on the most recent bar.

    Returns SwingSignal if the latest bar t qualifies; otherwise None.
    """
    if df.shape[0] < max(RSI_PERIOD + 5, SWING_LEFT_WINDOW + 3):
        return None

    close = df["c"].reset_index(drop=True)
    low = df["l"].reset_index(drop=True)
    rsi = wilder_rsi(close, RSI_PERIOD)

    t = len(df) - 1
    a_day = t - 2
    if a_day < 0 or t - SWING_LEFT_WINDOW < 0:
        return None

    window_vals = low.iloc[t - SWING_LEFT_WINDOW : t + 1]
    if window_vals.shape[0] < (SWING_LEFT_WINDOW + 1):
        return None

    l_t = float(low.iloc[t])
    min_win = float(np.nanmin(window_vals.values))
    # Allow tiny tolerance to handle float representation
    if not (np.isclose(l_t, min_win, rtol=0, atol=1e-6) or l_t <= min_win + 1e-6):
        return None

    rsi_a = float(rsi.iloc[a_day]) if not pd.isna(rsi.iloc[a_day]) else np.nan
    rsi_t = float(rsi.iloc[t]) if not pd.isna(rsi.iloc[t]) else np.nan
    if np.isnan(rsi_a) or np.isnan(rsi_t):
        return None

    if not (rsi_a < RSI_A_MAX and (rsi_t > rsi_a + RSI_DELTA_MIN)):
        return None

    p_a = float(close.iloc[a_day])
    last_close = float(close.iloc[t])

    return SwingSignal(
        symbol="",  # filled by caller
        t_index=t,
        a_index=a_day,
        date_t=pd.to_datetime(df["t"].iloc[t]).to_pydatetime(),
        date_a=pd.to_datetime(df["t"].iloc[a_day]).to_pydatetime(),
        last_close=last_close,
        rsi_t=rsi_t,
        rsi_a=rsi_a,
        p_a=p_a,
        b_low=l_t,
    )


def scan_symbols(symbols: List[str]) -> List[Tuple[SwingSignal, Dict[str, str]]]:
    """Return list of (SwingSignal, suggestion) for symbols that signal on last bar.

    suggestion is a dict of formatted fields to render in output (str values),
    including strikes and DTE guidance for the requested structure(s).
    """
    bars_map = fetch_daily_bars(symbols)
    hits: List[Tuple[SwingSignal, Dict[str, str]]] = []

    for sym, df in bars_map.items():
        sig = detect_latest_swing_signal(df)
        if not sig:
            continue
        sig.symbol = sym

        # Build option suggestions
        suggestions: Dict[str, str] = {}
        current_price = sig.last_close
        p_a = sig.p_a

        allowed = {x for x in ["butterfly", "vertical"] if SWING_OPTIONS_STRUCTURE in (x, "both")}
        if "butterfly" in allowed:
            k2 = round_to_liquid_strike(p_a)
            k1 = round_to_liquid_strike(current_price)
            k3 = float(f"{(k2 + (k2 - k1)):.2f}")
            suggestions["butterfly"] = (
                f"K1={k1:.2f}, K2={k2:.2f}, K3={k3:.2f}, DTE {BUTTERFLY_DTE_RANGE}"
            )
        if "vertical" in allowed:
            k1v = round_to_liquid_strike(current_price)
            k2v = round_to_liquid_strike(p_a)
            suggestions["vertical"] = (
                f"K1={k1v:.2f}, K2={k2v:.2f}, DTE {VERTICAL_DTE_RANGE}"
            )

        hits.append((sig, suggestions))

    # Sort by strongest RSI lift (rsi_t - rsi_a) descending, then by most recent date (should be same)
    hits.sort(key=lambda x: (x[0].rsi_t - x[0].rsi_a), reverse=True)
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
    parser = argparse.ArgumentParser(description="Swing RSI (daily) scanner with options suggestions")
    parser.add_argument("--force-run", action="store_true", help="Run regardless of market open time gate")
    parser.add_argument(
        "--structure",
        choices=["butterfly", "vertical", "both"],
        default=None,
        help="Override options suggestion structure (defaults to SWING_OPTIONS_STRUCTURE env)",
    )
    args = parser.parse_args()

    fail_if_missing_env()

    if not args.force_run and not is_open_window_daily():
        logging.info("Not in the open window; exiting. Use --force-run to override.")
        return

    # Optional override of structure via CLI
    if args.structure:
        global SWING_OPTIONS_STRUCTURE
        SWING_OPTIONS_STRUCTURE = args.structure

    syms = get_sp500_symbols()
    logging.info(f"Universe size: {len(syms)} symbols")

    results = scan_symbols(syms)
    now_ny = datetime.now(timezone.utc).astimezone(NY).strftime("%Y-%m-%d %H:%M")
    mode = SWING_OPTIONS_STRUCTURE
    header = (
        f"📅 Swing RSI scan — {now_ny} ET\n"
        f"Signals: {len(results)}  |  Options: {mode}"
    )

    if not results:
        send_telegram(header + "\n(no swing signals)")
        return

    lines: List[str] = []
    for sig, sug in results:
        a_dt = sig.date_a.astimezone(NY).strftime("%Y-%m-%d")
        t_dt = sig.date_t.astimezone(NY).strftime("%Y-%m-%d")
        core = (
            f"{sig.symbol:<6}  t={t_dt} A={a_dt}  RSI_t={sig.rsi_t:4.1f} RSI_A={sig.rsi_a:4.1f}  "
            f"Px={sig.last_close:.2f}  P_A={sig.p_a:.2f}  B(low)={sig.b_low:.2f}"
        )
        extras: List[str] = []
        if SWING_OPTIONS_STRUCTURE in ("butterfly", "both") and "butterfly" in sug:
            extras.append(f"butterfly: {sug['butterfly']}")
        if SWING_OPTIONS_STRUCTURE in ("vertical", "both") and "vertical" in sug:
            extras.append(f"vertical:  {sug['vertical']}")
        if extras:
            core += "\n    " + "\n    ".join(extras)
        lines.append(core)

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
