#!/usr/bin/env python3
#%%
"""
Daily RSI(14) Divergence Screener — Offline Study (Jupyter-style)
-----------------------------------------------------------------
- OFFLINE ONLY: uses local daily OHLCV files under `data/sp500_daily/`.
- No Internet: no Wikipedia, no yfinance; no database refresh.
- Results are shown in DataFrames and per-candidate plots.
- Structured with #%% cells for step-by-step exploration in VS Code/Jupyter.

At the top, we compute and print the latest date for which local prices are
available across the cache, so you know what day the analysis refers to.
"""

#%% Config
import logging
import os
import time
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Dict, List, Optional, Tuple
from zoneinfo import ZoneInfo

import numpy as np
import pandas as pd


# General config (tweak here)
RSI_PERIOD: int = int(os.getenv("RSI_PERIOD", "14"))
LOOKBACK_DAYS: int = int(os.getenv("LOOKBACK_DAYS", "400"))
MAX_SYMBOLS_PER_BATCH: int = int(os.getenv("MAX_SYMBOLS_PER_BATCH", "60"))

# Offline mode only: no network concurrency settings needed
YF_BATCH_PAUSE: float = 0.0

# Divergence detection tuning
DIVERGENCE_TYPES = {t.strip().lower() for t in os.getenv("DIVERGENCE_TYPES", "bullish").split(",") if t.strip()}
PIVOT_WINDOW: int = int(os.getenv("DIVERGENCE_PIVOT_WINDOW", "3"))
RECENT_BARS: int = int(os.getenv("DIVERGENCE_RECENT_BARS", "20"))

# Universe control (offline)
# - Set UNIVERSE to a custom list (Yahoo-style tickers) like:
#   ["AAPL", "MSFT", "NVDA", "AMZN", "GOOGL"]
# - If None, scans the local folder `data/sp500_daily/` to derive available symbols.
UNIVERSE: Optional[List[str]] = None

# For quick experimentation, limit scan to the first N symbols (after sorting)
DEBUG_LIMIT: Optional[int] = None  # e.g., 50; set to None for full universe

NY = ZoneInfo("America/New_York")
LOGLEVEL = os.getenv("LOGLEVEL", "INFO").upper()
logging.basicConfig(level=LOGLEVEL, format="%(asctime)s %(levelname)s %(message)s")

DATA_BASE = Path(os.getenv("DATA_BASE", "data")).expanduser()
LOCAL_DAILY_DIR = DATA_BASE / "sp500_daily"
USE_LOCAL_DATA = True  # forced ON for offline script

# Reduce yfinance logging noise
logging.getLogger("yfinance").setLevel(logging.ERROR)
logging.getLogger("yfinance.data").setLevel(logging.ERROR)


#%% Offline: no local data refresh (intentionally omitted)


#%% Helpers: symbols and batching
def _yahoo_symbol(symbol: str) -> str:
    """Map Wikipedia/Alpaca-style tickers to Yahoo format.

    Example: BRK.B -> BRK-B ; BF.B -> BF-B
    """
    return symbol.replace(".", "-").strip().upper()


# No HTTP session in offline script


#%% Data helpers (offline only)


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


def get_local_universe_from_files() -> List[str]:
    """Return available symbols by scanning `LOCAL_DAILY_DIR` for files.

    Uses file basenames (without extension) as Yahoo-style tickers, e.g., 'BRK-B'.
    """
    if not LOCAL_DAILY_DIR.exists():
        logging.warning("Local daily dir not found: %s", LOCAL_DAILY_DIR)
        return []
    syms: List[str] = []
    for fp in LOCAL_DAILY_DIR.iterdir():
        if not fp.is_file():
            continue
        if fp.suffix.lower() not in {".parquet", ".csv"}:
            continue
        sym = fp.stem.strip().upper()
        if sym:
            syms.append(sym)
    return sorted({s for s in syms if s})


def _latest_date_in_file(fp: Path) -> Optional[pd.Timestamp]:
    try:
        if fp.suffix.lower() == ".parquet":
            df = pd.read_parquet(fp, columns=["date"])  # type: ignore[arg-type]
        else:
            df = pd.read_csv(fp, usecols=["date"])  # type: ignore[list-item]
        if df is None or df.empty or "date" not in df.columns:
            return None
        dts = pd.to_datetime(df["date"], errors="coerce")
        dts = dts.dropna()
        return None if dts.empty else pd.Timestamp(dts.max()).tz_localize("UTC")
    except Exception:
        return None


def get_latest_local_data_date() -> Optional[datetime]:
    """Scan local files and return the most recent daily bar date as a timezone-aware UTC datetime."""
    if not LOCAL_DAILY_DIR.exists():
        return None
    latest: Optional[pd.Timestamp] = None
    for fp in LOCAL_DAILY_DIR.iterdir():
        if not fp.is_file() or fp.suffix.lower() not in {".parquet", ".csv"}:
            continue
        ts = _latest_date_in_file(fp)
        if ts is None:
            continue
        if latest is None or ts > latest:
            latest = ts
    return None if latest is None else latest.to_pydatetime()


def fetch_daily_bars(symbols: List[str]) -> Dict[str, pd.DataFrame]:
    """Fetch recent daily OHLCV bars for given symbols from local cache only.

    Returns a dict of DataFrames with columns: t (UTC), o, h, l, c, v
    """
    start_cutoff = (datetime.now(timezone.utc) - timedelta(days=LOOKBACK_DAYS))
    out: Dict[str, pd.DataFrame] = {}
    if not LOCAL_DAILY_DIR.exists():
        logging.warning("Local data folder missing: %s", LOCAL_DAILY_DIR)
        return out
    for s in symbols:
        ysym = _yahoo_symbol(s)
        df_local = _read_local_ticker(ysym)
        if df_local is None or df_local.empty:
            continue
        # ensure cutoff applied (already applied in _read_local_ticker, but safe)
        df_local = df_local[df_local["t"] >= start_cutoff]
        out[s] = df_local[["t", "o", "h", "l", "c", "v"]]
    return out


#%% Technicals: RSI, MACD and pivots
def wilder_rsi(close: pd.Series, period: int = 14) -> pd.Series:
    delta = close.diff()
    gains = (delta.clip(lower=0)).abs()
    losses = (-delta.clip(upper=0)).abs()
    avg_gain = gains.ewm(alpha=1 / period, min_periods=period, adjust=False).mean()
    avg_loss = losses.ewm(alpha=1 / period, min_periods=period, adjust=False).mean()
    rs = avg_gain / avg_loss.replace(0, np.nan)
    return 100 - (100 / (1 + rs))


def macd(close: pd.Series, fast: int = 12, slow: int = 26, signal: int = 9) -> Tuple[pd.Series, pd.Series, pd.Series]:
    """Compute MACD line, signal line, and histogram."""
    ema_fast = close.ewm(span=fast, adjust=False).mean()
    ema_slow = close.ewm(span=slow, adjust=False).mean()
    macd_line = ema_fast - ema_slow
    signal_line = macd_line.ewm(span=signal, adjust=False).mean()
    hist = macd_line - signal_line
    return macd_line, signal_line, hist


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


#%% Run: build universe and scan (offline)
def build_universe() -> List[str]:
    if UNIVERSE is not None:
        syms = [
            _yahoo_symbol(s)  # normalize to Yahoo style for local files
            for s in UNIVERSE
            if isinstance(s, str) and s.strip()
        ]
    else:
        syms = get_local_universe_from_files()
    syms = sorted({s for s in syms if s})
    if DEBUG_LIMIT is not None:
        syms = syms[: int(DEBUG_LIMIT)]
    return syms


#%% Data Snapshot: latest available local date
LATEST_DATA_DT_UTC: Optional[datetime] = get_latest_local_data_date()
LATEST_DATA_DATE_STR: str = (
    (LATEST_DATA_DT_UTC.astimezone(NY).strftime("%Y-%m-%d")) if LATEST_DATA_DT_UTC else "unknown"
)
print(f"Latest local daily data date (ET): {LATEST_DATA_DATE_STR}")


#%% Optional: visualize a specific symbol's price and RSI with pivots
def plot_symbol_with_rsi(symbol: str, lookback_days: int = LOOKBACK_DAYS):
    """Quick visualization helper (offline).

    Separate subplots: Price (top, wide), MACD, Volume, RSI (bottom, narrow).
    Display range limited to the most recent 30 bars.
    """
    try:
        import matplotlib.pyplot as plt
    except Exception as e:
        print(f"matplotlib not available: {e}")
        return

    data = fetch_daily_bars([symbol]).get(symbol)
    if data is None or data.empty:
        print(f"No data for {symbol}")
        return

    # Prepare series
    c = pd.to_numeric(data["c"], errors="coerce").reset_index(drop=True)
    o = pd.to_numeric(data["o"], errors="coerce").reset_index(drop=True)
    v = pd.to_numeric(data["v"], errors="coerce").reset_index(drop=True)
    t = pd.to_datetime(data["t"])  # tz-aware UTC
    r = wilder_rsi(c, RSI_PERIOD)
    m_line, m_signal, m_hist = macd(c)

    # Limit to last 30 bars
    window = 30
    start = max(0, len(c) - window)
    c_s, o_s, v_s, r_s = c.iloc[start:], o.iloc[start:], v.iloc[start:], r.iloc[start:]
    m_line_s, m_signal_s, m_hist_s = m_line.iloc[start:], m_signal.iloc[start:], m_hist.iloc[start:]
    t_s = t.iloc[start:]

    # Create subplots with height ratios (RSI above MACD)
    import matplotlib.gridspec as gridspec
    from matplotlib.dates import AutoDateLocator, ConciseDateFormatter
    from matplotlib.ticker import FuncFormatter

    with plt.style.context("seaborn-v0_8-whitegrid"):
        fig = plt.figure(figsize=(12, 8), dpi=160)
        gs = gridspec.GridSpec(4, 1, height_ratios=[7, 2, 2, 2], hspace=0.06)

        ax_price = fig.add_subplot(gs[0])
        ax_rsi = fig.add_subplot(gs[1], sharex=ax_price)
        ax_macd = fig.add_subplot(gs[2], sharex=ax_price)
        ax_vol = fig.add_subplot(gs[3], sharex=ax_price)

        # Price
        ax_price.plot(t_s, c_s, color="#111111", linewidth=1.3, label="Close")
        ax_price.set_ylabel("Price")
        ax_price.margins(x=0.01)
        ax_price.legend(loc="upper left", fontsize=8, frameon=False)

        # RSI
        ax_rsi.fill_between(t_s, 30, 70, color="#cccccc", alpha=0.15)
        ax_rsi.plot(t_s, r_s, color="#1f77b4", linewidth=1.2, label=f"RSI({RSI_PERIOD})")
        ax_rsi.axhline(50, color="#888888", linestyle="--", linewidth=0.8, alpha=0.7)
        ax_rsi.axhline(30, color="#d62728", linestyle=":", alpha=0.7)
        ax_rsi.axhline(70, color="#2ca02c", linestyle=":", alpha=0.7)
        ax_rsi.set_ylim(0, 100)
        ax_rsi.set_ylabel("RSI")
        ax_rsi.legend(loc="upper left", fontsize=8, frameon=False)

        # MACD
        macd_colors = ["#2ca02c" if h >= 0 else "#d62728" for h in m_hist_s]
        ax_macd.bar(t_s, m_hist_s, color=macd_colors, alpha=0.35, width=0.6)
        ax_macd.plot(t_s, m_line_s, color="#1f77b4", linewidth=1.1, label="MACD")
        ax_macd.plot(t_s, m_signal_s, color="#ff7f0e", linewidth=1.1, label="Signal")
        ax_macd.axhline(0, color="#666666", linewidth=0.8, alpha=0.6)
        ax_macd.set_ylabel("MACD")
        ax_macd.legend(loc="upper left", fontsize=8, frameon=False)

        # Volume
        vol_colors = np.where(c_s >= o_s, "#2ca02c", "#d62728")
        ax_vol.bar(t_s, v_s, color=vol_colors, alpha=0.6, width=0.6)
        ax_vol.set_ylabel("Vol")
        ax_vol.yaxis.set_major_formatter(
            FuncFormatter(lambda x, pos: f"{x/1e6:.1f}M" if x >= 1e6 else (f"{x/1e3:.0f}K" if x >= 1e3 else f"{int(x)}"))
        )

        # Shared x formatting
        locator = AutoDateLocator(minticks=3, maxticks=7)
        formatter = ConciseDateFormatter(locator)
        # Remove offset text like "2025-Oct" from subplot axes
        try:
            formatter.show_offset = False  # type: ignore[attr-defined]
        except Exception:
            pass
        ax_vol.xaxis.set_major_locator(locator)
        ax_vol.xaxis.set_major_formatter(formatter)
        for ax in [ax_price, ax_rsi, ax_macd, ax_vol]:
            # Hide any residual offset text generated by the formatter
            try:
                ax.get_xaxis().get_offset_text().set_visible(False)
            except Exception:
                pass
        for ax in [ax_price, ax_rsi, ax_macd]:
            plt.setp(ax.get_xticklabels(), visible=False)

        # Title and layout
        ax_price.set_title(f"{symbol} — Price, RSI, MACD, Volume (last 30 bars)")
        fig.tight_layout()
        plt.show()


#%% Plot explanations for each candidate
def _to_utc_ts(dt_val) -> pd.Timestamp:
    ts = pd.Timestamp(dt_val)
    if ts.tz is None:
        ts = ts.tz_localize("UTC")
    else:
        ts = ts.tz_convert("UTC")
    return ts


def _find_index_for_dt(t_series: pd.Series, target_dt: datetime) -> Optional[int]:
    if t_series is None or len(t_series) == 0 or target_dt is None:
        return None
    try:
        ts_target = _to_utc_ts(target_dt)
        idx = pd.Index(pd.to_datetime(t_series))
        pos = idx.get_indexer([ts_target])
        if pos is not None and len(pos) and pos[0] != -1:
            return int(pos[0])
        # Fallback to nearest within 2 days
        diffs = (idx.view("int64") - ts_target.value).astype("int64").abs()
        near = int(diffs.argmin())
        near_dt = idx[near]
        tol_days = abs((near_dt - ts_target).total_seconds()) / 86400.0
        return near if tol_days <= 2.5 else None
    except Exception:
        return None


def _plot_divergence_explanation(symbol: str, data: pd.DataFrame, row: pd.Series) -> None:
    """Separate subplots: Price, MACD, Volume, RSI with pivot links on Price and RSI.

    Display range limited to the most recent 30 bars.
    """
    if data is None or data.empty:
        return
    try:
        import matplotlib.pyplot as plt
    except Exception as e:
        logging.error("matplotlib not available: %s", e)
        return

    c = pd.to_numeric(data["c"], errors="coerce").reset_index(drop=True)
    t = pd.to_datetime(data["t"])  # tz-aware UTC
    o = pd.to_numeric(data["o"], errors="coerce").reset_index(drop=True)
    v = pd.to_numeric(data["v"], errors="coerce").reset_index(drop=True)
    r = wilder_rsi(c, RSI_PERIOD)
    m_line, m_signal, m_hist = macd(c)

    typ = str(row.get("type", "")).strip().lower()
    color = "green" if typ == "bullish" else "red" if typ == "bearish" else "blue"

    dt1 = row.get("pivot_start_dt")
    dt2 = row.get("pivot_dt")
    i1 = _find_index_for_dt(t, dt1)
    i2 = _find_index_for_dt(t, dt2)
    if i1 is None or i2 is None:
        return

    p1 = float(row.get("p1")) if row.get("p1") is not None else float(c.iloc[i1])
    p2 = float(row.get("p2")) if row.get("p2") is not None else float(c.iloc[i2])
    r1 = float(row.get("r1")) if row.get("r1") is not None else float(r.iloc[i1])
    r2 = float(row.get("r2")) if row.get("r2") is not None else float(r.iloc[i2])

    # Limit to last 30 bars for plotting
    window = 30
    start = max(0, len(c) - window)
    t_s = t.iloc[start:]
    c_s = c.iloc[start:]
    o_s = o.iloc[start:]
    v_s = v.iloc[start:]
    r_s = r.iloc[start:]
    m_line_s, m_signal_s, m_hist_s = m_line.iloc[start:], m_signal.iloc[start:], m_hist.iloc[start:]

    # Determine if pivots are within the slice; adjust indices for slice
    i1_s = i1 - start
    i2_s = i2 - start
    pivots_in_range = (0 <= i1_s < len(c_s)) and (0 <= i2_s < len(c_s))

    import matplotlib.gridspec as gridspec
    from matplotlib.dates import AutoDateLocator, ConciseDateFormatter
    from matplotlib.ticker import FuncFormatter

    with plt.style.context("seaborn-v0_8-whitegrid"):
        fig = plt.figure(figsize=(11.5, 8.0), dpi=160)
        gs = gridspec.GridSpec(4, 1, height_ratios=[7, 2, 2, 2], hspace=0.06)

        ax_price = fig.add_subplot(gs[0])
        ax_rsi = fig.add_subplot(gs[1], sharex=ax_price)
        ax_macd = fig.add_subplot(gs[2], sharex=ax_price)
        ax_vol = fig.add_subplot(gs[3], sharex=ax_price)

        # Price
        ax_price.plot(t_s, c_s, color="#111111", linewidth=1.3, label="Close")
        if pivots_in_range:
            ax_price.scatter([t_s.iloc[i1_s], t_s.iloc[i2_s]], [p1, p2], color=color, s=42, zorder=3, marker="o", edgecolors="white")
            ax_price.plot([t_s.iloc[i1_s], t_s.iloc[i2_s]], [p1, p2], color=color, linestyle="--", linewidth=1.6)
        ax_price.set_ylabel("Price")
        ax_price.margins(x=0.01)
        ax_price.legend(loc="upper left", fontsize=8, frameon=False)

        # RSI
        ax_rsi.fill_between(t_s, 30, 70, color="#cccccc", alpha=0.15)
        ax_rsi.plot(t_s, r_s, color="#1f77b4", linewidth=1.2, label=f"RSI({RSI_PERIOD})")
        if pivots_in_range:
            ax_rsi.scatter([t_s.iloc[i1_s], t_s.iloc[i2_s]], [r1, r2], color=color, s=36, zorder=3, marker="o", edgecolors="white")
            ax_rsi.plot([t_s.iloc[i1_s], t_s.iloc[i2_s]], [r1, r2], color=color, linestyle="--", linewidth=1.4)
        ax_rsi.axhline(50, color="#888888", linestyle="--", linewidth=0.8, alpha=0.7)
        ax_rsi.axhline(30, color="#d62728", linestyle=":", alpha=0.7)
        ax_rsi.axhline(70, color="#2ca02c", linestyle=":", alpha=0.7)
        ax_rsi.set_ylim(0, 100)
        ax_rsi.set_ylabel("RSI")
        ax_rsi.legend(loc="upper left", fontsize=8, frameon=False)

        # MACD
        macd_colors = ["#2ca02c" if h >= 0 else "#d62728" for h in m_hist_s]
        ax_macd.bar(t_s, m_hist_s, color=macd_colors, alpha=0.35, width=0.6)
        ax_macd.plot(t_s, m_line_s, color="#1f77b4", linewidth=1.1, label="MACD")
        ax_macd.plot(t_s, m_signal_s, color="#ff7f0e", linewidth=1.1, label="Signal")
        ax_macd.axhline(0, color="#666666", linewidth=0.8, alpha=0.6)
        ax_macd.set_ylabel("MACD")
        ax_macd.legend(loc="upper left", fontsize=8, frameon=False)

        # Volume
        vol_colors = np.where(c_s >= o_s, "#2ca02c", "#d62728")
        ax_vol.bar(t_s, v_s, color=vol_colors, alpha=0.6, width=0.6)
        ax_vol.set_ylabel("Vol")
        ax_vol.yaxis.set_major_formatter(
            FuncFormatter(lambda x, pos: f"{x/1e6:.1f}M" if x >= 1e6 else (f"{x/1e3:.0f}K" if x >= 1e3 else f"{int(x)}"))
        )

        # X-axis formatting on bottom subplot
        locator = AutoDateLocator(minticks=3, maxticks=7)
        formatter = ConciseDateFormatter(locator)
        # Remove offset text like "2025-Oct" from subplot axes
        try:
            formatter.show_offset = False  # type: ignore[attr-defined]
        except Exception:
            pass
        ax_vol.xaxis.set_major_locator(locator)
        ax_vol.xaxis.set_major_formatter(formatter)
        for ax in [ax_price, ax_rsi, ax_macd, ax_vol]:
            try:
                ax.get_xaxis().get_offset_text().set_visible(False)
            except Exception:
                pass
        for ax in [ax_price, ax_rsi, ax_macd]:
            plt.setp(ax.get_xticklabels(), visible=False)

        pivot_str = pd.to_datetime(row.get("pivot_dt")).tz_convert(NY).strftime("%Y-%m-%d")
        info_line = (
            f"{typ.capitalize()} divergence • pivots="
            f"{pd.to_datetime(row.get('pivot_start_dt')).tz_convert(NY).strftime('%Y-%m-%d')}->{pivot_str}"
        )
        ax_price.set_title(f"{symbol} — Price, RSI, MACD, Volume (last 30 bars) • {info_line}")

        fig.tight_layout()
        plt.show()


def plot_all_signal_explanations(results: pd.DataFrame) -> None:
    if results is None or results.empty:
        print("No signals to plot.")
        return
    symbols = results["symbol"].astype(str).tolist()
    bars = fetch_daily_bars(sorted(set(symbols)))
    for _, row in results.iterrows():
        sym = str(row.get("symbol", ""))
        df = bars.get(sym)
        if df is None or df.empty:
            continue
        _plot_divergence_explanation(sym, df, row)

#%% Entry-point for quick local run
if __name__ == "__main__":
    # Build universe strictly from local cache (or custom UNIVERSE)
    syms = build_universe()
    print(f"Universe size (local): {len(syms)} symbols")

    # Scan
    results = scan_for_divergences(syms)
    now_ny = datetime.now(timezone.utc).astimezone(NY).strftime("%Y-%m-%d %H:%M")
    print(f"Completed daily RSI divergence scan — {now_ny} ET (data as of {LATEST_DATA_DATE_STR})")
    if results.empty:
        print("No divergence signals found.")
    else:
        display_cols = [
            "symbol","type","strength","pivot_start_dt","pivot_dt","last_rsi","last_price",
            "price_drop_pct","rsi_gain","price_rise_pct","rsi_drop"
        ]
        display_cols = [c for c in display_cols if c in results.columns]
        print(results[display_cols].head(30))
        # Plot explanation for each candidate (price + RSI with pivot links)
        try:
            plot_all_signal_explanations(results)
        except Exception as e:
            logging.error("Plotting failed: %s", e)


# %%
