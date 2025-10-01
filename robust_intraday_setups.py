#!/usr/bin/env python3
"""Robust intraday swing trading scanner covering setups A1–A6.

This script implements the algorithms described in ``robust_intraday_setups.md``.
Each setup is implemented as a modular detector producing high-conviction
signals when strict criteria are met. The script focuses on RTH (regular
trading hours) minute data and daily context sourced from the Alpaca Market
Data API.

The scanner can be run for a single setup or all setups. Results are printed
to stdout and optionally delivered to a Telegram endpoint when credentials are
provided. The implementation emphasises composable feature engineering so that
setups can share precomputed context (VWAP, RVOL, strength score components,
etc.).
"""

from __future__ import annotations

import argparse
import dataclasses
import io
import json
import logging
import math
import os
import sys
import time
from collections import defaultdict
from dataclasses import dataclass, field
from datetime import date, datetime, timedelta, timezone
from typing import Any, Dict, List, Optional, Sequence

import numpy as np
import pandas as pd
import requests
from zoneinfo import ZoneInfo


# ================================================================
# Configuration (env overrides available)
# ================================================================

ALPACA_KEY = os.getenv("ALPACA_API_KEY_ID", "")
ALPACA_SECRET = os.getenv("ALPACA_API_SECRET_KEY", "")
ALPACA_DATA_BASE = os.getenv("ALPACA_DATA_BASE", "https://data.alpaca.markets")
ALPACA_TRADE_BASE = os.getenv("ALPACA_TRADE_BASE", "https://paper-api.alpaca.markets")

TELEGRAM_TOKEN = os.getenv("TELEGRAM_BOT_TOKEN", "")
TELEGRAM_CHAT_ID = os.getenv("TELEGRAM_CHAT_ID", "")

LOGLEVEL = os.getenv("LOGLEVEL", "INFO").upper()

DEFAULT_SETUP_NAMES = ["A1", "A2", "A3", "A4", "A5", "A6"]

NY = ZoneInfo("America/New_York")


# ================================================================
# Dataclasses and domain models
# ================================================================


@dataclass
class StrengthComponents:
    """Container for the 10-point strength score items.

    The scanner tallies these booleans to keep signals rare/high conviction.
    """

    catalyst: bool = False
    gap_context: bool = False
    rvol: bool = False
    trend_alignment: bool = False
    sector_confirmation: bool = False
    location: bool = False
    order_flow: bool = False
    vwap_behavior: bool = False
    structure: bool = False
    risk_reward: bool = False

    def score(self) -> int:
        return sum(int(v) for v in dataclasses.asdict(self).values())

    def as_dict(self) -> Dict[str, bool]:
        return dataclasses.asdict(self)


@dataclass
class SetupSignal:
    symbol: str
    setup: str
    timestamp: datetime
    trigger_level: float
    direction: str
    strength: StrengthComponents
    metadata: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "symbol": self.symbol,
            "setup": self.setup,
            "timestamp": self.timestamp.isoformat(),
            "trigger": self.trigger_level,
            "direction": self.direction,
            "strength": self.strength.score(),
            "components": self.strength.as_dict(),
            "metadata": self.metadata,
        }


@dataclass
class EventsCalendar:
    """Lightweight holder for catalysts/halts loaded from optional CSV files."""

    earnings: Dict[str, List[date]] = field(default_factory=lambda: defaultdict(list))
    macro: Dict[str, List[date]] = field(default_factory=lambda: defaultdict(list))
    halts: Dict[str, List[Dict[str, Any]]] = field(default_factory=lambda: defaultdict(list))

    @classmethod
    def from_paths(
        cls,
        earnings_path: Optional[str],
        macro_path: Optional[str],
        halts_path: Optional[str],
    ) -> "EventsCalendar":
        cal = cls()
        if earnings_path and os.path.isfile(earnings_path):
            df = pd.read_csv(earnings_path)
            for _, row in df.iterrows():
                sym = str(row.get("symbol", "")).strip().upper()
                dt = row.get("date")
                if sym and pd.notna(dt):
                    cal.earnings[sym].append(pd.to_datetime(dt).date())
        if macro_path and os.path.isfile(macro_path):
            df = pd.read_csv(macro_path)
            for _, row in df.iterrows():
                tag = str(row.get("tag", "")) or "macro"
                dt = row.get("date")
                if pd.notna(dt):
                    cal.macro[tag].append(pd.to_datetime(dt).date())
        if halts_path and os.path.isfile(halts_path):
            df = pd.read_csv(halts_path)
            for _, row in df.iterrows():
                sym = str(row.get("symbol", "")).strip().upper()
                reopen = row.get("reopen")
                halt_time = row.get("halt")
                direction = str(row.get("direction", "")).lower() or "unknown"
                if sym and pd.notna(reopen):
                    cal.halts[sym].append(
                        {
                            "halt": pd.to_datetime(halt_time) if pd.notna(halt_time) else None,
                            "reopen": pd.to_datetime(reopen),
                            "direction": direction,
                        }
                    )
        return cal


# ================================================================
# Network helpers
# ================================================================


def fail_if_missing_env(require_keys: bool = True) -> None:
    if require_keys:
        missing = []
        if not ALPACA_KEY:
            missing.append("ALPACA_API_KEY_ID")
        if not ALPACA_SECRET:
            missing.append("ALPACA_API_SECRET_KEY")
        if missing:
            raise SystemExit(f"Missing required env vars: {', '.join(missing)}")


def chunk_symbols(symbols: Sequence[str], max_batch_len: int = 200) -> List[List[str]]:
    batches: List[List[str]] = []
    cur: List[str] = []
    budget = 15000
    for sym in symbols:
        added = len(sym) + (1 if cur else 0)
        if len(",".join(cur)) + added > budget or len(cur) >= max_batch_len:
            if cur:
                batches.append(cur)
            cur = [sym]
        else:
            cur.append(sym)
    if cur:
        batches.append(cur)
    return batches


def alpaca_get(path: str, params: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
    headers = {"APCA-API-KEY-ID": ALPACA_KEY, "APCA-API-SECRET-KEY": ALPACA_SECRET}
    url = f"{ALPACA_DATA_BASE}{path}"
    for attempt in range(5):
        resp = requests.get(url, headers=headers, params=params, timeout=30)
        if resp.status_code == 429:
            time.sleep(min(2 ** attempt, 5))
            continue
        resp.raise_for_status()
        return resp.json()
    raise RuntimeError(f"Failed to fetch {path} after retries")


def fetch_sp500_symbols() -> List[str]:
    wiki_url = "https://en.wikipedia.org/wiki/List_of_S%26P_500_companies"
    headers = {
        "User-Agent": "Mozilla/5.0 (X11; Linux x86_64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/125 Safari/537.36",
        "Accept-Language": "en-US,en;q=0.9",
        "Referer": "https://en.wikipedia.org/",
        "Cache-Control": "no-cache",
    }
    try:
        resp = requests.get(wiki_url, headers=headers, timeout=30)
        resp.raise_for_status()
        tables = pd.read_html(resp.text)
        df = next((t for t in tables if "Symbol" in t.columns), None)
        if df is None:
            raise RuntimeError("No 'Symbol' column found")
        return sorted({s.strip().upper() for s in df["Symbol"].astype(str) if s})
    except Exception as exc:  # pragma: no cover - network fallback
        logging.warning("Wikipedia fetch failed (%s); falling back to public CSV", exc)
        fallback = "https://raw.githubusercontent.com/datasets/s-and-p-500-companies/main/data/constituents.csv"
        resp = requests.get(fallback, headers={"User-Agent": headers["User-Agent"]}, timeout=30)
        resp.raise_for_status()
        df = pd.read_csv(io.StringIO(resp.text))
        col = "Symbol" if "Symbol" in df.columns else "symbol"
        return sorted({s.strip().upper() for s in df[col].astype(str) if s})


def fetch_daily_bars(symbols: Sequence[str], lookback_days: int) -> Dict[str, pd.DataFrame]:
    since = datetime.now(timezone.utc) - timedelta(days=lookback_days * 2)
    start = since.isoformat(timespec="seconds").replace("+00:00", "Z")
    out: Dict[str, List[Dict[str, Any]]] = defaultdict(list)
    for batch in chunk_symbols(symbols):
        next_token: Optional[str] = None
        for _ in range(6):
            params = {
                "timeframe": "1Day",
                "symbols": ",".join(batch),
                "start": start,
                "adjustment": "raw",
                "feed": "iex",
                "sort": "asc",
                "limit": 1000,
            }
            if next_token:
                params["page_token"] = next_token
            data = alpaca_get("/v2/stocks/bars", params)
            bars = data.get("bars", {})
            for sym, rows in bars.items():
                out[sym].extend(rows)
            next_token = data.get("next_page_token")
            if not next_token:
                break
            time.sleep(0.2)
    frames: Dict[str, pd.DataFrame] = {}
    for sym, rows in out.items():
        if not rows:
            continue
        df = pd.DataFrame(rows)
        df.rename(columns={"t": "timestamp", "o": "open", "h": "high", "l": "low", "c": "close", "v": "volume"}, inplace=True)
        df["timestamp"] = pd.to_datetime(df["timestamp"], utc=True)
        df.sort_values("timestamp", inplace=True)
        df.drop_duplicates("timestamp", keep="last", inplace=True)
        frames[sym] = df.tail(lookback_days)
    return frames


def fetch_intraday_bars(
    symbols: Sequence[str],
    start: datetime,
    end: datetime,
) -> Dict[str, pd.DataFrame]:
    out: Dict[str, List[Dict[str, Any]]] = defaultdict(list)
    start_iso = start.astimezone(timezone.utc).isoformat(timespec="seconds").replace("+00:00", "Z")
    end_iso = end.astimezone(timezone.utc).isoformat(timespec="seconds").replace("+00:00", "Z")
    for batch in chunk_symbols(symbols, max_batch_len=50):
        next_token: Optional[str] = None
        for _ in range(12):
            params = {
                "timeframe": "1Min",
                "symbols": ",".join(batch),
                "start": start_iso,
                "end": end_iso,
                "adjustment": "raw",
                "feed": "iex",
                "sort": "asc",
                "limit": 10000,
            }
            if next_token:
                params["page_token"] = next_token
            data = alpaca_get("/v2/stocks/bars", params)
            bars = data.get("bars", {})
            for sym, rows in bars.items():
                out[sym].extend(rows)
            next_token = data.get("next_page_token")
            if not next_token:
                break
            time.sleep(0.2)
    frames: Dict[str, pd.DataFrame] = {}
    for sym, rows in out.items():
        if not rows:
            continue
        df = pd.DataFrame(rows)
        df.rename(columns={"t": "timestamp", "o": "open", "h": "high", "l": "low", "c": "close", "v": "volume"}, inplace=True)
        df["timestamp"] = pd.to_datetime(df["timestamp"], utc=True)
        df.sort_values("timestamp", inplace=True)
        df.drop_duplicates("timestamp", keep="last", inplace=True)
        frames[sym] = df
    return frames


# ================================================================
# Feature engineering helpers
# ================================================================


def compute_atr(df: pd.DataFrame, period: int) -> pd.Series:
    high = df["high"].astype(float)
    low = df["low"].astype(float)
    close = df["close"].astype(float)
    prev_close = close.shift(1)
    tr = pd.concat([
        high - low,
        (high - prev_close).abs(),
        (low - prev_close).abs(),
    ], axis=1).max(axis=1)
    return tr.rolling(window=period, min_periods=period).mean()


def compute_vwap(df: pd.DataFrame) -> pd.Series:
    pv = (df["close"] * df["volume"]).cumsum()
    vol = df["volume"].cumsum().replace(0, np.nan)
    return pv / vol


def minute_index(series: pd.Series, session_start: datetime) -> pd.Series:
    delta = (series - session_start).dt.total_seconds() / 60.0
    return delta.round().astype(int)


def compute_relative_volume(
    df: pd.DataFrame,
    daily_context: pd.DataFrame,
    session_minutes: int,
) -> pd.Series:
    """Estimate RVOL using average daily volume as proxy for intraday curve."""

    avg_daily_vol = daily_context["volume"].tail(20).median()
    if math.isnan(avg_daily_vol) or avg_daily_vol == 0:
        avg_daily_vol = daily_context["volume"].tail(5).mean()
    cum_vol = df["volume"].cumsum()
    elapsed = np.arange(1, len(df) + 1)
    expected = (avg_daily_vol / session_minutes) * elapsed
    expected = np.where(expected == 0, np.nan, expected)
    rvol = cum_vol / expected
    return pd.Series(rvol, index=df.index)


def calc_zscore(series: pd.Series, window: int) -> pd.Series:
    rolling_mean = series.rolling(window=window, min_periods=window).mean()
    rolling_std = series.rolling(window=window, min_periods=window).std(ddof=0)
    return (series - rolling_mean) / rolling_std.replace(0, np.nan)


def find_round_number(price: float) -> float:
    if price >= 200:
        step = 5
    elif price >= 50:
        step = 2
    elif price >= 20:
        step = 1
    elif price >= 5:
        step = 0.5
    else:
        step = 0.25
    return round(price / step) * step


def is_near(value: float, target: float, tolerance: float) -> bool:
    return abs(value - target) <= tolerance


def holds_above(series: pd.Series, level: float, window: int) -> bool:
    tail = series.tail(window)
    return bool(len(tail) == window and (tail > level).all())


def holds_below(series: pd.Series, level: float, window: int) -> bool:
    tail = series.tail(window)
    return bool(len(tail) == window and (tail < level).all())


def break_and_hold(df: pd.DataFrame, level: float, direction: str, hold_minutes: int) -> Optional[pd.Timestamp]:
    compare = df["close"]
    if direction == "up":
        breached = compare > level
    else:
        breached = compare < level
    if not breached.any():
        return None
    idx = breached.idxmax()
    tail = compare.loc[idx:].head(hold_minutes)
    if len(tail) < hold_minutes:
        return None
    if direction == "up" and (tail > level).all():
        return df.loc[idx, "timestamp"]
    if direction == "down" and (tail < level).all():
        return df.loc[idx, "timestamp"]
    return None


def shallow_pullbacks(df: pd.DataFrame, atr_series: pd.Series, window: int = 10) -> bool:
    closes = df["close"].tail(window)
    atr = atr_series.tail(window).mean()
    if math.isnan(atr) or atr == 0 or len(closes) < window:
        return False
    local_max = closes.max()
    drawdown = local_max - closes.min()
    return drawdown <= 1.5 * atr


def last_higher_low(df: pd.DataFrame, since: int = 10) -> Optional[float]:
    tail = df.tail(since)
    lows = tail["low"].values
    if len(lows) == 0:
        return None
    return float(np.min(lows))


def last_lower_high(df: pd.DataFrame, since: int = 10) -> Optional[float]:
    tail = df.tail(since)
    highs = tail["high"].values
    if len(highs) == 0:
        return None
    return float(np.max(highs))


# ================================================================
# Setup detectors
# ================================================================


@dataclass
class SymbolContext:
    symbol: str
    daily: pd.DataFrame
    intraday: pd.DataFrame
    session_start: datetime
    session_end: datetime
    opening_range_minutes: int = 15

    def __post_init__(self) -> None:
        self.intraday = self.intraday.copy()
        ts = pd.to_datetime(self.intraday["timestamp"], utc=True)
        self.intraday["timestamp"] = ts.dt.tz_convert(self.session_start.tzinfo)
        mask = (self.intraday["timestamp"] >= self.session_start) & (self.intraday["timestamp"] <= self.session_end)
        self.intraday = self.intraday.loc[mask].reset_index(drop=True)
        if self.intraday.empty:
            raise ValueError("No intraday data inside RTH window")
        self.intraday["minute_index"] = minute_index(self.intraday["timestamp"], self.session_start)
        self.intraday["vwap"] = compute_vwap(self.intraday)
        self.intraday["cum_volume"] = self.intraday["volume"].cumsum()
        self.intraday["atr15"] = compute_atr(self.intraday, period=15)
        self.intraday["rvol"] = compute_relative_volume(self.intraday, self.daily, session_minutes=390)
        signed_volume = np.sign(self.intraday["close"].diff().fillna(0)) * self.intraday["volume"]
        self.intraday["signed_volume"] = signed_volume
        self.intraday["imbalance_sum"] = signed_volume.rolling(window=15, min_periods=5).sum()
        self.intraday["imbalance_z"] = calc_zscore(self.intraday["imbalance_sum"], window=60)
        price_minus_vwap = self.intraday["close"] - self.intraday["vwap"]
        self.intraday["vwap_z"] = calc_zscore(price_minus_vwap, window=60)

    @property
    def prior_day(self) -> pd.Series:
        return self.daily.iloc[-2]

    @property
    def today_open(self) -> float:
        return float(self.intraday.iloc[0]["open"])

    @property
    def prior_close(self) -> float:
        return float(self.prior_day["close"])

    @property
    def atr14(self) -> float:
        atr = compute_atr(self.daily, period=14).iloc[-1]
        return float(atr) if pd.notna(atr) else float("nan")

    def opening_range(self) -> tuple[float, float, datetime]:
        head = self.intraday.head(self.opening_range_minutes)
        if len(head) < self.opening_range_minutes:
            raise ValueError("Insufficient bars for opening range")
        return float(head["high"].max()), float(head["low"].min()), head.iloc[-1]["timestamp"]

    def initial_balance(self) -> tuple[float, float]:
        ib = self.intraday.head(60)
        if len(ib) < 60:
            raise ValueError("Insufficient bars for initial balance")
        return float(ib["high"].max()), float(ib["low"].min())

    def prior_levels(self) -> Dict[str, float]:
        return {
            "high": float(self.prior_day["high"]),
            "low": float(self.prior_day["low"]),
            "close": float(self.prior_day["close"]),
        }


class SetupDetectors:
    def __init__(self, context: SymbolContext, params: Dict[str, Any], events: EventsCalendar, today: date):
        self.ctx = context
        self.params = params
        self.events = events
        self.today = today

    def strength_base(self) -> StrengthComponents:
        return StrengthComponents()

    # ------------------------------------------------------------
    # A1 – NR7 Opening Range Breakout
    # ------------------------------------------------------------
    def detect_a1(self) -> List[SetupSignal]:
        cfg = self.params.get("A1", {})
        req_nr7 = cfg.get("require_NR7_yesterday", True)
        or_minutes = int(cfg.get("use_OR_minutes", 15))
        min_gap = float(cfg.get("min_gap_ATR_ratio", 0.75))
        min_rvol = float(cfg.get("min_RVOL_open", 2.0))
        hold_minutes = int(cfg.get("break_hold_minutes", 3))

        daily = self.ctx.daily
        if len(daily) < 8:
            return []
        ranges = daily["high"] - daily["low"]
        nr7 = ranges.iloc[-2] == ranges.tail(7).min()
        if req_nr7 and not nr7:
            return []

        atr = self.ctx.atr14
        if math.isnan(atr) or atr == 0:
            return []

        prior_close = self.ctx.prior_close
        gap_ratio = abs(self.ctx.today_open - prior_close) / atr
        if gap_ratio < min_gap:
            return []

        or_high, or_low, or_end = self.ctx.opening_range()
        or_slice = self.ctx.intraday[self.ctx.intraday["timestamp"] <= or_end]
        or_rvol = float(or_slice["rvol"].iloc[-1]) if not or_slice.empty else float("nan")
        if math.isnan(or_rvol) or or_rvol < min_rvol:
            return []

        price_series = self.ctx.intraday
        post_or = price_series[price_series["timestamp"] >= or_end]
        signals: List[SetupSignal] = []

        def build_strength(direction: str, entry_idx: int, stop_level: float) -> StrengthComponents:
            comps = self.strength_base()
            comps.gap_context = gap_ratio >= min_gap
            comps.rvol = or_rvol >= min_rvol
            trend_ma = self.ctx.daily["close"].rolling(window=20).mean().iloc[-2]
            last_close = self.ctx.daily["close"].iloc[-2]
            comps.trend_alignment = (last_close >= trend_ma) if direction == "long" else (last_close <= trend_ma)
            comps.location = True if direction == "long" and or_high >= self.ctx.prior_levels()["high"] else True
            hold_series = price_series.loc[entry_idx : entry_idx + hold_minutes]
            vwap_tail = hold_series["vwap"]
            if direction == "long":
                comps.vwap_behavior = holds_above(vwap_tail, level=or_high, window=len(vwap_tail))
            else:
                comps.vwap_behavior = holds_below(vwap_tail, level=or_low, window=len(vwap_tail))
            risk = abs(price_series.loc[entry_idx, "close"] - stop_level)
            range_or = or_high - or_low
            comps.risk_reward = bool(range_or and risk and range_or / risk >= 2.5)
            imbalance = price_series["imbalance_z"].iloc[: entry_idx + 1].tail(3)
            comps.order_flow = bool((imbalance > 1.5).all()) if direction == "long" else bool((imbalance < -1.5).all())
            return comps

        if not post_or.empty:
            for idx in post_or.index:
                close_price = float(post_or.loc[idx, "close"])
                if close_price > or_high:
                    ts = break_and_hold(post_or.loc[idx:], or_high, "up", hold_minutes)
                    if ts is None:
                        continue
                    stop = or_low
                    comps = build_strength("long", idx, stop)
                    if comps.score() >= int(cfg.get("min_strength", 8)):
                        signals.append(
                            SetupSignal(
                                symbol=self.ctx.symbol,
                                setup="A1_LONG",
                                timestamp=ts.to_pydatetime(),
                                trigger_level=or_high,
                                direction="long",
                                strength=comps,
                                metadata={
                                    "gap_ratio": round(gap_ratio, 2),
                                    "or_high": round(or_high, 4),
                                    "or_low": round(or_low, 4),
                                },
                            )
                        )
                    break

            for idx in post_or.index:
                close_price = float(post_or.loc[idx, "close"])
                if close_price < or_low:
                    ts = break_and_hold(post_or.loc[idx:], or_low, "down", hold_minutes)
                    if ts is None:
                        continue
                    stop = or_high
                    comps = build_strength("short", idx, stop)
                    if comps.score() >= int(cfg.get("min_strength", 8)):
                        signals.append(
                            SetupSignal(
                                symbol=self.ctx.symbol,
                                setup="A1_SHORT",
                                timestamp=ts.to_pydatetime(),
                                trigger_level=or_low,
                                direction="short",
                                strength=comps,
                                metadata={
                                    "gap_ratio": round(gap_ratio, 2),
                                    "or_high": round(or_high, 4),
                                    "or_low": round(or_low, 4),
                                },
                            )
                        )
                    break
        return signals

    # ------------------------------------------------------------
    # A2 – Event-Driven Opening Drive
    # ------------------------------------------------------------
    def detect_a2(self) -> List[SetupSignal]:
        cfg = self.params.get("A2", {})
        min_gap_pct = float(cfg.get("min_gap_pct", 3.0))
        min_rvol = float(cfg.get("min_RVOL_open", 2.5))
        vwap_hold_window = int(cfg.get("vwap_hold_lookback_min", 5))
        require_event = bool(cfg.get("event_required", True))

        today_events = self.events.earnings.get(self.ctx.symbol, [])
        catalyst = self.today in today_events
        if require_event and not catalyst:
            return []

        prior_close = self.ctx.prior_close
        gap_pct = ((self.ctx.today_open - prior_close) / prior_close) * 100.0

        or_high, or_low, or_end = self.ctx.opening_range()
        or_slice = self.ctx.intraday[self.ctx.intraday["timestamp"] <= or_end]
        or_rvol = float(or_slice["rvol"].iloc[-1]) if not or_slice.empty else float("nan")
        if math.isnan(or_rvol) or or_rvol < min_rvol:
            return []

        post_or = self.ctx.intraday[self.ctx.intraday["timestamp"] >= or_end]
        vwap_series = self.ctx.intraday["vwap"]

        signals: List[SetupSignal] = []

        if gap_pct >= min_gap_pct:
            holds = holds_above(vwap_series, level=or_low, window=vwap_hold_window)
            if holds:
                trig = max(or_high, float(post_or["high"].head(5).max()))
                ts = break_and_hold(post_or, trig, "up", hold_minutes=vwap_hold_window)
                if ts is not None:
                    comps = self.strength_base()
                    comps.catalyst = catalyst
                    comps.gap_context = True
                    comps.rvol = True
                    comps.vwap_behavior = True
                    comps.trend_alignment = True
                    comps.location = trig >= self.ctx.prior_levels()["high"]
                    comps.order_flow = bool(self.ctx.intraday["imbalance_z"].tail(5).mean() > 1.5)
                    atr = self.ctx.atr14
                    stop = last_higher_low(self.ctx.intraday, since=10) or or_low
                    risk = trig - stop if stop is not None else float("nan")
                    comps.risk_reward = bool(not math.isnan(risk) and risk > 0 and (trig + 2 * risk) / risk >= 2.5)
                    if comps.score() >= int(cfg.get("min_strength", 8)):
                        signals.append(
                            SetupSignal(
                                symbol=self.ctx.symbol,
                                setup="A2_LONG",
                                timestamp=ts.to_pydatetime(),
                                trigger_level=trig,
                                direction="long",
                                strength=comps,
                                metadata={
                                    "gap_pct": round(gap_pct, 2),
                                    "or_high": round(or_high, 4),
                                    "or_low": round(or_low, 4),
                                },
                            )
                        )

        if gap_pct <= -min_gap_pct:
            holds = holds_below(vwap_series, level=or_high, window=vwap_hold_window)
            if holds:
                trig = min(or_low, float(post_or["low"].head(5).min()))
                ts = break_and_hold(post_or, trig, "down", hold_minutes=vwap_hold_window)
                if ts is not None:
                    comps = self.strength_base()
                    comps.catalyst = catalyst
                    comps.gap_context = True
                    comps.rvol = True
                    comps.vwap_behavior = True
                    comps.trend_alignment = True
                    comps.location = trig <= self.ctx.prior_levels()["low"]
                    comps.order_flow = bool(self.ctx.intraday["imbalance_z"].tail(5).mean() < -1.5)
                    stop = last_lower_high(self.ctx.intraday, since=10) or or_high
                    risk = stop - trig if stop is not None else float("nan")
                    comps.risk_reward = bool(not math.isnan(risk) and risk > 0 and (stop - 2 * risk) / risk >= 2.5)
                    if comps.score() >= int(cfg.get("min_strength", 8)):
                        signals.append(
                            SetupSignal(
                                symbol=self.ctx.symbol,
                                setup="A2_SHORT",
                                timestamp=ts.to_pydatetime(),
                                trigger_level=trig,
                                direction="short",
                                strength=comps,
                                metadata={
                                    "gap_pct": round(gap_pct, 2),
                                    "or_high": round(or_high, 4),
                                    "or_low": round(or_low, 4),
                                },
                            )
                        )
        return signals

    # ------------------------------------------------------------
    # A3 – Order Flow Imbalance Trend Day
    # ------------------------------------------------------------
    def detect_a3(self) -> List[SetupSignal]:
        cfg = self.params.get("A3", {})
        min_sigma = float(cfg.get("min_imbalance_sigma", 2.0))
        min_rvol = float(cfg.get("min_RVOL_open", 1.5))
        hold_minutes = int(cfg.get("hold_minutes", 5))

        df = self.ctx.intraday
        if len(df) < 90:
            return []

        imb = df["imbalance_z"].fillna(0)
        last_sigma = float(imb.iloc[-1])
        or_high, or_low, or_end = self.ctx.opening_range()
        or_slice = df[df["timestamp"] <= or_end]
        or_rvol = float(or_slice["rvol"].iloc[-1]) if not or_slice.empty else float("nan")
        if math.isnan(or_rvol) or or_rvol < min_rvol:
            return []

        signals: List[SetupSignal] = []

        if last_sigma >= min_sigma:
            failed = holds_above(df["close"], level=df["vwap"].iloc[-1], window=hold_minutes)
            if failed:
                trigger = float(df["high"].tail(hold_minutes).max())
                stop = last_higher_low(df, since=hold_minutes) or or_low
                comps = self.strength_base()
                comps.order_flow = True
                comps.rvol = True
                comps.vwap_behavior = True
                comps.trend_alignment = True
                comps.location = trigger >= or_high
                risk = trigger - stop if stop is not None else float("nan")
                comps.risk_reward = bool(not math.isnan(risk) and risk > 0 and (trigger + 2 * risk) / risk >= 2.5)
                if comps.score() >= int(cfg.get("min_strength", 8)):
                    signals.append(
                        SetupSignal(
                            symbol=self.ctx.symbol,
                            setup="A3_LONG",
                            timestamp=df.iloc[-1]["timestamp"].to_pydatetime(),
                            trigger_level=trigger,
                            direction="long",
                            strength=comps,
                            metadata={"imbalance_sigma": round(last_sigma, 2)},
                        )
                    )

        if last_sigma <= -min_sigma:
            failed = holds_below(df["close"], level=df["vwap"].iloc[-1], window=hold_minutes)
            if failed:
                trigger = float(df["low"].tail(hold_minutes).min())
                stop = last_lower_high(df, since=hold_minutes) or or_high
                comps = self.strength_base()
                comps.order_flow = True
                comps.rvol = True
                comps.vwap_behavior = True
                comps.trend_alignment = True
                comps.location = trigger <= or_low
                risk = stop - trigger if stop is not None else float("nan")
                comps.risk_reward = bool(not math.isnan(risk) and risk > 0 and (stop - 2 * risk) / risk >= 2.5)
                if comps.score() >= int(cfg.get("min_strength", 8)):
                    signals.append(
                        SetupSignal(
                            symbol=self.ctx.symbol,
                            setup="A3_SHORT",
                            timestamp=df.iloc[-1]["timestamp"].to_pydatetime(),
                            trigger_level=trigger,
                            direction="short",
                            strength=comps,
                            metadata={"imbalance_sigma": round(last_sigma, 2)},
                        )
                    )
        return signals

    # ------------------------------------------------------------
    # A4 – Liquidity Sweep and Reclaim
    # ------------------------------------------------------------
    def detect_a4(self) -> List[SetupSignal]:
        cfg = self.params.get("A4", {})
        max_minutes = int(cfg.get("max_sweep_close_time_min", 5))
        min_sigma = float(cfg.get("min_RVOL_spike_sigma", 2.0))

        prior = self.ctx.prior_levels()
        df = self.ctx.intraday
        signals: List[SetupSignal] = []

        rolling_vol = df["volume"].rolling(window=60, min_periods=10).median()
        for idx, row in df.iterrows():
            ts = row["timestamp"]
            recent = df.loc[:idx]
            vol = row["volume"]
            baseline = rolling_vol.loc[idx]
            vol_sigma = (vol - baseline) / (baseline if baseline else np.nan)
            if math.isnan(vol_sigma) or vol_sigma < min_sigma:
                continue
            # Long: sweep prior low
            if row["low"] < prior["low"]:
                window_df = df[(df["timestamp"] >= ts) & (df["timestamp"] <= ts + timedelta(minutes=max_minutes))]
                closes = window_df["close"]
                if not closes.empty and (closes > prior["low"]).any():
                    comps = self.strength_base()
                    comps.location = True
                    comps.rvol = True
                    comps.vwap_behavior = bool(df.loc[idx: idx + max_minutes, "vwap"].iloc[0] <= row["close"])
                    comps.order_flow = bool(df["imbalance_z"].loc[idx:].head(3).mean() > 0)
                    comps.risk_reward = True
                    if comps.score() >= int(cfg.get("min_strength", 8)):
                        signals.append(
                            SetupSignal(
                                symbol=self.ctx.symbol,
                                setup="A4_LONG",
                                timestamp=ts.to_pydatetime(),
                                trigger_level=float(row["close"]),
                                direction="long",
                                strength=comps,
                                metadata={"swept_level": prior["low"]},
                            )
                        )
            # Short: sweep prior high
            if row["high"] > prior["high"]:
                window_df = df[(df["timestamp"] >= ts) & (df["timestamp"] <= ts + timedelta(minutes=max_minutes))]
                closes = window_df["close"]
                if not closes.empty and (closes < prior["high"]).any():
                    comps = self.strength_base()
                    comps.location = True
                    comps.rvol = True
                    comps.vwap_behavior = bool(df.loc[idx: idx + max_minutes, "vwap"].iloc[0] >= row["close"])
                    comps.order_flow = bool(df["imbalance_z"].loc[idx:].head(3).mean() < 0)
                    comps.risk_reward = True
                    if comps.score() >= int(cfg.get("min_strength", 8)):
                        signals.append(
                            SetupSignal(
                                symbol=self.ctx.symbol,
                                setup="A4_SHORT",
                                timestamp=ts.to_pydatetime(),
                                trigger_level=float(row["close"]),
                                direction="short",
                                strength=comps,
                                metadata={"swept_level": prior["high"]},
                            )
                        )
        return signals

    # ------------------------------------------------------------
    # A5 – VWAP Extreme
    # ------------------------------------------------------------
    def detect_a5(self) -> List[SetupSignal]:
        cfg = self.params.get("A5", {})
        min_abs_z = float(cfg.get("min_abs_zscore", 2.5))
        mode = str(cfg.get("choose_mode", "auto")).lower()
        exhaustion_range = float(cfg.get("exhaustion_range_mult", 1.5))
        exhaustion_vol = float(cfg.get("exhaustion_vol_mult", 2.0))

        df = self.ctx.intraday
        if len(df) < 60:
            return []

        signals: List[SetupSignal] = []
        last_row = df.iloc[-1]
        last_z = float(last_row["vwap_z"])
        atr15 = df["atr15"].iloc[-1]
        vol60 = df["volume"].rolling(window=60, min_periods=30).mean().iloc[-1]

        def exhaustion(bar: pd.Series) -> bool:
            range_ok = (bar["high"] - bar["low"]) >= exhaustion_range * atr15 if atr15 else False
            vol_ok = bar["volume"] >= exhaustion_vol * vol60 if vol60 else False
            return bool(range_ok and vol_ok)

        def add_signal(tag: str, direction: str, trigger: float, comps: StrengthComponents) -> None:
            signals.append(
                SetupSignal(
                    symbol=self.ctx.symbol,
                    setup=tag,
                    timestamp=last_row["timestamp"].to_pydatetime(),
                    trigger_level=trigger,
                    direction=direction,
                    strength=comps,
                    metadata={"vwap_z": round(last_z, 2)},
                )
            )

        comps = self.strength_base()
        comps.vwap_behavior = True
        comps.order_flow = bool(abs(df["imbalance_z"].tail(10).mean()) < 1.0)
        comps.rvol = bool(df["rvol"].iloc[-1] >= 1.5)
        comps.trend_alignment = True
        comps.location = True
        comps.risk_reward = True

        if mode in ("mean_revert", "auto") and last_z <= -min_abs_z and exhaustion(last_row):
            add_signal("A5_LONG_MR", "long", float(last_row["close"]), comps)
        if mode in ("mean_revert", "auto") and last_z >= min_abs_z and exhaustion(last_row):
            add_signal("A5_SHORT_MR", "short", float(last_row["close"]), comps)

        if mode in ("trend", "auto"):
            if holds_above(df["close"], df["vwap"].iloc[-1], window=10) and shallow_pullbacks(df, df["atr15"], window=15):
                add_signal("A5_LONG_TREND", "long", float(last_row["close"]), comps)
            if holds_below(df["close"], df["vwap"].iloc[-1], window=10) and shallow_pullbacks(df, df["atr15"], window=15):
                add_signal("A5_SHORT_TREND", "short", float(last_row["close"]), comps)
        return [sig for sig in signals if sig.strength.score() >= int(cfg.get("min_strength", 8))]

    # ------------------------------------------------------------
    # A6 – Halt/Resume Continuation
    # ------------------------------------------------------------
    def detect_a6(self) -> List[SetupSignal]:
        cfg = self.params.get("A6", {})
        watch = bool(cfg.get("watch_for_halts", True))
        if not watch:
            return []
        rr_minutes = int(cfg.get("reopen_range_minutes", 3))
        min_rvol = float(cfg.get("min_RVOL_post_reopen", 3.0))
        hold_minutes = int(cfg.get("hold_minutes", 3))

        events = self.events.halts.get(self.ctx.symbol, [])
        if not events:
            return []

        df = self.ctx.intraday
        signals: List[SetupSignal] = []
        for event in events:
            reopen = event.get("reopen")
            if reopen is None:
                continue
            reopen = reopen.tz_convert(self.ctx.session_start.tzinfo)
            post = df[df["timestamp"] >= reopen]
            if post.empty:
                continue
            rr = post.head(rr_minutes)
            rr_high = float(rr["high"].max())
            rr_low = float(rr["low"].min())
            post_rvol = float(post["rvol"].iloc[min(len(post) - 1, rr_minutes)])
            if math.isnan(post_rvol) or post_rvol < min_rvol:
                continue

            ts_up = break_and_hold(post, rr_high, "up", hold_minutes)
            if ts_up is not None:
                comps = self.strength_base()
                comps.catalyst = True
                comps.rvol = True
                comps.vwap_behavior = True
                comps.order_flow = True
                comps.location = True
                comps.risk_reward = True
                if comps.score() >= int(cfg.get("min_strength", 8)):
                    signals.append(
                        SetupSignal(
                            symbol=self.ctx.symbol,
                            setup="A6_LONG",
                            timestamp=ts_up.to_pydatetime(),
                            trigger_level=rr_high,
                            direction="long",
                            strength=comps,
                            metadata={"reopen_range_high": rr_high, "reopen_time": reopen.isoformat()},
                        )
                    )

            ts_down = break_and_hold(post, rr_low, "down", hold_minutes)
            if ts_down is not None:
                comps = self.strength_base()
                comps.catalyst = True
                comps.rvol = True
                comps.vwap_behavior = True
                comps.order_flow = True
                comps.location = True
                comps.risk_reward = True
                if comps.score() >= int(cfg.get("min_strength", 8)):
                    signals.append(
                        SetupSignal(
                            symbol=self.ctx.symbol,
                            setup="A6_SHORT",
                            timestamp=ts_down.to_pydatetime(),
                            trigger_level=rr_low,
                            direction="short",
                            strength=comps,
                            metadata={"reopen_range_low": rr_low, "reopen_time": reopen.isoformat()},
                        )
                    )
        return signals


# ================================================================
# Runtime orchestration
# ================================================================


def build_params() -> Dict[str, Dict[str, Any]]:
    return {
        "A1": {
            "require_NR7_yesterday": True,
            "use_OR_minutes": 15,
            "min_gap_ATR_ratio": 0.75,
            "min_RVOL_open": 2.0,
            "break_hold_minutes": 3,
            "min_strength": 8,
        },
        "A2": {
            "min_gap_pct": 3.0,
            "min_RVOL_open": 2.5,
            "vwap_hold_lookback_min": 5,
            "event_required": True,
            "min_strength": 8,
        },
        "A3": {
            "min_imbalance_sigma": 2.0,
            "min_RVOL_open": 1.5,
            "hold_minutes": 5,
            "min_strength": 8,
        },
        "A4": {
            "max_sweep_close_time_min": 5,
            "min_RVOL_spike_sigma": 2.0,
            "min_strength": 8,
        },
        "A5": {
            "min_abs_zscore": 2.5,
            "choose_mode": "auto",
            "exhaustion_range_mult": 1.5,
            "exhaustion_vol_mult": 2.0,
            "min_strength": 8,
        },
        "A6": {
            "watch_for_halts": True,
            "reopen_range_minutes": 3,
            "min_RVOL_post_reopen": 3.0,
            "hold_minutes": 3,
            "min_strength": 8,
        },
    }


def send_telegram_message(text: str) -> None:
    if not TELEGRAM_TOKEN or not TELEGRAM_CHAT_ID:
        logging.info("Telegram credentials missing; skipping send")
        return
    url = f"https://api.telegram.org/bot{TELEGRAM_TOKEN}/sendMessage"
    payload = {"chat_id": TELEGRAM_CHAT_ID, "text": text, "parse_mode": "Markdown"}
    resp = requests.post(url, json=payload, timeout=30)
    if resp.status_code >= 400:
        logging.warning("Telegram send failed: %s", resp.text)


def summarise_signals(signals: Sequence[SetupSignal]) -> str:
    if not signals:
        return "No qualifying robust intraday setups today."
    lines = ["*Robust Intraday Setups*", ""]
    for sig in sorted(signals, key=lambda s: (s.setup, s.symbol)):
        comps = sig.strength
        parts = [
            f"{sig.setup} {sig.direction.upper()} {sig.symbol}",
            f"time={sig.timestamp.astimezone(NY).strftime('%H:%M')}",
            f"trigger={sig.trigger_level:.2f}",
            f"strength={comps.score()}/10",
        ]
        lines.append(" | ".join(parts))
    return "\n".join(lines)


def default_symbol_limit() -> int:
    env = os.getenv("INTRADAY_SYMBOL_LIMIT")
    if env and env.isdigit():
        return int(env)
    return 60


def parse_args(argv: Optional[Sequence[str]] = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Scan for robust intraday setups A1–A6")
    parser.add_argument("--setups", nargs="*", default=DEFAULT_SETUP_NAMES, help="Setups to run (e.g. A1 A3) or omit for all")
    parser.add_argument("--symbols", nargs="*", default=[], help="Overrides default symbol universe")
    parser.add_argument("--symbols-file", help="Optional file with symbols, one per line")
    parser.add_argument("--lookback-days", type=int, default=40, help="Daily lookback for ATR/NR7")
    parser.add_argument("--session-date", help="ISO date to scan (default: today)")
    parser.add_argument("--symbol-limit", type=int, default=default_symbol_limit(), help="Limit number of symbols to scan")
    parser.add_argument("--dry-run", action="store_true", help="Do not send Telegram notifications")
    parser.add_argument("--earnings-csv", help="Optional catalyst CSV with columns: symbol,date")
    parser.add_argument("--macro-csv", help="Optional macro CSV with columns: tag,date")
    parser.add_argument("--halts-csv", help="Optional halts CSV with columns: symbol,halt,reopen,direction")
    return parser.parse_args(argv)


def load_symbols(args: argparse.Namespace) -> List[str]:
    symbols: List[str] = []
    if args.symbols:
        symbols.extend([s.upper() for s in args.symbols])
    if args.symbols_file and os.path.isfile(args.symbols_file):
        with open(args.symbols_file, "r", encoding="utf-8") as fh:
            symbols.extend([line.strip().upper() for line in fh if line.strip()])
    if not symbols:
        symbols = fetch_sp500_symbols()
    symbols = sorted({s for s in symbols if s})
    if args.symbol_limit:
        symbols = symbols[: args.symbol_limit]
    return symbols


def build_symbol_contexts(
    symbols: Sequence[str],
    intraday_map: Dict[str, pd.DataFrame],
    daily_map: Dict[str, pd.DataFrame],
    session_start: datetime,
    session_end: datetime,
) -> Dict[str, SymbolContext]:
    contexts: Dict[str, SymbolContext] = {}
    for sym in symbols:
        ddf = daily_map.get(sym)
        idf = intraday_map.get(sym)
        if ddf is None or idf is None or ddf.empty or idf.empty or len(ddf) < 2:
            continue
        try:
            contexts[sym] = SymbolContext(
                symbol=sym,
                daily=ddf,
                intraday=idf,
                session_start=session_start,
                session_end=session_end,
            )
        except ValueError as exc:
            logging.debug("Skipping %s: %s", sym, exc)
    return contexts


def run_scan(args: argparse.Namespace) -> List[SetupSignal]:
    fail_if_missing_env()
    logging.info("Loading symbol universe…")
    symbols = load_symbols(args)
    if not symbols:
        raise SystemExit("No symbols to scan")
    logging.info("Scanning %d symbols", len(symbols))

    session_date = (
        pd.to_datetime(args.session_date).date() if args.session_date else datetime.now(NY).date()
    )
    open_time = datetime.strptime("09:30", "%H:%M").time()
    close_time = datetime.strptime("16:00", "%H:%M").time()
    session_start = datetime.combine(session_date, open_time, tzinfo=NY)
    session_end = datetime.combine(session_date, close_time, tzinfo=NY)

    logging.info("Fetching daily bars…")
    daily_map = fetch_daily_bars(symbols, lookback_days=args.lookback_days)
    logging.info("Fetching intraday bars…")
    intraday_map = fetch_intraday_bars(symbols, start=session_start - timedelta(hours=2), end=session_end + timedelta(minutes=10))

    contexts = build_symbol_contexts(symbols, intraday_map, daily_map, session_start, session_end)
    params = build_params()
    events = EventsCalendar.from_paths(args.earnings_csv, args.macro_csv, args.halts_csv)

    selected_setups = set(s.upper() for s in args.setups) if args.setups else set(DEFAULT_SETUP_NAMES)
    signals: List[SetupSignal] = []
    for sym, ctx in contexts.items():
        detectors = SetupDetectors(ctx, params, events, session_date)
        if "A1" in selected_setups:
            signals.extend(detectors.detect_a1())
        if "A2" in selected_setups:
            signals.extend(detectors.detect_a2())
        if "A3" in selected_setups:
            signals.extend(detectors.detect_a3())
        if "A4" in selected_setups:
            signals.extend(detectors.detect_a4())
        if "A5" in selected_setups:
            signals.extend(detectors.detect_a5())
        if "A6" in selected_setups:
            signals.extend(detectors.detect_a6())
    return signals


def main(argv: Optional[Sequence[str]] = None) -> int:
    logging.basicConfig(level=LOGLEVEL, format="%(asctime)s %(levelname)s %(message)s")
    args = parse_args(argv)
    try:
        signals = run_scan(args)
    except Exception as exc:  # pragma: no cover - runtime guard
        logging.exception("Scan failed: %s", exc)
        return 1

    summary = summarise_signals(signals)
    print(summary)
    if signals:
        as_json = [sig.to_dict() for sig in signals]
        print("\nJSON:")
        print(json.dumps(as_json, indent=2))

    if not args.dry_run:
        send_telegram_message(summary)

    return 0


if __name__ == "__main__":  # pragma: no cover
    sys.exit(main())
