#%% Setup
from __future__ import annotations

from pathlib import Path
from typing import Iterable

import pandas as pd


DATA_ROOT = Path("data/sp500_daily")

if not DATA_ROOT.exists():
    raise FileNotFoundError(
        f"Unable to locate data directory: {DATA_ROOT.resolve()}",
    )


#%% Helpers
def _resolve_first_timestamp(frame: pd.DataFrame) -> pd.Timestamp:
    """
    Locate the earliest timestamp in the frame, handling common layouts.

    The function supports data stored with a DatetimeIndex or a dedicated
    column named like ``date`` or ``timestamp`` (case-insensitive).
    """
    if isinstance(frame.index, pd.DatetimeIndex):
        first_timestamp = frame.index.min()
        if pd.isna(first_timestamp):
            raise ValueError("Datetime index does not contain any valid timestamps.")
        return pd.to_datetime(first_timestamp)

    candidate_columns: Iterable[str] = (
        col for col in frame.columns if col.lower() in {"date", "timestamp"}
    )

    for column in candidate_columns:
        series = pd.to_datetime(frame[column], errors="coerce")
        first_timestamp = series.min()
        if pd.notna(first_timestamp):
            return first_timestamp

    raise ValueError(
        "No datetime information found. Expected a DatetimeIndex or a column named "
        "'date' or 'timestamp'.",
    )


def load_first_year_for_ticker(parquet_path: Path) -> int:
    """Return the first calendar year found in the Parquet file."""
    frame = pd.read_parquet(parquet_path)
    first_timestamp = _resolve_first_timestamp(frame)
    return first_timestamp.year


#%% Oldest year per ticker
oldest_year_by_ticker: dict[str, int] = {}

for parquet_path in sorted(DATA_ROOT.glob("*.parquet")):
    ticker = parquet_path.stem.upper()
    try:
        oldest_year_by_ticker[ticker] = load_first_year_for_ticker(parquet_path)
    except ValueError as exc:
        print(f"Skipping {ticker}: {exc}")

for ticker, year in sorted(oldest_year_by_ticker.items()):
    print(f"{ticker}: {year}")

# %%

#%% Ticker study (local history + metadata)
# Provide a ticker (Yahoo format, e.g., "AAPL") to inspect local history
# and compute a market cap using shares_outstanding from metadata when available.

META_PATH = Path("data/metadata/sp500_constituents.parquet")


def _read_local_history(ticker: str, base_dir: Path = DATA_ROOT) -> pd.DataFrame:
    """Read a per-ticker local Parquet/CSV and return columns: date, open, high, low, close, volume.

    Returns empty DataFrame if the file is missing or unreadable.
    """
    ticker = ticker.strip().upper()
    fp_parquet = base_dir / f"{ticker}.parquet"
    fp_csv = base_dir / f"{ticker}.csv"
    df = pd.DataFrame()
    try:
        if fp_parquet.exists():
            df = pd.read_parquet(fp_parquet)
        elif fp_csv.exists():
            df = pd.read_csv(fp_csv, parse_dates=["date"])  # type: ignore[arg-type]
        else:
            return pd.DataFrame()
    except Exception:
        return pd.DataFrame()

    if df is None or df.empty:
        return pd.DataFrame()

    # Normalize expected columns if present
    cols = {c.lower(): c for c in df.columns}
    expected = ["date", "open", "high", "low", "close", "volume"]
    missing = [c for c in expected if c not in cols]
    # Some datasets might have 'adj_close' — keep close priority if present
    if missing:
        # If critical columns are missing, just return as-is for caller to inspect
        return df

    out = df[[cols[c] for c in expected]].copy()
    out.columns = expected
    out["date"] = pd.to_datetime(out["date"], errors="coerce")
    return out.dropna(subset=["date"]).sort_values("date")


def _load_meta_row(ticker: str, meta_path: Path = META_PATH) -> pd.Series | None:
    """Load the metadata row for the given ticker from the local parquet/csv.

    Prefers column "symbol_yahoo"; falls back to "symbol"/"symbol_wiki" if needed.
    Returns a Series or None if not found.
    """
    if not meta_path.exists():
        return None
    try:
        if meta_path.suffix.lower() == ".parquet":
            meta = pd.read_parquet(meta_path)
        else:
            meta = pd.read_csv(meta_path)
    except Exception:
        return None
    if meta is None or meta.empty:
        return None

    lower_map = {c.lower(): c for c in meta.columns}
    sym_col = lower_map.get("symbol_yahoo") or lower_map.get("symbol") or lower_map.get("symbol_wiki")
    if not sym_col:
        return None
    t = ticker.strip().upper()
    rowset = meta.loc[meta[sym_col].astype(str).str.upper() == t]
    if rowset.empty:
        return None
    # If multiple, take the first
    return rowset.iloc[0]


def _compute_local_market_cap(meta_row: pd.Series | None, hist: pd.DataFrame) -> float | None:
    """Compute market cap using shares_outstanding from metadata and last close in history.

    Returns None if inputs are insufficient.
    """
    if meta_row is None or hist is None or hist.empty:
        return None
    # Try to locate shares_outstanding (tolerate case variants)
    try:
        shares = None
        for key in ("shares_outstanding", "sharesOutstanding", "shares", "impliedSharesOutstanding"):
            if key in meta_row.index:
                val = meta_row[key]
                if val is not None and not pd.isna(val):
                    shares = float(val)
                    break
        if shares is None:
            # Also check case-insensitive index
            idx_lower = {str(k).lower(): k for k in meta_row.index}
            k = idx_lower.get("shares_outstanding") or idx_lower.get("sharesoutstanding")
            if k is not None:
                val = meta_row[k]
                if val is not None and not pd.isna(val):
                    shares = float(val)
        if shares is None:
            return None

        close_col = "close" if "close" in hist.columns else ("adj_close" if "adj_close" in hist.columns else None)
        if close_col is None:
            return None
        last_close = pd.to_numeric(hist[close_col], errors="coerce").dropna()
        if last_close.empty:
            return None
        price = float(last_close.iloc[-1])
        return price * shares
    except Exception:
        return None


# Example usage: set a ticker here to inspect
TICKER: str | None = "GNRC"  # e.g., "AAPL" or "MSFT"; set to None to skip

if TICKER:
    tkr = TICKER.strip().upper()
    print(f"\n=== Ticker study: {tkr} ===")
    hist = _read_local_history(tkr)
    if hist is None or hist.empty:
        print("[warn] No local history found.")
    else:
        first_dt = pd.to_datetime(hist["date"]).min()
        last_dt = pd.to_datetime(hist["date"]).max()
        print(f"History rows: {len(hist):,}  range: {first_dt.date()} -> {last_dt.date()}")

    meta_row = _load_meta_row(tkr)
    if meta_row is None:
        print("[warn] No metadata row found in data/metadata/sp500_constituents.parquet")
    else:
        # Print a small subset of metadata if available
        def _get(row: pd.Series, name: str) -> str:
            return str(row.get(name)) if name in row.index else "n/a"

        # Tolerate case variants for common fields
        idx_lower = {str(k).lower(): k for k in meta_row.index}
        def _get_ci(row: pd.Series, name: str) -> str:
            k = idx_lower.get(name.lower())
            return str(row[k]) if k is not None else "n/a"

        company = _get_ci(meta_row, "company")
        sector = _get_ci(meta_row, "sector")
        sub_industry = _get_ci(meta_row, "sub_industry")
        shares = _get_ci(meta_row, "shares_outstanding")
        print(f"Company: {company}\nSector: {sector}\nSub-industry: {sub_industry}\nShares outstanding: {shares}")

    cap = _compute_local_market_cap(meta_row, hist if isinstance(hist, pd.DataFrame) else pd.DataFrame())
    print("Market cap (local calc):", f"{cap:,.0f}" if cap is not None else "n/a")

# %%
