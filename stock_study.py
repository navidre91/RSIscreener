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
