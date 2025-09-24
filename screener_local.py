#!/usr/bin/env python3
"""Local-only RSI screener that prints matches to stdout."""

from datetime import datetime, timezone

from screener import (
    ALPACA_KEY,
    ALPACA_SECRET,
    ALERT_MODE,
    NY,
    get_sp500_symbols,
    is_regular_hours_now,
    scan,
)


def ensure_alpaca_credentials() -> None:
    missing = []
    if not ALPACA_KEY:
        missing.append("ALPACA_API_KEY_ID")
    if not ALPACA_SECRET:
        missing.append("ALPACA_API_SECRET_KEY")
    if missing:
        raise SystemExit("Missing required env vars: " + ", ".join(missing))


def main() -> None:
    ensure_alpaca_credentials()

    if not is_regular_hours_now():
        print("Market not in regular hours window; exiting.")
        return

    symbols = get_sp500_symbols()
    matches = scan(symbols)

    now_ny = datetime.now(timezone.utc).astimezone(NY).strftime("%Y-%m-%d %H:%M")
    mode_desc = "crossed up > 30" if ALERT_MODE == "cross" else "RSI > 30"
    header = f"5m RSI scan ({mode_desc}) - {now_ny} ET\nMatches: {len(matches)}"
    print(header)

    if not matches:
        print("No new signals this interval.")
        return

    for symbol, rsi_value, last_price in matches:
        print(f"{symbol:<6}  RSI={rsi_value:5.1f}  Px={last_price:.2f}")


if __name__ == "__main__":
    main()
