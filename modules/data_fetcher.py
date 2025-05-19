import pandas as pd
import ccxt
import time
from datetime import datetime, timezone, timedelta
from typing import Tuple

def fetch_data_with_retry(
    symbol: str,
    interval: str,
    start_date: datetime,
    end_date: datetime,
    max_retries: int = 3
) -> pd.DataFrame:
    exchange = ccxt.binance({
        'enableRateLimit': True,
        'options': {'adjustForTimeDifference': True}
    })

    # Ensure we never fetch incomplete (future) candles
    now = datetime.now(timezone.utc)
    if interval.endswith("m"):
        delta = timedelta(minutes=int(interval[:-1]))
    elif interval.endswith("h"):
        delta = timedelta(hours=int(interval[:-1]))
    elif interval.endswith("d"):
        delta = timedelta(days=int(interval[:-1]))
    else:
        raise ValueError(f"Unsupported interval: {interval}")

    end_date = min(end_date, now - delta)

    start_ms = int(start_date.timestamp() * 1000)
    end_ms = int(end_date.timestamp() * 1000)
    all_data = []

    while start_ms < end_ms:
        for attempt in range(max_retries):
            try:
                print(f"[Attempt {attempt+1}] Fetching {symbol} | Interval: {interval} | Since: {start_ms} ({datetime.fromtimestamp(start_ms / 1000, tz=timezone.utc)} UTC)")

                ohlcv = exchange.fetch_ohlcv(symbol, interval, since=start_ms, limit=1000)
                if not ohlcv:
                    raise ValueError("No data returned from exchange")

                all_data.extend(ohlcv)

                last_timestamp = ohlcv[-1][0]
                if last_timestamp == start_ms:
                    print("Same timestamp received repeatedly. Avoiding infinite loop.")
                    start_ms += 60_000  # skip 1 minute to break loop
                else:
                    start_ms = last_timestamp + 1
                break

            except ccxt.RateLimitExceeded:
                print("Rate limit hit, sleeping...")
                time.sleep((attempt + 1) * 2)
            except Exception as e:
                if attempt == max_retries - 1:
                    raise RuntimeError(f"Failed after {max_retries} attempts: {str(e)}")
                print(f"Retrying after error: {e}")
                time.sleep(1)

    df = pd.DataFrame(
        all_data,
        columns=['timestamp', 'open', 'high', 'low', 'close', 'volume']
    )
    df['timestamp'] = pd.to_datetime(df['timestamp'], unit='ms', utc=True)
    return df
