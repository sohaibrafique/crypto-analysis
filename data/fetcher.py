import ccxt
import pandas as pd
import numpy as np
from typing import Optional

# Singleton exchange instance
_exchange_instance = None

def get_exchange():
    """Initialize and return exchange instance with error handling"""
    global _exchange_instance
    if _exchange_instance is None:
        try:
            _exchange_instance = ccxt.binance({
                'enableRateLimit': True,
                'options': {
                    'defaultType': 'spot'  # or 'future' for derivatives
                }
            })
            # Verify exchange is loaded
            _exchange_instance.load_markets()
            print(f"Exchange initialized: {_exchange_instance.name}")
        except Exception as e:
            raise RuntimeError(f"Failed to initialize exchange: {str(e)}")
    return _exchange_instance

def fetch_ohlcv(
    symbol: str = 'BTC/USDT',
    timeframe: str = '4h',
    limit: int = 1000
) -> pd.DataFrame:
    """Fetch OHLCV data with robust error handling"""
    try:
        exchange = get_exchange()
        
        # Validate symbol exists
        if symbol not in exchange.markets:
            raise ValueError(f"Symbol {symbol} not available")
        
        # Fetch data
        data = exchange.fetch_ohlcv(symbol, timeframe, limit=limit)
        
        if not data:
            raise ValueError("Empty response from exchange")
            
        # Convert to DataFrame
        df = pd.DataFrame(
            data,
            columns=['timestamp', 'open', 'high', 'low', 'close', 'volume']
        )
        df['timestamp'] = pd.to_datetime(df['timestamp'], unit='ms')
        df.set_index('timestamp', inplace=True)
        
        # Basic validation
        if len(df) < 5:
            raise ValueError(f"Insufficient data points: {len(df)}")
        if df.isnull().values.any():
            df = df.ffill().bfill()
            
        return df.astype(float)
        
    except ccxt.NetworkError as e:
        raise ConnectionError(f"Network error: {str(e)}")
    except ccxt.ExchangeError as e:
        raise RuntimeError(f"Exchange error: {str(e)}")
    except Exception as e:
        raise RuntimeError(f"Data fetch failed: {str(e)}")