import pandas as pd
import numpy as np
from typing import Dict

def calculate_technical_indicators(df):
    """Robust indicator calculation with TA-Lib fallback"""
    df = df.copy()
    
    # Validate input
    if len(df) < 50:
        df = df.copy()  # Ensure we don't modify original
        padding = 50 - len(df)
        df = pd.concat([df.iloc[0:1].copy().assign(**{c:np.nan for c in df.columns}).iloc[-padding:], df])
        print(f"Warning: Extended dataframe from {len(df)-padding} to 50 rows")
    
    try:
        # Try TA-Lib first
        try:
            import talib
            df['ema20'] = talib.EMA(df['close'], timeperiod=20)
            df['ema50'] = talib.EMA(df['close'], timeperiod=50)
        except:
            print("TA-Lib failed, using pandas fallback")
            df['ema20'] = df['close'].ewm(span=20, adjust=False).mean()
            df['ema50'] = df['close'].ewm(span=50, adjust=False).mean()
        
        # Support/Resistance
        df['support'] = df['low'].rolling(50, min_periods=1).min()
        df['resistance'] = df['high'].rolling(50, min_periods=1).max()
        
        # Volume
        df['volume_ma20'] = df['volume'].rolling(20, min_periods=1).mean()
        
        # Clean any remaining NaNs
        df = df.ffill().bfill()
        
        # Final validation
        required = ['ema20', 'ema50', 'support', 'resistance', 'volume_ma20']
        if not all(col in df.columns for col in required):
            missing = [col for col in required if col not in df.columns]
            raise ValueError(f"Missing columns after calculation: {missing}")
            
        return df
    
    except Exception as e:
        raise RuntimeError(f"Indicator calculation failed: {str(e)}\nColumns: {df.columns.tolist()}")