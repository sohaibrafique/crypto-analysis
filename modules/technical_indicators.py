import pandas as pd
import talib
import numpy as np

def ichimoku_cloud(df):
    """
    Calculate Ichimoku Cloud components
    Returns: DataFrame with new columns
    """
    # Conversion Line
    nine_period_high = df['high'].rolling(window=9).max()
    nine_period_low = df['low'].rolling(window=9).min()
    df['tenkan_sen'] = (nine_period_high + nine_period_low) / 2
    
    # Base Line
    twenty_six_period_high = df['high'].rolling(window=26).max()
    twenty_six_period_low = df['low'].rolling(window=26).min()
    df['kijun_sen'] = (twenty_six_period_high + twenty_six_period_low) / 2
    
    # Leading Span A
    df['senkou_span_a'] = ((df['tenkan_sen'] + df['kijun_sen']) / 2).shift(26)
    
    # Leading Span B
    fifty_two_period_high = df['high'].rolling(window=52).max()
    fifty_two_period_low = df['low'].rolling(window=52).min()
    df['senkou_span_b'] = ((fifty_two_period_high + fifty_two_period_low) / 2).shift(26)
    
    return df

def vwap_calculation(df):
    """
    Calculate Volume Weighted Average Price
    """
    df['typical_price'] = (df['high'] + df['low'] + df['close']) / 3
    df['cumulative_volume'] = df['volume'].cumsum()
    df['cumulative_typical'] = (df['typical_price'] * df['volume']).cumsum()
    return df

def advanced_indicators(df):
    # Volatility Measures
    df['ATR'] = talib.ATR(df['high'], df['low'], df['close'], timeperiod=14)
    df['VIX'] = talib.NATR(df['high'], df['low'], df['close'], timeperiod=14)
    
    # Momentum Indicators
    df['RSI'] = talib.RSI(df['close'], timeperiod=14)
    df['MACD'], df['MACD_signal'], _ = talib.MACD(df['close'])
    
    # Market Structure
    df = ichimoku_cloud(df)
    df['ichimoku'] = df.apply(lambda row: 'bullish' if row['tenkan_sen'] > row['kijun_sen'] else 'bearish', axis=1)
    
    # Volume-based Indicators
    df = vwap_calculation(df)
    df['VWAP'] = df['cumulative_typical'] / df['cumulative_volume']
    df['OBV'] = talib.OBV(df['close'], df['volume'])
    
    return df

def compute_indicators(df):
    """Calculate technical indicators with validation"""
    if not isinstance(df, pd.DataFrame):
        raise ValueError("Input must be a pandas DataFrame")
    
    required_cols = ['open', 'high', 'low', 'close', 'volume']
    missing = [col for col in required_cols if col not in df.columns]
    if missing:
        raise ValueError(f"Missing required columns: {missing}")
    
    try:
        # Calculate indicators
        df['EMA20'] = talib.EMA(df['close'], timeperiod=20)
        df['EMA50'] = talib.EMA(df['close'], timeperiod=50)
        df['volume_ma20'] = df['volume'].rolling(20).mean()

        df = advanced_indicators(df)
        
        return df.dropna()
        
    except Exception as e:
        raise RuntimeError(f"Indicator calculation failed: {str(e)}")
    
def calculate_indicators(df):
    # EMAs with trend confirmation
    df["EMA20"] = df["close"].ewm(span=20, adjust=False).mean()
    df["EMA50"] = df["close"].ewm(span=50, adjust=False).mean()
    df["EMA200"] = df["close"].ewm(span=200, adjust=False).mean()  # Added long-term trend indicator
    
    # Standard RSI calculation using EMA
    delta = df["close"].diff()
    gain = delta.where(delta > 0, 0).ewm(alpha=1/14, adjust=False).mean()
    loss = -delta.where(delta < 0, 0).ewm(alpha=1/14, adjust=False).mean()
    rs = gain / loss
    df["RSI"] = 100 - (100 / (1 + rs))
    
    # MACD with improved signal line
    exp12 = df["close"].ewm(span=12, adjust=False).mean()
    exp26 = df["close"].ewm(span=26, adjust=False).mean()
    df["MACD"] = exp12 - exp26
    df["Signal"] = df["MACD"].ewm(span=9, adjust=False).mean()
    df["MACD_Hist"] = df["MACD"] - df["Signal"]
    
    # Dynamic Support/Resistance using recent swing points
    df['Support'] = df['low'].rolling(50, center=True).min()
    df['Resistance'] = df['high'].rolling(50, center=True).max()
    
    return df

def detect_divergence(df, lookback=5):
    bullish_div = bearish_div = False
    price_highs = df['high'].rolling(lookback).max()
    price_lows = df['low'].rolling(lookback).min()
    macd_highs = df['MACD'].rolling(lookback).max()
    macd_lows = df['MACD'].rolling(lookback).min()

    # Bearish divergence
    if (price_highs.iloc[-1] > price_highs.iloc[-2] and 
        macd_highs.iloc[-1] < macd_highs.iloc[-2]):
        bearish_div = True
        
    # Bullish divergence
    if (price_lows.iloc[-1] < price_lows.iloc[-2] and 
        macd_lows.iloc[-1] > macd_lows.iloc[-2]):
        bullish_div = True
        
    return bullish_div, bearish_div
