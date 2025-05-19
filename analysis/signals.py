def detect_candlestick_patterns(df):
    """Identify price action patterns"""
    # Engulfing Pattern
    df['bullish_engulfing'] = (
        (df['close'] > df['open']) & 
        (df['close'].shift(1) < df['open'].shift(1)) & 
        (df['close'] > df['open'].shift(1)) & 
        (df['open'] < df['close'].shift(1))
    )
    
    # Pinbar Pattern
    body_size = abs(df['close'] - df['open'])
    total_range = df['high'] - df['low']
    df['bullish_pinbar'] = (
        (body_size / total_range < 0.3) & 
        (df['close'] > (df['high'] + df['low']) / 2))
    
    return df

def detect_trend(df, lookback=3):
    """Determine market trend"""
    df['higher_high'] = df['high'] > df['high'].rolling(lookback).max()
    df['higher_low'] = df['low'] > df['low'].rolling(lookback).max()
    df['uptrend'] = df['higher_high'] & df['higher_low']
    return df