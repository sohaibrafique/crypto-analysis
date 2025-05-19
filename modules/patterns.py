import pandas as pd
import talib
import pandas as pd
import numpy as np

def detect_higher_highs(df, window=3, confirmation_bars=1):
    """
    Detect higher highs in price action
    :param df: DataFrame with OHLC data
    :param window: Lookback window for pattern detection
    :param confirmation_bars: Number of bars needed to confirm the pattern
    :return: Series with True/False for higher highs
    """
    higher_highs = pd.Series(False, index=df.index)
    
    for i in range(window, len(df)):
        current_high = df['high'].iloc[i]
        prev_highs = df['high'].iloc[i-window:i]
        
        # Check if current high is higher than previous highs
        if (current_high > prev_highs.max()) and \
           (df['high'].iloc[i-confirmation_bars:i].idxmax() == i):
            higher_highs.iloc[i] = True
            
    return higher_highs

def detect_lower_lows(df, window=3, confirmation_bars=1):
    """
    Detect lower lows in price action
    :param df: DataFrame with OHLC data
    :param window: Lookback window for pattern detection
    :param confirmation_bars: Number of bars needed to confirm the pattern
    :return: Series with True/False for lower lows
    """
    lower_lows = pd.Series(False, index=df.index)
    
    for i in range(window, len(df)):
        current_low = df['low'].iloc[i]
        prev_lows = df['low'].iloc[i-window:i]
        
        # Check if current low is lower than previous lows
        if (current_low < prev_lows.min()) and \
           (df['low'].iloc[i-confirmation_bars:i].idxmin() == i):
            lower_lows.iloc[i] = True
            
    return lower_lows

def detect_shark_pattern(df):
    """
    Detect Shark harmonic pattern using zigzag indicators
    Returns: Series with True/False for pattern presence
    """
    # Implement zigzag detection first
    df['zigzag'] = zigzag(df['high'], df['low'])
    
    # Pattern detection logic
    patterns = []
    for i in range(len(df)-5, len(df)):
        if is_shark(df, i):
            patterns.append(True)
        else:
            patterns.append(False)
    return pd.Series(patterns, index=df.index)

def detect_bat_pattern(df):
    """
    Detect Bat harmonic pattern
    Returns: Series with True/False for pattern presence
    """
    patterns = []
    for i in range(len(df)-5, len(df)):
        if is_bat(df, i):
            patterns.append(True)
        else:
            patterns.append(False)
    return pd.Series(patterns, index=df.index)

def zigzag(high, low, percent=5):
    """
    Basic zigzag implementation
    """
    peaks = []
    current_dir = None
    last_extreme = None
    
    for i in range(1, len(high)):
        # Detection logic
        pass
    return peaks

def is_shark(df, i):
    """
    Shark pattern detection at index i
    """
    # Implementation rules
    pass

def is_bat(df, i):
    """
    Bat pattern detection at index i
    """
    # Implementation rules
    pass

def advanced_pattern_detection(df):
    # Harmonic Patterns
    df['shark_pattern'] = detect_shark_pattern(df)
    df['bat_pattern'] = detect_bat_pattern(df)
    
    # Market Structure
    df['higher_highs'] = detect_higher_highs(df)
    df['lower_lows'] = detect_lower_lows(df)
    
    # Candlestick Patterns
    df['three_black_crows'] = talib.CDL3BLACKCROWS(df['open'], df['high'], df['low'], df['close'])
    df['evening_star'] = talib.CDLEVENINGSTAR(df['open'], df['high'], df['low'], df['close'])
    
    return df

def detect_higher_highs_with_trend(df, window=3, trend_confirmation=0.02):
    """
    More sophisticated version that requires:
    1. Higher highs pattern
    2. Price above EMA20 (uptrend confirmation)
    3. Minimum price increase threshold
    """
    df['higher_high'] = False
    
    # Calculate EMA for trend confirmation
    df['EMA20'] = df['close'].ewm(span=20, adjust=False).mean()
    
    for i in range(window, len(df)):
        # Basic higher high condition
        if df['high'].iloc[i] <= df['high'].iloc[i-1]:
            continue
            
        # Trend confirmation (price above EMA20)
        if df['close'].iloc[i] < df['EMA20'].iloc[i]:
            continue
            
        # Minimum price increase threshold (e.g., 2%)
        price_increase = (df['high'].iloc[i] - df['high'].iloc[i-1]) / df['high'].iloc[i-1]
        if price_increase < trend_confirmation:
            continue
            
        df['higher_high'].iloc[i] = True
        
    return df['higher_high']

def detect_lower_lows_with_trend(df, window=3, trend_confirmation=0.02):
    """
    Enhanced version that requires:
    1. Lower lows pattern
    2. Price below EMA20 (downtrend confirmation)
    3. Minimum price decrease threshold
    """
    df['lower_low'] = False
    df['EMA20'] = df['close'].ewm(span=20, adjust=False).mean()
    
    for i in range(window, len(df)):
        if df['low'].iloc[i] >= df['low'].iloc[i-1]:
            continue
            
        if df['close'].iloc[i] > df['EMA20'].iloc[i]:
            continue
            
        price_decrease = (df['low'].iloc[i-1] - df['low'].iloc[i]) / df['low'].iloc[i-1]
        if price_decrease < trend_confirmation:
            continue
            
        df['lower_low'].iloc[i] = True
        
    return df['lower_low']

def detect_patterns(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df['pattern'] = None
    
    for i in range(2, len(df)):
        if df['close'].iloc[i] > df['open'].iloc[i] and df['close'].iloc[i-1] < df['open'].iloc[i-1]:
            df.at[df.index[i], 'pattern'] = 'Bullish Engulfing'
    
    df = advanced_pattern_detection(df)

    # Add market structure patterns
    df['higher_highs'] = detect_higher_highs_with_trend(df)
    df['lower_lows'] = detect_lower_lows_with_trend(df)
    
    # Detect trend continuations
    df['uptrend_confirmed'] = df['higher_highs'] & ~df['lower_lows']
    df['downtrend_confirmed'] = df['lower_lows'] & ~df['higher_highs']

    return df