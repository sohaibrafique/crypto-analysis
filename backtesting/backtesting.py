import pandas as pd
from analysis.signals import detect_trend, detect_candlestick_patterns

class Backtester:
    def __init__(self, initial_balance=10000):
        self.initial_balance = initial_balance
        
    def run(self, df):
        """Backtest a simple strategy"""
        df = detect_trend(df)
        df = detect_candlestick_patterns(df)
        
        df['signal'] = 0
        df['returns'] = 0.0
        
        # Strategy Rules
        df.loc[(df['uptrend']) & (df['bullish_engulfing']), 'signal'] = 1
        df.loc[(df['close'] > df['resistance']) & (df['bullish_pinbar']), 'signal'] = -1
        
        # Calculate Returns
        df['returns'] = df['close'].pct_change() * df['signal'].shift(1)
        df['equity'] = (1 + df['returns']).cumprod() * self.initial_balance
        
        return df[['close', 'signal', 'returns', 'equity']]