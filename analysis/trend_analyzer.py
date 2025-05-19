import pandas as pd
import numpy as np

class TrendAnalyzer:
    def __init__(self, ema_period=20, volatility_window=14):
        self.ema_period = ema_period
        self.volatility_window = volatility_window

    def detect_market_regime(self, df):
        """Enhanced market regime detection with volatility analysis"""
        df = df.copy()
        
        # Ensure required columns exist
        if 'close' not in df.columns:
            raise ValueError("DataFrame missing 'close' column")
        
        # Calculate volatility
        df['returns'] = df['close'].pct_change()
        df['volatility'] = df['returns'].rolling(self.volatility_window).std()
        
        # Determine trend direction
        df['ema'] = df['close'].ewm(span=self.ema_period, adjust=False).mean()
        df['trend'] = np.where(df['close'] > df['ema'], 'up', 'down')
        
        # Determine market regime
        conditions = [
            (df['volatility'] < 0.01) & (df['trend'] == 'up'),
            (df['volatility'] < 0.01) & (df['trend'] == 'down'),
            (df['volatility'] >= 0.01)
        ]
        choices = ['bullish-weak', 'bearish-weak', 'volatile']
        df['regime'] = np.select(conditions, choices, default='neutral')
        
        # Clean up intermediate columns
        df.drop(['returns', 'volatility', 'trend'], axis=1, inplace=True)
        
        return df