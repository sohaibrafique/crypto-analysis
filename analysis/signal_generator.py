# analysis/signal_generator.py
import pandas as pd  # <-- Add this at the top
from typing import Dict

class SignalGenerator:
    REQUIRED_COLUMNS = {'close', 'ema20', 'volume', 'volume_ma20'}

    def generate(self, df):
        """Generate signals with column validation"""
        required_cols = ['regime', 'close', 'ema20', 'volume']
        missing = [col for col in required_cols if col not in df.columns]
        if missing:
            raise ValueError(f"Signal generation requires: {missing}")

        """Generate signals with confidence scores"""
        signals = pd.DataFrame(index=df.index)
        
        # Bullish conditions
        signals['bull_strength'] = (
            (df['close'] > df['ema20']).astype(int) +
            (df['volume'] > df['volume_ma20']).astype(int) +
            (df['close'] > df['open']).astype(int)
        )
        
        # Bearish conditions
        signals['bear_strength'] = (
            (df['close'] < df['ema20']).astype(int) +
            (df['volume'] > df['volume_ma20']).astype(int) +
            (df['close'] < df['open']).astype(int))
        
        # Combined signal
        signals['signal'] = signals['bull_strength'] - signals['bear_strength']
        
        return signals

    def _validate_input(self, df):
        """Updated validation with NaN handling"""
        if not isinstance(df, pd.DataFrame):
            raise TypeError("Input must be a pandas DataFrame")
            
        missing = self.REQUIRED_COLUMNS - set(df.columns)
        if missing:
            raise KeyError(f"Missing columns: {missing}")

        # Instead of rejecting, clean minor NaNs
        nan_count = df[self.REQUIRED_COLUMNS].isnull().sum().sum()
        if nan_count > 0:
            print(f"Warning: {nan_count} NaN values found - filling with last valid")
            df[self.REQUIRED_COLUMNS] = df[self.REQUIRED_COLUMNS].ffill()
            
        if df[self.REQUIRED_COLUMNS].isnull().any().any():
            raise ValueError("Critical NaN values remain after cleaning")

