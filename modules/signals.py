import pandas as pd

def generate_combined_strategy_signals(df):
    signals = []
    for i in range(1, len(df)):
        ema_trend = df['EMA20'].iloc[i] > df['EMA50'].iloc[i]
        bullish_candle = df['close'].iloc[i] > df['open'].iloc[i]
        increasing_volume = df['volume'].iloc[i] > df['volume'].iloc[i - 1]

        if ema_trend and bullish_candle and increasing_volume:
            signals.append({'timestamp': df.index[i], 'signal': 'Buy'})
        elif not ema_trend and not bullish_candle and not increasing_volume:
            signals.append({'timestamp': df.index[i], 'signal': 'Sell'})

    return pd.DataFrame(signals)  # timestamp stays a column