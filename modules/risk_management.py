def calculate_risk_parameters(df):
    current_price = df['close'].iloc[-1]
    atr = df['high'].combine(df['low'], max).rolling(14).mean()  # Simplified ATR
    
    return {
        'stop_loss': {
            'technical': df['Support'].iloc[-1],
            'volatility': current_price - 2*atr.iloc[-1],
            'psychological': round(current_price * 0.95, 2)
        },
        'position_size': {
            'conservative': min(0.02/(0.01*current_price), 0.1),  # 2% risk, 10% max
            'moderate': min(0.05/(0.015*current_price), 0.2),
            'aggressive': min(0.1/(0.02*current_price), 0.3)
        }
    }