import pandas as pd
from modules.signals import generate_combined_strategy_signals

def backtest_strategy(df, initial_balance=10000):
    signals = generate_combined_strategy_signals(df)
    portfolio = {
        'balance': initial_balance,
        'position': 0,
        'trades': []
    }
    
    for i, row in signals.iterrows():
        if row['signal'] == 'BUY' and portfolio['balance'] > 0:
            portfolio['position'] = portfolio['balance'] / row['close']
            portfolio['balance'] = 0
            portfolio['trades'].append({
                'type': 'BUY',
                'price': row['close'],
                'timestamp': row['timestamp']
            })
        elif row['signal'] == 'SELL' and portfolio['position'] > 0:
            portfolio['balance'] = portfolio['position'] * row['close']
            portfolio['position'] = 0
            portfolio['trades'].append({
                'type': 'SELL',
                'price': row['close'],
                'timestamp': row['timestamp']
            })
    
    return {
        'final_balance': portfolio['balance'],
        'return_pct': (portfolio['balance'] / initial_balance - 1) * 100,
        'trades': portfolio['trades']
    }