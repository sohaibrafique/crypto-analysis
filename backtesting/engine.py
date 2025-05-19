"""
Backtesting engine with clear trade logging
"""
from dataclasses import dataclass
from typing import List
import pandas as pd

@dataclass
class Trade:
    entry_time: pd.Timestamp
    exit_time: pd.Timestamp
    entry_price: float
    exit_price: float
    pnl: float
    direction: str  # "long" or "short"

class BacktestEngine:
    def __init__(self, initial_balance=10000):
        self.initial_balance = initial_balance
        self.trades: List[Trade] = []
        
    def run(self, df, strategy):
        """Execute backtest with clear phases"""
        self._validate_input(df)
        signals = self._generate_signals(df, strategy)
        self._simulate_trades(df, signals)
        return self._generate_report()

    def _generate_signals(self, df, strategy):
        """Phase 1: Signal Generation"""
        return strategy.generate_signals(df)

    def _simulate_trades(self, df, signals):
        """Phase 2: Trade Simulation"""
        position = None
        for i, row in signals.iterrows():
            if row.signal == 1 and not position:  # Buy signal
                position = Trade(
                    entry_time=row.name,
                    entry_price=row.close,
                    direction="long"
                )
            elif row.signal == -1 and position:  # Sell signal
                position.exit_time = row.name
                position.exit_price = row.close
                position.pnl = self._calculate_pnl(position)
                self.trades.append(position)
                position = None

    def _generate_report(self):
        """Phase 3: Performance Reporting"""
        report = {
            "total_trades": len(self.trades),
            "win_rate": None,
            "max_drawdown": None,
            "sharpe_ratio": None
        }
        # Add calculation logic here
        return report