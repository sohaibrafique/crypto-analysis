"""
Central database management with automatic schema migrations
"""
import sqlite3
from pathlib import Path
from datetime import datetime

class DatabaseManager:
    def __init__(self, db_path="data/cache.db"):
        self.db_path = Path(db_path)
        self._ensure_directory_exists()
        self.conn = self._create_connection()
        self._initialize_schema()

    def _ensure_directory_exists(self):
        self.db_path.parent.mkdir(exist_ok=True)

    def _create_connection(self):
        return sqlite3.connect(str(self.db_path))

    def _initialize_schema(self):
        with self.conn:
            # OHLCV data storage
            self.conn.execute("""
            CREATE TABLE IF NOT EXISTS ohlcv (
                symbol TEXT,
                timeframe TEXT,
                timestamp TIMESTAMP,
                open REAL,
                high REAL,
                low REAL,
                close REAL,
                volume REAL,
                PRIMARY KEY (symbol, timeframe, timestamp)
            )""")
            
            # Backtest results
            self.conn.execute("""
            CREATE TABLE IF NOT EXISTS backtest_results (
                strategy_id TEXT,
                symbol TEXT,
                timeframe TEXT,
                sharpe_ratio REAL,
                max_drawdown REAL,
                win_rate REAL,
                timestamp TIMESTAMP DEFAULT CURRENT_TIMESTAMP
            )""")

    def save_ohlcv(self, symbol, timeframe, df):
        """Save OHLCV data with conflict resolution"""
        df = df.reset_index()
        df['symbol'] = symbol
        df['timeframe'] = timeframe
        df.to_sql('ohlcv', self.conn, if_exists='append', index=False)
        self.conn.commit()