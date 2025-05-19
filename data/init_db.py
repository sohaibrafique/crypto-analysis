import sqlite3
import os
from pathlib import Path

def init_database():
    # Get absolute path to the data directory
    current_dir = Path(__file__).parent
    db_path = current_dir / 'cache.db'
    
    conn = sqlite3.connect(str(db_path))
    conn.execute("""
    CREATE TABLE IF NOT EXISTS metadata (
        pair TEXT,
        timeframe TEXT,
        last_updated TIMESTAMP,
        PRIMARY KEY (pair, timeframe)
    )""")
    conn.commit()
    conn.close()
    print(f"Database initialized at: {db_path}")

if __name__ == "__main__":
    init_database()