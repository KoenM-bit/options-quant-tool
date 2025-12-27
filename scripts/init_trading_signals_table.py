"""
Initialize database table for trading signals.

Creates the `trading_signals` table to store generated signals
from the hourly pipeline for paper trading execution.
"""

import sys
from pathlib import Path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from sqlalchemy import create_engine, text
from src.config import settings

def create_trading_signals_table():
    """Create trading_signals table if it doesn't exist."""
    
    engine = create_engine(settings.database_url)
    
    create_table_query = """
    CREATE TABLE IF NOT EXISTS trading_signals (
        id SERIAL PRIMARY KEY,
        ticker VARCHAR(20) NOT NULL,
        market VARCHAR(10) NOT NULL,
        t_end TIMESTAMP NOT NULL,
        close_end FLOAT NOT NULL,
        atr_pct_last FLOAT NOT NULL,
        proba_accum FLOAT NOT NULL,
        strategy_version VARCHAR(10) NOT NULL,
        generated_at TIMESTAMP NOT NULL,
        status VARCHAR(20) NOT NULL DEFAULT 'pending',
        executed_at TIMESTAMP,
        entry_price FLOAT,
        stop_price FLOAT,
        target_price FLOAT,
        exit_price FLOAT,
        exit_reason VARCHAR(20),
        pnl_eur FLOAT,
        notes TEXT,
        UNIQUE(ticker, t_end, strategy_version)
    );
    
    CREATE INDEX IF NOT EXISTS idx_signals_status 
        ON trading_signals(status, generated_at);
    
    CREATE INDEX IF NOT EXISTS idx_signals_ticker 
        ON trading_signals(ticker, generated_at DESC);
    
    CREATE INDEX IF NOT EXISTS idx_signals_generated 
        ON trading_signals(generated_at DESC);
    """
    
    with engine.begin() as conn:
        conn.execute(text(create_table_query))
    
    print("✅ trading_signals table created")


if __name__ == "__main__":
    create_trading_signals_table()
