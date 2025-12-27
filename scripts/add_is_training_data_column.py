#!/usr/bin/env python3
"""
Add is_training_data column to bronze_ohlcv_intraday table.
This column is used by build_accum_distrib_events.py to filter training data.
"""

import logging
from sqlalchemy import create_engine, text

from src.config import settings

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


def main():
    """Add is_training_data column to bronze_ohlcv_intraday table."""
    logger.info("Adding is_training_data column to bronze_ohlcv_intraday table...")
    
    engine = create_engine(settings.database_url)
    
    with engine.connect() as conn:
        # Check if column already exists
        result = conn.execute(text("""
            SELECT column_name 
            FROM information_schema.columns 
            WHERE table_name = 'bronze_ohlcv_intraday' 
            AND column_name = 'is_training_data'
        """))
        
        if result.fetchone():
            logger.info("✅ Column 'is_training_data' already exists")
            return
        
        # Add the column
        logger.info("Adding column 'is_training_data' with default value TRUE...")
        conn.execute(text("""
            ALTER TABLE bronze_ohlcv_intraday 
            ADD COLUMN is_training_data BOOLEAN DEFAULT TRUE NOT NULL
        """))
        
        logger.info("✅ Successfully added 'is_training_data' column")
        logger.info("   All existing rows have been set to TRUE (training data)")


if __name__ == "__main__":
    main()
