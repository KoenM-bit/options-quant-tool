#!/usr/bin/env python3
"""
Add trade_date column to FD bronze tables.
This brings FD tables in line with BD tables for consistent date handling.
"""
import logging
from datetime import datetime, timedelta
from sqlalchemy import text
from src.utils.db import get_db_session

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def add_trade_date_columns():
    """Add trade_date columns to bronze_fd_options and bronze_fd_overview."""
    
    with get_db_session() as session:
        logger.info("Adding trade_date column to bronze_fd_options...")
        
        # Add column to bronze_fd_options
        try:
            session.execute(text("""
                ALTER TABLE bronze_fd_options
                ADD COLUMN IF NOT EXISTS trade_date DATE
            """))
            session.commit()
            logger.info("✅ Added trade_date column to bronze_fd_options")
        except Exception as e:
            logger.error(f"Failed to add column to bronze_fd_options: {e}")
            session.rollback()
            raise
        
        # Add column to bronze_fd_overview
        logger.info("Adding trade_date column to bronze_fd_overview...")
        try:
            session.execute(text("""
                ALTER TABLE bronze_fd_overview
                ADD COLUMN IF NOT EXISTS trade_date DATE
            """))
            session.commit()
            logger.info("✅ Added trade_date column to bronze_fd_overview")
        except Exception as e:
            logger.error(f"Failed to add column to bronze_fd_overview: {e}")
            session.rollback()
            raise
        
        # Backfill bronze_fd_options using peildatum from bronze_fd_overview
        logger.info("Backfilling trade_date in bronze_fd_options from bronze_fd_overview.peildatum...")
        try:
            # First try to match with overview data by ticker and similar scrape time
            result = session.execute(text("""
                UPDATE bronze_fd_options opt
                SET trade_date = ov.peildatum
                FROM bronze_fd_overview ov
                WHERE opt.ticker = ov.ticker
                  AND opt.trade_date IS NULL
                  AND ov.peildatum IS NOT NULL
                  AND DATE(opt.scraped_at) = DATE(ov.scraped_at)
            """))
            matched_rows = result.rowcount
            session.commit()
            logger.info(f"✅ Backfilled {matched_rows} rows in bronze_fd_options from overview peildatum")
            
            # For any remaining rows without a match, use derived date logic as fallback
            result2 = session.execute(text("""
                UPDATE bronze_fd_options
                SET trade_date = (
                    CASE 
                        WHEN EXTRACT(DOW FROM scraped_at) = 6 THEN DATE(scraped_at) - INTERVAL '1 day'
                        WHEN EXTRACT(DOW FROM scraped_at) = 0 THEN DATE(scraped_at) - INTERVAL '2 days'
                        ELSE DATE(scraped_at)
                    END
                )::date
                WHERE trade_date IS NULL
            """))
            fallback_rows = result2.rowcount
            session.commit()
            logger.info(f"✅ Backfilled {fallback_rows} remaining rows using derived date logic")
        except Exception as e:
            logger.error(f"Failed to backfill bronze_fd_options: {e}")
            session.rollback()
            raise
        
        # Backfill bronze_fd_overview using peildatum (direct match)
        logger.info("Backfilling trade_date in bronze_fd_overview...")
        try:
            result = session.execute(text("""
                UPDATE bronze_fd_overview
                SET trade_date = peildatum
                WHERE trade_date IS NULL AND peildatum IS NOT NULL
            """))
            session.commit()
            logger.info(f"✅ Backfilled {result.rowcount} rows in bronze_fd_overview")
        except Exception as e:
            logger.error(f"Failed to backfill bronze_fd_overview: {e}")
            session.rollback()
            raise
        
        # Make columns NOT NULL and add indexes
        logger.info("Setting NOT NULL constraints and adding indexes...")
        try:
            session.execute(text("""
                ALTER TABLE bronze_fd_options
                ALTER COLUMN trade_date SET NOT NULL
            """))
            
            session.execute(text("""
                CREATE INDEX IF NOT EXISTS idx_bronze_fd_options_trade_date
                ON bronze_fd_options(trade_date)
            """))
            
            session.execute(text("""
                ALTER TABLE bronze_fd_overview
                ALTER COLUMN trade_date SET NOT NULL
            """))
            
            session.execute(text("""
                CREATE INDEX IF NOT EXISTS idx_bronze_fd_overview_trade_date
                ON bronze_fd_overview(trade_date)
            """))
            
            session.commit()
            logger.info("✅ Added NOT NULL constraints and indexes")
        except Exception as e:
            logger.error(f"Failed to add constraints: {e}")
            session.rollback()
            raise
        
        # Verify results
        logger.info("\n📊 Verification:")
        
        fd_options_count = session.execute(text("""
            SELECT COUNT(*), MIN(trade_date), MAX(trade_date)
            FROM bronze_fd_options
        """)).fetchone()
        logger.info(f"bronze_fd_options: {fd_options_count[0]} rows, dates {fd_options_count[1]} to {fd_options_count[2]}")
        
        fd_overview_count = session.execute(text("""
            SELECT COUNT(*), MIN(trade_date), MAX(trade_date)
            FROM bronze_fd_overview
        """)).fetchone()
        logger.info(f"bronze_fd_overview: {fd_overview_count[0]} rows, dates {fd_overview_count[1]} to {fd_overview_count[2]}")
        
        logger.info("\n✅ Migration complete!")


if __name__ == "__main__":
    add_trade_date_columns()
