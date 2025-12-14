#!/usr/bin/env python3
"""
Export all historical OHLCV data (bronze/silver/gold) to MinIO.
"""

import sys
from pathlib import Path

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from datetime import datetime, timedelta
from sqlalchemy import text
from src.utils.db import get_db_session
from src.utils.logger import setup_logger

logger = setup_logger(__name__)


def get_date_range():
    """Get min and max trade dates from bronze_ohlcv."""
    with get_db_session() as session:
        query = text("""
            SELECT 
                MIN(trade_date) as min_date,
                MAX(trade_date) as max_date,
                COUNT(DISTINCT trade_date) as total_dates,
                COUNT(DISTINCT ticker) as total_tickers
            FROM bronze_ohlcv
        """)
        result = session.execute(query).fetchone()
        return result


def get_all_dates():
    """Get all unique trade dates from bronze_ohlcv."""
    with get_db_session() as session:
        query = text("""
            SELECT DISTINCT trade_date 
            FROM bronze_ohlcv 
            ORDER BY trade_date
        """)
        result = session.execute(query).fetchall()
        return [row[0] for row in result]


def export_date(trade_date, ticker='AD.AS'):
    """Export OHLCV data for a specific date using the export script."""
    import subprocess
    
    date_str = trade_date.strftime('%Y-%m-%d')
    cmd = [
        'python',
        str(project_root / 'scripts' / 'export_parquet_simple.py'),
        '--date', date_str,
        '--ticker', ticker,
        '--layer', 'ohlcv'  # Only export OHLCV layers (bronze/silver/gold)
    ]
    
    logger.info(f"Exporting {date_str}...")
    result = subprocess.run(cmd, capture_output=True, text=True)
    
    if result.returncode != 0:
        logger.error(f"❌ Failed to export {date_str}: {result.stderr}")
        return False
    else:
        # Count records from output
        if "Total records:" in result.stdout:
            total = result.stdout.split("Total records:")[1].split()[0]
            logger.info(f"✅ {date_str}: {total} records")
        return True


def main():
    logger.info("="*60)
    logger.info("Starting historical OHLCV export to MinIO")
    logger.info("="*60)
    
    # Get date range
    stats = get_date_range()
    logger.info(f"\n📊 Database Statistics:")
    logger.info(f"  Date range: {stats[0]} to {stats[1]}")
    logger.info(f"  Total dates: {stats[2]}")
    logger.info(f"  Total tickers: {stats[3]}")
    
    # Get all dates
    dates = get_all_dates()
    logger.info(f"\n📅 Exporting {len(dates)} dates...")
    
    # Export each date
    success_count = 0
    failed_dates = []
    
    for i, trade_date in enumerate(dates, 1):
        logger.info(f"\n[{i}/{len(dates)}] Processing {trade_date}...")
        if export_date(trade_date):
            success_count += 1
        else:
            failed_dates.append(trade_date)
    
    # Summary
    logger.info("\n" + "="*60)
    logger.info("Export Summary")
    logger.info("="*60)
    logger.info(f"✅ Successful: {success_count}/{len(dates)}")
    if failed_dates:
        logger.info(f"❌ Failed: {len(failed_dates)}")
        for date in failed_dates:
            logger.info(f"  - {date}")
    logger.info("="*60)


if __name__ == '__main__':
    main()
