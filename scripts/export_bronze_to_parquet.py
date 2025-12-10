"""
Export Bronze Layer to Partitioned Parquet Files in MinIO
===========================================================
Exports bronze_fd_options and bronze_fd_overview to MinIO with proper partitioning:

Structure:
- bronze/options_chain/v1/date=YYYY-MM-DD/ticker=AD.AS/as_of=YYYY-MM-DDTHH-MM-SSZ/data.parquet
- bronze/options_overview/v1/date=YYYY-MM-DD/ticker=AD.AS/as_of=YYYY-MM-DDTHH-MM-SSZ/data.parquet

Benefits:
- Proper data lake structure with versioning (v1)
- Partitioned by date for efficient time-series queries
- Partitioned by ticker for multi-asset support
- as_of timestamp for data lineage and reprocessing
- Columnar parquet format for fast analytics
"""

import sys
import os
from pathlib import Path
from datetime import datetime
from typing import List, Dict, Tuple
import logging

# Support both Docker and local execution
if os.path.exists('/opt/airflow'):
    sys.path.insert(0, '/opt/airflow/dags/..')
else:
    # Running locally - add project root to path
    project_root = Path(__file__).parent.parent
    sys.path.insert(0, str(project_root))

import pandas as pd
from sqlalchemy import text

from src.utils.db import get_db_session
from src.utils.minio_client import get_minio_client
from src.config import settings

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Temp directory for parquet files before upload
TEMP_DIR = Path("/tmp/bronze_parquet_export")
TEMP_DIR.mkdir(parents=True, exist_ok=True)


def get_distinct_trade_dates() -> List[Tuple[str, str, datetime]]:
    """
    Get distinct trade dates (peildatum) and the LATEST scrape timestamp for each.
    This ensures we export only the most recent data for each trading day.
    
    Returns:
        List of (trade_date, ticker, latest_scraped_at) tuples
    """
    with get_db_session() as session:
        query = text("""
            SELECT 
                peildatum as trade_date,
                ticker,
                MAX(scraped_at) as latest_scraped_at
            FROM bronze_fd_overview
            GROUP BY peildatum, ticker
            ORDER BY trade_date, ticker
        """)
        
        result = session.execute(query)
        rows = result.fetchall()
        
    logger.info(f"Found {len(rows)} distinct trade dates with latest scrapes")
    return rows


def export_options_chain_partition(
    trade_date: str,
    ticker: str,
    latest_scraped_at: datetime
) -> str:
    """
    Export ALL options chain data for a specific trade date (using latest scrape).
    Gets all options that were scraped on the same date as the latest overview scrape.
    
    Structure: bronze/options_chain/v1/date=YYYY-MM-DD/ticker=TICKER/as_of=YYYY-MM-DDTHH-MM-SSZ/data.parquet
    
    Returns:
        S3 path where file was uploaded
    """
    logger.info(f"Exporting options chain for {ticker} on {trade_date} (latest scrape: {latest_scraped_at})")
    
    # Query ALL options data from the same scrape day as the latest overview
    with get_db_session() as session:
        query = text("""
            SELECT 
                ticker,
                symbol_code,
                scraped_at,
                source_url,
                option_type,
                expiry_date,
                strike,
                naam,
                isin,
                laatste as last_price,
                bid,
                ask,
                volume,
                open_interest,
                underlying_price,
                created_at,
                updated_at
            FROM bronze_fd_options
            WHERE ticker = :ticker
              AND DATE(scraped_at) = DATE(:latest_scraped_at)
            ORDER BY expiry_date, strike, option_type
        """)
        
        df = pd.read_sql(query, session.connection(), params={
            'ticker': ticker,
            'latest_scraped_at': latest_scraped_at
        })
    
    if df.empty:
        logger.warning(f"No options data found for {ticker} on scrape date {latest_scraped_at.date()}")
        return None
    
    # Format as_of timestamp for path (ISO 8601 format)
    as_of_str = latest_scraped_at.strftime('%Y-%m-%dT%H-%M-%SZ')
    
    # Build S3 path
    s3_path = f"bronze/options_chain/v1/date={trade_date}/ticker={ticker}/as_of={as_of_str}/data.parquet"
    
    # Save to temp file
    temp_file = TEMP_DIR / f"options_chain_{trade_date}_{ticker}_{as_of_str}.parquet"
    df.to_parquet(temp_file, index=False, compression='snappy')
    
    # Upload to MinIO
    minio_client = get_minio_client()
    minio_client.upload_file(
        local_path=str(temp_file),
        object_name=s3_path
    )
    
    # Cleanup
    temp_file.unlink()
    
    logger.info(f"✅ Uploaded {len(df)} options contracts to {s3_path}")
    return s3_path


def export_options_overview_partition(
    trade_date: str,
    ticker: str,
    latest_scraped_at: datetime
) -> str:
    """
    Export options overview for a specific trade date (using latest scrape only).
    This deduplicates accumulated data by taking only the most recent scrape.
    
    Structure: bronze/options_overview/v1/date=YYYY-MM-DD/ticker=TICKER/as_of=YYYY-MM-DDTHH-MM-SSZ/data.parquet
    
    Returns:
        S3 path where file was uploaded
    """
    logger.info(f"Exporting options overview for {ticker} on {trade_date} (latest scrape: {latest_scraped_at})")
    
    # Query ONLY the latest scrape data for this trade date
    with get_db_session() as session:
        query = text("""
            SELECT 
                ticker,
                symbol_code,
                onderliggende_waarde,
                koers,
                vorige,
                delta,
                delta_pct,
                hoog,
                laag,
                volume_underlying,
                tijd,
                peildatum,
                totaal_volume,
                totaal_volume_calls,
                totaal_volume_puts,
                totaal_oi,
                totaal_oi_calls,
                totaal_oi_puts,
                call_put_ratio,
                scraped_at,
                source_url,
                created_at,
                updated_at
            FROM bronze_fd_overview
            WHERE peildatum = :trade_date
              AND ticker = :ticker
              AND scraped_at = :latest_scraped_at
        """)
        
        df = pd.read_sql(query, session.connection(), params={
            'trade_date': trade_date,
            'ticker': ticker,
            'latest_scraped_at': latest_scraped_at
        })
    
    if df.empty:
        logger.warning(f"No overview data found for {ticker} on {trade_date}")
        return None
    
    # Format as_of timestamp for path
    as_of_str = latest_scraped_at.strftime('%Y-%m-%dT%H-%M-%SZ')
    
    # Build S3 path
    s3_path = f"bronze/options_overview/v1/date={trade_date}/ticker={ticker}/as_of={as_of_str}/data.parquet"
    
    # Save to temp file
    temp_file = TEMP_DIR / f"options_overview_{trade_date}_{ticker}_{as_of_str}.parquet"
    df.to_parquet(temp_file, index=False, compression='snappy')
    
    # Upload to MinIO
    minio_client = get_minio_client()
    minio_client.upload_file(
        local_path=str(temp_file),
        object_name=s3_path
    )
    
    # Cleanup
    temp_file.unlink()
    
    logger.info(f"✅ Uploaded {len(df)} overview records to {s3_path}")
    return s3_path


def export_all_bronze_to_parquet(limit_dates: int = None):
    """
    Export all bronze data to partitioned parquet files in MinIO.
    
    Args:
        limit_dates: Optional limit on number of dates to process (for testing)
    """
    logger.info("="*60)
    logger.info("🚀 STARTING BRONZE PARQUET EXPORT")
    logger.info("="*60)
    
    stats = {
        'options_chain_files': 0,
        'options_overview_files': 0,
        'total_options_rows': 0,
        'total_overview_rows': 0,
        'errors': 0
    }
    
    # Get distinct trade dates with latest scrape timestamp
    trade_dates = get_distinct_trade_dates()
    
    if limit_dates:
        trade_dates = trade_dates[-limit_dates:]
        logger.info(f"⚠️  Limited to {limit_dates} most recent dates")
    
    logger.info(f"\n📅 Processing {len(trade_dates)} trade dates (deduplicating to latest scrape per date)")
    
    # Export each trade date (with latest scrape data only)
    for i, (trade_date, ticker, latest_scraped_at) in enumerate(trade_dates, 1):
        try:
            logger.info(f"\n[{i}/{len(trade_dates)}] Processing {trade_date} | {ticker}")
            logger.info(f"   Latest scrape: {latest_scraped_at}")
            
            # Export overview (latest scrape only)
            overview_path = export_options_overview_partition(trade_date, ticker, latest_scraped_at)
            if overview_path:
                stats['options_overview_files'] += 1
            
            # Export options chain (all options from same scrape day)
            chain_path = export_options_chain_partition(trade_date, ticker, latest_scraped_at)
            if chain_path:
                stats['options_chain_files'] += 1
            
        except Exception as e:
            logger.error(f"❌ Error processing {trade_date} | {ticker}: {e}")
            stats['errors'] += 1
            continue
    
    # Summary
    logger.info("\n" + "="*60)
    logger.info("🎉 BRONZE PARQUET EXPORT COMPLETED")
    logger.info("="*60)
    logger.info(f"📊 Options chain files: {stats['options_chain_files']}")
    logger.info(f"📊 Options overview files: {stats['options_overview_files']}")
    logger.info(f"❌ Errors: {stats['errors']}")
    logger.info("\n✅ Data structure in MinIO:")
    logger.info("   bronze/options_chain/v1/date=YYYY-MM-DD/ticker=TICKER/as_of=TIMESTAMP/data.parquet")
    logger.info("   bronze/options_overview/v1/date=YYYY-MM-DD/ticker=TICKER/as_of=TIMESTAMP/data.parquet")


def main():
    """Main entry point"""
    import argparse
    
    parser = argparse.ArgumentParser(description='Export bronze layer to partitioned parquet files in MinIO')
    parser.add_argument('--limit', type=int, help='Limit number of dates to process (for testing)', default=None)
    parser.add_argument('--date', type=str, help='Export specific date only (YYYY-MM-DD)', default=None)
    
    args = parser.parse_args()
    
    if args.date:
        # Export specific date
        logger.info(f"Exporting specific date: {args.date}")
        # Get all trade dates and filter
        trade_dates = get_distinct_trade_dates()
        date_entries = [(d, t, s) for d, t, s in trade_dates if str(d) == args.date]
        
        for trade_date, ticker, latest_scraped_at in date_entries:
            export_options_overview_partition(trade_date, ticker, latest_scraped_at)
            export_options_chain_partition(trade_date, ticker, latest_scraped_at)
    else:
        # Export all
        export_all_bronze_to_parquet(limit_dates=args.limit)


if __name__ == '__main__':
    main()
