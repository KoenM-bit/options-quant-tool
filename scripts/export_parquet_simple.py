"""
Simple Parquet Export for Testing
Exports bronze and silver data to parquet with date partitioning and uploads to MinIO.

MinIO structure:
  s3://options-data/bronze/bd_options/date=YYYY-MM-DD/ticker=TICKER/data.parquet
  s3://options-data/bronze/bd_underlying/date=YYYY-MM-DD/ticker=TICKER/data.parquet
  s3://options-data/bronze/fd_options/date=YYYY-MM-DD/ticker=TICKER/data.parquet
  s3://options-data/bronze/fd_overview/date=YYYY-MM-DD/ticker=TICKER/data.parquet
  s3://options-data/silver/options_chain_merged/date=YYYY-MM-DD/ticker=TICKER/data.parquet
  s3://options-data/gold/daily_summary/date=YYYY-MM-DD/ticker=TICKER/data.parquet

Usage:
    python scripts/export_parquet_simple.py --date 2025-12-10 --layer bronze
    python scripts/export_parquet_simple.py --date 2025-12-10 --layer silver
    python scripts/export_parquet_simple.py --date 2025-12-10 --layer gold
    python scripts/export_parquet_simple.py --date 2025-12-10 --layer all
"""

import sys
sys.path.insert(0, '/opt/airflow/dags/..')

import argparse
import pandas as pd
from pathlib import Path
from datetime import date, datetime
from sqlalchemy import text
import logging
import tempfile

from src.utils.db import get_db_session
from src.utils.minio_client import get_minio_client
from src.config import settings

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Get MinIO client
minio_client = get_minio_client()


def export_bronze_bd(trade_date: date, ticker: str) -> int:
    """Export bronze_bd_options and bronze_bd_underlying for a specific date."""
    logger.info(f"Exporting bronze BD data for {trade_date}, ticker={ticker}")
    
    total_records = 0
    
    with get_db_session() as session:
        # Export options
        options_query = text("""
            SELECT *
            FROM bronze_bd_options
            WHERE trade_date = :trade_date AND ticker = :ticker
            ORDER BY option_type, strike, expiry_date
        """)
        
        df_options = pd.read_sql(options_query, session.connection(), params={
            'trade_date': trade_date,
            'ticker': ticker
        })
        
        if len(df_options) > 0:
            # Write to temp file
            with tempfile.NamedTemporaryFile(mode='wb', suffix='.parquet', delete=False) as tmp:
                df_options.to_parquet(tmp.name, index=False, engine='pyarrow')
                tmp_path = tmp.name
            
            # Upload to MinIO
            s3_path = f"bronze/bd_options/date={trade_date}/ticker={ticker}/data.parquet"
            minio_client.upload_file(tmp_path, s3_path)
            logger.info(f"✅ Exported {len(df_options)} BD option records to s3://{minio_client.bucket}/{s3_path}")
            
            # Cleanup temp file
            Path(tmp_path).unlink()
            total_records += len(df_options)
        
        # Export underlying
        underlying_query = text("""
            SELECT *
            FROM bronze_bd_underlying
            WHERE trade_date = :trade_date AND ticker = :ticker
            ORDER BY scraped_at DESC
        """)
        
        df_underlying = pd.read_sql(underlying_query, session.connection(), params={
            'trade_date': trade_date,
            'ticker': ticker
        })
        
        if len(df_underlying) > 0:
            # Write to temp file
            with tempfile.NamedTemporaryFile(mode='wb', suffix='.parquet', delete=False) as tmp:
                df_underlying.to_parquet(tmp.name, index=False, engine='pyarrow')
                tmp_path = tmp.name
            
            # Upload to MinIO
            s3_path = f"bronze/bd_underlying/date={trade_date}/ticker={ticker}/data.parquet"
            minio_client.upload_file(tmp_path, s3_path)
            logger.info(f"✅ Exported {len(df_underlying)} BD underlying records to s3://{minio_client.bucket}/{s3_path}")
            
            # Cleanup temp file
            Path(tmp_path).unlink()
            total_records += len(df_underlying)
        
        return total_records


def export_bronze_fd(trade_date: date, ticker: str) -> int:
    """Export bronze_fd_options and bronze_fd_overview for a specific date."""
    logger.info(f"Exporting bronze FD data for {trade_date}, ticker={ticker}")
    
    total_records = 0
    
    with get_db_session() as session:
        # Export options (use trade_date column)
        options_query = text("""
            SELECT *
            FROM bronze_fd_options
            WHERE ticker = :ticker
              AND trade_date = :trade_date
            ORDER BY option_type, strike, expiry_date
        """)
        
        df_options = pd.read_sql(options_query, session.connection(), params={
            'trade_date': trade_date,
            'ticker': ticker
        })
        
        if len(df_options) > 0:
            # Write to temp file
            with tempfile.NamedTemporaryFile(mode='wb', suffix='.parquet', delete=False) as tmp:
                df_options.to_parquet(tmp.name, index=False, engine='pyarrow')
                tmp_path = tmp.name
            
            # Upload to MinIO
            s3_path = f"bronze/fd_options/date={trade_date}/ticker={ticker}/data.parquet"
            minio_client.upload_file(tmp_path, s3_path)
            logger.info(f"✅ Exported {len(df_options)} FD option records to s3://{minio_client.bucket}/{s3_path}")
            
            # Cleanup
            Path(tmp_path).unlink()
            total_records += len(df_options)
        
        # Export overview
        overview_query = text("""
            SELECT *
            FROM bronze_fd_overview
            WHERE ticker = :ticker AND trade_date = :trade_date
            ORDER BY scraped_at DESC
        """)
        
        df_overview = pd.read_sql(overview_query, session.connection(), params={
            'trade_date': trade_date,
            'ticker': ticker
        })
        
        if len(df_overview) > 0:
            # Write to temp file
            with tempfile.NamedTemporaryFile(mode='wb', suffix='.parquet', delete=False) as tmp:
                df_overview.to_parquet(tmp.name, index=False, engine='pyarrow')
                tmp_path = tmp.name
            
            # Upload to MinIO
            s3_path = f"bronze/fd_overview/date={trade_date}/ticker={ticker}/data.parquet"
            minio_client.upload_file(tmp_path, s3_path)
            logger.info(f"✅ Exported {len(df_overview)} FD overview records to s3://{minio_client.bucket}/{s3_path}")
            
            # Cleanup
            Path(tmp_path).unlink()
            total_records += len(df_overview)
        
        return total_records


def export_silver(trade_date: date, ticker: str) -> int:
    """Export silver_bd_options_enriched for a specific date."""
    logger.info(f"Exporting silver BD options data for {trade_date}, ticker={ticker}")
    
    with get_db_session() as session:
        query = text("""
            SELECT *
            FROM silver_bd_options_enriched
            WHERE trade_date = :trade_date AND ticker = :ticker
            ORDER BY option_type, strike, expiry_date
        """)
        
        df = pd.read_sql(query, session.connection(), params={
            'trade_date': trade_date,
            'ticker': ticker
        })
        
        if len(df) > 0:
            # Write to temp file
            with tempfile.NamedTemporaryFile(mode='wb', suffix='.parquet', delete=False) as tmp:
                df.to_parquet(tmp.name, index=False, engine='pyarrow')
                tmp_path = tmp.name
            
            # Upload to MinIO
            s3_path = f"silver/bd_options_enriched/date={trade_date}/ticker={ticker}/data.parquet"
            minio_client.upload_file(tmp_path, s3_path)
            logger.info(f"✅ Exported {len(df)} silver BD records to s3://{minio_client.bucket}/{s3_path}")
            
            # Cleanup
            Path(tmp_path).unlink()
            return len(df)
        else:
            logger.warning(f"No silver data found for {trade_date}, {ticker}")
            return 0


def export_gold(trade_date: date, ticker: str) -> int:
    """Export gold_daily_summary_test for a specific date (skip if table doesn't exist)."""
    logger.info(f"Exporting gold data for {trade_date}, ticker={ticker}")
    
    with get_db_session() as session:
        # Check if table exists first
        check_query = text("""
            SELECT EXISTS (
                SELECT FROM information_schema.tables 
                WHERE table_schema = 'public' 
                AND table_name = 'gold_daily_summary_test'
            )
        """)
        table_exists = session.execute(check_query).scalar()
        
        if not table_exists:
            logger.warning("⚠️  gold_daily_summary_test table doesn't exist, skipping gold export")
            return 0
        
        query = text("""
            SELECT *
            FROM gold_daily_summary_test
            WHERE trade_date = :trade_date AND ticker = :ticker
        """)
        
        df = pd.read_sql(query, session.connection(), params={
            'trade_date': trade_date,
            'ticker': ticker
        })
        
        if len(df) > 0:
            # Write to temp file
            with tempfile.NamedTemporaryFile(mode='wb', suffix='.parquet', delete=False) as tmp:
                df.to_parquet(tmp.name, index=False, engine='pyarrow')
                tmp_path = tmp.name
            
            # Upload to MinIO
            s3_path = f"gold/daily_summary/date={trade_date}/ticker={ticker}/data.parquet"
            minio_client.upload_file(tmp_path, s3_path)
            logger.info(f"✅ Exported {len(df)} gold records to s3://{minio_client.bucket}/{s3_path}")
            
            # Cleanup
            Path(tmp_path).unlink()
            return len(df)
        else:
            logger.warning(f"No gold data found for {trade_date}, {ticker}")
            return 0


def main():
    parser = argparse.ArgumentParser(description='Export data to parquet with date partitioning')
    parser.add_argument('--date', type=str, required=True, help='Trade date (YYYY-MM-DD)')
    parser.add_argument('--ticker', type=str, default='AD.AS', help='Ticker symbol')
    parser.add_argument('--layer', type=str, choices=['bronze', 'silver', 'gold', 'all'], 
                        default='all', help='Which layer to export')
    
    args = parser.parse_args()
    
    # Parse date
    trade_date = datetime.strptime(args.date, '%Y-%m-%d').date()
    ticker = args.ticker
    
    logger.info("="*60)
    logger.info(f"Starting parquet export for {trade_date}, ticker={ticker}, layer={args.layer}")
    logger.info("="*60)
    
    total_records = 0
    
    if args.layer in ['bronze', 'all']:
        logger.info("\n--- Exporting Bronze Layer ---")
        total_records += export_bronze_bd(trade_date, ticker)
        total_records += export_bronze_fd(trade_date, ticker)
    
    if args.layer in ['silver', 'all']:
        logger.info("\n--- Exporting Silver Layer ---")
        total_records += export_silver(trade_date, ticker)
    
    if args.layer in ['gold', 'all']:
        logger.info("\n--- Exporting Gold Layer ---")
        total_records += export_gold(trade_date, ticker)
    
    logger.info("="*60)
    logger.info(f"✅ Export complete! Total records: {total_records}")
    logger.info(f"MinIO bucket: s3://{minio_client.bucket}/")
    logger.info("="*60)


if __name__ == '__main__':
    main()
