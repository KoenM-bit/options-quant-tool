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
import os
from pathlib import Path

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

import argparse
import pandas as pd
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
    """Export bronze_fd_options and bronze_fd_overview for a specific date.
    
    Note: FD data is published with 1-day delay. If no data exists for the requested date,
    we'll export the latest available data instead (typically previous day).
    """
    logger.info(f"Exporting bronze FD data for {trade_date}, ticker={ticker}")
    
    total_records = 0
    
    with get_db_session() as session:
        # First check if data exists for requested date, otherwise get latest
        check_query = text("""
            SELECT MAX(trade_date) as latest_date
            FROM bronze_fd_options
            WHERE ticker = :ticker
              AND trade_date <= :trade_date
        """)
        
        result = session.execute(check_query, {'ticker': ticker, 'trade_date': trade_date}).fetchone()
        actual_date = result[0] if result and result[0] else None
        
        if actual_date and actual_date != trade_date:
            logger.info(f"📅 No FD data for {trade_date}, using latest available: {actual_date}")
        elif not actual_date:
            logger.warning(f"No FD data found for {ticker} up to {trade_date}")
            return 0
        else:
            actual_date = trade_date
        
        # Export options (use actual available date)
        options_query = text("""
            SELECT *
            FROM bronze_fd_options
            WHERE ticker = :ticker
              AND trade_date = :trade_date
            ORDER BY option_type, strike, expiry_date
        """)
        
        df_options = pd.read_sql(options_query, session.connection(), params={
            'trade_date': actual_date,
            'ticker': ticker
        })
        
        if len(df_options) > 0:
            # Write to temp file
            with tempfile.NamedTemporaryFile(mode='wb', suffix='.parquet', delete=False) as tmp:
                df_options.to_parquet(tmp.name, index=False, engine='pyarrow')
                tmp_path = tmp.name
            
            # Upload to MinIO - use actual_date for partitioning
            s3_path = f"bronze/fd_options/date={actual_date}/ticker={ticker}/data.parquet"
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
            'trade_date': actual_date,
            'ticker': ticker
        })
        
        if len(df_overview) > 0:
            # Write to temp file
            with tempfile.NamedTemporaryFile(mode='wb', suffix='.parquet', delete=False) as tmp:
                df_overview.to_parquet(tmp.name, index=False, engine='pyarrow')
                tmp_path = tmp.name
            
            # Upload to MinIO - use actual_date for partitioning
            s3_path = f"bronze/fd_overview/date={actual_date}/ticker={ticker}/data.parquet"
            minio_client.upload_file(tmp_path, s3_path)
            logger.info(f"✅ Exported {len(df_overview)} FD overview records to s3://{minio_client.bucket}/{s3_path}")
            
            # Cleanup
            Path(tmp_path).unlink()
            total_records += len(df_overview)
        
        return total_records


def export_silver(trade_date: date, ticker: str) -> int:
    """Export silver star schema (dims + facts) for a specific date.
    
    Exports:
    - dim_underlying (per ticker)
    - dim_option_contract (per ticker)
    - fact_option_timeseries (per date, per ticker) - Intraday BD data
    - fact_option_eod (per date, per ticker) - End-of-day FD data with OI
    - fact_market_overview (per date, per ticker) - Daily market totals
    """
    logger.info(f"Exporting silver star schema for {trade_date}, ticker={ticker}")
    
    total_records = 0
    
    with get_db_session() as session:
        # 1. Export dim_underlying (full refresh - small table)
        logger.info("Exporting dim_underlying...")
        query_underlying = text("""
            SELECT *
            FROM dim_underlying
            WHERE ticker = :ticker
            ORDER BY ticker
        """)
        
        df_underlying = pd.read_sql(query_underlying, session.connection(), params={'ticker': ticker})
        
        if len(df_underlying) > 0:
            with tempfile.NamedTemporaryFile(mode='wb', suffix='.parquet', delete=False) as tmp:
                df_underlying.to_parquet(tmp.name, index=False, engine='pyarrow')
                tmp_path = tmp.name
            
            s3_path = f"silver/dim_underlying/ticker={ticker}/data.parquet"
            minio_client.upload_file(tmp_path, s3_path)
            logger.info(f"✅ Exported {len(df_underlying)} dim_underlying records to s3://{minio_client.bucket}/{s3_path}")
            Path(tmp_path).unlink()
            total_records += len(df_underlying)
        
        # 2. Export dim_option_contract (full refresh for this ticker)
        logger.info("Exporting dim_option_contract...")
        query_contract = text("""
            SELECT *
            FROM dim_option_contract
            WHERE ticker = :ticker
            ORDER BY expiration_date, strike, call_put
        """)
        
        df_contract = pd.read_sql(query_contract, session.connection(), params={'ticker': ticker})
        
        if len(df_contract) > 0:
            with tempfile.NamedTemporaryFile(mode='wb', suffix='.parquet', delete=False) as tmp:
                df_contract.to_parquet(tmp.name, index=False, engine='pyarrow')
                tmp_path = tmp.name
            
            s3_path = f"silver/dim_option_contract/ticker={ticker}/data.parquet"
            minio_client.upload_file(tmp_path, s3_path)
            logger.info(f"✅ Exported {len(df_contract)} dim_option_contract records to s3://{minio_client.bucket}/{s3_path}")
            Path(tmp_path).unlink()
            total_records += len(df_contract)
        
        # 3. Export fact_option_timeseries (partitioned by date)
        logger.info("Exporting fact_option_timeseries...")
        query_fact = text("""
            SELECT f.*
            FROM fact_option_timeseries f
            JOIN dim_option_contract c ON f.option_id = c.option_id
            WHERE f.trade_date = :trade_date AND c.ticker = :ticker
            ORDER BY f.ts
        """)
        
        df_fact = pd.read_sql(query_fact, session.connection(), params={
            'trade_date': trade_date,
            'ticker': ticker
        })
        
        if len(df_fact) > 0:
            with tempfile.NamedTemporaryFile(mode='wb', suffix='.parquet', delete=False) as tmp:
                df_fact.to_parquet(tmp.name, index=False, engine='pyarrow')
                tmp_path = tmp.name
            
            s3_path = f"silver/fact_option_timeseries/date={trade_date}/ticker={ticker}/data.parquet"
            minio_client.upload_file(tmp_path, s3_path)
            logger.info(f"✅ Exported {len(df_fact)} fact_option_timeseries records to s3://{minio_client.bucket}/{s3_path}")
            Path(tmp_path).unlink()
            total_records += len(df_fact)
        else:
            logger.warning(f"No fact_option_timeseries data found for {trade_date}, {ticker}")
        
        # 4. Export fact_option_eod (partitioned by date)
        # Note: FD data may lag by 1 day, export latest available
        logger.info("Exporting fact_option_eod...")
        
        # Check for latest available EOD data
        check_eod = text("""
            SELECT MAX(e.trade_date) as latest_date
            FROM fact_option_eod e
            JOIN dim_option_contract c ON e.option_id = c.option_id
            WHERE c.ticker = :ticker AND e.trade_date <= :trade_date
        """)
        
        result_eod = session.execute(check_eod, {'ticker': ticker, 'trade_date': trade_date}).fetchone()
        actual_eod_date = result_eod[0] if result_eod and result_eod[0] else None
        
        if actual_eod_date and actual_eod_date != trade_date:
            logger.info(f"📅 No EOD data for {trade_date}, using latest available: {actual_eod_date}")
        
        if actual_eod_date:
            query_eod = text("""
                SELECT e.*
                FROM fact_option_eod e
                JOIN dim_option_contract c ON e.option_id = c.option_id
                WHERE e.trade_date = :trade_date AND c.ticker = :ticker
                ORDER BY e.ts
            """)
            
            df_eod = pd.read_sql(query_eod, session.connection(), params={
                'trade_date': actual_eod_date,
                'ticker': ticker
            })
            
            if len(df_eod) > 0:
                with tempfile.NamedTemporaryFile(mode='wb', suffix='.parquet', delete=False) as tmp:
                    df_eod.to_parquet(tmp.name, index=False, engine='pyarrow')
                    tmp_path = tmp.name
                
                s3_path = f"silver/fact_option_eod/date={actual_eod_date}/ticker={ticker}/data.parquet"
                minio_client.upload_file(tmp_path, s3_path)
                logger.info(f"✅ Exported {len(df_eod)} fact_option_eod records to s3://{minio_client.bucket}/{s3_path}")
                Path(tmp_path).unlink()
                total_records += len(df_eod)
        else:
            logger.warning(f"No fact_option_eod data found for {ticker} up to {trade_date}")
        
        # 5. Export fact_market_overview (partitioned by date)
        # Note: Market overview also lags by 1 day, export latest available
        logger.info("Exporting fact_market_overview...")
        
        # Check for latest available market overview data
        check_overview = text("""
            SELECT MAX(o.trade_date) as latest_date
            FROM fact_market_overview o
            JOIN dim_underlying u ON o.underlying_id = u.underlying_id
            WHERE u.ticker = :ticker AND o.trade_date <= :trade_date
        """)
        
        result_overview = session.execute(check_overview, {'ticker': ticker, 'trade_date': trade_date}).fetchone()
        actual_overview_date = result_overview[0] if result_overview and result_overview[0] else None
        
        if actual_overview_date and actual_overview_date != trade_date:
            logger.info(f"📅 No market overview for {trade_date}, using latest available: {actual_overview_date}")
        
        if actual_overview_date:
            query_overview = text("""
                SELECT o.*
            FROM fact_market_overview o
            JOIN dim_underlying u ON o.underlying_id = u.underlying_id
            WHERE o.trade_date = :trade_date AND u.ticker = :ticker
        """)
        
            df_overview = pd.read_sql(query_overview, session.connection(), params={
                'trade_date': actual_overview_date,
                'ticker': ticker
            })
            
            if len(df_overview) > 0:
                with tempfile.NamedTemporaryFile(mode='wb', suffix='.parquet', delete=False) as tmp:
                    df_overview.to_parquet(tmp.name, index=False, engine='pyarrow')
                    tmp_path = tmp.name
                
                s3_path = f"silver/fact_market_overview/date={actual_overview_date}/ticker={ticker}/data.parquet"
                minio_client.upload_file(tmp_path, s3_path)
                logger.info(f"✅ Exported {len(df_overview)} fact_market_overview records to s3://{minio_client.bucket}/{s3_path}")
                Path(tmp_path).unlink()
                total_records += len(df_overview)
        else:
            logger.warning(f"No fact_market_overview data found for {ticker} up to {trade_date}")
        
        logger.info(f"✅ Total silver star schema records exported: {total_records}")
        return total_records


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


def export_bronze_ohlcv(trade_date: date, ticker: str) -> int:
    """Export bronze_ohlcv (stock OHLCV data) for a specific date."""
    logger.info(f"Exporting bronze OHLCV data for {trade_date}, ticker={ticker}")
    
    with get_db_session() as session:
        query = text("""
            SELECT *
            FROM bronze_ohlcv
            WHERE trade_date = :trade_date AND ticker = :ticker
            ORDER BY trade_date
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
            s3_path = f"bronze/ohlcv/date={trade_date}/ticker={ticker}/data.parquet"
            minio_client.upload_file(tmp_path, s3_path)
            logger.info(f"✅ Exported {len(df)} OHLCV records to s3://{minio_client.bucket}/{s3_path}")
            
            # Cleanup
            Path(tmp_path).unlink()
            return len(df)
        else:
            logger.warning(f"No OHLCV data found for {trade_date}, {ticker}")
            return 0


def export_technical_indicators(trade_date: date, ticker: str) -> int:
    """Export fact_technical_indicators for a specific date.
    
    Note: Technical indicators may lag if OHLCV data isn't available yet.
    Will export latest available data if requested date doesn't exist.
    """
    logger.info(f"Exporting technical indicators for {trade_date}, ticker={ticker}")
    
    with get_db_session() as session:
        # Check for latest available technical indicators
        check_query = text("""
            SELECT MAX(trade_date) as latest_date
            FROM fact_technical_indicators
            WHERE ticker = :ticker AND trade_date <= :trade_date
        """)
        
        result = session.execute(check_query, {'ticker': ticker, 'trade_date': trade_date}).fetchone()
        actual_date = result[0] if result and result[0] else None
        
        if actual_date and actual_date != trade_date:
            logger.info(f"📅 No technical indicators for {trade_date}, using latest available: {actual_date}")
        elif not actual_date:
            logger.warning(f"No technical indicators found for {ticker} up to {trade_date}")
            return 0
        else:
            actual_date = trade_date
        
        query = text("""
            SELECT *
            FROM fact_technical_indicators
            WHERE trade_date = :trade_date AND ticker = :ticker
            ORDER BY trade_date
        """)
        
        df = pd.read_sql(query, session.connection(), params={
            'trade_date': actual_date,
            'ticker': ticker
        })
        
        if len(df) > 0:
            # Write to temp file
            with tempfile.NamedTemporaryFile(mode='wb', suffix='.parquet', delete=False) as tmp:
                df.to_parquet(tmp.name, index=False, engine='pyarrow')
                tmp_path = tmp.name
            
            # Upload to MinIO - use actual_date for partitioning
            s3_path = f"silver/technical_indicators/date={actual_date}/ticker={ticker}/data.parquet"
            minio_client.upload_file(tmp_path, s3_path)
            logger.info(f"✅ Exported {len(df)} technical indicator records to s3://{minio_client.bucket}/{s3_path}")
            
            # Cleanup
            Path(tmp_path).unlink()
            return len(df)
        else:
            logger.warning(f"No technical indicators found for {actual_date}, {ticker}")
            return 0


def export_market_regime(trade_date: date, ticker: str) -> int:
    """Export fact_market_regime for a specific date."""
    logger.info(f"Exporting market regime for {trade_date}, ticker={ticker}")
    
    with get_db_session() as session:
        query = text("""
            SELECT *
            FROM fact_market_regime
            WHERE trade_date = :trade_date AND ticker = :ticker
            ORDER BY trade_date
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
            s3_path = f"gold/market_regime/date={trade_date}/ticker={ticker}/data.parquet"
            minio_client.upload_file(tmp_path, s3_path)
            logger.info(f"✅ Exported {len(df)} market regime records to s3://{minio_client.bucket}/{s3_path}")
            
            # Cleanup
            Path(tmp_path).unlink()
            return len(df)
        else:
            logger.warning(f"No market regime found for {trade_date}, {ticker}")
            return 0


def main():
    parser = argparse.ArgumentParser(description='Export data to parquet with date partitioning')
    parser.add_argument('--date', type=str, required=True, help='Trade date (YYYY-MM-DD)')
    parser.add_argument('--ticker', type=str, default='AD.AS', help='Ticker symbol')
    parser.add_argument('--layer', type=str, choices=['bronze', 'silver', 'gold', 'all', 'ohlcv'], 
                        default='all', help='Which layer to export')
    
    args = parser.parse_args()
    
    # Parse date
    trade_date = datetime.strptime(args.date, '%Y-%m-%d').date()
    ticker = args.ticker
    
    logger.info("="*60)
    logger.info(f"Starting parquet export for {trade_date}, ticker={ticker}, layer={args.layer}")
    logger.info("="*60)
    
    total_records = 0
    
    if args.layer in ['bronze', 'all', 'ohlcv']:
        logger.info("\n--- Exporting Bronze Layer ---")
        if args.layer != 'ohlcv':
            total_records += export_bronze_bd(trade_date, ticker)
            total_records += export_bronze_fd(trade_date, ticker)
        total_records += export_bronze_ohlcv(trade_date, ticker)
    
    if args.layer in ['silver', 'all', 'ohlcv']:
        logger.info("\n--- Exporting Silver Layer ---")
        if args.layer != 'ohlcv':
            total_records += export_silver(trade_date, ticker)
        total_records += export_technical_indicators(trade_date, ticker)
    
    if args.layer in ['gold', 'all', 'ohlcv']:
        logger.info("\n--- Exporting Gold Layer ---")
        if args.layer != 'ohlcv':
            total_records += export_gold(trade_date, ticker)
        total_records += export_market_regime(trade_date, ticker)
    
    logger.info("="*60)
    logger.info(f"✅ Export complete! Total records: {total_records}")
    logger.info(f"MinIO bucket: s3://{minio_client.bucket}/")
    logger.info("="*60)


if __name__ == '__main__':
    main()
