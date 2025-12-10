#!/usr/bin/env python3
"""
Export historical migrated data to MinIO as Parquet files.

Exports:
- bronze_bd_options (source='mariadb_migration')
- bronze_bd_underlying (source_url='mariadb_migration')
- silver_bd_options_enriched (all historical data)

Uses same partitioning structure as current pipeline:
  s3://options-data/bronze/bd_options/date=YYYY-MM-DD/ticker=TICKER/data.parquet
  s3://options-data/bronze/bd_underlying/date=YYYY-MM-DD/ticker=TICKER/data.parquet
  s3://options-data/silver/bd_options_enriched/date=YYYY-MM-DD/ticker=TICKER/data.parquet
"""

import sys
sys.path.insert(0, '/Users/koenmarijt/Documents/Projects/ahold-options')

import argparse
import pandas as pd
from pathlib import Path
from datetime import date
from sqlalchemy import text
import logging
import tempfile

from src.utils.db import get_db_session
from src.utils.minio_client import get_minio_client

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


def get_migrated_dates():
    """Get list of unique trade dates from migrated data."""
    with get_db_session() as session:
        query = text("""
            SELECT DISTINCT trade_date
            FROM bronze_bd_options
            WHERE source = 'mariadb_migration'
            ORDER BY trade_date
        """)
        result = session.execute(query)
        dates = [row[0] for row in result]
    return dates


def export_bronze_bd_for_date(trade_date: date, ticker: str, minio_client, dry_run: bool = False) -> dict:
    """Export bronze BD data for a specific date."""
    stats = {'options': 0, 'underlying': 0}
    
    with get_db_session() as session:
        # Export bronze_bd_options
        options_query = text("""
            SELECT 
                ticker, symbol_code, issue_id, trade_date, option_type,
                expiry_date, expiry_text, strike, bid, ask,
                last_price, volume, last_timestamp, last_date_text,
                source, source_url, scraped_at, created_at, updated_at
            FROM bronze_bd_options
            WHERE trade_date = :trade_date 
              AND ticker = :ticker
              AND source = 'mariadb_migration'
            ORDER BY option_type, strike, expiry_date
        """)
        
        df_options = pd.read_sql(options_query, session.connection(), params={
            'trade_date': trade_date,
            'ticker': ticker
        })
        
        if len(df_options) > 0:
            stats['options'] = len(df_options)
            
            if not dry_run:
                # Write to temp file
                with tempfile.NamedTemporaryFile(mode='wb', suffix='.parquet', delete=False) as tmp:
                    df_options.to_parquet(tmp.name, index=False, engine='pyarrow')
                    tmp_path = tmp.name
                
                # Upload to MinIO
                s3_path = f"bronze/bd_options/date={trade_date}/ticker={ticker}/data.parquet"
                minio_client.upload_file(tmp_path, s3_path)
                logger.info(f"  ✅ Bronze options: {len(df_options)} records → {s3_path}")
                
                # Cleanup
                Path(tmp_path).unlink()
            else:
                logger.info(f"  [DRY RUN] Would export {len(df_options)} bronze options records")
        
        # Export bronze_bd_underlying
        underlying_query = text("""
            SELECT 
                ticker, trade_date, isin, name, last_price, bid, ask,
                volume, last_timestamp_text, scraped_at, source_url,
                created_at, updated_at
            FROM bronze_bd_underlying
            WHERE DATE(trade_date) = :trade_date 
              AND ticker = :ticker
              AND source_url = 'mariadb_migration'
            ORDER BY scraped_at DESC
        """)
        
        df_underlying = pd.read_sql(underlying_query, session.connection(), params={
            'trade_date': trade_date,
            'ticker': ticker
        })
        
        if len(df_underlying) > 0:
            stats['underlying'] = len(df_underlying)
            
            if not dry_run:
                # Write to temp file
                with tempfile.NamedTemporaryFile(mode='wb', suffix='.parquet', delete=False) as tmp:
                    df_underlying.to_parquet(tmp.name, index=False, engine='pyarrow')
                    tmp_path = tmp.name
                
                # Upload to MinIO
                s3_path = f"bronze/bd_underlying/date={trade_date}/ticker={ticker}/data.parquet"
                minio_client.upload_file(tmp_path, s3_path)
                logger.info(f"  ✅ Bronze underlying: {len(df_underlying)} records → {s3_path}")
                
                # Cleanup
                Path(tmp_path).unlink()
            else:
                logger.info(f"  [DRY RUN] Would export {len(df_underlying)} bronze underlying records")
    
    return stats


def export_silver_for_date(trade_date: date, ticker: str, minio_client, dry_run: bool = False) -> int:
    """Export silver enriched data for a specific date."""
    with get_db_session() as session:
        query = text("""
            SELECT 
                ticker, trade_date, option_type, strike, expiry_date,
                symbol_code, issue_id, bid, ask, mid_price, last_price,
                underlying_price, volume, underlying_volume, last_timestamp,
                days_to_expiry, moneyness, delta, gamma, theta, vega, rho,
                implied_volatility, source_url, scraped_at, transformed_at
            FROM silver_bd_options_enriched
            WHERE trade_date = :trade_date AND ticker = :ticker
            ORDER BY option_type, strike, expiry_date
        """)
        
        df = pd.read_sql(query, session.connection(), params={
            'trade_date': trade_date,
            'ticker': ticker
        })
        
        if len(df) > 0:
            if not dry_run:
                # Write to temp file
                with tempfile.NamedTemporaryFile(mode='wb', suffix='.parquet', delete=False) as tmp:
                    df.to_parquet(tmp.name, index=False, engine='pyarrow')
                    tmp_path = tmp.name
                
                # Upload to MinIO
                s3_path = f"silver/bd_options_enriched/date={trade_date}/ticker={ticker}/data.parquet"
                minio_client.upload_file(tmp_path, s3_path)
                logger.info(f"  ✅ Silver enriched: {len(df)} records → {s3_path}")
                
                # Cleanup
                Path(tmp_path).unlink()
            else:
                logger.info(f"  [DRY RUN] Would export {len(df)} silver enriched records")
            
            return len(df)
        
        return 0


def main():
    parser = argparse.ArgumentParser(
        description='Export historical migrated data to MinIO as Parquet'
    )
    parser.add_argument('--dry-run', action='store_true',
                       help='Show what would be exported without uploading')
    parser.add_argument('--ticker', default='AD.AS',
                       help='Ticker to export (default: AD.AS)')
    parser.add_argument('--layer', choices=['bronze', 'silver', 'all'], default='all',
                       help='Which layer to export')
    
    args = parser.parse_args()
    
    logger.info("=" * 80)
    logger.info("HISTORICAL DATA PARQUET EXPORT TO MINIO")
    logger.info("=" * 80)
    logger.info(f"Ticker: {args.ticker}")
    logger.info(f"Layer: {args.layer}")
    logger.info(f"Dry run: {args.dry_run}")
    logger.info("=" * 80)
    
    # Get MinIO client
    minio_client = get_minio_client() if not args.dry_run else None
    
    # Get list of dates to export
    logger.info("Finding migrated trade dates...")
    dates = get_migrated_dates()
    logger.info(f"✅ Found {len(dates)} trade dates: {dates[0]} to {dates[-1]}")
    
    # Export each date
    total_stats = {
        'bronze_options': 0,
        'bronze_underlying': 0,
        'silver_enriched': 0
    }
    
    for i, trade_date in enumerate(dates, 1):
        logger.info(f"\n[{i}/{len(dates)}] Exporting {trade_date}...")
        
        # Export bronze layer
        if args.layer in ['bronze', 'all']:
            bronze_stats = export_bronze_bd_for_date(
                trade_date, 
                args.ticker, 
                minio_client,
                dry_run=args.dry_run
            )
            total_stats['bronze_options'] += bronze_stats['options']
            total_stats['bronze_underlying'] += bronze_stats['underlying']
        
        # Export silver layer
        if args.layer in ['silver', 'all']:
            silver_count = export_silver_for_date(
                trade_date,
                args.ticker,
                minio_client,
                dry_run=args.dry_run
            )
            total_stats['silver_enriched'] += silver_count
    
    # Summary
    logger.info("\n" + "=" * 80)
    logger.info("✅ EXPORT COMPLETE")
    logger.info("=" * 80)
    logger.info(f"Trade dates exported: {len(dates)}")
    
    if args.layer in ['bronze', 'all']:
        logger.info(f"Bronze options records: {total_stats['bronze_options']}")
        logger.info(f"Bronze underlying records: {total_stats['bronze_underlying']}")
    
    if args.layer in ['silver', 'all']:
        logger.info(f"Silver enriched records: {total_stats['silver_enriched']}")
    
    logger.info("=" * 80)


if __name__ == '__main__':
    main()
