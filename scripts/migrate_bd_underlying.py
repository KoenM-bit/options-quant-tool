#!/usr/bin/env python3
"""
Migrate historical BD underlying prices from MariaDB to PostgreSQL.

Strategy:
- Extract one record per trading day (rn=2, same as options)
- Map spot_price to last_price in bronze_bd_underlying
- ticker='AD.AS', isin='NL0011794037', name='Ahold Delhaize'
- Skip 2025-12-10 (already in DB from current scraper)
"""

import argparse
import logging
import os
from datetime import datetime
from typing import List, Dict, Optional

import mysql.connector
import psycopg2
from psycopg2.extras import execute_batch
from dotenv import load_dotenv

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


def connect_mysql() -> mysql.connector.MySQLConnection:
    """Connect to MariaDB source database."""
    load_dotenv('.env.migration')
    
    config = {
        "host": os.getenv("MYSQL_HOST", "192.168.1.201"),
        "port": int(os.getenv("MYSQL_PORT", 3306)),
        "user": os.getenv("MYSQL_USER", "remoteuser"),
        "password": os.getenv("MYSQL_PASSWORD"),
        "database": os.getenv("MYSQL_DATABASE", "optionsdb"),
    }
    
    return mysql.connector.connect(**config)


def connect_postgres() -> psycopg2.extensions.connection:
    """Connect to PostgreSQL target database."""
    load_dotenv('.env.migration')
    
    config = {
        "host": os.getenv("POSTGRES_HOST", "localhost"),
        "port": int(os.getenv("POSTGRES_PORT", 5432)),
        "user": os.getenv("POSTGRES_USER", "airflow"),
        "password": os.getenv("POSTGRES_PASSWORD", "airflow"),
        "database": os.getenv("POSTGRES_DB", "ahold_options"),
    }
    
    return psycopg2.connect(**config)


def fetch_underlying_prices():
    """
    Fetch one underlying price per trading day from MariaDB.
    Uses rn=2 (2nd-to-last scrape) same as options data.
    """
    conn = connect_mysql()
    cursor = conn.cursor(dictionary=True)
    
    query = """
    WITH ranked_scrapes AS (
        SELECT 
            ticker,
            spot_price as underlying_price,
            DATE(created_at) as trade_date,
            created_at as scrape_time,
            ROW_NUMBER() OVER (PARTITION BY DATE(created_at) ORDER BY created_at DESC) as rn
        FROM option_prices_live
        WHERE spot_price IS NOT NULL
          AND ticker = 'AD.AS'
    )
    SELECT DISTINCT
        ticker,
        underlying_price,
        trade_date,
        scrape_time
    FROM ranked_scrapes r
    WHERE r.rn = 2
      AND r.trade_date < '2025-12-10'  -- Skip today, already in DB
    ORDER BY trade_date
    """
    
    cursor.execute(query)
    rows = cursor.fetchall()
    
    cursor.close()
    conn.close()
    
    return rows


def transform_to_bronze_underlying(mysql_row: dict) -> dict:
    """
    Transform MariaDB row to PostgreSQL bronze_bd_underlying format.
    """
    return {
        'ticker': mysql_row['ticker'],  # 'AD.AS'
        'trade_date': mysql_row['scrape_time'],  # Full timestamp for trade_date column
        'isin': 'NL0011794037',  # Ahold Delhaize ISIN
        'name': 'Ahold Delhaize',  # Company name
        'last_price': float(mysql_row['underlying_price']),
        'bid': None,  # Not available in MariaDB
        'ask': None,  # Not available in MariaDB
        'volume': None,  # Not available in MariaDB
        'last_timestamp_text': None,  # Not available in MariaDB
        'scraped_at': mysql_row['scrape_time'],
        'source_url': 'mariadb_migration',  # Using source_url to mark migration
    }


def insert_to_postgres(records: List[dict], dry_run: bool = False) -> int:
    """
    Insert transformed records into PostgreSQL bronze_bd_underlying table.
    """
    if dry_run:
        logger.info("DRY RUN - would insert records:")
        for i, rec in enumerate(records[:3]):  # Show first 3
            logger.info(f"  Record {i+1}: {rec}")
        return 0
    
    conn = connect_postgres()
    cursor = conn.cursor()
    
    insert_query = """
    INSERT INTO bronze_bd_underlying (
        ticker, trade_date, isin, name,
        last_price, bid, ask, volume,
        last_timestamp_text, scraped_at, source_url
    ) VALUES (
        %(ticker)s, %(trade_date)s, %(isin)s, %(name)s,
        %(last_price)s, %(bid)s, %(ask)s, %(volume)s,
        %(last_timestamp_text)s, %(scraped_at)s, %(source_url)s
    )
    """
    
    try:
        execute_batch(cursor, insert_query, records, page_size=100)
        conn.commit()
        inserted = cursor.rowcount
        cursor.close()
        conn.close()
        return inserted
    except Exception as e:
        logger.error(f"❌ Insert failed: {e}")
        conn.rollback()
        cursor.close()
        conn.close()
        raise


def main():
    parser = argparse.ArgumentParser(
        description='Migrate BD underlying prices from MariaDB to PostgreSQL'
    )
    parser.add_argument('--dry-run', action='store_true',
                       help='Preview records without inserting')
    parser.add_argument('--limit', type=int,
                       help='Limit number of records to migrate (for testing)')
    
    args = parser.parse_args()
    
    logger.info("=" * 80)
    logger.info("MARIADB → POSTGRESQL UNDERLYING MIGRATION")
    logger.info("=" * 80)
    logger.info(f"Source: option_prices_live.spot_price (MariaDB)")
    logger.info(f"Target: bronze_bd_underlying (PostgreSQL)")
    logger.info(f"Strategy: 2nd-to-last scrape per day (rn=2)")
    logger.info(f"Dry run: {args.dry_run}")
    logger.info("=" * 80)
    
    try:
        # Step 1: Fetch from MariaDB
        logger.info("Fetching underlying prices from MariaDB...")
        mysql_rows = fetch_underlying_prices()
        logger.info(f"✅ Fetched {len(mysql_rows)} records from MariaDB")
        
        if args.limit:
            mysql_rows = mysql_rows[:args.limit]
            logger.info(f"⚠️  Limited to {args.limit} records for testing")
        
        # Step 2: Transform
        logger.info("Transforming records...")
        transformed = []
        for row in mysql_rows:
            transformed_row = transform_to_bronze_underlying(row)
            if transformed_row:
                transformed.append(transformed_row)
        
        logger.info(f"✅ Transformed {len(transformed)} records")
        
        # Show sample
        if transformed:
            logger.info("\nSample transformed record:")
            sample = transformed[0]
            for key, value in sample.items():
                logger.info(f"  {key:20s}: {value}")
        
        # Step 3: Insert
        logger.info(f"\nInserting {len(transformed)} records into bronze_bd_underlying...")
        inserted = insert_to_postgres(transformed, dry_run=args.dry_run)
        
        if not args.dry_run:
            logger.info(f"✅ Inserted {inserted} records successfully")
        
        # Summary
        logger.info("\n" + "=" * 80)
        logger.info("✅ MIGRATION COMPLETE")
        logger.info("=" * 80)
        logger.info(f"Source records: {len(mysql_rows)}")
        logger.info(f"Transformed: {len(transformed)}")
        logger.info(f"Inserted: {inserted}")
        logger.info("=" * 80)
        
    except Exception as e:
        logger.error(f"❌ Migration failed: {e}")
        raise


if __name__ == '__main__':
    main()
