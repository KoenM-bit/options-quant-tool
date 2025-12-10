"""
Migrate historical BD options data from MariaDB to PostgreSQL Bronze layer.

Source: option_prices_live (MariaDB optionsdb)
Target: bronze_bd_options (PostgreSQL ahold_options)

Strategy:
- Extract 2nd-to-last scrape per day (rn=2) for accurate market-hours bid/ask
- Parse Dutch month names to 3rd Friday of month
- Skip underlying_price (will add bronze_bd_underlying table later)
- Migrate ~4,400 records (18 days × ~250 contracts)
"""

import os
import sys
from datetime import datetime, date
from calendar import monthcalendar, FRIDAY
import mysql.connector
import psycopg2
from psycopg2.extras import execute_batch
from dotenv import load_dotenv
import logging

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


# Dutch month name to English mapping
DUTCH_MONTHS = {
    'januari': 1, 'january': 1,
    'februari': 2, 'february': 2,
    'maart': 3, 'march': 3,
    'april': 4,
    'mei': 5, 'may': 5,
    'juni': 6, 'june': 6,
    'juli': 7, 'july': 7,
    'augustus': 8, 'august': 8,
    'september': 9,
    'oktober': 10, 'october': 10,
    'november': 11,
    'december': 12
}


def get_third_friday(year: int, month: int) -> date:
    """
    Get the 3rd Friday of a given month (standard options expiry).
    
    Args:
        year: Year (e.g., 2029)
        month: Month (1-12)
    
    Returns:
        date object for 3rd Friday
    """
    # Get calendar for the month
    cal = monthcalendar(year, month)
    
    # Find all Fridays in the month
    fridays = [week[FRIDAY] for week in cal if week[FRIDAY] != 0]
    
    # Return 3rd Friday (index 2)
    if len(fridays) >= 3:
        return date(year, month, fridays[2])
    else:
        # Fallback: last day of month (shouldn't happen)
        logger.warning(f"Could not find 3rd Friday for {year}-{month}, using last Friday")
        return date(year, month, fridays[-1])


def parse_expiry_text(expiry_text: str) -> date:
    """
    Parse Dutch expiry text to actual expiry date (3rd Friday).
    
    Examples:
        'December 2029' → 2029-12-19 (3rd Friday)
        'Juni 2026' → 2026-06-19 (3rd Friday)
        'Maart 2026' → 2026-03-21 (3rd Friday)
    
    Args:
        expiry_text: Dutch month + year string
    
    Returns:
        date object for 3rd Friday of that month
    """
    try:
        # Split and clean
        parts = expiry_text.strip().lower().split()
        
        if len(parts) != 2:
            logger.error(f"Invalid expiry format: '{expiry_text}'")
            return None
        
        month_name, year_str = parts
        
        # Map Dutch month to number
        month_num = DUTCH_MONTHS.get(month_name)
        if not month_num:
            logger.error(f"Unknown month name: '{month_name}'")
            return None
        
        year = int(year_str)
        
        # Calculate 3rd Friday
        expiry_date = get_third_friday(year, month_num)
        
        logger.debug(f"Parsed '{expiry_text}' → {expiry_date}")
        return expiry_date
        
    except Exception as e:
        logger.error(f"Error parsing expiry '{expiry_text}': {e}")
        return None


def connect_mysql():
    """Connect to MariaDB source"""
    load_dotenv('.env.migration')
    
    config = {
        "host": os.getenv("MYSQL_HOST", "192.168.1.201"),
        "user": os.getenv("MYSQL_USER", "remoteuser"),
        "password": os.getenv("MYSQL_PASSWORD"),
        "database": os.getenv("MYSQL_DATABASE", "optionsdb"),
        "port": int(os.getenv("MYSQL_PORT", 3306)),
    }
    
    return mysql.connector.connect(**config)


def connect_postgres():
    """Connect to PostgreSQL target"""
    load_dotenv('.env.migration')
    
    config = {
        "host": os.getenv("POSTGRES_HOST", "localhost"),
        "port": int(os.getenv("POSTGRES_PORT", 5432)),
        "user": os.getenv("POSTGRES_USER", "airflow"),
        "password": os.getenv("POSTGRES_PASSWORD", "airflow"),
        "database": os.getenv("POSTGRES_DB", "ahold_options"),
    }
    
    return psycopg2.connect(**config)


def fetch_second_to_last_snapshot():
    """
    Fetch 2nd-to-last scrape per day from option_prices_live.
    This gives us accurate market-hours bid/ask data (17:15-17:30).
    """
    conn = connect_mysql()
    cursor = conn.cursor(dictionary=True)
    
    logger.info("Fetching 2nd-to-last snapshot per day from MariaDB...")
    
    query = '''
    WITH all_timestamps AS (
        SELECT DISTINCT
            DATE(created_at) as trade_date,
            created_at as ts
        FROM option_prices_live
    ),
    ranked AS (
        SELECT 
            trade_date,
            ts,
            ROW_NUMBER() OVER (PARTITION BY trade_date ORDER BY ts DESC) as rn
        FROM all_timestamps
    )
    SELECT 
        opl.ticker,
        opl.issue_id,
        DATE(opl.created_at) as trade_date,
        opl.type as option_type,
        opl.expiry as expiry_text,
        opl.strike,
        opl.bid,
        opl.ask,
        opl.last_price,
        opl.volume,
        opl.last_time as last_timestamp,
        opl.created_at as scraped_at,
        opl.spot_price
    FROM ranked r
    INNER JOIN option_prices_live opl 
        ON DATE(opl.created_at) = r.trade_date 
        AND opl.created_at = r.ts
    WHERE r.rn = 2  -- 2nd to last = market hours snapshot
      AND r.trade_date < '2025-12-10'  -- Exclude 2025-12-10 (already in DB from current scraper)
    ORDER BY trade_date DESC, option_type, strike
    '''
    
    cursor.execute(query)
    rows = cursor.fetchall()
    
    logger.info(f"✅ Fetched {len(rows):,} records from MariaDB")
    
    cursor.close()
    conn.close()
    
    return rows


def transform_to_bronze_bd(mysql_row: dict) -> dict:
    """
    Transform MariaDB row to PostgreSQL bronze_bd_options format.
    """
    # Parse expiry text to actual date
    expiry_date = parse_expiry_text(mysql_row['expiry_text'])
    
    if not expiry_date:
        logger.warning(f"Skipping record with invalid expiry: {mysql_row['expiry_text']}")
        return None
    
    return {
        'ticker': mysql_row['ticker'],  # 'AD.AS'
        'symbol_code': 'AH',  # Fixed - all AD.AS
        'issue_id': str(mysql_row['issue_id']),
        'trade_date': mysql_row['trade_date'],
        'option_type': mysql_row['option_type'],  # 'Call' or 'Put'
        'expiry_date': expiry_date,  # Parsed 3rd Friday
        'expiry_text': mysql_row['expiry_text'],  # Original text
        'strike': float(mysql_row['strike']),
        'bid': float(mysql_row['bid']) if mysql_row['bid'] else None,
        'ask': float(mysql_row['ask']) if mysql_row['ask'] else None,
        'last_price': float(mysql_row['last_price']) if mysql_row['last_price'] else None,
        'volume': int(mysql_row['volume']) if mysql_row['volume'] else None,
        'last_timestamp': mysql_row['last_timestamp'],
        'last_date_text': str(mysql_row['last_timestamp']) if mysql_row['last_timestamp'] else None,
        'source': 'mariadb_migration',
        'source_url': None,  # No URLs in old data
        'scraped_at': mysql_row['scraped_at'],
        # NOTE: underlying_price (spot_price) NOT included - will add bronze_bd_underlying later
    }


def insert_to_postgres(records: list, dry_run=False):
    """
    Insert transformed records into PostgreSQL bronze_bd_options.
    """
    if dry_run:
        logger.info(f"DRY RUN: Would insert {len(records)} records")
        logger.info(f"Sample record: {records[0] if records else 'None'}")
        return 0
    
    conn = connect_postgres()
    cursor = conn.cursor()
    
    logger.info(f"Inserting {len(records)} records into bronze_bd_options...")
    
    insert_query = '''
    INSERT INTO bronze_bd_options (
        ticker, symbol_code, issue_id, trade_date,
        option_type, expiry_date, expiry_text, strike,
        bid, ask, last_price, volume,
        last_timestamp, last_date_text,
        source, source_url, scraped_at,
        created_at, updated_at
    ) VALUES (
        %(ticker)s, %(symbol_code)s, %(issue_id)s, %(trade_date)s,
        %(option_type)s, %(expiry_date)s, %(expiry_text)s, %(strike)s,
        %(bid)s, %(ask)s, %(last_price)s, %(volume)s,
        %(last_timestamp)s, %(last_date_text)s,
        %(source)s, %(source_url)s, %(scraped_at)s,
        NOW(), NOW()
    )
    '''
    
    try:
        execute_batch(cursor, insert_query, records, page_size=100)
        conn.commit()
        
        inserted_count = cursor.rowcount
        logger.info(f"✅ Inserted {inserted_count:,} records successfully")
        
        cursor.close()
        conn.close()
        
        return inserted_count
        
    except Exception as e:
        logger.error(f"❌ Insert failed: {e}")
        conn.rollback()
        cursor.close()
        conn.close()
        raise


def main():
    """Main migration workflow"""
    import argparse
    
    parser = argparse.ArgumentParser(description='Migrate historical BD options from MariaDB to PostgreSQL')
    parser.add_argument('--dry-run', action='store_true', help='Preview migration without inserting')
    parser.add_argument('--limit', type=int, help='Limit number of records (for testing)')
    
    args = parser.parse_args()
    
    logger.info("="*80)
    logger.info("MARIADB → POSTGRESQL MIGRATION")
    logger.info("="*80)
    logger.info(f"Source: option_prices_live (MariaDB)")
    logger.info(f"Target: bronze_bd_options (PostgreSQL)")
    logger.info(f"Strategy: 2nd-to-last scrape per day (rn=2)")
    logger.info(f"Dry run: {args.dry_run}")
    logger.info("="*80)
    
    try:
        # 1. Fetch data from MariaDB
        mysql_rows = fetch_second_to_last_snapshot()
        
        if args.limit:
            mysql_rows = mysql_rows[:args.limit]
            logger.info(f"⚠️  Limited to {args.limit} records for testing")
        
        # 2. Transform to PostgreSQL format
        logger.info("Transforming records...")
        transformed = []
        skipped = 0
        
        for row in mysql_rows:
            record = transform_to_bronze_bd(row)
            if record:
                transformed.append(record)
            else:
                skipped += 1
        
        logger.info(f"✅ Transformed {len(transformed):,} records (skipped {skipped} invalid)")
        
        # Show sample
        if transformed:
            logger.info("\nSample transformed record:")
            sample = transformed[0]
            for key, value in sample.items():
                logger.info(f"  {key:<20}: {value}")
        
        # 3. Insert to PostgreSQL
        if transformed:
            inserted = insert_to_postgres(transformed, dry_run=args.dry_run)
            
            logger.info("\n" + "="*80)
            logger.info("✅ MIGRATION COMPLETE")
            logger.info("="*80)
            logger.info(f"Source records: {len(mysql_rows):,}")
            logger.info(f"Transformed: {len(transformed):,}")
            logger.info(f"Inserted: {inserted:,}")
            logger.info(f"Skipped: {skipped}")
            logger.info("="*80)
        else:
            logger.warning("⚠️  No records to migrate!")
        
    except Exception as e:
        logger.error(f"❌ Migration failed: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)


if __name__ == '__main__':
    main()
