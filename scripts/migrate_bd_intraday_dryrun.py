"""
DRY RUN: Preview migration from MariaDB option_prices_live to PostgreSQL bronze_bd_options

Strategy: Select the "2nd to last" scrape per contract per trading day
Rationale: Last scrape might have incomplete bid/ask data, 2nd to last ensures we have complete data

This script will:
1. Show which dates exist in MariaDB option_prices_live but not in PostgreSQL bronze_bd_options
2. Show sample records that would be migrated (2nd to last scrape per contract)
3. Show column mapping
4. Show total records to migrate
"""

import mysql.connector
import psycopg2
from datetime import datetime
import logging

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# MariaDB connection
MARIADB_CONFIG = {
    'host': '192.168.1.201',
    'port': 3306,
    'user': 'remoteuser',
    'password': 'T3l3foon32#123',
    'database': 'optionsdb'
}

# PostgreSQL connection
POSTGRES_CONFIG = {
    'host': '192.168.1.201',
    'port': 5433,
    'user': 'airflow',
    'password': 'airflow',
    'database': 'ahold_options'
}


def get_existing_dates():
    """Get dates that already exist in PostgreSQL bronze_bd_options"""
    conn = psycopg2.connect(**POSTGRES_CONFIG)
    cur = conn.cursor()
    
    cur.execute("""
        SELECT DISTINCT trade_date 
        FROM bronze_bd_options 
        ORDER BY trade_date
    """)
    
    existing_dates = [row[0] for row in cur.fetchall()]
    cur.close()
    conn.close()
    
    return existing_dates


def get_available_dates():
    """Get all dates from MariaDB option_prices_live"""
    conn = mysql.connector.connect(**MARIADB_CONFIG)
    cur = conn.cursor()
    
    cur.execute("""
        SELECT DATE(created_at) as trade_date, 
               COUNT(*) as total_scrapes,
               COUNT(DISTINCT CONCAT(ticker, type, strike, expiry)) as unique_contracts,
               MIN(created_at) as first_scrape,
               MAX(created_at) as last_scrape
        FROM option_prices_live
        GROUP BY DATE(created_at)
        ORDER BY trade_date DESC
    """)
    
    dates = cur.fetchall()
    cur.close()
    conn.close()
    
    return dates


def show_second_to_last_logic():
    """Show how the 2nd to last scrape selection works"""
    conn = mysql.connector.connect(**MARIADB_CONFIG)
    cur = conn.cursor(dictionary=True)
    
    # Example: show all scrapes for one contract on 2025-12-12
    cur.execute("""
        SELECT 
            ticker,
            type,
            strike,
            expiry,
            created_at,
            last_price,
            bid,
            ask,
            volume,
            ROW_NUMBER() OVER (ORDER BY created_at DESC) as scrape_rank
        FROM option_prices_live
        WHERE DATE(created_at) = '2025-12-12'
          AND ticker = 'AD.AS'
          AND type = 'Call'
          AND strike = 34.0
          AND expiry = 'December 2025'
        ORDER BY created_at DESC
        LIMIT 5
    """)
    
    scrapes = cur.fetchall()
    cur.close()
    conn.close()
    
    return scrapes


def get_sample_migration_data():
    """Get sample records that would be migrated using 2nd to last logic"""
    conn = mysql.connector.connect(**MARIADB_CONFIG)
    cur = conn.cursor(dictionary=True)
    
    # Query to get 2nd to last scrape per contract per day
    cur.execute("""
        WITH ranked_scrapes AS (
            SELECT 
                *,
                ROW_NUMBER() OVER (
                    PARTITION BY DATE(created_at), ticker, type, strike, expiry 
                    ORDER BY created_at DESC
                ) as scrape_rank
            FROM option_prices_live
            WHERE DATE(created_at) = '2025-12-12'
        )
        SELECT 
            ticker,
            type as option_type,
            strike,
            expiry as expiry_text,
            last_price,
            bid,
            ask,
            volume,
            issue_id,
            created_at,
            spot_price as underlying_price,
            scrape_rank
        FROM ranked_scrapes
        WHERE scrape_rank = 2  -- 2nd to last scrape
        ORDER BY ticker, option_type, strike
        LIMIT 10
    """)
    
    samples = cur.fetchall()
    cur.close()
    conn.close()
    
    return samples


def main():
    logger.info("=" * 80)
    logger.info("DRY RUN: MariaDB option_prices_live → PostgreSQL bronze_bd_options")
    logger.info("=" * 80)
    
    # Get existing dates
    logger.info("\n📊 Checking existing dates in PostgreSQL...")
    existing_dates = get_existing_dates()
    logger.info(f"Found {len(existing_dates)} dates already in bronze_bd_options")
    if existing_dates:
        logger.info(f"   From: {min(existing_dates)}")
        logger.info(f"   To: {max(existing_dates)}")
    
    # Get available dates from MariaDB
    logger.info("\n📊 Checking available dates in MariaDB...")
    available_dates = get_available_dates()
    logger.info(f"Found {len(available_dates)} dates in option_prices_live")
    
    # Show missing dates
    missing_dates = []
    for date_info in available_dates:
        trade_date = date_info[0]
        if trade_date not in existing_dates:
            missing_dates.append(date_info)
    
    logger.info(f"\n📋 Found {len(missing_dates)} dates to migrate:")
    logger.info(f"   From: {missing_dates[-1][0] if missing_dates else 'N/A'}")
    logger.info(f"   To: {missing_dates[0][0] if missing_dates else 'N/A'}")
    
    # Show first 10 missing dates
    logger.info("\n📅 First 10 dates to migrate:")
    logger.info(f"{'Date':<15} {'Total Scrapes':<15} {'Unique Contracts':<20} {'First Scrape':<25} {'Last Scrape':<25}")
    logger.info("-" * 100)
    for date_info in missing_dates[:10]:
        logger.info(f"{date_info[0]!s:<15} {date_info[1]:<15} {date_info[2]:<20} {date_info[3]!s:<25} {date_info[4]!s:<25}")
    
    # Show the 2nd to last scrape logic
    logger.info("\n" + "=" * 80)
    logger.info("📊 EXAMPLE: 2nd to Last Scrape Selection Logic")
    logger.info("=" * 80)
    logger.info("For contract: AD.AS Call Strike 34.0 December 2025 on 2025-12-12")
    logger.info("Showing last 5 scrapes (we select rank=2):")
    logger.info("")
    
    scrapes = show_second_to_last_logic()
    if scrapes:
        logger.info(f"{'Rank':<6} {'Time':<20} {'Last':<10} {'Bid':<10} {'Ask':<10} {'Volume':<10}")
        logger.info("-" * 66)
        for scrape in scrapes:
            logger.info(f"{scrape.get('scrape_rank', 'N/A'):<6} "
                       f"{str(scrape['created_at']):<20} "
                       f"{scrape['last_price'] or 'NULL':<10} "
                       f"{scrape['bid'] or 'NULL':<10} "
                       f"{scrape['ask'] or 'NULL':<10} "
                       f"{scrape['volume'] or 'NULL':<10}")
        logger.info("\n➡️  We select RANK=2 (2nd to last) to ensure complete bid/ask data")
    
    # Show sample migration records
    logger.info("\n" + "=" * 80)
    logger.info("📋 SAMPLE: Records that would be migrated (2nd to last per contract)")
    logger.info("=" * 80)
    
    samples = get_sample_migration_data()
    if samples:
        logger.info(f"Showing 10 sample records from 2025-12-12:")
        logger.info("")
        logger.info(f"{'Ticker':<8} {'Type':<6} {'Strike':<8} {'Last':<8} {'Bid':<8} {'Ask':<8} {'Volume':<8} {'Rank':<6}")
        logger.info("-" * 58)
        for sample in samples:
            logger.info(f"{sample['ticker']:<8} "
                       f"{sample['option_type']:<6} "
                       f"{sample['strike']:<8.1f} "
                       f"{sample['last_price'] or 'NULL':<8} "
                       f"{sample['bid'] or 'NULL':<8} "
                       f"{sample['ask'] or 'NULL':<8} "
                       f"{sample['volume'] or 'NULL':<8} "
                       f"{sample['scrape_rank']:<6}")
    
    # Show column mapping
    logger.info("\n" + "=" * 80)
    logger.info("📋 COLUMN MAPPING")
    logger.info("=" * 80)
    logger.info(f"{'MariaDB (option_prices_live)':<40} → {'PostgreSQL (bronze_bd_options)':<40}")
    logger.info("-" * 80)
    logger.info(f"{'ticker':<40} → {'ticker':<40}")
    logger.info(f"{'type (Call/Put)':<40} → {'option_type':<40}")
    logger.info(f"{'strike':<40} → {'strike':<40}")
    logger.info(f"{'expiry (text)':<40} → {'expiry_text':<40}")
    logger.info(f"{'(convert expiry text to date)':<40} → {'expiry_date':<40}")
    logger.info(f"{'last_price':<40} → {'last_price':<40}")
    logger.info(f"{'bid':<40} → {'bid':<40}")
    logger.info(f"{'ask':<40} → {'ask':<40}")
    logger.info(f"{'volume':<40} → {'volume':<40}")
    logger.info(f"{'issue_id':<40} → {'issue_id':<40}")
    logger.info(f"{'(generate from issue_id)':<40} → {'symbol_code':<40}")
    logger.info(f"{'DATE(created_at)':<40} → {'trade_date':<40}")
    logger.info(f"{'created_at (2nd to last)':<40} → {'scraped_at':<40}")
    logger.info(f"{'spot_price':<40} → {'(not mapped - use underlying)':<40}")
    logger.info(f"{'(static: beursduivel)':<40} → {'source':<40}")
    logger.info(f"{'(static: beursduivel.be)':<40} → {'source_url':<40}")
    
    # Show total migration estimate
    logger.info("\n" + "=" * 80)
    logger.info("📊 MIGRATION ESTIMATE")
    logger.info("=" * 80)
    
    total_contracts = sum(date_info[2] for date_info in missing_dates)
    logger.info(f"Dates to migrate: {len(missing_dates)}")
    logger.info(f"Total contracts to migrate: ~{total_contracts}")
    logger.info(f"(One record per contract per day, using 2nd to last scrape)")
    
    logger.info("\n" + "=" * 80)
    logger.info("✅ Dry run complete!")
    logger.info("=" * 80)


if __name__ == "__main__":
    main()
