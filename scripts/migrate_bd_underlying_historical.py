"""
Migrate underlying prices from MariaDB option_prices_live to PostgreSQL bronze_bd_underlying

Strategy: Extract underlying price from the 2nd to last scrape per trading day
(same timing as the option prices we migrated)
"""

import mysql.connector
import psycopg2
from psycopg2.extras import execute_batch
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
    """Get dates that already exist in PostgreSQL bronze_bd_underlying"""
    conn = psycopg2.connect(**POSTGRES_CONFIG)
    cur = conn.cursor()
    cur.execute("SELECT DISTINCT trade_date FROM bronze_bd_underlying ORDER BY trade_date")
    existing_dates = set([row[0] for row in cur.fetchall()])
    cur.close()
    conn.close()
    return existing_dates


def get_underlying_prices():
    """Get underlying prices from 2nd to last scrape per day"""
    conn = mysql.connector.connect(**MARIADB_CONFIG)
    cur = conn.cursor(dictionary=True)
    
    # Get the spot price from the same scrape time we used for options (2nd to last)
    query = """
        WITH ranked_scrapes AS (
            SELECT 
                DATE(created_at) as trade_date,
                ticker,
                spot_price,
                created_at,
                ROW_NUMBER() OVER (
                    PARTITION BY DATE(created_at), ticker 
                    ORDER BY created_at DESC
                ) as scrape_rank
            FROM option_prices_live
            WHERE spot_price IS NOT NULL
        )
        SELECT 
            trade_date,
            ticker,
            spot_price as last_price,
            spot_price as bid,
            spot_price as ask,
            created_at
        FROM ranked_scrapes
        WHERE scrape_rank = 2
        ORDER BY trade_date, ticker
    """
    
    logger.info("Fetching underlying prices from MariaDB...")
    cur.execute(query)
    prices = cur.fetchall()
    cur.close()
    conn.close()
    
    return prices


def insert_underlying_prices(prices):
    """Bulk insert underlying prices into PostgreSQL"""
    if not prices:
        logger.warning("No prices to insert")
        return 0
    
    conn = psycopg2.connect(**POSTGRES_CONFIG)
    cur = conn.cursor()
    
    insert_query = """
        INSERT INTO bronze_bd_underlying (
            ticker, trade_date, last_price, bid, ask,
            source_url, scraped_at
        ) VALUES (
            %(ticker)s, %(trade_date)s, %(last_price)s, %(bid)s, %(ask)s,
            %(source_url)s, %(scraped_at)s
        )
        ON CONFLICT (ticker, trade_date, scraped_at)
        DO UPDATE SET
            last_price = EXCLUDED.last_price,
            bid = EXCLUDED.bid,
            ask = EXCLUDED.ask,
            updated_at = NOW()
    """
    
    # Transform data
    transformed = []
    for price in prices:
        transformed.append({
            'ticker': price['ticker'],
            'trade_date': price['created_at'],  # Use full timestamp for trade_date
            'last_price': float(price['last_price']),
            'bid': float(price['bid']),
            'ask': float(price['ask']),
            'source_url': 'https://www.beursduivel.be',
            'scraped_at': price['created_at']
        })
    
    execute_batch(cur, insert_query, transformed, page_size=100)
    conn.commit()
    
    rowcount = cur.rowcount
    cur.close()
    conn.close()
    
    return rowcount


def main():
    logger.info("=" * 80)
    logger.info("MIGRATE: Underlying prices from option_prices_live → bronze_bd_underlying")
    logger.info("=" * 80)
    
    # Get existing dates
    existing_dates = get_existing_dates()
    logger.info(f"Found {len(existing_dates)} dates already in PostgreSQL")
    
    # Get all underlying prices
    all_prices = get_underlying_prices()
    logger.info(f"Found {len(all_prices)} underlying price records in MariaDB")
    
    # Filter out existing dates
    missing_prices = [p for p in all_prices if p['trade_date'] not in existing_dates]
    
    if not missing_prices:
        logger.info("✅ All underlying prices already migrated!")
        return
    
    logger.info(f"Found {len(missing_prices)} records to migrate")
    
    # Show date range
    dates = sorted(set(p['trade_date'] for p in missing_prices))
    logger.info(f"\n📋 Will migrate {len(dates)} dates:")
    logger.info(f"   From: {dates[0]}")
    logger.info(f"   To: {dates[-1]}")
    
    # Insert
    logger.info("\n💾 Inserting into PostgreSQL...")
    inserted = insert_underlying_prices(missing_prices)
    logger.info(f"✅ Inserted {inserted} underlying price records")
    
    logger.info("\n" + "=" * 80)
    logger.info("✅ Migration complete!")
    logger.info("=" * 80)


if __name__ == "__main__":
    main()
