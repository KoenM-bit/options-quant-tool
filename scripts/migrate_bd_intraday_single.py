"""
Test migration: Single record from MariaDB option_prices_live to PostgreSQL bronze_bd_options

Strategy: Select the 2nd to last scrape for one contract on 2025-12-11
"""

import mysql.connector
import psycopg2
from datetime import datetime, timedelta
import logging
from dateutil import parser

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

# Dutch month mapping for expiry conversion
DUTCH_MONTHS = {
    'januari': 'january',
    'februari': 'february',
    'maart': 'march',
    'april': 'april',
    'mei': 'may',
    'juni': 'june',
    'juli': 'july',
    'augustus': 'august',
    'september': 'september',
    'oktober': 'october',
    'november': 'november',
    'december': 'december'
}


def convert_expiry_to_date(expiry_text):
    """
    Convert Dutch expiry text to date.
    Format: "December 2025" or "Januari 2026"
    Result: 3rd Friday of that month (options expiry date)
    """
    expiry_lower = expiry_text.lower().strip()
    
    # Replace Dutch month with English
    for dutch, english in DUTCH_MONTHS.items():
        if dutch in expiry_lower:
            expiry_lower = expiry_lower.replace(dutch, english)
            break
    
    # Parse "month year" format
    try:
        # Parse to first day of month
        dt = parser.parse(expiry_lower, dayfirst=False)
        
        # Calculate 3rd Friday of the month
        # Find first day of month and its weekday
        first_day = dt.replace(day=1)
        
        # Calculate days until first Friday
        # weekday(): Monday=0, Friday=4
        days_until_friday = (4 - first_day.weekday()) % 7
        
        # First Friday
        first_friday = first_day + timedelta(days=days_until_friday)
        
        # Third Friday (2 weeks after first Friday)
        third_friday = first_friday + timedelta(weeks=2)
        
        return third_friday.date()
    except Exception as e:
        logger.error(f"Failed to parse expiry '{expiry_text}': {e}")
        return None


def get_single_test_record():
    """Get 2nd to last scrape for one contract on 2025-12-11"""
    conn = mysql.connector.connect(**MARIADB_CONFIG)
    cur = conn.cursor(dictionary=True)
    
    cur.execute("""
        WITH ranked_scrapes AS (
            SELECT 
                *,
                ROW_NUMBER() OVER (
                    PARTITION BY DATE(created_at), ticker, type, strike, expiry 
                    ORDER BY created_at DESC
                ) as scrape_rank
            FROM option_prices_live
            WHERE DATE(created_at) = '2025-12-11'
              AND ticker = 'AD.AS'
              AND type = 'Put'
              AND strike = 34.0
              AND expiry = 'December 2025'
        )
        SELECT * 
        FROM ranked_scrapes
        WHERE scrape_rank = 2
        LIMIT 1
    """)
    
    record = cur.fetchone()
    cur.close()
    conn.close()
    
    return record


def transform_record(maria_record):
    """Transform MariaDB record to PostgreSQL format"""
    
    # Convert expiry text to date
    expiry_date = convert_expiry_to_date(maria_record['expiry'])
    
    # Map ticker to symbol_code
    ticker_to_symbol = {
        'AD.AS': 'AH',
        'MT.AS': 'MT'
    }
    symbol_code = ticker_to_symbol.get(maria_record['ticker'], maria_record['ticker'])
    
    # Format expiry_text with (AEX / symbol) suffix
    expiry_text = f"{maria_record['expiry']} (AEX / {symbol_code})"
    
    pg_record = {
        'ticker': maria_record['ticker'],
        'symbol_code': symbol_code,
        'issue_id': maria_record['issue_id'],
        'trade_date': maria_record['created_at'].date(),
        'option_type': maria_record['type'],
        'expiry_date': expiry_date,
        'expiry_text': expiry_text,
        'strike': float(maria_record['strike']),
        'bid': float(maria_record['bid']) if maria_record['bid'] else None,
        'ask': float(maria_record['ask']) if maria_record['ask'] else None,
        'last_price': float(maria_record['last_price']) if maria_record['last_price'] else None,
        'volume': int(maria_record['volume']) if maria_record['volume'] else None,
        'last_timestamp': maria_record.get('last_time'),
        'last_date_text': None,  # Not in MariaDB
        'source': 'beursduivel',
        'source_url': 'https://www.beursduivel.be',
        'scraped_at': maria_record['created_at'],
    }
    
    return pg_record


def insert_record(pg_record):
    """Insert record into PostgreSQL"""
    conn = psycopg2.connect(**POSTGRES_CONFIG)
    cur = conn.cursor()
    
    insert_query = """
        INSERT INTO bronze_bd_options (
            ticker, symbol_code, issue_id, trade_date, option_type,
            expiry_date, expiry_text, strike, bid, ask, last_price,
            volume, last_timestamp, last_date_text, source, source_url, scraped_at
        ) VALUES (
            %(ticker)s, %(symbol_code)s, %(issue_id)s, %(trade_date)s, %(option_type)s,
            %(expiry_date)s, %(expiry_text)s, %(strike)s, %(bid)s, %(ask)s, %(last_price)s,
            %(volume)s, %(last_timestamp)s, %(last_date_text)s, %(source)s, %(source_url)s, %(scraped_at)s
        )
        ON CONFLICT (ticker, trade_date, option_type, strike, expiry_date)
        DO UPDATE SET
            bid = EXCLUDED.bid,
            ask = EXCLUDED.ask,
            last_price = EXCLUDED.last_price,
            volume = EXCLUDED.volume,
            scraped_at = EXCLUDED.scraped_at,
            updated_at = NOW()
    """
    
    cur.execute(insert_query, pg_record)
    conn.commit()
    
    rowcount = cur.rowcount
    cur.close()
    conn.close()
    
    return rowcount


def verify_insert():
    """Verify the inserted record"""
    conn = psycopg2.connect(**POSTGRES_CONFIG)
    cur = conn.cursor()
    
    cur.execute("""
        SELECT ticker, option_type, strike, expiry_date, expiry_text,
               bid, ask, last_price, volume, trade_date, scraped_at
        FROM bronze_bd_options
        WHERE trade_date = '2025-12-11'
          AND ticker = 'AD.AS'
          AND option_type = 'Put'
          AND strike = 34.0
        ORDER BY scraped_at DESC
        LIMIT 1
    """)
    
    record = cur.fetchone()
    cur.close()
    conn.close()
    
    return record


def main():
    logger.info("=" * 80)
    logger.info("TEST MIGRATION: Single record from option_prices_live → bronze_bd_options")
    logger.info("=" * 80)
    
    # Get test record
    logger.info("\n📊 Fetching test record from MariaDB...")
    maria_record = get_single_test_record()
    
    if not maria_record:
        logger.error("❌ No test record found!")
        return
    
    logger.info(f"✅ Found test record:")
    logger.info(f"   Ticker: {maria_record['ticker']}")
    logger.info(f"   Type: {maria_record['type']}")
    logger.info(f"   Strike: {maria_record['strike']}")
    logger.info(f"   Expiry: {maria_record['expiry']}")
    logger.info(f"   Trade Date: {maria_record['created_at'].date()}")
    logger.info(f"   Scraped At: {maria_record['created_at']}")
    logger.info(f"   Bid: {maria_record['bid']}")
    logger.info(f"   Ask: {maria_record['ask']}")
    logger.info(f"   Last Price: {maria_record['last_price']}")
    logger.info(f"   Volume: {maria_record['volume']}")
    logger.info(f"   Issue ID: {maria_record['issue_id']}")
    
    # Transform record
    logger.info("\n🔄 Transforming record...")
    pg_record = transform_record(maria_record)
    
    logger.info(f"✅ Transformed:")
    logger.info(f"   Expiry Date: {pg_record['expiry_date']}")
    logger.info(f"   Symbol Code: {pg_record['symbol_code']}")
    
    # Insert record
    logger.info("\n💾 Inserting into PostgreSQL...")
    rowcount = insert_record(pg_record)
    
    if rowcount > 0:
        logger.info(f"✅ Inserted {rowcount} record(s)")
    else:
        logger.warning("⚠️  No rows inserted (might already exist)")
    
    # Verify insert
    logger.info("\n🔍 Verifying insert...")
    verified = verify_insert()
    
    if verified:
        logger.info("✅ Record verified in PostgreSQL:")
        logger.info(f"   Ticker: {verified[0]}")
        logger.info(f"   Type: {verified[1]}")
        logger.info(f"   Strike: {verified[2]}")
        logger.info(f"   Expiry Date: {verified[3]}")
        logger.info(f"   Expiry Text: {verified[4]}")
        logger.info(f"   Bid: {verified[5]}")
        logger.info(f"   Ask: {verified[6]}")
        logger.info(f"   Last Price: {verified[7]}")
        logger.info(f"   Volume: {verified[8]}")
        logger.info(f"   Trade Date: {verified[9]}")
        logger.info(f"   Scraped At: {verified[10]}")
    else:
        logger.error("❌ Failed to verify record")
    
    logger.info("\n" + "=" * 80)
    logger.info("✅ Test migration complete!")
    logger.info("=" * 80)


if __name__ == "__main__":
    main()
