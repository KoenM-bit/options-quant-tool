"""
BULK MIGRATION: All missing BD intraday data from MariaDB option_prices_live to PostgreSQL bronze_bd_options

Strategy: Select 2nd to last scrape per contract per trading day
Rationale: Ensures complete bid/ask data while avoiding potential incomplete final scrape

This script will:
1. Get all dates that exist in MariaDB but not in PostgreSQL
2. For each date, select 2nd to last scrape per contract
3. Bulk insert using batch operations
4. Verify migration results
"""

import mysql.connector
import psycopg2
from psycopg2.extras import execute_batch
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

# Dutch month mapping
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


class BulkBDMigration:
    def __init__(self):
        self.maria_conn = None
        self.pg_conn = None
        
    def connect(self):
        """Establish database connections"""
        self.maria_conn = mysql.connector.connect(**MARIADB_CONFIG)
        self.pg_conn = psycopg2.connect(**POSTGRES_CONFIG)
        logger.info(f"✅ Connected to MariaDB: {MARIADB_CONFIG['host']}:{MARIADB_CONFIG['port']}")
        logger.info(f"✅ Connected to PostgreSQL: {POSTGRES_CONFIG['host']}:{POSTGRES_CONFIG['port']}")
    
    def close(self):
        """Close database connections"""
        if self.maria_conn:
            self.maria_conn.close()
            logger.info("Closed MariaDB connection")
        if self.pg_conn:
            self.pg_conn.close()
            logger.info("Closed PostgreSQL connection")
    
    def convert_expiry_to_date(self, expiry_text):
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
    
    def get_existing_dates(self):
        """Get dates that already exist in PostgreSQL"""
        cur = self.pg_conn.cursor()
        cur.execute("SELECT DISTINCT trade_date FROM bronze_bd_options ORDER BY trade_date")
        existing_dates = [row[0] for row in cur.fetchall()]
        cur.close()
        return set(existing_dates)
    
    def get_missing_contracts(self):
        """Get all 2nd to last scrapes for dates missing in PostgreSQL"""
        existing_dates = self.get_existing_dates()
        logger.info(f"📊 Found {len(existing_dates)} existing dates in PostgreSQL")
        
        cur = self.maria_conn.cursor(dictionary=True)
        
        # Get all 2nd to last scrapes using window function
        query = """
            WITH ranked_scrapes AS (
                SELECT 
                    ticker,
                    type,
                    strike,
                    expiry,
                    last_price,
                    bid,
                    ask,
                    volume,
                    issue_id,
                    created_at,
                    DATE(created_at) as trade_date,
                    ROW_NUMBER() OVER (
                        PARTITION BY DATE(created_at), ticker, type, strike, expiry 
                        ORDER BY created_at DESC
                    ) as scrape_rank
                FROM option_prices_live
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
                trade_date
            FROM ranked_scrapes
            WHERE scrape_rank = 2
            ORDER BY trade_date, ticker, option_type, strike
        """
        
        logger.info("📊 Fetching all 2nd to last scrapes from MariaDB...")
        cur.execute(query)
        all_contracts = cur.fetchall()
        cur.close()
        
        # Filter out dates that already exist in PostgreSQL
        missing_contracts = [
            contract for contract in all_contracts 
            if contract['trade_date'] not in existing_dates
        ]
        
        logger.info(f"📊 Found {len(all_contracts)} total contracts in MariaDB")
        logger.info(f"📊 Found {len(missing_contracts)} contracts to migrate")
        
        return missing_contracts
    
    def transform_contract(self, maria_contract):
        """Transform MariaDB contract to PostgreSQL format"""
        expiry_date = self.convert_expiry_to_date(maria_contract['expiry_text'])
        
        # Map ticker to symbol_code
        ticker_to_symbol = {
            'AD.AS': 'AH',
            'MT.AS': 'MT'
        }
        symbol_code = ticker_to_symbol.get(maria_contract['ticker'], maria_contract['ticker'])
        
        # Format expiry_text with (AEX / symbol) suffix
        expiry_text = f"{maria_contract['expiry_text']} (AEX / {symbol_code})"
        
        return {
            'ticker': maria_contract['ticker'],
            'symbol_code': symbol_code,
            'issue_id': maria_contract['issue_id'],
            'trade_date': maria_contract['trade_date'],
            'option_type': maria_contract['option_type'],
            'expiry_date': expiry_date,
            'expiry_text': expiry_text,
            'strike': float(maria_contract['strike']),
            'bid': float(maria_contract['bid']) if maria_contract['bid'] else None,
            'ask': float(maria_contract['ask']) if maria_contract['ask'] else None,
            'last_price': float(maria_contract['last_price']) if maria_contract['last_price'] else None,
            'volume': int(maria_contract['volume']) if maria_contract['volume'] else None,
            'last_timestamp': None,
            'last_date_text': None,
            'source': 'beursduivel',
            'source_url': 'https://www.beursduivel.be',
            'scraped_at': maria_contract['created_at'],
        }
    
    def insert_contracts(self, contracts):
        """Bulk insert contracts into PostgreSQL"""
        if not contracts:
            logger.warning("⚠️  No contracts to insert")
            return 0
        
        cur = self.pg_conn.cursor()
        
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
        
        # Execute in batches
        execute_batch(cur, insert_query, contracts, page_size=500)
        self.pg_conn.commit()
        
        rowcount = cur.rowcount
        cur.close()
        
        return rowcount
    
    def verify_migration(self):
        """Verify the migration results"""
        cur = self.pg_conn.cursor()
        
        cur.execute("""
            SELECT 
                COUNT(*) as total_records,
                COUNT(DISTINCT trade_date) as unique_dates,
                MIN(trade_date) as min_date,
                MAX(trade_date) as max_date,
                COUNT(DISTINCT expiry_date) as unique_expiries
            FROM bronze_bd_options
        """)
        
        stats = cur.fetchone()
        cur.close()
        
        return stats
    
    def migrate(self):
        """Execute the full migration"""
        try:
            self.connect()
            
            logger.info("=" * 80)
            logger.info("Starting BULK migration of all missing BD intraday data")
            logger.info("=" * 80)
            
            # Get missing contracts
            missing_contracts = self.get_missing_contracts()
            
            if not missing_contracts:
                logger.info("✅ No contracts to migrate - all data is up to date!")
                return
            
            # Show date range
            dates = sorted(set(c['trade_date'] for c in missing_contracts))
            logger.info(f"\n📋 Will migrate {len(dates)} dates:")
            logger.info(f"   From: {dates[0]}")
            logger.info(f"   To: {dates[-1]}")
            logger.info(f"   Total contracts: {len(missing_contracts)}")
            
            # Transform all contracts
            logger.info(f"\n🔄 Transforming {len(missing_contracts)} contracts...")
            transformed = []
            for i, contract in enumerate(missing_contracts, 1):
                if i % 1000 == 0:
                    logger.info(f"   Transformed {i}/{len(missing_contracts)} contracts...")
                transformed.append(self.transform_contract(contract))
            
            logger.info(f"✅ Transformed all {len(transformed)} contracts")
            
            # Insert into PostgreSQL
            logger.info("\n💾 Inserting into PostgreSQL in batches...")
            inserted = self.insert_contracts(transformed)
            logger.info(f"✅ Inserted {inserted} contracts into bronze_bd_options")
            
            # Verify migration
            logger.info("\n🔍 Verifying migration...")
            stats = self.verify_migration()
            
            logger.info("\n📊 Final Statistics:")
            logger.info(f"   Date range: {stats[2]} to {stats[3]}")
            logger.info(f"   Total contracts: {stats[0]}")
            logger.info(f"   Unique dates: {stats[1]}")
            logger.info(f"   Unique expiries: {stats[4]}")
            
            logger.info("\n" + "=" * 80)
            logger.info(f"✅ Bulk migration complete: {inserted} contracts inserted across {len(dates)} dates")
            logger.info("=" * 80)
            
        finally:
            self.close()


def main():
    migration = BulkBDMigration()
    migration.migrate()


if __name__ == "__main__":
    main()
