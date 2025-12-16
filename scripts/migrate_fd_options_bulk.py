"""
Bulk migration of all missing option contracts from MariaDB fd_option_contracts to PostgreSQL bronze_fd_options
"""

import os
import sys
from datetime import datetime
from typing import List, Dict, Any
import logging

import mysql.connector
import psycopg2
from psycopg2.extras import execute_batch

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


class BulkOptionsMigration:
    """Migrate all missing historical option contracts"""
    
    def __init__(self):
        self.mysql_config = {
            "host": "192.168.1.201",
            "user": "remoteuser",
            "password": "T3l3foon32#123",
            "database": "optionsdb",
            "port": 3306,
        }
        
        self.postgres_config = {
            "host": "192.168.1.201",
            "port": 5433,
            "user": "airflow",
            "password": "airflow",
            "database": "ahold_options",
        }
        
        self.mysql_conn = None
        self.postgres_conn = None
        
    def connect_mysql(self):
        try:
            self.mysql_conn = mysql.connector.connect(**self.mysql_config)
            logger.info(f"✅ Connected to MariaDB: {self.mysql_config['host']}:{self.mysql_config['port']}")
            return True
        except mysql.connector.Error as e:
            logger.error(f"❌ MariaDB connection failed: {e}")
            return False
            
    def connect_postgres(self):
        try:
            self.postgres_conn = psycopg2.connect(**self.postgres_config)
            logger.info(f"✅ Connected to PostgreSQL: {self.postgres_config['host']}:{self.postgres_config['port']}")
            return True
        except psycopg2.Error as e:
            logger.error(f"❌ PostgreSQL connection failed: {e}")
            return False
            
    def get_existing_dates(self) -> set:
        """Get dates that already exist in PostgreSQL"""
        cursor = self.postgres_conn.cursor()
        cursor.execute("""
            SELECT DISTINCT trade_date 
            FROM bronze_fd_options 
            WHERE ticker = 'AD.AS'
            ORDER BY trade_date
        """)
        existing = {row[0] for row in cursor.fetchall()}
        cursor.close()
        return existing
        
    def get_underlying_prices(self) -> Dict[str, float]:
        """Get all underlying prices from bronze_fd_overview"""
        cursor = self.postgres_conn.cursor()
        cursor.execute("""
            SELECT ticker, trade_date, koers 
            FROM bronze_fd_overview
            ORDER BY scraped_at DESC
        """)
        
        # Build dict: (ticker, trade_date) -> price
        prices = {}
        for row in cursor.fetchall():
            key = (row[0], row[1])
            if key not in prices:  # Take first (most recent)
                prices[key] = row[2]
        
        cursor.close()
        logger.info(f"📊 Loaded {len(prices)} underlying prices")
        return prices
        
    def get_missing_contracts(self) -> List[Dict[str, Any]]:
        """Get all contracts from MariaDB that don't exist in PostgreSQL"""
        existing_dates = self.get_existing_dates()
        logger.info(f"📊 Found {len(existing_dates)} existing dates in PostgreSQL")
        
        cursor = self.mysql_conn.cursor(dictionary=True)
        cursor.execute("""
            SELECT * FROM fd_option_contracts 
            ORDER BY peildatum ASC, scraped_at DESC
        """)
        all_contracts = cursor.fetchall()
        cursor.close()
        
        # Filter out existing dates
        missing_contracts = [row for row in all_contracts if row['peildatum'] not in existing_dates]
        
        logger.info(f"📊 Found {len(missing_contracts)} contracts to migrate from MariaDB")
        return missing_contracts
        
    def transform_contract(self, mysql_row: Dict[str, Any], underlying_prices: Dict) -> Dict[str, Any]:
        """Transform MySQL row to PostgreSQL format"""
        trade_date = mysql_row['peildatum']
        ticker = mysql_row['ticker']
        
        # Get underlying price from cache
        underlying_price = underlying_prices.get((ticker, trade_date))
        
        return {
            'ticker': ticker,
            'symbol_code': mysql_row['symbol_code'],
            'trade_date': trade_date,
            'scraped_at': mysql_row['scraped_at'],
            'source_url': mysql_row.get('source', 'https://www.fd.nl/beurs/opties'),
            'option_type': mysql_row['type'],  # 'Call' or 'Put'
            'expiry_date': mysql_row['expiry'],
            'strike': float(mysql_row['strike']) if mysql_row['strike'] else None,
            'naam': None,  # Not in MariaDB
            'isin': None,  # Not in MariaDB
            'laatste': float(mysql_row['last']) if mysql_row.get('last') else None,
            'bid': float(mysql_row['bid']) if mysql_row.get('bid') else None,
            'ask': float(mysql_row['ask']) if mysql_row.get('ask') else None,
            'volume': int(mysql_row['volume']) if mysql_row.get('volume') else None,
            'open_interest': int(mysql_row['open_interest']) if mysql_row.get('open_interest') else None,
            'underlying_price': underlying_price,
        }
        
    def insert_contracts(self, data: List[Dict[str, Any]]) -> int:
        """Insert contracts into PostgreSQL in batches"""
        if not data:
            return 0
            
        cursor = self.postgres_conn.cursor()
        
        query = """
        INSERT INTO bronze_fd_options 
        (created_at, updated_at, ticker, symbol_code, trade_date, scraped_at, source_url,
         option_type, expiry_date, strike, naam, isin, laatste, bid, ask, 
         volume, open_interest, underlying_price)
        VALUES (NOW(), NOW(), %(ticker)s, %(symbol_code)s, %(trade_date)s, %(scraped_at)s, %(source_url)s,
                %(option_type)s, %(expiry_date)s, %(strike)s, %(naam)s, %(isin)s, %(laatste)s, 
                %(bid)s, %(ask)s, %(volume)s, %(open_interest)s, %(underlying_price)s)
        """
        
        execute_batch(cursor, query, data, page_size=500)
        self.postgres_conn.commit()
        
        inserted = cursor.rowcount
        cursor.close()
        
        logger.info(f"✅ Inserted {inserted} contracts into bronze_fd_options")
        return inserted
        
    def verify_migration(self):
        """Verify migration was successful"""
        cursor = self.postgres_conn.cursor()
        cursor.execute("""
            SELECT 
                MIN(trade_date) as earliest,
                MAX(trade_date) as latest,
                COUNT(*) as total_contracts,
                COUNT(DISTINCT trade_date) as unique_dates,
                COUNT(DISTINCT expiry_date) as unique_expiries
            FROM bronze_fd_options 
            WHERE ticker = 'AD.AS'
        """)
        
        stats = cursor.fetchone()
        cursor.close()
        
        logger.info(f"\n📊 Final Statistics:")
        logger.info(f"   Date range: {stats[0]} to {stats[1]}")
        logger.info(f"   Total contracts: {stats[2]}")
        logger.info(f"   Unique dates: {stats[3]}")
        logger.info(f"   Unique expiries: {stats[4]}")
        
    def migrate_all(self):
        """Migrate all missing contracts"""
        logger.info(f"\n{'='*80}")
        logger.info(f"Starting BULK migration of all missing option contracts")
        logger.info(f"{'='*80}\n")
        
        # Get underlying prices cache
        logger.info("📦 Loading underlying prices...")
        underlying_prices = self.get_underlying_prices()
        
        # Get missing contracts
        missing_contracts = self.get_missing_contracts()
        if not missing_contracts:
            logger.info("✅ No missing contracts - PostgreSQL is already up to date!")
            return 0
            
        # Show summary
        dates = sorted(set(row['peildatum'] for row in missing_contracts))
        logger.info(f"\n📋 Will migrate {len(dates)} dates:")
        logger.info(f"   From: {dates[0]}")
        logger.info(f"   To: {dates[-1]}")
        logger.info(f"   Total contracts: {len(missing_contracts)}")
        
        # Transform data
        logger.info(f"\n🔄 Transforming {len(missing_contracts)} contracts...")
        transformed_data = []
        for i, row in enumerate(missing_contracts, 1):
            if i % 1000 == 0:
                logger.info(f"   Transformed {i}/{len(missing_contracts)} contracts...")
            transformed_data.append(self.transform_contract(row, underlying_prices))
        
        logger.info(f"✅ Transformed all {len(transformed_data)} contracts")
        
        # Insert into PostgreSQL
        logger.info(f"\n💾 Inserting into PostgreSQL in batches...")
        inserted = self.insert_contracts(transformed_data)
        
        # Verify
        logger.info(f"\n🔍 Verifying migration...")
        self.verify_migration()
        
        logger.info(f"\n{'='*80}")
        logger.info(f"✅ Bulk migration complete: {inserted} contracts inserted across {len(dates)} dates")
        logger.info(f"{'='*80}\n")
        
        return inserted
        
    def close_connections(self):
        """Close database connections"""
        if self.mysql_conn:
            self.mysql_conn.close()
            logger.info("Closed MariaDB connection")
        if self.postgres_conn:
            self.postgres_conn.close()
            logger.info("Closed PostgreSQL connection")


def main():
    """Run bulk migration"""
    migration = BulkOptionsMigration()
    
    try:
        if not migration.connect_mysql():
            return 1
        if not migration.connect_postgres():
            return 1
            
        migration.migrate_all()
        
    except Exception as e:
        logger.error(f"❌ Error: {e}", exc_info=True)
        return 1
    finally:
        migration.close_connections()
    
    return 0


if __name__ == "__main__":
    sys.exit(main())
