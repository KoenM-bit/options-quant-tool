"""
Migrate a single option contract from MariaDB fd_option_contracts to PostgreSQL bronze_fd_options

This migrates just ONE row for testing.
"""

import os
import sys
from datetime import datetime
from typing import Dict, Any
import logging

import mysql.connector
import psycopg2

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


class SingleContractMigration:
    """Migrate a single option contract for testing"""
    
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
            logger.info(f"✅ Connected to MariaDB")
            return True
        except mysql.connector.Error as e:
            logger.error(f"❌ MariaDB connection failed: {e}")
            return False
            
    def connect_postgres(self):
        try:
            self.postgres_conn = psycopg2.connect(**self.postgres_config)
            logger.info(f"✅ Connected to PostgreSQL")
            return True
        except psycopg2.Error as e:
            logger.error(f"❌ PostgreSQL connection failed: {e}")
            return False
            
    def get_one_contract(self, date: str = '2025-12-10') -> Dict[str, Any]:
        """Get one sample contract from MariaDB"""
        cursor = self.mysql_conn.cursor(dictionary=True)
        cursor.execute("""
            SELECT * FROM fd_option_contracts 
            WHERE peildatum = %s
            ORDER BY scraped_at DESC, id
            LIMIT 1
        """, (date,))
        contract = cursor.fetchone()
        cursor.close()
        return contract
        
    def get_underlying_price(self, ticker: str, trade_date: str) -> float:
        """Get underlying price from bronze_fd_overview"""
        cursor = self.postgres_conn.cursor()
        cursor.execute("""
            SELECT koers FROM bronze_fd_overview 
            WHERE ticker = %s AND trade_date = %s
            ORDER BY scraped_at DESC
            LIMIT 1
        """, (ticker, trade_date))
        result = cursor.fetchone()
        cursor.close()
        return result[0] if result else None
        
    def transform_contract(self, mysql_row: Dict[str, Any]) -> Dict[str, Any]:
        """Transform MySQL row to PostgreSQL format"""
        trade_date = mysql_row['peildatum']
        ticker = mysql_row['ticker']
        
        # Get underlying price from overview
        underlying_price = self.get_underlying_price(ticker, trade_date)
        
        return {
            'ticker': ticker,
            'symbol_code': mysql_row['symbol_code'],
            'trade_date': trade_date,
            'scraped_at': mysql_row['scraped_at'],
            'source_url': mysql_row.get('source', 'https://www.fd.nl/beurs/opties'),
            'option_type': mysql_row['type'],  # 'Call' or 'Put'
            'expiry_date': mysql_row['expiry'],
            'strike': float(mysql_row['strike']),
            'naam': None,  # Not in MariaDB
            'isin': None,  # Not in MariaDB
            'laatste': float(mysql_row['last']) if mysql_row.get('last') else None,
            'bid': float(mysql_row['bid']) if mysql_row.get('bid') else None,
            'ask': float(mysql_row['ask']) if mysql_row.get('ask') else None,
            'volume': int(mysql_row['volume']) if mysql_row.get('volume') else None,
            'open_interest': int(mysql_row['open_interest']) if mysql_row.get('open_interest') else None,
            'underlying_price': underlying_price,
        }
        
    def insert_contract(self, data: Dict[str, Any]) -> bool:
        """Insert one contract into PostgreSQL"""
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
        
        try:
            cursor.execute(query, data)
            self.postgres_conn.commit()
            logger.info(f"✅ Inserted 1 contract")
            cursor.close()
            return True
        except Exception as e:
            logger.error(f"❌ Insert failed: {e}")
            self.postgres_conn.rollback()
            cursor.close()
            return False
            
    def verify_insertion(self, ticker: str, trade_date: str, strike: float, option_type: str):
        """Verify the contract was inserted"""
        cursor = self.postgres_conn.cursor()
        cursor.execute("""
            SELECT 
                ticker, trade_date, option_type, expiry_date, strike,
                laatste, bid, ask, volume, open_interest, underlying_price
            FROM bronze_fd_options 
            WHERE ticker = %s AND trade_date = %s AND strike = %s AND option_type = %s
            ORDER BY scraped_at DESC
            LIMIT 1
        """, (ticker, trade_date, strike, option_type))
        
        row = cursor.fetchone()
        cursor.close()
        
        if row:
            logger.info(f"\n📊 Verification - Contract inserted:")
            logger.info(f"  Ticker: {row[0]}, Date: {row[1]}, Type: {row[2]}")
            logger.info(f"  Expiry: {row[3]}, Strike: {row[4]}")
            logger.info(f"  Last: {row[5]}, Bid: {row[6]}, Ask: {row[7]}")
            logger.info(f"  Volume: {row[8]}, OI: {row[9]}")
            logger.info(f"  Underlying Price: {row[10]}")
        else:
            logger.warning(f"⚠️  Contract not found after insertion")
            
    def migrate_one_contract(self):
        """Migrate one contract for testing"""
        logger.info(f"\n{'='*80}")
        logger.info(f"Migrating ONE contract for testing")
        logger.info(f"{'='*80}\n")
        
        # Get one contract from MariaDB
        contract = self.get_one_contract()
        if not contract:
            logger.error("❌ No contract found in MariaDB")
            return False
            
        logger.info(f"📋 Contract from MariaDB:")
        logger.info(f"  Date: {contract['peildatum']}, Ticker: {contract['ticker']}")
        logger.info(f"  Type: {contract['type']}, Expiry: {contract['expiry']}, Strike: {contract['strike']}")
        logger.info(f"  Last: {contract['last']}, Bid: {contract['bid']}, Ask: {contract['ask']}")
        logger.info(f"  Volume: {contract['volume']}, OI: {contract['open_interest']}")
        
        # Transform
        logger.info(f"\n🔄 Transforming...")
        transformed = self.transform_contract(contract)
        
        # Insert
        logger.info(f"💾 Inserting into PostgreSQL...")
        if not self.insert_contract(transformed):
            return False
            
        # Verify
        logger.info(f"\n🔍 Verifying...")
        self.verify_insertion(
            transformed['ticker'],
            transformed['trade_date'],
            transformed['strike'],
            transformed['option_type']
        )
        
        logger.info(f"\n{'='*80}")
        logger.info(f"✅ Test migration complete!")
        logger.info(f"{'='*80}\n")
        
        return True
        
    def close_connections(self):
        if self.mysql_conn:
            self.mysql_conn.close()
        if self.postgres_conn:
            self.postgres_conn.close()


def main():
    migration = SingleContractMigration()
    
    try:
        if not migration.connect_mysql():
            return 1
        if not migration.connect_postgres():
            return 1
            
        if not migration.migrate_one_contract():
            return 1
        
    except Exception as e:
        logger.error(f"❌ Error: {e}", exc_info=True)
        return 1
    finally:
        migration.close_connections()
    
    return 0


if __name__ == "__main__":
    sys.exit(main())
