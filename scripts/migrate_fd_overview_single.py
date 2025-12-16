"""
Migrate a single date from MariaDB fd_option_overview to PostgreSQL bronze_fd_overview

Usage:
    python scripts/migrate_fd_overview_single.py --date 2025-12-10
"""

import os
import sys
from datetime import datetime
from typing import List, Dict, Any
import logging
import argparse

import mysql.connector
import psycopg2
from psycopg2.extras import execute_batch

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


class SingleDateMigration:
    """Migrate data for a single date"""
    
    def __init__(self):
        # MariaDB (source) configuration
        self.mysql_config = {
            "host": "192.168.1.201",
            "user": "remoteuser",
            "password": "T3l3foon32#123",
            "database": "optionsdb",
            "port": 3306,
        }
        
        # PostgreSQL (destination) configuration
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
        """Connect to MariaDB source database"""
        try:
            self.mysql_conn = mysql.connector.connect(**self.mysql_config)
            logger.info(f"✅ Connected to MariaDB: {self.mysql_config['host']}:{self.mysql_config['port']}")
            return True
        except mysql.connector.Error as e:
            logger.error(f"❌ MariaDB connection failed: {e}")
            return False
            
    def connect_postgres(self):
        """Connect to PostgreSQL destination database"""
        try:
            self.postgres_conn = psycopg2.connect(**self.postgres_config)
            logger.info(f"✅ Connected to PostgreSQL: {self.postgres_config['host']}:{self.postgres_config['port']}")
            return True
        except psycopg2.Error as e:
            logger.error(f"❌ PostgreSQL connection failed: {e}")
            return False
            
    def check_date_exists(self, target_date: str) -> bool:
        """Check if date already exists in PostgreSQL"""
        cursor = self.postgres_conn.cursor()
        cursor.execute("""
            SELECT COUNT(*) FROM bronze_fd_overview 
            WHERE peildatum = %s AND ticker = 'AD.AS'
        """, (target_date,))
        count = cursor.fetchone()[0]
        cursor.close()
        return count > 0
        
    def get_data_for_date(self, target_date: str) -> List[Dict[str, Any]]:
        """Get data from MariaDB for specific date"""
        cursor = self.mysql_conn.cursor(dictionary=True)
        cursor.execute("""
            SELECT * FROM fd_option_overview 
            WHERE peildatum = %s
            ORDER BY scraped_at DESC
        """, (target_date,))
        data = cursor.fetchall()
        cursor.close()
        
        logger.info(f"📊 Found {len(data)} records in MariaDB for {target_date}")
        return data
        
    def transform_row(self, mysql_row: Dict[str, Any]) -> Dict[str, Any]:
        """Transform MySQL row to PostgreSQL format"""
        scraped_at = mysql_row['scraped_at'] if 'scraped_at' in mysql_row else mysql_row.get('peildatum')
        
        # Map ticker to full company name
        ticker_to_name = {
            'AD.AS': 'Ahold Delhaize Koninklijke*',
            'MT.AS': 'ArcelorMittal*'
        }
        onderliggende_waarde = ticker_to_name.get(mysql_row['ticker'], mysql_row['ticker'])
        
        return {
            'ticker': mysql_row['ticker'],
            'symbol_code': mysql_row['symbol_code'],
            'onderliggende_waarde': onderliggende_waarde,
            'koers': float(mysql_row['koers']) if mysql_row.get('koers') else None,
            'vorige': float(mysql_row['vorige']) if mysql_row.get('vorige') else None,
            'delta': float(mysql_row['delta']) if mysql_row.get('delta') else None,
            'delta_pct': float(mysql_row['delta_pct']) if mysql_row.get('delta_pct') else None,
            'hoog': float(mysql_row['hoog']) if mysql_row.get('hoog') else None,
            'laag': float(mysql_row['laag']) if mysql_row.get('laag') else None,
            'volume_underlying': int(mysql_row['volume_ul']) if mysql_row.get('volume_ul') else None,
            'tijd': mysql_row.get('tijd'),
            'peildatum': mysql_row.get('peildatum'),
            'trade_date': mysql_row.get('peildatum'),  # trade_date should match peildatum
            'totaal_volume': int(mysql_row['totaal_volume']) if mysql_row.get('totaal_volume') else None,
            'totaal_volume_calls': int(mysql_row['totaal_volume_calls']) if mysql_row.get('totaal_volume_calls') else None,
            'totaal_volume_puts': int(mysql_row['totaal_volume_puts']) if mysql_row.get('totaal_volume_puts') else None,
            'totaal_oi': int(mysql_row['totaal_oi_opening']) if mysql_row.get('totaal_oi_opening') else None,
            'totaal_oi_calls': int(mysql_row['totaal_oi_calls']) if mysql_row.get('totaal_oi_calls') else None,
            'totaal_oi_puts': int(mysql_row['totaal_oi_puts']) if mysql_row.get('totaal_oi_puts') else None,
            'call_put_ratio': float(mysql_row['call_put_ratio']) if mysql_row.get('call_put_ratio') else None,
            'scraped_at': scraped_at,
            'source_url': mysql_row.get('source', 'https://www.fd.nl/beurs/ahold-delhaize'),
        }
        
    def insert_data(self, data: List[Dict[str, Any]]) -> int:
        """Insert data into PostgreSQL"""
        if not data:
            return 0
            
        cursor = self.postgres_conn.cursor()
        
        query = """
        INSERT INTO bronze_fd_overview 
        (created_at, updated_at, ticker, symbol_code, onderliggende_waarde, koers, vorige, delta, delta_pct, 
         hoog, laag, volume_underlying, tijd, peildatum, trade_date, totaal_volume, totaal_volume_calls, 
         totaal_volume_puts, totaal_oi, totaal_oi_calls, totaal_oi_puts, call_put_ratio, 
         scraped_at, source_url)
        VALUES (NOW(), NOW(), %(ticker)s, %(symbol_code)s, %(onderliggende_waarde)s, %(koers)s, %(vorige)s, 
                %(delta)s, %(delta_pct)s, %(hoog)s, %(laag)s, %(volume_underlying)s, %(tijd)s, 
                %(peildatum)s, %(trade_date)s, %(totaal_volume)s, %(totaal_volume_calls)s, %(totaal_volume_puts)s, 
                %(totaal_oi)s, %(totaal_oi_calls)s, %(totaal_oi_puts)s, %(call_put_ratio)s, 
                %(scraped_at)s, %(source_url)s)
        """
        
        execute_batch(cursor, query, data, page_size=100)
        self.postgres_conn.commit()
        
        inserted = cursor.rowcount
        cursor.close()
        
        logger.info(f"✅ Inserted {inserted} rows into bronze_fd_overview")
        return inserted
        
    def verify_insertion(self, target_date: str):
        """Verify data was inserted correctly"""
        cursor = self.postgres_conn.cursor()
        cursor.execute("""
            SELECT 
                ticker,
                peildatum,
                koers,
                totaal_volume,
                totaal_oi,
                totaal_volume_calls,
                totaal_volume_puts,
                call_put_ratio,
                scraped_at
            FROM bronze_fd_overview 
            WHERE peildatum = %s AND ticker = 'AD.AS'
            ORDER BY scraped_at DESC
        """, (target_date,))
        
        rows = cursor.fetchall()
        cursor.close()
        
        if rows:
            logger.info(f"\n📊 Verification - Data inserted for {target_date}:")
            for row in rows:
                logger.info(f"  Ticker: {row[0]}, Date: {row[1]}, Price: {row[2]:.2f}, "
                          f"Vol: {row[3]}, OI: {row[4]}, C/P Ratio: {row[7]:.2f}, Scraped: {row[8]}")
        else:
            logger.warning(f"⚠️  No data found after insertion for {target_date}")
            
    def migrate_date(self, target_date: str, force: bool = False):
        """Migrate data for a specific date"""
        logger.info(f"\n{'='*80}")
        logger.info(f"Starting migration for date: {target_date}")
        logger.info(f"{'='*80}\n")
        
        # Check if date already exists
        if not force and self.check_date_exists(target_date):
            logger.warning(f"⚠️  Date {target_date} already exists in PostgreSQL. Use --force to overwrite.")
            return 0
            
        # Get data from MariaDB
        mysql_data = self.get_data_for_date(target_date)
        if not mysql_data:
            logger.warning(f"⚠️  No data found in MariaDB for {target_date}")
            return 0
            
        # Show preview
        sample = mysql_data[0]
        logger.info(f"\n📋 Preview of data to migrate:")
        logger.info(f"  Ticker: {sample['ticker']}")
        logger.info(f"  Date: {sample['peildatum']}")
        logger.info(f"  Price: {sample['koers']:.2f}")
        logger.info(f"  Total Volume: {sample['totaal_volume']}")
        logger.info(f"  Total OI: {sample['totaal_oi_opening']}")
        logger.info(f"  Call/Put Ratio: {sample['call_put_ratio']:.2f}")
        logger.info(f"  Scraped At: {sample['scraped_at']}")
        
        # Transform data
        logger.info(f"\n🔄 Transforming {len(mysql_data)} records...")
        transformed_data = [self.transform_row(row) for row in mysql_data]
        
        # Insert into PostgreSQL
        logger.info(f"💾 Inserting into PostgreSQL...")
        inserted = self.insert_data(transformed_data)
        
        # Verify
        logger.info(f"\n🔍 Verifying insertion...")
        self.verify_insertion(target_date)
        
        logger.info(f"\n{'='*80}")
        logger.info(f"✅ Migration complete for {target_date}: {inserted} records inserted")
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
    """Run single date migration"""
    parser = argparse.ArgumentParser(description='Migrate single date from MariaDB to PostgreSQL')
    parser.add_argument('--date', required=True, help='Date to migrate (YYYY-MM-DD)')
    parser.add_argument('--force', action='store_true', help='Force migration even if date exists')
    
    args = parser.parse_args()
    
    migration = SingleDateMigration()
    
    try:
        if not migration.connect_mysql():
            return 1
        if not migration.connect_postgres():
            return 1
            
        migration.migrate_date(args.date, force=args.force)
        
    except Exception as e:
        logger.error(f"❌ Error: {e}", exc_info=True)
        return 1
    finally:
        migration.close_connections()
    
    return 0


if __name__ == "__main__":
    sys.exit(main())
