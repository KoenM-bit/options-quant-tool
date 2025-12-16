"""
Bulk migration of all missing dates from MariaDB fd_option_overview to PostgreSQL bronze_fd_overview

This script migrates all dates that don't yet exist in PostgreSQL.
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


class BulkMigration:
    """Migrate all missing historical data"""
    
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
            
    def get_existing_dates(self) -> set:
        """Get dates that already exist in PostgreSQL"""
        cursor = self.postgres_conn.cursor()
        cursor.execute("""
            SELECT DISTINCT peildatum 
            FROM bronze_fd_overview 
            WHERE ticker = 'AD.AS'
            ORDER BY peildatum
        """)
        existing = {row[0] for row in cursor.fetchall()}
        cursor.close()
        return existing
        
    def get_missing_data(self) -> List[Dict[str, Any]]:
        """Get data from MariaDB that doesn't exist in PostgreSQL"""
        existing_dates = self.get_existing_dates()
        logger.info(f"📊 Found {len(existing_dates)} existing dates in PostgreSQL")
        
        cursor = self.mysql_conn.cursor(dictionary=True)
        cursor.execute("""
            SELECT * FROM fd_option_overview 
            ORDER BY peildatum ASC
        """)
        all_data = cursor.fetchall()
        cursor.close()
        
        # Filter out existing dates
        missing_data = [row for row in all_data if row['peildatum'] not in existing_dates]
        
        logger.info(f"📊 Found {len(missing_data)} records to migrate from MariaDB")
        return missing_data
        
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
        
    def verify_migration(self):
        """Verify migration was successful"""
        cursor = self.postgres_conn.cursor()
        cursor.execute("""
            SELECT 
                MIN(peildatum) as earliest,
                MAX(peildatum) as latest,
                COUNT(*) as total_records,
                COUNT(DISTINCT peildatum) as unique_dates
            FROM bronze_fd_overview 
            WHERE ticker = 'AD.AS'
        """)
        
        stats = cursor.fetchone()
        cursor.close()
        
        logger.info(f"\n📊 Final Statistics:")
        logger.info(f"   Date range: {stats[0]} to {stats[1]}")
        logger.info(f"   Total records: {stats[2]}")
        logger.info(f"   Unique dates: {stats[3]}")
        
    def migrate_all(self):
        """Migrate all missing data"""
        logger.info(f"\n{'='*80}")
        logger.info(f"Starting BULK migration of all missing dates")
        logger.info(f"{'='*80}\n")
        
        # Get missing data
        missing_data = self.get_missing_data()
        if not missing_data:
            logger.info("✅ No missing data - PostgreSQL is already up to date!")
            return 0
            
        # Show summary
        dates = sorted(set(row['peildatum'] for row in missing_data))
        logger.info(f"\n📋 Will migrate {len(dates)} dates:")
        logger.info(f"   From: {dates[0]}")
        logger.info(f"   To: {dates[-1]}")
        logger.info(f"   Total records: {len(missing_data)}")
        
        # Transform data
        logger.info(f"\n🔄 Transforming {len(missing_data)} records...")
        transformed_data = [self.transform_row(row) for row in missing_data]
        
        # Insert into PostgreSQL
        logger.info(f"💾 Inserting into PostgreSQL...")
        inserted = self.insert_data(transformed_data)
        
        # Verify
        logger.info(f"\n🔍 Verifying migration...")
        self.verify_migration()
        
        logger.info(f"\n{'='*80}")
        logger.info(f"✅ Bulk migration complete: {inserted} records inserted across {len(dates)} dates")
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
    migration = BulkMigration()
    
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
