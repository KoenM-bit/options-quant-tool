"""
DRY RUN: Preview migration from MariaDB fd_option_contracts to PostgreSQL bronze_fd_options
"""

import os
import sys
from datetime import datetime
from typing import List, Dict, Any
import logging

import mysql.connector
import psycopg2
from tabulate import tabulate

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


class OptionsMigrationDryRun:
    """Preview options chain migration"""
    
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
        
    def preview_migration(self):
        """Show what will be migrated"""
        existing_dates = self.get_existing_dates()
        logger.info(f"📊 Found {len(existing_dates)} existing dates in PostgreSQL: {sorted(existing_dates)}")
        
        # Get summary from MariaDB
        cursor = self.mysql_conn.cursor(dictionary=True)
        cursor.execute("""
            SELECT 
                peildatum,
                COUNT(*) as num_contracts,
                COUNT(DISTINCT expiry) as num_expiries,
                COUNT(CASE WHEN type = 'Call' THEN 1 END) as num_calls,
                COUNT(CASE WHEN type = 'Put' THEN 1 END) as num_puts,
                SUM(volume) as total_volume,
                SUM(open_interest) as total_oi
            FROM fd_option_contracts
            GROUP BY peildatum
            ORDER BY peildatum
        """)
        all_dates = cursor.fetchall()
        cursor.close()
        
        # Filter missing dates
        missing_dates = [row for row in all_dates if row['peildatum'] not in existing_dates]
        
        if not missing_dates:
            logger.info("✅ No missing data - PostgreSQL is up to date!")
            return
            
        print("\n" + "="*100)
        print("DRY RUN PREVIEW - Dates to be migrated:")
        print("="*100 + "\n")
        
        summary_data = []
        for row in missing_dates:
            summary_data.append([
                row['peildatum'],
                row['num_contracts'],
                row['num_expiries'],
                row['num_calls'],
                row['num_puts'],
                row['total_volume'] or 0,
                row['total_oi'] or 0
            ])
        
        print(tabulate(
            summary_data,
            headers=['Date', '# Contracts', '# Expiries', '# Calls', '# Puts', 'Total Vol', 'Total OI'],
            tablefmt='grid'
        ))
        
        print(f"\n📊 SUMMARY:")
        print(f"   Dates to migrate: {len(missing_dates)}")
        print(f"   Date range: {missing_dates[0]['peildatum']} to {missing_dates[-1]['peildatum']}")
        print(f"   Total contracts: {sum(row['num_contracts'] for row in missing_dates)}")
        
        print(f"\n📋 COLUMN MAPPING (MariaDB → PostgreSQL):")
        mappings = [
            ("ticker", "ticker", "✓"),
            ("symbol_code", "symbol_code", "✓"),
            ("peildatum", "trade_date", "✓ (renamed)"),
            ("expiry", "expiry_date", "✓"),
            ("strike", "strike", "✓"),
            ("type", "option_type", "✓ (Call/Put)"),
            ("last", "laatste", "✓ (last price)"),
            ("bid", "bid", "✓"),
            ("ask", "ask", "✓"),
            ("volume", "volume", "✓"),
            ("open_interest", "open_interest", "✓"),
            ("scraped_at", "scraped_at", "✓"),
            ("source", "source_url", "✓"),
            ("", "underlying_price", "⚠️ Will be NULL (get from overview)"),
            ("", "naam", "⚠️ Will be NULL (not in MariaDB)"),
            ("", "isin", "⚠️ Will be NULL (not in MariaDB)"),
        ]
        
        print(tabulate(mappings, headers=['MariaDB Column', 'PostgreSQL Column', 'Status'], tablefmt='grid'))
        
        print("\n" + "="*100)
        print("⚠️  This is a DRY RUN - no data has been inserted")
        print("="*100 + "\n")
        
    def close_connections(self):
        if self.mysql_conn:
            self.mysql_conn.close()
            logger.info("Closed MariaDB connection")
        if self.postgres_conn:
            self.postgres_conn.close()
            logger.info("Closed PostgreSQL connection")


def main():
    migration = OptionsMigrationDryRun()
    
    try:
        if not migration.connect_mysql():
            return 1
        if not migration.connect_postgres():
            return 1
            
        migration.preview_migration()
        
    except Exception as e:
        logger.error(f"❌ Error: {e}", exc_info=True)
        return 1
    finally:
        migration.close_connections()
    
    return 0


if __name__ == "__main__":
    sys.exit(main())
