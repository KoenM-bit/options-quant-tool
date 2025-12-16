"""
DRY RUN: Preview migration from MariaDB fd_option_overview to PostgreSQL bronze_fd_overview

This script shows what will be migrated WITHOUT actually inserting data.
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


class MigrationDryRun:
    """Preview migration without making changes"""
    
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
        logger.info(f"📊 Found {len(existing_dates)} existing dates in PostgreSQL: {sorted(existing_dates)}")
        
        cursor = self.mysql_conn.cursor(dictionary=True)
        cursor.execute("""
            SELECT * FROM fd_option_overview 
            ORDER BY peildatum DESC
        """)
        all_data = cursor.fetchall()
        cursor.close()
        
        # Filter out existing dates
        missing_data = [row for row in all_data if row['peildatum'] not in existing_dates]
        
        logger.info(f"📊 Found {len(missing_data)} records to migrate from MariaDB")
        return missing_data
        
    def preview_migration(self):
        """Show what will be migrated"""
        missing_data = self.get_missing_data()
        
        if not missing_data:
            logger.info("✅ No missing data - PostgreSQL is up to date!")
            return
            
        print("\n" + "="*100)
        print("DRY RUN PREVIEW - Records to be migrated:")
        print("="*100 + "\n")
        
        # Group by date
        by_date = {}
        for row in missing_data:
            date = row['peildatum']
            if date not in by_date:
                by_date[date] = []
            by_date[date].append(row)
        
        # Show summary
        summary_data = []
        for date in sorted(by_date.keys()):
            rows = by_date[date]
            sample = rows[0]
            summary_data.append([
                date,
                sample['ticker'],
                f"{sample['koers']:.2f}" if sample['koers'] else 'NULL',
                sample['totaal_volume'] or 0,
                sample['totaal_oi_opening'] or 0,
                sample['totaal_volume_calls'] or 0,
                sample['totaal_volume_puts'] or 0,
                f"{sample['call_put_ratio']:.2f}" if sample['call_put_ratio'] else 'NULL',
                sample['scraped_at']
            ])
        
        print(tabulate(
            summary_data,
            headers=['Date', 'Ticker', 'Price', 'Total Vol', 'Total OI', 'Call Vol', 'Put Vol', 'C/P Ratio', 'Scraped At'],
            tablefmt='grid'
        ))
        
        print(f"\n📊 SUMMARY:")
        print(f"   Total records to migrate: {len(missing_data)}")
        print(f"   Date range: {min(by_date.keys())} to {max(by_date.keys())}")
        print(f"   Unique dates: {len(by_date)}")
        
        # Show column mapping
        print(f"\n📋 COLUMN MAPPING (MariaDB → PostgreSQL):")
        mappings = [
            ("ticker", "ticker", "✓"),
            ("symbol_code", "symbol_code", "✓"),
            ("koers", "koers", "✓ (underlying price)"),
            ("vorige", "vorige", "✓ (previous close)"),
            ("delta", "delta", "✓ (price change)"),
            ("delta_pct", "delta_pct", "✓ (% change)"),
            ("hoog", "hoog", "✓ (high)"),
            ("laag", "laag", "✓ (low)"),
            ("volume_ul", "volume_underlying", "✓ (underlying volume)"),
            ("tijd", "tijd", "✓ (time)"),
            ("peildatum", "peildatum", "✓ (trade date)"),
            ("totaal_volume", "totaal_volume", "✓ (total options volume)"),
            ("totaal_volume_calls", "totaal_volume_calls", "✓"),
            ("totaal_volume_puts", "totaal_volume_puts", "✓"),
            ("totaal_oi_opening", "totaal_oi", "✓ (open interest)"),
            ("totaal_oi_calls", "totaal_oi_calls", "✓"),
            ("totaal_oi_puts", "totaal_oi_puts", "✓"),
            ("call_put_ratio", "call_put_ratio", "✓"),
            ("scraped_at", "scraped_at", "✓"),
            ("source", "source_url", "✓"),
        ]
        
        print(tabulate(mappings, headers=['MariaDB Column', 'PostgreSQL Column', 'Status'], tablefmt='grid'))
        
        print("\n" + "="*100)
        print("⚠️  This is a DRY RUN - no data has been inserted")
        print("="*100 + "\n")
        
    def close_connections(self):
        """Close database connections"""
        if self.mysql_conn:
            self.mysql_conn.close()
            logger.info("Closed MariaDB connection")
        if self.postgres_conn:
            self.postgres_conn.close()
            logger.info("Closed PostgreSQL connection")


def main():
    """Run dry run preview"""
    migration = MigrationDryRun()
    
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
