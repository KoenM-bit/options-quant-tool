"""
Migration script to transfer historical options data from old MySQL database
to new PostgreSQL Bronze layer.

This script:
1. Connects to old MySQL database (optionsdb)
2. Reads historical data from fd_option_overview
3. Transforms it to match Bronze layer schema
4. Inserts into PostgreSQL bronze_fd_overview and bronze_fd_options
"""

import os
import sys
from datetime import datetime
from typing import List, Dict, Any
import logging

import mysql.connector
import psycopg2
from psycopg2.extras import execute_batch
from dotenv import load_dotenv

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


class HistoricalDataMigration:
    """Migrates historical options data from MySQL to PostgreSQL"""
    
    def __init__(self):
        load_dotenv('.env.migration')
        
        # MySQL (source) configuration
        self.mysql_config = {
            "host": os.getenv("MYSQL_HOST", "192.168.1.201"),
            "user": os.getenv("MYSQL_USER", "remoteuser"),
            "password": os.getenv("MYSQL_PASSWORD", "T3l3foon32#123"),
            "database": os.getenv("MYSQL_DATABASE", "optionsdb"),
            "port": int(os.getenv("MYSQL_PORT", 3306)),
        }
        
        # PostgreSQL (destination) configuration
        self.postgres_config = {
            "host": os.getenv("POSTGRES_HOST", "localhost"),
            "port": int(os.getenv("POSTGRES_PORT", 5432)),
            "user": os.getenv("POSTGRES_USER", "airflow"),
            "password": os.getenv("POSTGRES_PASSWORD", "airflow"),
            "database": os.getenv("POSTGRES_DB", "ahold_options"),
        }
        
        self.mysql_conn = None
        self.postgres_conn = None
        
    def connect_mysql(self):
        """Connect to MySQL source database"""
        try:
            self.mysql_conn = mysql.connector.connect(**self.mysql_config)
            logger.info(f"✅ Connected to MySQL: {self.mysql_config['host']}:{self.mysql_config['port']}")
            return True
        except mysql.connector.Error as e:
            logger.error(f"❌ MySQL connection failed: {e}")
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
            
    def get_mysql_table_schema(self, table_name: str) -> List[Dict[str, Any]]:
        """Get schema of MySQL table"""
        cursor = self.mysql_conn.cursor(dictionary=True)
        cursor.execute(f"DESCRIBE {table_name}")
        schema = cursor.fetchall()
        cursor.close()
        return schema
        
    def get_mysql_data(self, table_name: str, limit: int = None) -> List[Dict[str, Any]]:
        """Fetch all data from MySQL table"""
        cursor = self.mysql_conn.cursor(dictionary=True)
        
        query = f"SELECT * FROM {table_name} ORDER BY scraped_at DESC"
        if limit:
            query += f" LIMIT {limit}"
            
        cursor.execute(query)
        data = cursor.fetchall()
        cursor.close()
        
        logger.info(f"📊 Fetched {len(data)} rows from {table_name}")
        return data
        
    def get_distinct_dates(self, table_name: str) -> List[str]:
        """Get list of distinct dates in the MySQL data"""
        cursor = self.mysql_conn.cursor()
        
        query = f"""
        SELECT DISTINCT peildatum as date 
        FROM {table_name} 
        ORDER BY date
        """
        
        cursor.execute(query)
        dates = [row[0] for row in cursor.fetchall()]
        cursor.close()
        
        logger.info(f"📅 Found {len(dates)} distinct dates")
        return dates
        
    def check_existing_data_postgres(self, peildatum: str) -> bool:
        """Check if data for this date already exists in PostgreSQL"""
        cursor = self.postgres_conn.cursor()
        
        # Check if overview data exists for this peildatum
        query = """
        SELECT COUNT(*) FROM bronze_fd_overview 
        WHERE peildatum = %s
        """
        
        cursor.execute(query, (peildatum,))
        count = cursor.fetchone()[0]
        cursor.close()
        
        return count > 0
        
    def transform_to_bronze_overview(self, mysql_row: Dict[str, Any]) -> Dict[str, Any]:
        """Transform MySQL row to Bronze overview format - exact MySQL → PostgreSQL mapping"""
        
        # Get scraped_at from MySQL
        scraped_at = mysql_row['scraped_at'] if 'scraped_at' in mysql_row else mysql_row.get('peildatum')
        
        return {
            'ticker': mysql_row['ticker'],  # MySQL: ticker (AD.AS)
            'symbol_code': mysql_row['symbol_code'],  # MySQL: symbol_code (AEX.AH/O)
            'onderliggende_waarde': mysql_row['ticker'],  # Use ticker for onderliggende_waarde
            'koers': float(mysql_row['koers']),
            'vorige': float(mysql_row['vorige']) if mysql_row.get('vorige') else None,
            'delta': float(mysql_row['delta']) if mysql_row.get('delta') else None,
            'delta_pct': float(mysql_row['delta_pct']) if mysql_row.get('delta_pct') else None,
            'hoog': float(mysql_row['hoog']) if mysql_row.get('hoog') else None,
            'laag': float(mysql_row['laag']) if mysql_row.get('laag') else None,
            'volume_underlying': int(mysql_row['volume_ul']) if mysql_row.get('volume_ul') else None,
            'tijd': mysql_row.get('tijd'),
            'peildatum': mysql_row.get('peildatum'),
            'totaal_volume': int(mysql_row['totaal_volume']) if mysql_row.get('totaal_volume') else None,
            'totaal_volume_calls': int(mysql_row['totaal_volume_calls']) if mysql_row.get('totaal_volume_calls') else None,
            'totaal_volume_puts': int(mysql_row['totaal_volume_puts']) if mysql_row.get('totaal_volume_puts') else None,
            'totaal_oi': int(mysql_row['totaal_oi_opening']) if mysql_row.get('totaal_oi_opening') else None,  # MySQL: totaal_oi_opening → PostgreSQL: totaal_oi
            'totaal_oi_calls': int(mysql_row['totaal_oi_calls']) if mysql_row.get('totaal_oi_calls') else None,
            'totaal_oi_puts': int(mysql_row['totaal_oi_puts']) if mysql_row.get('totaal_oi_puts') else None,
            'call_put_ratio': float(mysql_row['call_put_ratio']) if mysql_row.get('call_put_ratio') else None,
            'scraped_at': scraped_at,
            'source_url': mysql_row.get('source', 'https://www.fd.nl/beurs/ahold-delhaize'),
        }
        
    def transform_to_bronze_options(self, mysql_row: Dict[str, Any], underlying_price: float = None) -> Dict[str, Any]:
        """Transform MySQL fd_option_contracts row to Bronze options format - NEW SCHEMA (no Greeks in bronze)"""
        
        return {
            'ticker': mysql_row['ticker'],  # AD.AS
            'symbol_code': mysql_row['symbol_code'],  # AEX.AH/O
            'scraped_at': mysql_row['scraped_at'],
            'source_url': mysql_row.get('source', 'https://beurs.fd.nl/derivaten/opties/'),
            'option_type': mysql_row['type'],  # 'Call' or 'Put'
            'expiry_date': mysql_row['expiry'],
            'strike': float(mysql_row['strike']),
            'naam': None,  # Not in MySQL
            'isin': None,  # Not in MySQL
            'laatste': float(mysql_row['last']) if mysql_row.get('last') else None,
            'bid': float(mysql_row['bid']) if mysql_row.get('bid') else None,
            'ask': float(mysql_row['ask']) if mysql_row.get('ask') else None,
            'volume': int(mysql_row['volume']) if mysql_row.get('volume') else None,
            'open_interest': int(mysql_row['open_interest']) if mysql_row.get('open_interest') else None,
            'underlying_price': underlying_price,  # Get from overview data
        }
        
    def insert_bronze_overview(self, data: List[Dict[str, Any]]) -> int:
        """Insert data into bronze_fd_overview - matches actual Bronze schema"""
        if not data:
            return 0
            
        cursor = self.postgres_conn.cursor()
        
        query = """
        INSERT INTO bronze_fd_overview 
        (created_at, updated_at, ticker, symbol_code, onderliggende_waarde, koers, vorige, delta, delta_pct, 
         hoog, laag, volume_underlying, tijd, peildatum, totaal_volume, totaal_volume_calls, 
         totaal_volume_puts, totaal_oi, totaal_oi_calls, totaal_oi_puts, call_put_ratio, 
         scraped_at, source_url)
        VALUES (NOW(), NOW(), %(ticker)s, %(symbol_code)s, %(onderliggende_waarde)s, %(koers)s, %(vorige)s, 
                %(delta)s, %(delta_pct)s, %(hoog)s, %(laag)s, %(volume_underlying)s, %(tijd)s, 
                %(peildatum)s, %(totaal_volume)s, %(totaal_volume_calls)s, %(totaal_volume_puts)s, 
                %(totaal_oi)s, %(totaal_oi_calls)s, %(totaal_oi_puts)s, %(call_put_ratio)s, 
                %(scraped_at)s, %(source_url)s)
        """
        
        execute_batch(cursor, query, data, page_size=100)
        inserted = cursor.rowcount
        self.postgres_conn.commit()
        cursor.close()
        
        logger.info(f"✅ Inserted {inserted} rows into bronze_fd_overview")
        return inserted
        
    def insert_bronze_options(self, data: List[Dict[str, Any]]) -> int:
        """Insert data into bronze_fd_options"""
        if not data:
            return 0
            
        cursor = self.postgres_conn.cursor()
        
        query = """
        INSERT INTO bronze_fd_options 
        (created_at, updated_at, ticker, symbol_code, scraped_at, source_url, option_type, 
         expiry_date, strike, naam, isin, laatste, bid, ask, volume, open_interest, underlying_price)
        VALUES (NOW(), NOW(), %(ticker)s, %(symbol_code)s, %(scraped_at)s, %(source_url)s, 
                %(option_type)s, %(expiry_date)s, %(strike)s, %(naam)s, %(isin)s, %(laatste)s, 
                %(bid)s, %(ask)s, %(volume)s, %(open_interest)s, %(underlying_price)s)
        """
        
        execute_batch(cursor, query, data, page_size=100)
        inserted = cursor.rowcount
        self.postgres_conn.commit()
        cursor.close()
        
        logger.info(f"✅ Inserted {inserted} rows into bronze_fd_options")
        return inserted
        
    def migrate_date(self, target_date: str, dry_run: bool = False) -> Dict[str, int]:
        """Migrate data for a specific date (both overview and option contracts)"""
        logger.info(f"\n{'='*60}")
        logger.info(f"📅 Migrating data for date: {target_date}")
        logger.info(f"{'='*60}")
        
        cursor = self.mysql_conn.cursor(dictionary=True)
        
        # 1. Fetch overview data (fd_option_overview)
        query_overview = """
        SELECT * FROM fd_option_overview 
        WHERE peildatum = %s
        ORDER BY scraped_at DESC
        LIMIT 1
        """
        
        cursor.execute(query_overview, (target_date,))
        overview_rows = cursor.fetchall()
        
        if not overview_rows:
            logger.warning(f"⚠️  No overview data found for {target_date}")
            return {'overview': 0, 'options': 0}
            
        logger.info(f"📊 Found {len(overview_rows)} overview row(s) for {target_date}")
        
        # 2. Fetch option contracts (fd_option_contracts)
        query_options = """
        SELECT * FROM fd_option_contracts 
        WHERE peildatum = %s
        ORDER BY scraped_at DESC, expiry, strike
        """
        
        cursor.execute(query_options, (target_date,))
        contracts_rows = cursor.fetchall()
        cursor.close()
        
        logger.info(f"📊 Found {len(contracts_rows)} option contracts for {target_date}")
        
        # Get underlying price from overview for the options
        underlying_price = float(overview_rows[0]['koers']) if overview_rows else None
        
        # Transform data
        overview_data = []
        options_data = []
        
        # Transform overview data (1 row per day)
        for row in overview_rows:
            overview_data.append(self.transform_to_bronze_overview(row))
            
        # Transform option contracts (many rows per day)
        for row in contracts_rows:
            options_data.append(self.transform_to_bronze_options(row, underlying_price))
            
        logger.info(f"🔄 Transformed: {len(overview_data)} overview records, {len(options_data)} option contracts")
        
        if dry_run:
            logger.info("🔍 DRY RUN - No data will be inserted")
            logger.info(f"   Would insert {len(overview_data)} overview records")
            logger.info(f"   Would insert {len(options_data)} option contracts")
            return {'overview': len(overview_data), 'options': len(options_data)}
            
        # Insert into PostgreSQL
        overview_count = self.insert_bronze_overview(overview_data)
        options_count = self.insert_bronze_options(options_data)
        
        return {'overview': overview_count, 'options': options_count}
        
    def migrate_all(self, dry_run: bool = False, limit_dates: int = None):
        """Migrate all historical data"""
        logger.info("\n" + "="*60)
        logger.info("🚀 STARTING HISTORICAL DATA MIGRATION")
        logger.info("="*60)
        
        # Connect to databases
        if not self.connect_mysql():
            logger.error("❌ Cannot connect to MySQL. Aborting.")
            return
            
        if not self.connect_postgres():
            logger.error("❌ Cannot connect to PostgreSQL. Aborting.")
            return
            
        # Get MySQL table schema
        logger.info("\n📋 MySQL Table Schema:")
        schema = self.get_mysql_table_schema('fd_option_overview')
        for col in schema[:10]:  # Show first 10 columns
            logger.info(f"   {col['Field']}: {col['Type']}")
        if len(schema) > 10:
            logger.info(f"   ... and {len(schema) - 10} more columns")
            
        # Get distinct dates
        dates = self.get_distinct_dates('fd_option_overview')
        
        if limit_dates:
            dates = dates[:limit_dates]
            logger.info(f"⚠️  Limited to {limit_dates} most recent dates")
            
        logger.info(f"\n📅 Dates to migrate: {len(dates)}")
        for date in dates[:5]:
            logger.info(f"   - {date}")
        if len(dates) > 5:
            logger.info(f"   ... and {len(dates) - 5} more dates")
            
        # Migrate each date
        total_overview = 0
        total_options = 0
        
        for i, date in enumerate(dates, 1):
            try:
                result = self.migrate_date(str(date), dry_run=dry_run)
                total_overview += result['overview']
                total_options += result['options']
                
                logger.info(f"✅ Progress: {i}/{len(dates)} dates completed")
                
            except Exception as e:
                logger.error(f"❌ Error migrating {date}: {e}")
                continue
                
        # Summary
        logger.info("\n" + "="*60)
        logger.info("🎉 MIGRATION COMPLETED")
        logger.info("="*60)
        logger.info(f"📊 Total overview records: {total_overview}")
        logger.info(f"📊 Total option records: {total_options}")
        logger.info(f"📅 Total dates processed: {len(dates)}")
        
        # Close connections
        if self.mysql_conn:
            self.mysql_conn.close()
        if self.postgres_conn:
            self.postgres_conn.close()
            

def main():
    """Main entry point"""
    import argparse
    
    parser = argparse.ArgumentParser(description='Migrate historical options data from MySQL to PostgreSQL')
    parser.add_argument('--dry-run', action='store_true', help='Show what would be migrated without inserting')
    parser.add_argument('--limit', type=int, help='Limit number of dates to migrate (most recent first)')
    parser.add_argument('--date', type=str, help='Migrate specific date only (YYYY-MM-DD)')
    
    args = parser.parse_args()
    
    migrator = HistoricalDataMigration()
    
    if args.date:
        # Migrate specific date
        if not migrator.connect_mysql():
            sys.exit(1)
        if not migrator.connect_postgres():
            sys.exit(1)
            
        result = migrator.migrate_date(args.date, dry_run=args.dry_run)
        logger.info(f"\n✅ Migrated {result['overview']} overview + {result['options']} options for {args.date}")
        
    else:
        # Migrate all
        migrator.migrate_all(dry_run=args.dry_run, limit_dates=args.limit)


if __name__ == '__main__':
    main()
