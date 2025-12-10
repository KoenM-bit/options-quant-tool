"""
Inspect MariaDB option_prices_live table structure and sample data
to plan migration to Bronze BD layer.
"""

import os
import sys
import mysql.connector
from dotenv import load_dotenv
from datetime import datetime
import json

# Load environment
load_dotenv('.env.migration')

def connect_mysql():
    """Connect to MariaDB"""
    config = {
        "host": os.getenv("MYSQL_HOST", "192.168.1.201"),
        "user": os.getenv("MYSQL_USER", "remoteuser"),
        "password": os.getenv("MYSQL_PASSWORD"),
        "database": os.getenv("MYSQL_DATABASE", "optionsdb"),
        "port": int(os.getenv("MYSQL_PORT", 3306)),
    }
    
    return mysql.connector.connect(**config)


def inspect_table_structure(table_name='option_prices_live'):
    """Get table structure"""
    conn = connect_mysql()
    cursor = conn.cursor(dictionary=True)
    
    print(f"\n{'='*80}")
    print(f"TABLE STRUCTURE: {table_name}")
    print('='*80)
    
    cursor.execute(f"DESCRIBE {table_name}")
    columns = cursor.fetchall()
    
    print(f"\n{'Column':<30} {'Type':<20} {'Null':<10} {'Key':<10} {'Default':<15}")
    print('-'*80)
    for col in columns:
        print(f"{col['Field']:<30} {col['Type']:<20} {col['Null']:<10} {col['Key']:<10} {str(col['Default']):<15}")
    
    cursor.close()
    conn.close()
    
    return columns


def get_sample_data(table_name='option_prices_live', limit=5):
    """Get sample rows"""
    conn = connect_mysql()
    cursor = conn.cursor(dictionary=True)
    
    print(f"\n{'='*80}")
    print(f"SAMPLE DATA: {table_name} (limit {limit})")
    print('='*80)
    
    cursor.execute(f"SELECT * FROM {table_name} ORDER BY id DESC LIMIT {limit}")
    rows = cursor.fetchall()
    
    if rows:
        # Print first row in detail
        print(f"\nFirst row (detailed):")
        print('-'*80)
        for key, value in rows[0].items():
            print(f"{key:<30} : {value}")
        
        # Print all rows summary
        print(f"\nAll {len(rows)} rows:")
        print('-'*80)
        for i, row in enumerate(rows, 1):
            print(f"\nRow {i}:")
            print(json.dumps({k: str(v) for k, v in row.items()}, indent=2))
    else:
        print("No data found!")
    
    cursor.close()
    conn.close()
    
    return rows


def get_table_stats(table_name='option_prices_live'):
    """Get statistics about the table"""
    conn = connect_mysql()
    cursor = conn.cursor(dictionary=True)
    
    print(f"\n{'='*80}")
    print(f"TABLE STATISTICS: {table_name}")
    print('='*80)
    
    # Total rows
    cursor.execute(f"SELECT COUNT(*) as total FROM {table_name}")
    total = cursor.fetchone()['total']
    print(f"\nTotal rows: {total:,}")
    
    # Date range - try multiple timestamp columns
    try:
        cursor.execute(f"""
            SELECT 
                MIN(created_at) as earliest,
                MAX(created_at) as latest
            FROM {table_name}
        """)
        date_range = cursor.fetchone()
        print(f"Date range (created_at): {date_range['earliest']} to {date_range['latest']}")
    except:
        print("Date range: N/A (no created_at column)")
    
    # Distinct tickers
    try:
        cursor.execute(f"SELECT COUNT(DISTINCT ticker) as ticker_count FROM {table_name}")
        ticker_count = cursor.fetchone()['ticker_count']
        print(f"Distinct tickers: {ticker_count}")
    except:
        print("Distinct tickers: N/A")
    
    # Distinct dates
    try:
        cursor.execute(f"SELECT COUNT(DISTINCT DATE(created_at)) as date_count FROM {table_name}")
        date_count = cursor.fetchone()['date_count']
        print(f"Distinct dates: {date_count}")
    except:
        print("Distinct dates: N/A")
    
    # Check if there are option_type, strike, expiry columns
    cursor.execute(f"DESCRIBE {table_name}")
    columns = {row['Field'] for row in cursor.fetchall()}
    
    print(f"\nKey columns present:")
    for key_col in ['ticker', 'type', 'strike', 'expiry', 'bid', 'ask', 'last_price', 'volume', 'created_at', 'issue_id', 'spot_price']:
        status = '✅' if key_col in columns else '❌'
        print(f"  {status} {key_col}")
    
    cursor.close()
    conn.close()


def list_all_tables():
    """List all tables in the database"""
    conn = connect_mysql()
    cursor = conn.cursor()
    
    print(f"\n{'='*80}")
    print(f"ALL TABLES IN DATABASE: {os.getenv('MYSQL_DATABASE', 'optionsdb')}")
    print('='*80)
    
    cursor.execute("SHOW TABLES")
    tables = cursor.fetchall()
    
    for i, table in enumerate(tables, 1):
        table_name = table[0]
        cursor.execute(f"SELECT COUNT(*) FROM {table_name}")
        count = cursor.fetchone()[0]
        print(f"{i:2}. {table_name:<40} ({count:,} rows)")
    
    cursor.close()
    conn.close()


def main():
    import argparse
    
    parser = argparse.ArgumentParser(description='Inspect MariaDB options tables')
    parser.add_argument('--list-tables', action='store_true', help='List all tables in database')
    parser.add_argument('--table', default='option_prices_live', help='Table name to inspect')
    parser.add_argument('--limit', type=int, default=5, help='Number of sample rows')
    parser.add_argument('--structure-only', action='store_true', help='Show only table structure')
    
    args = parser.parse_args()
    
    try:
        if args.list_tables:
            list_all_tables()
        else:
            # Inspect specific table
            inspect_table_structure(args.table)
            
            if not args.structure_only:
                get_table_stats(args.table)
                get_sample_data(args.table, args.limit)
        
        print(f"\n{'='*80}")
        print("✅ Inspection complete!")
        print('='*80)
        
    except mysql.connector.Error as e:
        print(f"\n❌ MySQL Error: {e}")
        sys.exit(1)
    except Exception as e:
        print(f"\n❌ Error: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)


if __name__ == '__main__':
    main()
