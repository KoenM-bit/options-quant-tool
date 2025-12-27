#!/usr/bin/env python3
"""
Quick SQL query tool for bronze_euronext_options table.

Usage:
    python scripts/quick_oi_query.py "SELECT * FROM bronze_euronext_options LIMIT 5"
    python scripts/quick_oi_query.py --preset top-oi
    python scripts/quick_oi_query.py --preset by-expiry
"""

import os
import sys
from pathlib import Path
import argparse

# Add parent directory to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent))

from sqlalchemy import create_engine
from dotenv import load_dotenv
import pandas as pd

# Load environment variables
load_dotenv()

# Database connection from .env
DB_HOST = os.getenv("POSTGRES_HOST", "192.168.1.201")
DB_PORT = os.getenv("POSTGRES_PORT", "5433")
DB_NAME = os.getenv("POSTGRES_DB", "ahold_options")
DB_USER = os.getenv("POSTGRES_USER", "airflow")
DB_PASSWORD = os.getenv("POSTGRES_PASSWORD", "airflow")

DATABASE_URL = f"postgresql://{DB_USER}:{DB_PASSWORD}@{DB_HOST}:{DB_PORT}/{DB_NAME}"

# Preset queries
PRESETS = {
    "top-oi": """
        WITH latest_scrape AS (
            SELECT MAX(scraped_at) as max_scraped
            FROM bronze_euronext_options
        )
        SELECT 
            ticker,
            option_type,
            strike,
            expiration_date,
            open_interest,
            volume,
            settlement_price
        FROM bronze_euronext_options
        WHERE scraped_at >= (SELECT max_scraped - interval '1 day' FROM latest_scrape)
        ORDER BY open_interest DESC
        LIMIT 30
    """,
    
    "by-expiry": """
        WITH latest_scrape AS (
            SELECT MAX(scraped_at) as max_scraped
            FROM bronze_euronext_options
        )
        SELECT 
            expiration_date,
            actual_expiration_date,
            option_type,
            COUNT(*) as num_contracts,
            SUM(open_interest) as total_oi,
            SUM(volume) as total_volume
        FROM bronze_euronext_options
        WHERE scraped_at >= (SELECT max_scraped - interval '1 day' FROM latest_scrape)
        GROUP BY expiration_date, actual_expiration_date, option_type
        ORDER BY actual_expiration_date, option_type DESC
    """,
    
    "recent-activity": """
        SELECT 
            ticker,
            option_type,
            strike,
            expiration_date,
            volume,
            open_interest,
            scraped_at::date as date
        FROM bronze_euronext_options
        WHERE scraped_at::date >= CURRENT_DATE - interval '3 days'
            AND volume > 0
        ORDER BY scraped_at DESC, volume DESC
        LIMIT 50
    """,
    
    "strikes-near-money": """
        WITH latest_scrape AS (
            SELECT MAX(scraped_at) as max_scraped
            FROM bronze_euronext_options
        ),
        latest_data AS (
            SELECT *
            FROM bronze_euronext_options
            WHERE scraped_at >= (SELECT max_scraped - interval '1 day' FROM latest_scrape)
                AND underlying_last_price IS NOT NULL
        )
        SELECT 
            ticker,
            option_type,
            strike,
            expiration_date,
            open_interest,
            volume,
            underlying_last_price,
            strike - underlying_last_price as distance_from_spot,
            ROUND(((strike / underlying_last_price - 1) * 100)::numeric, 2) as moneyness_pct
        FROM latest_data
        WHERE ABS(strike - underlying_last_price) <= 5
        ORDER BY expiration_date, ABS(strike - underlying_last_price)
    """,
    
    "volume-leaders": """
        SELECT 
            ticker,
            option_type,
            strike,
            expiration_date,
            SUM(volume) as total_volume_7d,
            MAX(open_interest) as latest_oi,
            COUNT(DISTINCT scraped_at::date) as days_traded
        FROM bronze_euronext_options
        WHERE scraped_at::date >= CURRENT_DATE - interval '7 days'
            AND volume > 0
        GROUP BY ticker, option_type, strike, expiration_date
        ORDER BY total_volume_7d DESC
        LIMIT 30
    """,
    
    "data-freshness": """
        SELECT 
            MAX(scraped_at) as latest_scrape,
            MAX(scraped_at::date) as latest_date,
            COUNT(DISTINCT scraped_at::date) as days_available,
            COUNT(*) as total_records,
            COUNT(DISTINCT ticker || option_type || strike::text || expiration_date) as unique_contracts
        FROM bronze_euronext_options
    """
}


def main():
    parser = argparse.ArgumentParser(
        description="Quick SQL query tool for bronze_euronext_options",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=f"""
Available presets:
  {chr(10).join(f'  {name}: {doc}' for name, doc in [
      ('top-oi', 'Top 30 contracts by open interest'),
      ('by-expiry', 'OI and volume grouped by expiration'),
      ('recent-activity', 'Recent trading activity (last 3 days)'),
      ('strikes-near-money', 'Strikes within ±5 from underlying price'),
      ('volume-leaders', 'Top volume over last 7 days'),
      ('data-freshness', 'Database freshness check')
  ])}

Examples:
  python scripts/quick_oi_query.py --preset top-oi
  python scripts/quick_oi_query.py "SELECT COUNT(*) FROM bronze_euronext_options"
  python scripts/quick_oi_query.py --preset by-expiry --output csv
        """
    )
    
    parser.add_argument("query", nargs="?", help="Custom SQL query")
    parser.add_argument("--preset", choices=PRESETS.keys(), help="Use a preset query")
    parser.add_argument("--output", choices=["table", "csv", "json"], default="table",
                       help="Output format (default: table)")
    parser.add_argument("--no-index", action="store_true", help="Hide index in output")
    
    args = parser.parse_args()
    
    if not args.query and not args.preset:
        parser.print_help()
        sys.exit(1)
    
    # Get query
    if args.preset:
        query = PRESETS[args.preset]
        print(f"🔍 Running preset: {args.preset}\n")
    else:
        query = args.query
    
    # Execute query
    try:
        engine = create_engine(DATABASE_URL)
        df = pd.read_sql(query, engine)
        
        # Output results
        if args.output == "table":
            if df.empty:
                print("No results found.")
            else:
                print(df.to_string(index=not args.no_index))
                print(f"\n📊 Rows: {len(df)}")
        elif args.output == "csv":
            print(df.to_csv(index=not args.no_index))
        elif args.output == "json":
            print(df.to_json(orient="records", indent=2))
            
    except Exception as e:
        print(f"❌ Error: {e}", file=sys.stderr)
        sys.exit(1)


if __name__ == "__main__":
    main()
