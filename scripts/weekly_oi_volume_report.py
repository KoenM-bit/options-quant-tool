#!/usr/bin/env python3
"""
Weekly Open Interest and Volume Analysis for Euronext Options.

Run this script on Fridays (or any day) to analyze:
- Weekly OI changes by strike and expiration
- Volume patterns and liquidity
- Top strikes by OI and volume
- Put/Call OI distribution

Usage:
    python scripts/weekly_oi_volume_report.py
    python scripts/weekly_oi_volume_report.py --days 7  # Custom lookback period
    python scripts/weekly_oi_volume_report.py --ticker AD.AS  # Specific ticker
"""

import os
import sys
from datetime import datetime, timedelta
from pathlib import Path
import argparse

# Add parent directory to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent))

from sqlalchemy import create_engine, text
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


def get_engine():
    """Create database engine."""
    return create_engine(DATABASE_URL)


def print_section(title):
    """Print a formatted section header."""
    print("\n" + "=" * 80)
    print(f" {title}")
    print("=" * 80)


def query_current_oi_by_strike(engine, ticker=None, min_oi=10):
    """Get current OI by strike and expiration (most recent data)."""
    ticker_filter = f"AND ticker = '{ticker}'" if ticker else ""
    
    query = f"""
    WITH latest_scrape AS (
        SELECT MAX(scraped_at) as max_scraped
        FROM bronze_euronext_options
        WHERE 1=1 {ticker_filter}
    )
    SELECT 
        ticker,
        option_type,
        strike,
        expiration_date,
        actual_expiration_date,
        open_interest,
        volume,
        settlement_price,
        underlying_last_price,
        ROUND(((strike / NULLIF(underlying_last_price, 0) - 1) * 100)::numeric, 2) as moneyness_pct,
        scraped_at::date as data_date
    FROM bronze_euronext_options
    WHERE scraped_at >= (SELECT max_scraped - interval '1 day' FROM latest_scrape)
        AND open_interest >= {min_oi}
        {ticker_filter}
    ORDER BY 
        ticker,
        actual_expiration_date,
        option_type DESC,
        open_interest DESC
    """
    
    return pd.read_sql(query, engine)


def query_oi_changes(engine, days=7, ticker=None):
    """Calculate OI changes over the past N days."""
    ticker_filter = f"AND ticker = '{ticker}'" if ticker else ""
    
    query = f"""
    WITH date_range AS (
        SELECT 
            MAX(scraped_at::date) as latest_date,
            MAX(scraped_at::date) - interval '{days} days' as start_date
        FROM bronze_euronext_options
        WHERE 1=1 {ticker_filter}
    ),
    latest_oi AS (
        SELECT DISTINCT ON (ticker, option_type, strike, expiration_date)
            ticker,
            option_type,
            strike,
            expiration_date,
            open_interest as oi_latest,
            scraped_at::date as latest_date
        FROM bronze_euronext_options
        WHERE scraped_at::date = (SELECT latest_date FROM date_range)
            {ticker_filter}
        ORDER BY ticker, option_type, strike, expiration_date, scraped_at DESC
    ),
    past_oi AS (
        SELECT DISTINCT ON (ticker, option_type, strike, expiration_date)
            ticker,
            option_type,
            strike,
            expiration_date,
            open_interest as oi_past,
            scraped_at::date as past_date
        FROM bronze_euronext_options
        WHERE scraped_at::date >= (SELECT start_date FROM date_range)
            AND scraped_at::date < (SELECT latest_date FROM date_range)
            {ticker_filter}
        ORDER BY ticker, option_type, strike, expiration_date, scraped_at
    )
    SELECT 
        l.ticker,
        l.option_type,
        l.strike,
        l.expiration_date,
        COALESCE(p.oi_past, 0) as oi_{days}d_ago,
        l.oi_latest as oi_current,
        l.oi_latest - COALESCE(p.oi_past, 0) as oi_change,
        CASE 
            WHEN COALESCE(p.oi_past, 0) > 0 THEN
                ROUND(((l.oi_latest - p.oi_past)::numeric / p.oi_past * 100), 1)
            ELSE NULL
        END as oi_change_pct,
        l.latest_date,
        p.past_date
    FROM latest_oi l
    LEFT JOIN past_oi p 
        ON l.ticker = p.ticker
        AND l.option_type = p.option_type
        AND l.strike = p.strike
        AND l.expiration_date = p.expiration_date
    WHERE l.oi_latest > 0 OR COALESCE(p.oi_past, 0) > 0
    ORDER BY ABS(l.oi_latest - COALESCE(p.oi_past, 0)) DESC
    """
    
    return pd.read_sql(query, engine)


def query_volume_summary(engine, days=7, ticker=None):
    """Get volume summary for the past N days."""
    ticker_filter = f"AND ticker = '{ticker}'" if ticker else ""
    
    query = f"""
    SELECT 
        ticker,
        option_type,
        strike,
        expiration_date,
        actual_expiration_date,
        SUM(COALESCE(volume, 0)) as total_volume,
        AVG(COALESCE(volume, 0))::int as avg_daily_volume,
        MAX(open_interest) as latest_oi,
        COUNT(DISTINCT scraped_at::date) as days_with_data,
        MAX(scraped_at::date) as latest_date
    FROM bronze_euronext_options
    WHERE scraped_at::date >= CURRENT_DATE - interval '{days} days'
        {ticker_filter}
    GROUP BY ticker, option_type, strike, expiration_date, actual_expiration_date
    HAVING SUM(COALESCE(volume, 0)) > 0
    ORDER BY total_volume DESC
    LIMIT 50
    """
    
    return pd.read_sql(query, engine)


def query_put_call_oi_ratio(engine, ticker=None):
    """Calculate put/call OI ratio by expiration."""
    ticker_filter = f"AND ticker = '{ticker}'" if ticker else ""
    
    query = f"""
    WITH latest_scrape AS (
        SELECT MAX(scraped_at) as max_scraped
        FROM bronze_euronext_options
        WHERE 1=1 {ticker_filter}
    )
    SELECT 
        ticker,
        expiration_date,
        actual_expiration_date,
        SUM(CASE WHEN option_type = 'C' THEN open_interest ELSE 0 END) as call_oi,
        SUM(CASE WHEN option_type = 'P' THEN open_interest ELSE 0 END) as put_oi,
        SUM(open_interest) as total_oi,
        ROUND(
            SUM(CASE WHEN option_type = 'P' THEN open_interest ELSE 0 END)::numeric /
            NULLIF(SUM(CASE WHEN option_type = 'C' THEN open_interest ELSE 0 END), 0),
            2
        ) as put_call_ratio
    FROM bronze_euronext_options
    WHERE scraped_at >= (SELECT max_scraped - interval '1 day' FROM latest_scrape)
        AND open_interest > 0
        {ticker_filter}
    GROUP BY ticker, expiration_date, actual_expiration_date
    ORDER BY actual_expiration_date
    """
    
    return pd.read_sql(query, engine)


def query_near_money_strikes(engine, ticker=None, moneyness_range=10):
    """Get strikes near the current underlying price (within +/- X%)."""
    ticker_filter = f"AND ticker = '{ticker}'" if ticker else ""
    
    query = f"""
    WITH latest_scrape AS (
        SELECT MAX(scraped_at) as max_scraped
        FROM bronze_euronext_options
        WHERE 1=1 {ticker_filter}
    )
    SELECT 
        ticker,
        option_type,
        strike,
        expiration_date,
        actual_expiration_date,
        open_interest,
        volume,
        settlement_price,
        underlying_last_price,
        ROUND(((strike / NULLIF(underlying_last_price, 0) - 1) * 100)::numeric, 2) as moneyness_pct,
        CASE 
            WHEN strike < underlying_last_price THEN 'ITM'
            WHEN strike = underlying_last_price THEN 'ATM'
            ELSE 'OTM'
        END as moneyness
    FROM bronze_euronext_options
    WHERE scraped_at >= (SELECT max_scraped - interval '1 day' FROM latest_scrape)
        AND open_interest > 0
        AND underlying_last_price IS NOT NULL
        AND ABS((strike / NULLIF(underlying_last_price, 0) - 1) * 100) <= {moneyness_range}
        {ticker_filter}
    ORDER BY 
        ticker,
        actual_expiration_date,
        option_type DESC,
        ABS(strike - underlying_last_price)
    """
    
    return pd.read_sql(query, engine)


def main():
    parser = argparse.ArgumentParser(description="Weekly OI and Volume Report for Euronext Options")
    parser.add_argument("--days", type=int, default=7, help="Lookback period in days (default: 7)")
    parser.add_argument("--ticker", type=str, default=None, help="Filter by ticker (e.g., AH-DAMS)")
    parser.add_argument("--min-oi", type=int, default=10, help="Minimum OI to include (default: 10)")
    parser.add_argument("--moneyness", type=float, default=10.0, help="Moneyness range in percent (default: 10)")
    args = parser.parse_args()
    
    engine = get_engine()
    
    # Print report header
    print("\n" + "█" * 80)
    print("  📊 WEEKLY OPEN INTEREST & VOLUME REPORT")
    print("█" * 80)
    print(f"  Report Date: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"  Lookback Period: {args.days} days")
    if args.ticker:
        print(f"  Ticker Filter: {args.ticker}")
    print("█" * 80)
    
    # 1. Put/Call OI Ratio by Expiration
    print_section("1. PUT/CALL OI RATIO BY EXPIRATION")
    pc_ratio_df = query_put_call_oi_ratio(engine, args.ticker)
    if not pc_ratio_df.empty:
        print(pc_ratio_df.to_string(index=False))
        print(f"\nTotal Contracts Tracked: {pc_ratio_df['total_oi'].sum():,}")
    else:
        print("No data available.")
    
    # 2. Top Strikes by Current OI
    print_section("2. TOP STRIKES BY CURRENT OPEN INTEREST")
    current_oi_df = query_current_oi_by_strike(engine, args.ticker, args.min_oi)
    if not current_oi_df.empty:
        print(current_oi_df.head(30).to_string(index=False))
    else:
        print("No data available.")
    
    # 3. OI Changes Over Past Week
    print_section(f"3. BIGGEST OI CHANGES (PAST {args.days} DAYS)")
    oi_changes_df = query_oi_changes(engine, args.days, args.ticker)
    if not oi_changes_df.empty:
        print("\n📈 Top OI Increases:")
        increases = oi_changes_df[oi_changes_df['oi_change'] > 0].head(15)
        if not increases.empty:
            print(increases.to_string(index=False))
        else:
            print("No increases found.")
        
        print("\n📉 Top OI Decreases:")
        decreases = oi_changes_df[oi_changes_df['oi_change'] < 0].head(15)
        if not decreases.empty:
            print(decreases.to_string(index=False))
        else:
            print("No decreases found.")
    else:
        print("No OI change data available.")
    
    # 4. Volume Summary
    print_section(f"4. TOP VOLUME STRIKES (PAST {args.days} DAYS)")
    volume_df = query_volume_summary(engine, args.days, args.ticker)
    if not volume_df.empty:
        print(volume_df.head(20).to_string(index=False))
    else:
        print("No volume data available.")
    
    # 5. Near-the-Money Strikes
    print_section(f"5. NEAR-THE-MONEY STRIKES (±{args.moneyness}%)")
    near_money_df = query_near_money_strikes(engine, args.ticker, args.moneyness)
    if not near_money_df.empty:
        print(near_money_df.to_string(index=False))
    else:
        print("No data available.")
    
    # Summary statistics
    print_section("6. SUMMARY STATISTICS")
    if not current_oi_df.empty:
        print(f"Total Unique Contracts: {len(current_oi_df)}")
        print(f"Total Open Interest: {current_oi_df['open_interest'].sum():,}")
        print(f"Average OI per Contract: {current_oi_df['open_interest'].mean():.0f}")
        print(f"Median OI: {current_oi_df['open_interest'].median():.0f}")
        
        if 'underlying_last_price' in current_oi_df.columns:
            last_prices = current_oi_df.dropna(subset=['underlying_last_price'])
            if not last_prices.empty:
                print(f"\nUnderlying Prices:")
                for ticker in last_prices['ticker'].unique():
                    price = last_prices[last_prices['ticker'] == ticker]['underlying_last_price'].iloc[0]
                    print(f"  {ticker}: €{price:.2f}")
    
    print("\n" + "█" * 80)
    print("  ✅ Report Complete")
    print("█" * 80 + "\n")


if __name__ == "__main__":
    main()
