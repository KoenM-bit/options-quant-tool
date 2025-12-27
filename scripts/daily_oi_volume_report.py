#!/usr/bin/env python3
"""
Daily OI and Volume Report - Simple Table View

Shows daily OI and volume data for ATM and surrounding strikes (±6 strikes)
in a clear table format for the current and next expiry months.

Usage:
    python scripts/daily_oi_volume_report.py
    python scripts/daily_oi_volume_report.py --ticker AH
    python scripts/daily_oi_volume_report.py --strike-range 8
    python scripts/daily_oi_volume_report.py --days 7
"""

import os
import sys
from datetime import datetime, timedelta
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


def get_engine():
    """Create database engine."""
    return create_engine(DATABASE_URL)


def print_section(title):
    """Print a formatted section header."""
    print("\n" + "=" * 120)
    print(f" {title}")
    print("=" * 120)


def get_atm_strikes_for_expiry(engine, ticker, expiry, strike_range=6):
    """Get ATM strikes for a specific expiry."""
    query = f"""
    WITH strike_data AS (
        SELECT DISTINCT 
            strike,
            MAX(open_interest) as max_oi
        FROM bronze_euronext_options
        WHERE ticker = '{ticker}'
            AND expiration_date = '{expiry}'
            AND open_interest > 0
        GROUP BY strike
        ORDER BY strike
    ),
    -- Estimate ATM from strike distribution
    atm_estimate AS (
        SELECT (MIN(strike) + MAX(strike)) / 2 as atm_price
        FROM strike_data
    )
    SELECT strike
    FROM strike_data, atm_estimate
    ORDER BY ABS(strike - atm_price)
    LIMIT {strike_range * 2 + 1}
    """
    
    df = pd.read_sql(query, engine)
    return sorted(df['strike'].tolist()) if not df.empty else []


def get_daily_data(engine, ticker, lookback_days=10):
    """
    Get daily OI and volume data for recent days.
    Returns: date, expiration_date, option_type, strike, open_interest, volume
    """
    
    query = f"""
    WITH recent_expiries AS (
        SELECT DISTINCT 
            expiration_date,
            actual_expiration_date
        FROM bronze_euronext_options
        WHERE ticker = '{ticker}'
            AND scraped_at::date >= CURRENT_DATE - interval '{lookback_days} days'
            AND actual_expiration_date IS NOT NULL
        ORDER BY actual_expiration_date
        LIMIT 2
    )
    SELECT 
        b.scraped_at::date as trade_date,
        b.expiration_date,
        b.actual_expiration_date,
        b.option_type,
        b.strike,
        b.open_interest,
        COALESCE(b.volume, 0) as volume
    FROM bronze_euronext_options b
    INNER JOIN recent_expiries re ON b.expiration_date = re.expiration_date
    WHERE b.ticker = '{ticker}'
        AND b.scraped_at::date >= CURRENT_DATE - interval '{lookback_days} days'
    ORDER BY 
        b.actual_expiration_date,
        b.scraped_at::date DESC,
        b.option_type DESC,
        b.strike
    """
    
    return pd.read_sql(query, engine)


def filter_atm_strikes(df, engine, ticker, strike_range=6):
    """Filter dataframe to only include ATM strikes for each expiry."""
    if df.empty:
        return df
    
    filtered_dfs = []
    for expiry in df['expiration_date'].unique():
        atm_strikes = get_atm_strikes_for_expiry(engine, ticker, expiry, strike_range)
        if atm_strikes:
            expiry_df = df[df['expiration_date'] == expiry]
            filtered_df = expiry_df[expiry_df['strike'].isin(atm_strikes)]
            filtered_dfs.append(filtered_df)
    
    return pd.concat(filtered_dfs) if filtered_dfs else pd.DataFrame()


def print_expiry_table(df, expiry_label):
    """Print data for one expiry in table format."""
    if df.empty:
        print(f"No data available for {expiry_label}")
        return
    
    print(f"\n{expiry_label}")
    print("-" * 120)
    
    # Get unique dates sorted
    dates = sorted(df['trade_date'].unique(), reverse=True)
    
    for opt_type in ['C', 'P']:
        type_df = df[df['option_type'] == opt_type].copy()
        if type_df.empty:
            continue
        
        print(f"\n{'CALLS' if opt_type == 'C' else 'PUTS'}:")
        print(f"{'Date':<12} {'Strike':>8} {'OI':>8} {'Volume':>8} {'OI Δ':>10} {'Vol Δ':>10}")
        print("-" * 120)
        
        # Get strikes sorted
        strikes = sorted(type_df['strike'].unique())
        
        for strike in strikes:
            strike_df = type_df[type_df['strike'] == strike].sort_values('trade_date', ascending=False)
            
            if strike_df.empty:
                continue
            
            # Print header for this strike
            print(f"\nStrike €{strike:.2f}:")
            
            prev_oi = None
            prev_vol = None
            
            for _, row in strike_df.iterrows():
                date_str = row['trade_date'].strftime('%Y-%m-%d')
                oi = row['open_interest'] if pd.notna(row['open_interest']) else 0
                vol = row['volume'] if pd.notna(row['volume']) else 0
                
                # Calculate changes
                if prev_oi is not None:
                    oi_change = oi - prev_oi
                    vol_change = vol - prev_vol
                    oi_change_str = f"{oi_change:+,.0f}"
                    vol_change_str = f"{vol_change:+,.0f}"
                else:
                    oi_change_str = "-"
                    vol_change_str = "-"
                
                print(f"  {date_str:<12} {strike:>8.2f} {oi:>8,.0f} {vol:>8,.0f} {oi_change_str:>10} {vol_change_str:>10}")
                
                prev_oi = oi
                prev_vol = vol


def print_summary_table(df):
    """Print summary statistics."""
    if df.empty:
        return
    
    print_section("SUMMARY BY DATE")
    
    # Group by date, expiry, and option type
    summary = df.groupby(['trade_date', 'expiration_date', 'option_type']).agg({
        'open_interest': 'sum',
        'volume': 'sum',
        'strike': 'count'
    }).reset_index()
    
    summary.columns = ['Date', 'Expiry', 'Type', 'Total_OI', 'Total_Vol', 'Num_Strikes']
    summary = summary.sort_values(['Date', 'Expiry', 'Type'], ascending=[False, True, False])
    
    print("\n" + summary.to_string(index=False))
    
    # Put/Call ratios
    print("\n\nPut/Call OI Ratios by Date:")
    print(f"{'Date':<12} {'Expiry':<15} {'Call OI':>12} {'Put OI':>12} {'P/C Ratio':>12}")
    print("-" * 70)
    
    for date in sorted(df['trade_date'].unique(), reverse=True):
        date_df = df[df['trade_date'] == date]
        for expiry in sorted(date_df['expiration_date'].unique()):
            expiry_df = date_df[date_df['expiration_date'] == expiry]
            call_oi = expiry_df[expiry_df['option_type'] == 'C']['open_interest'].sum()
            put_oi = expiry_df[expiry_df['option_type'] == 'P']['open_interest'].sum()
            pc_ratio = put_oi / max(call_oi, 1)
            
            print(f"{date.strftime('%Y-%m-%d'):<12} {expiry:<15} {call_oi:>12,.0f} {put_oi:>12,.0f} {pc_ratio:>12.2f}")


def main():
    parser = argparse.ArgumentParser(description="Daily OI and Volume Report")
    parser.add_argument("--ticker", type=str, default="AH", help="Ticker symbol (default: AH)")
    parser.add_argument("--strike-range", type=int, default=6, help="Number of strikes above/below ATM (default: 6)")
    parser.add_argument("--days", type=int, default=10, help="Lookback period in days (default: 10)")
    args = parser.parse_args()
    
    engine = get_engine()
    
    # Print report header
    print("\n" + "█" * 120)
    print("  📊 DAILY OI & VOLUME REPORT")
    print("█" * 120)
    print(f"  Report Date: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"  Ticker: {args.ticker}")
    print(f"  Analysis Period: Last {args.days} days")
    print(f"  Strike Range: ATM ± {args.strike_range} strikes")
    print("█" * 120)
    
    # Get data
    print("\n⏳ Fetching data from database...")
    df = get_daily_data(engine, args.ticker, args.days)
    
    if df.empty:
        print(f"\n❌ No data found for ticker {args.ticker}")
        return
    
    print(f"✅ Found {len(df)} records across {df['trade_date'].nunique()} days")
    print(f"   Available dates: {', '.join([d.strftime('%Y-%m-%d') for d in sorted(df['trade_date'].unique())])}")
    
    # Filter to ATM strikes
    print(f"\n⏳ Filtering to ATM ± {args.strike_range} strikes...")
    df_filtered = filter_atm_strikes(df, engine, args.ticker, args.strike_range)
    
    if df_filtered.empty:
        print("\n❌ No data after filtering for ATM strikes")
        return
    
    print(f"✅ Filtered to {len(df_filtered)} records")
    
    # Get expiries
    expiries = sorted(df_filtered['expiration_date'].unique())
    
    # Print data for each expiry
    for i, expiry in enumerate(expiries):
        expiry_df = df_filtered[df_filtered['expiration_date'] == expiry]
        actual_exp = expiry_df.iloc[0]['actual_expiration_date']
        
        label = f"{'CURRENT' if i == 0 else 'NEXT'} EXPIRY: {expiry} (Actual: {actual_exp.strftime('%Y-%m-%d')})"
        print_section(label)
        print_expiry_table(expiry_df, expiry)
    
    # Summary
    print_summary_table(df_filtered)
    
    print("\n" + "█" * 120)
    print("  ✅ Report Complete")
    print("█" * 120 + "\n")


if __name__ == "__main__":
    main()
