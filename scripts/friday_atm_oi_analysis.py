#!/usr/bin/env python3
"""
Friday ATM Options Analysis - OI and Volume Changes

Analyzes the last 4 Fridays of data for ATM and surrounding strikes (±6 strikes)
for the current expiry month and next expiry month.

Shows week-over-week changes in Open Interest and Volume.

Usage:
    python scripts/friday_atm_oi_analysis.py
    python scripts/friday_atm_oi_analysis.py --ticker AH-DAMS
    python scripts/friday_atm_oi_analysis.py --strike-range 8  # ±8 strikes instead of ±6
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
import numpy as np

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
    print("\n" + "=" * 100)
    print(f" {title}")
    print("=" * 100)


def get_last_fridays(n=4):
    """Get the last N Friday dates."""
    today = datetime.now().date()
    fridays = []
    
    # Start from today and go back
    current = today
    while len(fridays) < n:
        # Go back day by day
        if current.weekday() == 4:  # Friday is 4
            fridays.append(current)
        current = current - timedelta(days=1)
    
    return sorted(fridays)


def get_nearest_strikes(engine, ticker, expiry, underlying_price, strike_range=6):
    """Get the nearest strikes around the underlying price."""
    query = f"""
    WITH latest_data AS (
        SELECT DISTINCT strike
        FROM bronze_euronext_options
        WHERE ticker = '{ticker}'
            AND expiration_date = '{expiry}'
        ORDER BY strike
    )
    SELECT strike,
           ABS(strike - {underlying_price}) as distance
    FROM latest_data
    ORDER BY distance
    LIMIT {strike_range * 2 + 1}
    """
    
    df = pd.read_sql(query, engine)
    return sorted(df['strike'].tolist())


def get_friday_oi_volume_data(engine, ticker=None, strike_range=6):
    """
    Get OI and volume data for Fridays, focusing on ATM and nearby strikes
    for current and next expiry months.
    """
    ticker_filter = f"AND ticker = '{ticker}'" if ticker else ""
    
    # Get last 4 Fridays
    fridays = get_last_fridays(4)
    friday_dates_str = "', '".join([f.strftime('%Y-%m-%d') for f in fridays])
    
    query = f"""
    WITH friday_data AS (
        SELECT 
            ticker,
            option_type,
            strike,
            expiration_date,
            actual_expiration_date,
            open_interest,
            volume,
            underlying_last_price,
            scraped_at::date as trade_date,
            EXTRACT(YEAR FROM actual_expiration_date) as exp_year,
            EXTRACT(MONTH FROM actual_expiration_date) as exp_month
        FROM bronze_euronext_options
        WHERE scraped_at::date IN ('{friday_dates_str}')
            {ticker_filter}
            AND actual_expiration_date IS NOT NULL
            AND actual_expiration_date >= CURRENT_DATE
    ),
    -- Get the two nearest expiry months
    next_expiries AS (
        SELECT DISTINCT 
            exp_year,
            exp_month,
            actual_expiration_date,
            expiration_date
        FROM friday_data
        ORDER BY actual_expiration_date
        LIMIT 2
    ),
    -- Get latest underlying price
    latest_price AS (
        SELECT DISTINCT ON (ticker)
            ticker,
            underlying_last_price
        FROM friday_data
        WHERE underlying_last_price IS NOT NULL
        ORDER BY ticker, trade_date DESC
    )
    SELECT 
        fd.ticker,
        fd.option_type,
        fd.strike,
        fd.expiration_date,
        fd.actual_expiration_date,
        fd.open_interest,
        COALESCE(fd.volume, 0) as volume,
        fd.trade_date,
        lp.underlying_last_price,
        fd.strike - lp.underlying_last_price as distance_from_atm,
        ABS(fd.strike - lp.underlying_last_price) as abs_distance
    FROM friday_data fd
    INNER JOIN next_expiries ne 
        ON fd.exp_year = ne.exp_year 
        AND fd.exp_month = ne.exp_month
    LEFT JOIN latest_price lp 
        ON fd.ticker = lp.ticker
    WHERE lp.underlying_last_price IS NOT NULL
    ORDER BY 
        fd.actual_expiration_date,
        fd.trade_date DESC,
        fd.option_type DESC,
        abs_distance,
        fd.strike
    """
    
    df = pd.read_sql(query, engine)
    
    if df.empty:
        return df
    
    # Filter to only ATM ± strike_range for each expiry
    filtered_dfs = []
    for expiry in df['actual_expiration_date'].unique():
        expiry_df = df[df['actual_expiration_date'] == expiry].copy()
        
        # Get the strikes sorted by distance from ATM
        strikes_by_distance = expiry_df.groupby('strike')['abs_distance'].first().sort_values()
        atm_strikes = strikes_by_distance.head(strike_range * 2 + 1).index.tolist()
        
        filtered_df = expiry_df[expiry_df['strike'].isin(atm_strikes)]
        filtered_dfs.append(filtered_df)
    
    return pd.concat(filtered_dfs) if filtered_dfs else pd.DataFrame()


def calculate_weekly_changes(df):
    """Calculate week-over-week changes in OI and volume."""
    if df.empty:
        return df
    
    # Sort by date
    df = df.sort_values('trade_date')
    
    # Pivot to get OI and volume by date for each strike/type combo
    changes = []
    
    for (ticker, expiry, opt_type, strike), group in df.groupby(['ticker', 'actual_expiration_date', 'option_type', 'strike']):
        group = group.sort_values('trade_date')
        
        if len(group) < 2:
            continue
        
        dates = group['trade_date'].tolist()
        oi_values = group['open_interest'].tolist()
        vol_values = group['volume'].tolist()
        
        # Calculate changes between consecutive weeks
        for i in range(len(dates) - 1):
            oi_change = oi_values[i + 1] - oi_values[i]
            vol_change = vol_values[i + 1] - vol_values[i]
            oi_pct_change = (oi_change / oi_values[i] * 100) if oi_values[i] > 0 else None
            
            changes.append({
                'ticker': ticker,
                'expiration_date': expiry,
                'option_type': opt_type,
                'strike': strike,
                'from_date': dates[i],
                'to_date': dates[i + 1],
                'oi_from': oi_values[i],
                'oi_to': oi_values[i + 1],
                'oi_change': oi_change,
                'oi_pct_change': oi_pct_change,
                'vol_from': vol_values[i],
                'vol_to': vol_values[i + 1],
                'vol_change': vol_change,
                'underlying_price': group.iloc[0]['underlying_last_price'],
                'distance_from_atm': group.iloc[0]['distance_from_atm']
            })
    
    return pd.DataFrame(changes)


def format_change(value, is_percentage=False):
    """Format a change value with color indicators."""
    if pd.isna(value):
        return "N/A"
    
    if is_percentage:
        sign = "📈" if value > 0 else "📉" if value < 0 else "➡️"
        return f"{sign} {value:+.1f}%"
    else:
        sign = "📈" if value > 0 else "📉" if value < 0 else "➡️"
        return f"{sign} {value:+,.0f}"


def print_weekly_comparison_table(df, expiry_label):
    """Print a formatted table showing weekly OI and volume changes."""
    if df.empty:
        print(f"No data available for {expiry_label}")
        return
    
    print(f"\n{expiry_label}")
    print("-" * 100)
    
    # Get unique dates
    dates = sorted(df['to_date'].unique())
    
    for opt_type in ['C', 'P']:
        type_df = df[df['option_type'] == opt_type].copy()
        if type_df.empty:
            continue
        
        print(f"\n{'CALLS' if opt_type == 'C' else 'PUTS'}:")
        
        # Get strikes sorted by distance from ATM
        strikes = type_df.sort_values('distance_from_atm')['strike'].unique()
        
        for strike in strikes:
            strike_df = type_df[type_df['strike'] == strike].sort_values('to_date')
            
            if strike_df.empty:
                continue
            
            underlying = strike_df.iloc[0]['underlying_price']
            distance = strike_df.iloc[0]['distance_from_atm']
            moneyness = "ATM" if abs(distance) < 0.5 else ("ITM" if distance < 0 else "OTM")
            
            print(f"\n  Strike: €{strike:.2f} ({moneyness}, {distance:+.2f} from €{underlying:.2f})")
            
            # Print each week's data
            for _, row in strike_df.iterrows():
                from_date = row['from_date'].strftime('%m/%d')
                to_date = row['to_date'].strftime('%m/%d')
                
                oi_change_str = format_change(row['oi_change'])
                vol_change_str = format_change(row['vol_change'])
                oi_pct_str = format_change(row['oi_pct_change'], is_percentage=True) if pd.notna(row['oi_pct_change']) else "N/A"
                
                print(f"    {from_date}→{to_date}: OI {row['oi_from']:>5,.0f}→{row['oi_to']:>5,.0f} {oi_change_str:>12} ({oi_pct_str:>10}) | Vol {row['vol_from']:>4,.0f}→{row['vol_to']:>4,.0f} {vol_change_str:>10}")


def print_summary_stats(df, expiry_label):
    """Print summary statistics for the expiry."""
    if df.empty:
        return
    
    print(f"\n📊 Summary for {expiry_label}:")
    
    # Get most recent data
    latest_date = df['to_date'].max()
    latest_df = df[df['to_date'] == latest_date]
    
    total_oi_calls = latest_df[latest_df['option_type'] == 'C']['oi_to'].sum()
    total_oi_puts = latest_df[latest_df['option_type'] == 'P']['oi_to'].sum()
    total_vol_calls = latest_df[latest_df['option_type'] == 'C']['vol_to'].sum()
    total_vol_puts = latest_df[latest_df['option_type'] == 'P']['vol_to'].sum()
    
    print(f"  Current OI - Calls: {total_oi_calls:,.0f} | Puts: {total_oi_puts:,.0f} | P/C Ratio: {total_oi_puts/max(total_oi_calls,1):.2f}")
    print(f"  Last Week Vol - Calls: {total_vol_calls:,.0f} | Puts: {total_vol_puts:,.0f}")
    
    # Biggest OI changes
    biggest_increase = latest_df.nlargest(1, 'oi_change')
    biggest_decrease = latest_df.nsmallest(1, 'oi_change')
    
    if not biggest_increase.empty:
        row = biggest_increase.iloc[0]
        print(f"  Biggest OI Increase: {row['option_type']} €{row['strike']:.2f} {format_change(row['oi_change'])}")
    
    if not biggest_decrease.empty:
        row = biggest_decrease.iloc[0]
        print(f"  Biggest OI Decrease: {row['option_type']} €{row['strike']:.2f} {format_change(row['oi_change'])}")


def main():
    parser = argparse.ArgumentParser(description="Friday ATM Options OI & Volume Analysis")
    parser.add_argument("--ticker", type=str, default=None, help="Filter by ticker (e.g., AH-DAMS)")
    parser.add_argument("--strike-range", type=int, default=6, help="Number of strikes above/below ATM (default: 6)")
    args = parser.parse_args()
    
    engine = get_engine()
    
    # Print report header
    fridays = get_last_fridays(4)
    print("\n" + "█" * 100)
    print("  📊 FRIDAY ATM OPTIONS ANALYSIS - OI & VOLUME CHANGES")
    print("█" * 100)
    print(f"  Report Date: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"  Analysis Period: Last 4 Fridays: {', '.join([f.strftime('%Y-%m-%d') for f in fridays])}")
    print(f"  Strike Range: ATM ± {args.strike_range} strikes")
    if args.ticker:
        print(f"  Ticker Filter: {args.ticker}")
    print("█" * 100)
    
    # Get data
    print("\n⏳ Fetching Friday data from database...")
    df = get_friday_oi_volume_data(engine, args.ticker, args.strike_range)
    
    if df.empty:
        print("\n❌ No data found for the specified criteria.")
        print("   Make sure you have Friday data in bronze_euronext_options table.")
        return
    
    print(f"✅ Found {len(df)} records across {df['trade_date'].nunique()} Fridays")
    
    # Calculate changes
    changes_df = calculate_weekly_changes(df)
    
    if changes_df.empty:
        print("\n❌ Not enough data to calculate weekly changes.")
        return
    
    # Get the two expiry months
    expiries = sorted(changes_df['expiration_date'].unique())
    
    if len(expiries) == 0:
        print("\n❌ No expiry data found.")
        return
    
    # Current expiry
    print_section("CURRENT EXPIRY MONTH")
    current_expiry_df = changes_df[changes_df['expiration_date'] == expiries[0]]
    current_expiry_date = current_expiry_df.iloc[0]['expiration_date']
    print_weekly_comparison_table(current_expiry_df, f"Expiry: {current_expiry_date}")
    print_summary_stats(current_expiry_df, f"{current_expiry_date}")
    
    # Next expiry
    if len(expiries) > 1:
        print_section("NEXT EXPIRY MONTH")
        next_expiry_df = changes_df[changes_df['expiration_date'] == expiries[1]]
        next_expiry_date = next_expiry_df.iloc[0]['expiration_date']
        print_weekly_comparison_table(next_expiry_df, f"Expiry: {next_expiry_date}")
        print_summary_stats(next_expiry_df, f"{next_expiry_date}")
    
    # Overall summary
    print_section("OVERALL SUMMARY")
    
    latest_date = changes_df['to_date'].max()
    latest_changes = changes_df[changes_df['to_date'] == latest_date]
    
    print(f"\n📈 Top OI Increases (Last Week: {latest_date.strftime('%Y-%m-%d')}):")
    top_increases = latest_changes.nlargest(10, 'oi_change')[['expiration_date', 'option_type', 'strike', 'oi_change', 'oi_pct_change', 'vol_to']]
    for _, row in top_increases.iterrows():
        print(f"   {row['expiration_date']} {row['option_type']} €{row['strike']:.2f}: {format_change(row['oi_change'])} "
              f"({format_change(row['oi_pct_change'], True)}) | Vol: {row['vol_to']:,.0f}")
    
    print(f"\n📉 Top OI Decreases (Last Week: {latest_date.strftime('%Y-%m-%d')}):")
    top_decreases = latest_changes.nsmallest(10, 'oi_change')[['expiration_date', 'option_type', 'strike', 'oi_change', 'oi_pct_change', 'vol_to']]
    for _, row in top_decreases.iterrows():
        print(f"   {row['expiration_date']} {row['option_type']} €{row['strike']:.2f}: {format_change(row['oi_change'])} "
              f"({format_change(row['oi_pct_change'], True)}) | Vol: {row['vol_to']:,.0f}")
    
    print("\n" + "█" * 100)
    print("  ✅ Report Complete")
    print("█" * 100 + "\n")


if __name__ == "__main__":
    main()
