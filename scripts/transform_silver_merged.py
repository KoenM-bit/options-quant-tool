"""
Silver Layer Transformation with Beursduivel + FD Merge
========================================================
Merges pricing data from Beursduivel (PRIMARY) with metrics from FD (SECONDARY)
to achieve 90%+ Greeks coverage.

Strategy:
1. PRIMARY: Beursduivel for bid/ask/last_price (90%+ coverage)
2. SECONDARY: FD for open_interest, daily volume totals
3. FALLBACK: FD last_price if BD missing
4. UNDERLYING: Always from FD overview
"""

import sys
import os
from pathlib import Path
from datetime import datetime, date
from typing import Optional
import logging

# Support both Docker and local execution
if os.path.exists('/opt/airflow'):
    sys.path.insert(0, '/opt/airflow/dags/..')
else:
    project_root = Path(__file__).parent.parent
    sys.path.insert(0, str(project_root))

import pandas as pd
import numpy as np
from sqlalchemy import text

from src.utils.db import get_db_session
from src.analytics.black_scholes import BlackScholes
from src.analytics.risk_free_rate import get_risk_free_rate_for_date

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def transform_merged_options_chain(
    trade_date: str,
    ticker: str = "AD.AS"
) -> pd.DataFrame:
    """
    Transform options chain by merging Beursduivel (pricing) + FD (metrics).
    
    Args:
        trade_date: Date in YYYY-MM-DD format
        ticker: Stock ticker
    
    Returns:
        DataFrame with merged silver schema
    """
    logger.info(f"="*60)
    logger.info(f"Transforming MERGED silver layer for {ticker} on {trade_date}")
    logger.info(f"="*60)
    
    with get_db_session() as session:
        # Step 1: Get BD data (PRIMARY pricing source)
        logger.info("1️⃣ Fetching Beursduivel data (PRIMARY pricing)...")
        bd_query = text("""
            SELECT 
                ticker,
                symbol_code,
                option_type,
                expiry_date,
                strike,
                bid as bd_bid,
                ask as bd_ask,
                last_price as bd_last_price,
                volume as bd_volume,
                scraped_at as bd_scraped_at,
                source_url as bd_source_url
            FROM bronze_bd_options
            WHERE ticker = :ticker
              AND trade_date = :trade_date
            ORDER BY expiry_date, strike, option_type
        """)
        
        bd_df = pd.read_sql(bd_query, session.connection(), params={
            'ticker': ticker,
            'trade_date': trade_date
        })
        
        logger.info(f"   Beursduivel: {len(bd_df)} contracts")
        
        # Step 2: Get FD data (SECONDARY - open interest + fallback pricing)
        logger.info("2️⃣ Fetching FD data (SECONDARY metrics)...")
        fd_query = text("""
            WITH ranked_fd AS (
                SELECT 
                    ticker,
                    option_type,
                    expiry_date,
                    strike,
                    laatste as fd_last_price,
                    bid as fd_bid,
                    ask as fd_ask,
                    volume as fd_volume,
                    open_interest as fd_open_interest,
                    underlying_price as fd_underlying_price,
                    scraped_at as fd_scraped_at,
                    ROW_NUMBER() OVER (
                        PARTITION BY ticker, expiry_date, strike, option_type 
                        ORDER BY updated_at DESC
                    ) as rn
                FROM bronze_fd_options
                WHERE ticker = :ticker
                  AND DATE(scraped_at) >= CAST(:trade_date AS DATE)
                  AND DATE(scraped_at) <= CAST(:trade_date AS DATE) + INTERVAL '1 day'
            )
            SELECT 
                ticker, option_type, expiry_date, strike,
                fd_last_price, fd_bid, fd_ask, fd_volume,
                fd_open_interest, fd_underlying_price, fd_scraped_at
            FROM ranked_fd
            WHERE rn = 1
        """)
        
        fd_df = pd.read_sql(fd_query, session.connection(), params={
            'ticker': ticker,
            'trade_date': trade_date
        })
        
        logger.info(f"   FD: {len(fd_df)} contracts")
        
        # Step 3: Get underlying price from BD (synchronized with options data!)
        logger.info("3️⃣ Fetching underlying price from BD (synchronized)...")
        underlying_query = text("""
            SELECT last_price, bid, ask, last_timestamp_text
            FROM bronze_bd_underlying
            WHERE ticker = :ticker
              AND trade_date = :trade_date
            ORDER BY scraped_at DESC
            LIMIT 1
        """)
        
        underlying_result = session.execute(underlying_query, {
            'ticker': ticker,
            'trade_date': trade_date
        }).fetchone()
        
        if underlying_result:
            underlying_price = float(underlying_result[0]) if underlying_result[0] else None
            underlying_timestamp = underlying_result[3]
            logger.info(f"   Underlying: €{underlying_price} @ {underlying_timestamp}")
        else:
            # Fallback to FD overview if BD not available
            logger.info("   BD underlying not found, falling back to FD overview...")
            fd_overview_query = text("""
                SELECT koers as underlying_price
                FROM bronze_fd_overview
                WHERE ticker = :ticker
                  AND peildatum = :trade_date
                ORDER BY scraped_at DESC
                LIMIT 1
            """)
            
            fd_overview_result = session.execute(fd_overview_query, {
                'ticker': ticker,
                'trade_date': trade_date
            }).fetchone()
            
            underlying_price = float(fd_overview_result[0]) if fd_overview_result else None
            underlying_timestamp = f"{trade_date} (FD)"
            logger.info(f"   Underlying (FD): €{underlying_price}")
    
    if bd_df.empty:
        logger.warning(f"⚠️  No Beursduivel data for {trade_date}")
        return pd.DataFrame()
    
    # Step 4: Merge BD (left) with FD (right)
    logger.info("4️⃣ Merging BD + FD data...")
    
    # Convert option_type to match (BD uses Call/Put, need to check)
    bd_df['option_type_key'] = bd_df['option_type']
    fd_df['option_type_key'] = fd_df['option_type']
    
    # Merge on contract key
    df = pd.merge(
        bd_df,
        fd_df,
        how='left',
        on=['ticker', 'option_type_key', 'expiry_date', 'strike'],
        suffixes=('', '_fd')
    )
    
    logger.info(f"   Merged: {len(df)} contracts")
    logger.info(f"   With FD match: {df['fd_last_price'].notna().sum()} contracts")
    
    # Step 5: Build final silver schema with merged data
    logger.info("5️⃣ Building silver schema...")
    
    # A) Identity / time / lineage
    df['ticker'] = df['ticker']
    df['trade_date'] = pd.to_datetime(trade_date).date()  # The trading day this data represents
    df['symbol_code'] = df['symbol_code']
    df['as_of_ts'] = pd.to_datetime(df['bd_scraped_at'])
    df['as_of_date'] = df['as_of_ts'].dt.date
    df['source'] = 'beursduivel_primary_fd_secondary'
    df['source_url'] = df['bd_source_url']
    df['scrape_run_id'] = 'manual'
    df['ingested_at'] = pd.Timestamp.now(tz='UTC')
    
    # B) Contract fields
    df['option_type'] = df['option_type_key']
    df['expiry_date'] = pd.to_datetime(df['expiry_date']).dt.date
    df['strike'] = pd.to_numeric(df['strike'], errors='coerce')
    
    # Create contract key
    df['contract_key'] = (
        df['ticker'].astype(str) + '|' +
        df['expiry_date'].astype(str) + '|' +
        df['strike'].astype(str) + '|' +
        df['option_type'].astype(str)
    )
    
    # C) Quote fields - MERGE STRATEGY
    # Priority: BD > FD
    df['bid'] = df['bd_bid'].combine_first(pd.to_numeric(df['fd_bid'], errors='coerce'))
    df['ask'] = df['bd_ask'].combine_first(pd.to_numeric(df['fd_ask'], errors='coerce'))
    df['last_price'] = df['bd_last_price'].combine_first(pd.to_numeric(df['fd_last_price'], errors='coerce'))
    
    # Mid price and spread
    df['mid_price'] = np.where(
        df['bid'].notna() & df['ask'].notna(),
        (df['bid'] + df['ask']) / 2,
        np.nan
    )
    df['spread_abs'] = np.where(
        df['bid'].notna() & df['ask'].notna(),
        df['ask'] - df['bid'],
        np.nan
    )
    df['spread_pct'] = np.where(
        (df['spread_abs'].notna()) & (df['mid_price'] > 0),
        df['spread_abs'] / df['mid_price'],
        np.nan
    )
    
    # Volume: BD intraday volume
    df['volume'] = pd.to_numeric(df['bd_volume'], errors='coerce').fillna(0).astype(int)
    
    # Open Interest: Only from FD
    df['open_interest'] = pd.to_numeric(df['fd_open_interest'], errors='coerce').fillna(0).astype(int)
    
    # Underlying price: From overview (filled for all)
    df['underlying_price'] = underlying_price
    
    # D) Quality flags
    df['is_valid_quote'] = (
        (df['bid'].isna() | df['ask'].isna() | (df['bid'] <= df['ask'])) &
        (df['volume'] >= 0) &
        (df['open_interest'] >= 0) &
        (df['strike'] > 0)
    )
    
    # Data source flags
    df['has_bd_data'] = df['bd_bid'].notna() | df['bd_ask'].notna() | df['bd_last_price'].notna()
    df['has_fd_data'] = df['fd_last_price'].notna() | df['fd_open_interest'].notna()
    
    # Calculate row hash
    import hashlib
    def calc_hash(row):
        key = f"{row['contract_key']}|{row['last_price']}|{row['bid']}|{row['ask']}"
        return hashlib.md5(key.encode()).hexdigest()
    
    df['row_hash'] = df.apply(calc_hash, axis=1)
    
    # E) Calculate Greeks
    logger.info("6️⃣ Calculating Greeks...")
    risk_free_rate = get_risk_free_rate_for_date(pd.to_datetime(trade_date).date())
    
    greeks_list = []
    successful_greeks = 0
    
    for idx, row in df.iterrows():
        # Use last_price if available, otherwise use mid_price (bid+ask)/2
        price_for_greeks = row['last_price'] if pd.notna(row['last_price']) and row['last_price'] > 0 else row['mid_price']
        
        # For options where mid_price < intrinsic (arbitrage/bad data), use ask instead
        # This happens with deep ITM options where spread crosses intrinsic value
        if pd.notna(price_for_greeks) and pd.notna(row['underlying_price']) and pd.notna(row['strike']):
            if row['option_type'] == 'Call':
                intrinsic = max(row['underlying_price'] - row['strike'], 0)
            else:  # Put
                intrinsic = max(row['strike'] - row['underlying_price'], 0)
            
            # If mid < intrinsic, use ask price (what you'd pay to buy)
            if price_for_greeks < intrinsic and pd.notna(row['ask']):
                price_for_greeks = row['ask']
        
        has_price = pd.notna(price_for_greeks) and price_for_greeks > 0
        has_underlying = pd.notna(row['underlying_price']) and row['underlying_price'] > 0
        has_strike = pd.notna(row['strike']) and row['strike'] > 0
        
        if has_price and has_underlying and has_strike:
            try:
                days_to_expiry = (row['expiry_date'] - pd.to_datetime(trade_date).date()).days
                
                from src.analytics.black_scholes import calculate_option_metrics
                greeks = calculate_option_metrics(
                    option_price=float(price_for_greeks),
                    underlying_price=float(row['underlying_price']),
                    strike=float(row['strike']),
                    days_to_expiry=max(days_to_expiry, 1),
                    option_type='Call' if row['option_type'] == 'Call' else 'Put',
                    risk_free_rate=risk_free_rate
                )
                greeks_list.append(greeks)
                if greeks.get('implied_volatility'):
                    successful_greeks += 1
            except Exception as e:
                logger.debug(f"Failed to calculate Greeks for {row['contract_key']}: {e}")
                greeks_list.append({
                    'implied_volatility': None, 'delta': None, 'gamma': None,
                    'vega': None, 'theta': None, 'rho': None
                })
        else:
            greeks_list.append({
                'implied_volatility': None, 'delta': None, 'gamma': None,
                'vega': None, 'theta': None, 'rho': None
            })
    
    greeks_df = pd.DataFrame(greeks_list)
    df['iv'] = greeks_df['implied_volatility']
    df['delta'] = greeks_df['delta']
    df['gamma'] = greeks_df['gamma']
    df['vega'] = greeks_df['vega']
    df['theta'] = greeks_df['theta']
    df['rho'] = greeks_df['rho']
    
    logger.info(f"   Greeks calculated: {successful_greeks}/{len(df)} ({100*successful_greeks/len(df):.1f}%)")
    
    # F) Derived fields
    df['dte'] = (pd.to_datetime(df['expiry_date']) - pd.to_datetime(trade_date)).dt.days
    
    df['moneyness'] = np.where(
        (df['underlying_price'].notna()) & (df['strike'] > 0),
        df['underlying_price'] / df['strike'],
        np.nan
    )
    
    df['is_itm'] = np.where(
        df['option_type'] == 'Call',
        df['underlying_price'] > df['strike'],
        df['underlying_price'] < df['strike']
    )
    
    # Select final columns
    silver_columns = [
        # A) Identity / time / lineage
        'ticker', 'trade_date', 'symbol_code', 'as_of_ts', 'as_of_date', 'source', 'source_url',
        'scrape_run_id', 'ingested_at',
        # B) Contract fields
        'option_type', 'expiry_date', 'strike', 'contract_key',
        # C) Quote fields
        'last_price', 'bid', 'ask', 'mid_price', 'spread_abs', 'spread_pct',
        'volume', 'open_interest', 'underlying_price',
        # D) Quality helpers
        'is_valid_quote', 'has_bd_data', 'has_fd_data', 'row_hash',
        # E) Greeks
        'iv', 'delta', 'gamma', 'vega', 'theta', 'rho',
        # F) Derived fields
        'dte', 'moneyness', 'is_itm'
    ]
    
    result_df = df[silver_columns]
    
    # Summary
    logger.info(f"="*60)
    logger.info(f"✅ Silver transformation complete")
    logger.info(f"   Total contracts: {len(result_df)}")
    logger.info(f"   With pricing (bid/ask/last): {result_df['last_price'].notna().sum()}")
    logger.info(f"   With Greeks: {result_df['iv'].notna().sum()} ({100*result_df['iv'].notna().sum()/len(result_df):.1f}%)")
    logger.info(f"   BD primary: {result_df['has_bd_data'].sum()}")
    logger.info(f"   FD secondary: {result_df['has_fd_data'].sum()}")
    logger.info(f"="*60)
    
    return result_df


def save_to_silver_table(df: pd.DataFrame, trade_date: str, ticker: str = "AD.AS") -> int:
    """
    Save transformed silver data to silver_options_chain table.
    
    Args:
        df: Transformed silver DataFrame
        trade_date: Date string in YYYY-MM-DD format
        ticker: Stock ticker
    
    Returns:
        Number of rows inserted
    """
    if df.empty:
        logger.warning("No data to save")
        return 0
    
    from src.models.silver import SilverOptionsChain
    from sqlalchemy import text
    
    logger.info(f"💾 Saving {len(df)} contracts to silver_options_chain table...")
    
    with get_db_session() as session:
        # Delete existing data for this date/ticker (upsert pattern)
        delete_query = text("""
            DELETE FROM silver_options_chain 
            WHERE ticker = :ticker 
              AND trade_date = :trade_date
        """)
        
        result = session.execute(delete_query, {
            'ticker': ticker,
            'trade_date': trade_date
        })
        deleted = result.rowcount
        if deleted > 0:
            logger.info(f"   🗑️  Deleted {deleted} existing records for {trade_date}")
        
        # Insert new data
        inserted = 0
        errors = 0
        
        for idx, row in df.iterrows():
            try:
                now = pd.Timestamp.now(tz='UTC')
                record = SilverOptionsChain(
                    ticker=row['ticker'],
                    trade_date=row['trade_date'],
                    symbol_code=row.get('symbol_code'),
                    as_of_ts=row['as_of_ts'],
                    as_of_date=row['as_of_date'],
                    source=row['source'],
                    source_url=row.get('source_url'),
                    scrape_run_id=row.get('scrape_run_id'),
                    ingested_at=row.get('ingested_at') if pd.notna(row.get('ingested_at')) else now,
                    option_type=row['option_type'],
                    expiry_date=row['expiry_date'],
                    strike=row['strike'],
                    contract_key=row['contract_key'],
                    last_price=row.get('last_price'),
                    bid=row.get('bid'),
                    ask=row.get('ask'),
                    mid_price=row.get('mid_price'),
                    spread_abs=row.get('spread_abs'),
                    spread_pct=row.get('spread_pct'),
                    volume=int(row['volume']) if pd.notna(row.get('volume')) else 0,
                    open_interest=int(row['open_interest']) if pd.notna(row.get('open_interest')) else 0,
                    underlying_price=row.get('underlying_price'),
                    is_valid_quote=bool(row.get('is_valid_quote', True)),
                    has_bd_data=bool(row.get('has_bd_data', False)),
                    has_fd_data=bool(row.get('has_fd_data', False)),
                    row_hash=row.get('row_hash'),
                    iv=row.get('iv'),
                    delta=row.get('delta'),
                    gamma=row.get('gamma'),
                    vega=row.get('vega'),
                    theta=row.get('theta'),
                    rho=row.get('rho'),
                    dte=int(row['dte']) if pd.notna(row.get('dte')) else None,
                    moneyness=row.get('moneyness'),
                    is_itm=bool(row.get('is_itm')) if pd.notna(row.get('is_itm')) else None,
                    created_at=now,
                    updated_at=now
                )
                session.add(record)
                inserted += 1
            except Exception as e:
                logger.error(f"   ❌ Error inserting {row.get('contract_key')}: {e}")
                errors += 1
        
        session.commit()
        logger.info(f"   ✅ Inserted {inserted} contracts ({errors} errors)")
    
    return inserted


if __name__ == "__main__":
    import sys
    
    # Test with Dec 8 which has FD data (no BD yet - just testing the merge logic)
    test_date = sys.argv[1] if len(sys.argv) > 1 else "2025-12-08"
    save_to_db = '--save' in sys.argv
    
    logger.info(f"Testing merged silver transformation for {test_date}...")
    logger.info(f"Note: BD data only available for 2025-12-10, FD data for earlier dates")
    df = transform_merged_options_chain(test_date, ticker="AD.AS")
    
    if not df.empty:
        print(f"\n📊 Sample data (showing first 10 contracts):")
        print(df[['contract_key', 'last_price', 'bid', 'ask', 'open_interest', 'iv', 'delta', 'has_bd_data', 'has_fd_data']].head(10).to_string())
        
        # Calculate effective coverage
        valid_pricing = df[(df['bid'].notna()) & (df['ask'].notna())]
        with_greeks = valid_pricing[valid_pricing['iv'].notna()]
        
        print(f"\n📈 Coverage Summary:")
        print(f"   Total contracts: {len(df)}")
        print(f"   Contracts with bid: {df['bid'].notna().sum()}/{len(df)} ({100*df['bid'].notna().sum()/len(df):.1f}%)")
        print(f"   Contracts with ask: {df['ask'].notna().sum()}/{len(df)} ({100*df['ask'].notna().sum()/len(df):.1f}%)")
        print(f"   Contracts with bid AND ask: {len(valid_pricing)}/{len(df)} ({100*len(valid_pricing)/len(df):.1f}%)")
        print(f"   Contracts with last_price: {df['last_price'].notna().sum()}/{len(df)} ({100*df['last_price'].notna().sum()/len(df):.1f}%)")
        print(f"   Contracts with open_interest: {(df['open_interest'] > 0).sum()}/{len(df)} ({100*(df['open_interest'] > 0).sum()/len(df):.1f}%)")
        print(f"   ")
        print(f"   💎 Contracts with Greeks (IV): {df['iv'].notna().sum()}/{len(df)} ({100*df['iv'].notna().sum()/len(df):.1f}% overall)")
        print(f"   ✅ Effective Greeks coverage: {len(with_greeks)}/{len(valid_pricing)} = {100*len(with_greeks)/len(valid_pricing):.1f}% (of contracts with bid/ask)")
        print(f"   ")
        print(f"   📡 BD primary source: {df['has_bd_data'].sum()} contracts")
        print(f"   📡 FD secondary source: {df['has_fd_data'].sum()} contracts")
        
        # Save to database if requested
        if save_to_db:
            inserted = save_to_silver_table(df, test_date, ticker="AD.AS")
            print(f"\n💾 Saved {inserted} contracts to silver_options_chain table")
    else:
        logger.warning(f"⚠️  No data returned for {test_date}")
