"""
Export Silver Layer to Partitioned Parquet Files in MinIO
===========================================================
Reads bronze parquet, applies transformations, calculates Greeks, writes to silver.

Silver Structure:
- silver/options_chain/v1/date=YYYY-MM-DD/ticker=AD.AS/as_of=YYYY-MM-DDTHH-MM-SSZ/data.parquet
- silver/options_overview/v1/date=YYYY-MM-DD/ticker=AD.AS/as_of=YYYY-MM-DDTHH-MM-SSZ/data.parquet

Key Improvements:
- Cleaned and typed data
- IV and Greeks calculated
- Derived fields (DTE, moneyness, ITM flag)
- Quality flags (is_valid_quote)
- Standardized column names
"""

import sys
import os
from pathlib import Path
from datetime import datetime, date
from typing import List, Tuple, Optional
import logging
import hashlib

# Support both Docker and local execution
if os.path.exists('/opt/airflow'):
    sys.path.insert(0, '/opt/airflow/dags/..')
else:
    # Running locally - add project root to path
    project_root = Path(__file__).parent.parent
    sys.path.insert(0, str(project_root))

import pandas as pd
import numpy as np
from sqlalchemy import text

from src.utils.db import get_db_session
from src.utils.minio_client import get_minio_client
from src.analytics.black_scholes import calculate_option_metrics, BlackScholes
from src.analytics.risk_free_rate import get_risk_free_rate_for_date
from src.config import settings

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Temp directory for parquet files before upload
TEMP_DIR = Path("/tmp/silver_parquet_export")
TEMP_DIR.mkdir(parents=True, exist_ok=True)


def get_trade_dates_from_bronze() -> List[Tuple[str, str, datetime]]:
    """
    Get distinct trade dates from bronze layer (from MinIO or PostgreSQL).
    For now, we'll use PostgreSQL as source of truth.
    
    Returns:
        List of (trade_date, ticker, latest_scraped_at) tuples
    """
    with get_db_session() as session:
        query = text("""
            SELECT 
                peildatum as trade_date,
                ticker,
                MAX(scraped_at) as latest_scraped_at
            FROM bronze_fd_overview
            GROUP BY peildatum, ticker
            ORDER BY trade_date, ticker
        """)
        
        result = session.execute(query)
        rows = result.fetchall()
        
    logger.info(f"Found {len(rows)} distinct trade dates in bronze layer")
    return rows


def calculate_row_hash(row: pd.Series) -> str:
    """Calculate a stable hash for deduplication and change detection."""
    # Use canonical fields that define a unique quote
    key_fields = [
        str(row.get('ticker', '')),
        str(row.get('expiry_date', '')),
        str(row.get('strike', '')),
        str(row.get('option_type', '')),
        str(row.get('last_price', '')),
        str(row.get('bid', '')),
        str(row.get('ask', '')),
    ]
    hash_input = '|'.join(key_fields)
    return hashlib.md5(hash_input.encode()).hexdigest()


def transform_options_chain_to_silver(
    trade_date: str,
    ticker: str,
    latest_scraped_at: datetime,
    scrape_run_id: Optional[str] = None
) -> pd.DataFrame:
    """
    Read bronze options chain and transform to silver schema with Greeks.
    
    Returns:
        DataFrame with silver schema
    """
    logger.info(f"Transforming options chain for {ticker} on {trade_date}")
    
    # Read from bronze PostgreSQL (in production, would read from bronze parquet)
    with get_db_session() as session:
        query = text("""
            WITH ranked_options AS (
                SELECT 
                    ticker,
                    symbol_code,
                    scraped_at,
                    source_url,
                    option_type,
                    expiry_date,
                    strike,
                    naam,
                    isin,
                    laatste,
                    bid,
                    ask,
                    volume,
                    open_interest,
                    underlying_price,
                    created_at,
                    updated_at,
                    ROW_NUMBER() OVER (
                        PARTITION BY ticker, expiry_date, strike, option_type 
                        ORDER BY updated_at DESC, created_at DESC
                    ) as rn
                FROM bronze_fd_options
                WHERE ticker = :ticker
                  AND DATE(scraped_at) = DATE(:latest_scraped_at)
            )
            SELECT 
                ticker,
                symbol_code,
                scraped_at,
                source_url,
                option_type,
                expiry_date,
                strike,
                naam,
                isin,
                laatste as last_price,
                bid,
                ask,
                volume,
                open_interest,
                underlying_price,
                created_at,
                updated_at
            FROM ranked_options
            WHERE rn = 1
            ORDER BY expiry_date, strike, option_type
        """)
        
        df = pd.read_sql(query, session.connection(), params={
            'ticker': ticker,
            'latest_scraped_at': latest_scraped_at
        })
    
    if df.empty:
        logger.warning(f"No bronze data for {ticker} on {trade_date}")
        return pd.DataFrame()
    
    # A) Identity / time / lineage
    df['ticker'] = df['ticker']
    df['symbol_code'] = df['symbol_code']
    df['as_of_ts'] = pd.to_datetime(df['scraped_at'])
    df['as_of_date'] = df['as_of_ts'].dt.date
    df['source'] = 'beurs_fd'
    df['source_url'] = df['source_url']
    df['scrape_run_id'] = scrape_run_id or 'manual'
    df['ingested_at'] = pd.Timestamp.now(tz='UTC')
    
    # B) Contract fields
    # Transform Put/Call to C/P
    df['option_type'] = df['option_type'].map({'Call': 'C', 'Put': 'P'})
    df['expiry_date'] = pd.to_datetime(df['expiry_date']).dt.date
    df['strike'] = pd.to_numeric(df['strike'], errors='coerce')
    
    # Create stable contract key
    df['contract_key'] = (
        df['ticker'].astype(str) + '|' +
        df['expiry_date'].astype(str) + '|' +
        df['strike'].astype(str) + '|' +
        df['option_type'].astype(str)
    )
    
    # C) Quote fields (cleaned)
    df['last_price'] = pd.to_numeric(df['last_price'], errors='coerce')
    df['bid'] = pd.to_numeric(df['bid'], errors='coerce')
    df['ask'] = pd.to_numeric(df['ask'], errors='coerce')
    
    # Calculate mid price and spread
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
    
    df['volume'] = pd.to_numeric(df['volume'], errors='coerce').fillna(0).astype(int)
    df['open_interest'] = pd.to_numeric(df['open_interest'], errors='coerce').fillna(0).astype(int)
    df['underlying_price'] = pd.to_numeric(df['underlying_price'], errors='coerce')
    
    # D) Quality / dedupe helpers
    df['is_valid_quote'] = (
        (df['bid'].isna() | df['ask'].isna() | (df['bid'] <= df['ask'])) &
        (df['volume'] >= 0) &
        (df['open_interest'] >= 0) &
        (df['strike'] > 0)
    )
    
    # Calculate row hash for each contract
    df['row_hash'] = df.apply(calculate_row_hash, axis=1)
    
    # E) Calculate IV + Greeks (ENHANCED VERSION)
    logger.info(f"Calculating Greeks for {len(df)} contracts...")
    risk_free_rate = get_risk_free_rate_for_date(pd.to_datetime(trade_date).date())
    
    # Get underlying price from overview if not in options data
    if df['underlying_price'].isna().any():
        with get_db_session() as session:
            overview_query = text("""
                SELECT koers as underlying_price
                FROM bronze_fd_overview
                WHERE peildatum = :trade_date
                  AND ticker = :ticker
                ORDER BY scraped_at DESC
                LIMIT 1
            """)
            overview_result = session.execute(overview_query, {
                'trade_date': trade_date,
                'ticker': ticker
            }).fetchone()
            
            if overview_result:
                underlying_from_overview = float(overview_result[0])
                df['underlying_price'] = df['underlying_price'].fillna(underlying_from_overview)
                logger.info(f"  Filled missing underlying_price with {underlying_from_overview}")
    
    # PASS 1: Calculate IV for contracts with market prices
    greeks_list = []
    iv_estimates = []  # Collect IVs for estimating missing prices
    
    for idx, row in df.iterrows():
        has_price = pd.notna(row['last_price']) and row['last_price'] > 0
        has_underlying = pd.notna(row['underlying_price']) and row['underlying_price'] > 0
        has_strike = pd.notna(row['strike']) and row['strike'] > 0
        
        if has_price and has_underlying and has_strike:
            try:
                days_to_expiry = (row['expiry_date'] - pd.to_datetime(trade_date).date()).days
                
                greeks = calculate_option_metrics(
                    option_price=float(row['last_price']),
                    underlying_price=float(row['underlying_price']),
                    strike=float(row['strike']),
                    days_to_expiry=max(days_to_expiry, 1),
                    option_type='Call' if row['option_type'] == 'C' else 'Put',
                    risk_free_rate=risk_free_rate
                )
                greeks_list.append(greeks)
                
                # Collect valid IV for estimation
                if greeks.get('implied_volatility') and greeks['implied_volatility'] > 0:
                    iv_estimates.append(greeks['implied_volatility'])
                    
            except Exception as e:
                logger.debug(f"Failed to calculate Greeks for contract {idx}: {e}")
                greeks_list.append({
                    'implied_volatility': None,
                    'delta': None,
                    'gamma': None,
                    'vega': None,
                    'theta': None,
                    'rho': None,
                    'is_estimated': False
                })
        else:
            # Placeholder - will estimate in PASS 2
            greeks_list.append(None)
    
    # Calculate median IV for estimating illiquid contracts
    median_iv = np.median(iv_estimates) if iv_estimates else 0.30  # Default 30% if no IVs
    logger.info(f"  Pass 1: {len(iv_estimates)} contracts with IV (median: {median_iv:.2%})")
    
    # PASS 2: Estimate Greeks for contracts WITHOUT market prices
    estimated_count = 0
    for idx, row in df.iterrows():
        if greeks_list[idx] is None:  # No market price
            has_underlying = pd.notna(row['underlying_price']) and row['underlying_price'] > 0
            has_strike = pd.notna(row['strike']) and row['strike'] > 0
            
            if has_underlying and has_strike:
                try:
                    days_to_expiry = (row['expiry_date'] - pd.to_datetime(trade_date).date()).days
                    T = max(days_to_expiry, 1) / 365.0
                    
                    # Calculate theoretical price using median IV
                    if row['option_type'] == 'C':
                        theoretical_price = BlackScholes.call_price(
                            S=float(row['underlying_price']),
                            K=float(row['strike']),
                            T=T,
                            r=risk_free_rate,
                            sigma=median_iv
                        )
                    else:
                        theoretical_price = BlackScholes.put_price(
                            S=float(row['underlying_price']),
                            K=float(row['strike']),
                            T=T,
                            r=risk_free_rate,
                            sigma=median_iv
                        )
                    
                    # Now calculate Greeks using theoretical price
                    greeks = calculate_option_metrics(
                        option_price=theoretical_price,
                        underlying_price=float(row['underlying_price']),
                        strike=float(row['strike']),
                        days_to_expiry=max(days_to_expiry, 1),
                        option_type='Call' if row['option_type'] == 'C' else 'Put',
                        risk_free_rate=risk_free_rate
                    )
                    greeks['is_estimated'] = True
                    greeks['estimated_price'] = theoretical_price
                    greeks_list[idx] = greeks
                    estimated_count += 1
                    
                except Exception as e:
                    logger.debug(f"Failed to estimate Greeks for contract {idx}: {e}")
                    greeks_list[idx] = {
                        'implied_volatility': None,
                        'delta': None,
                        'gamma': None,
                        'vega': None,
                        'theta': None,
                        'rho': None,
                        'is_estimated': False
                    }
            else:
                greeks_list[idx] = {
                    'implied_volatility': None,
                    'delta': None,
                    'gamma': None,
                    'vega': None,
                    'theta': None,
                    'rho': None,
                    'is_estimated': False
                }
    
    logger.info(f"  Pass 2: {estimated_count} contracts with ESTIMATED Greeks")
    
    # Assign Greeks to dataframe
    greeks_df = pd.DataFrame(greeks_list)
    df['iv'] = greeks_df['implied_volatility']
    df['delta'] = greeks_df['delta']
    df['gamma'] = greeks_df['gamma']
    df['vega'] = greeks_df['vega']
    df['theta'] = greeks_df['theta']
    df['rho'] = greeks_df['rho']
    df['greeks_are_estimated'] = greeks_df.get('is_estimated', False).fillna(False)
    df['estimated_option_price'] = greeks_df.get('estimated_price', np.nan)
    
    # F) Extra derived fields
    df['dte'] = (pd.to_datetime(df['expiry_date']) - pd.to_datetime(trade_date)).dt.days
    
    df['moneyness'] = np.where(
        (df['underlying_price'].notna()) & (df['strike'] > 0),
        df['underlying_price'] / df['strike'],
        np.nan
    )
    
    df['is_itm'] = np.where(
        df['option_type'] == 'C',
        df['underlying_price'] > df['strike'],
        df['underlying_price'] < df['strike']
    )
    
    # Select final columns in order
    silver_columns = [
        # A) Identity / time / lineage
        'ticker', 'symbol_code', 'as_of_ts', 'as_of_date', 'source', 'source_url', 
        'scrape_run_id', 'ingested_at',
        # B) Contract fields
        'option_type', 'expiry_date', 'strike', 'contract_key',
        # C) Quote fields
        'last_price', 'bid', 'ask', 'mid_price', 'spread_abs', 'spread_pct',
        'volume', 'open_interest', 'underlying_price',
        # D) Quality helpers
        'is_valid_quote', 'row_hash',
        # E) IV + Greeks (with estimation support)
        'iv', 'delta', 'gamma', 'vega', 'theta', 'rho',
        'greeks_are_estimated', 'estimated_option_price',
        # F) Derived fields
        'dte', 'moneyness', 'is_itm'
    ]
    
    return df[silver_columns]


def transform_options_overview_to_silver(
    trade_date: str,
    ticker: str,
    latest_scraped_at: datetime,
    scrape_run_id: Optional[str] = None
) -> pd.DataFrame:
    """
    Read bronze options overview and transform to silver schema.
    
    Returns:
        DataFrame with silver schema
    """
    logger.info(f"Transforming options overview for {ticker} on {trade_date}")
    
    # Read from bronze PostgreSQL
    with get_db_session() as session:
        query = text("""
            SELECT 
                ticker,
                symbol_code,
                onderliggende_waarde,
                koers,
                vorige,
                delta,
                delta_pct,
                hoog,
                laag,
                volume_underlying,
                tijd,
                peildatum,
                totaal_volume,
                totaal_volume_calls,
                totaal_volume_puts,
                totaal_oi,
                totaal_oi_calls,
                totaal_oi_puts,
                call_put_ratio,
                scraped_at,
                source_url,
                created_at,
                updated_at
            FROM bronze_fd_overview
            WHERE peildatum = :trade_date
              AND ticker = :ticker
              AND scraped_at = :latest_scraped_at
        """)
        
        df = pd.read_sql(query, session.connection(), params={
            'trade_date': trade_date,
            'ticker': ticker,
            'latest_scraped_at': latest_scraped_at
        })
    
    if df.empty:
        logger.warning(f"No bronze overview data for {ticker} on {trade_date}")
        return pd.DataFrame()
    
    # Take first row (should only be one per ticker/date/scrape)
    row = df.iloc[0]
    
    # Build silver record
    silver_data = {
        # A) Identity / time / lineage
        'ticker': row['ticker'],
        'symbol_code': row['symbol_code'],
        'as_of_ts': pd.to_datetime(row['scraped_at']),
        'as_of_date': pd.to_datetime(row['scraped_at']).date(),
        'source': 'beurs_fd',
        'source_url': row['source_url'],
        'scrape_run_id': scrape_run_id or 'manual',
        'ingested_at': pd.Timestamp.now(tz='UTC'),
        
        # B) Underlying snapshot fields
        'underlying_name': row['onderliggende_waarde'],
        'underlying_last': pd.to_numeric(row['koers'], errors='coerce'),
        'underlying_prev': pd.to_numeric(row['vorige'], errors='coerce'),
        'underlying_change_abs': pd.to_numeric(row['delta'], errors='coerce'),
        'underlying_change_pct': pd.to_numeric(row['delta_pct'], errors='coerce'),
        'underlying_high': pd.to_numeric(row['hoog'], errors='coerce'),
        'underlying_low': pd.to_numeric(row['laag'], errors='coerce'),
        'underlying_volume': pd.to_numeric(row['volume_underlying'], errors='coerce'),
        
        # C) Options totals
        'total_options_volume': pd.to_numeric(row['totaal_volume'], errors='coerce'),
        'total_call_volume': pd.to_numeric(row['totaal_volume_calls'], errors='coerce'),
        'total_put_volume': pd.to_numeric(row['totaal_volume_puts'], errors='coerce'),
        'total_open_interest': pd.to_numeric(row['totaal_oi'], errors='coerce'),
        'total_call_open_interest': pd.to_numeric(row['totaal_oi_calls'], errors='coerce'),
        'total_put_open_interest': pd.to_numeric(row['totaal_oi_puts'], errors='coerce'),
        'call_put_ratio': pd.to_numeric(row['call_put_ratio'], errors='coerce'),
        
        # D) Market timing fields
        'market_time_str': row['tijd'],
        'trade_date': pd.to_datetime(row['peildatum']).date(),
    }
    
    return pd.DataFrame([silver_data])


def export_silver_options_chain(
    trade_date: str,
    ticker: str,
    latest_scraped_at: datetime,
    scrape_run_id: Optional[str] = None
) -> Optional[str]:
    """
    Export silver options chain to partitioned parquet.
    
    Structure: silver/options_chain/v1/date=YYYY-MM-DD/ticker=TICKER/as_of=TIMESTAMP/data.parquet
    """
    df = transform_options_chain_to_silver(trade_date, ticker, latest_scraped_at, scrape_run_id)
    
    if df.empty:
        return None
    
    # Format as_of timestamp for path
    as_of_str = latest_scraped_at.strftime('%Y-%m-%dT%H-%M-%SZ')
    
    # Build S3 path
    s3_path = f"silver/options_chain/v1/date={trade_date}/ticker={ticker}/as_of={as_of_str}/data.parquet"
    
    # Save to temp file
    temp_file = TEMP_DIR / f"silver_chain_{trade_date}_{ticker}_{as_of_str}.parquet"
    df.to_parquet(temp_file, index=False, compression='snappy')
    
    # Upload to MinIO
    minio_client = get_minio_client()
    minio_client.upload_file(
        local_path=str(temp_file),
        object_name=s3_path
    )
    
    # Cleanup
    temp_file.unlink()
    
    logger.info(f"✅ Uploaded {len(df)} silver chain contracts to {s3_path}")
    return s3_path


def export_silver_options_overview(
    trade_date: str,
    ticker: str,
    latest_scraped_at: datetime,
    scrape_run_id: Optional[str] = None
) -> Optional[str]:
    """
    Export silver options overview to partitioned parquet.
    
    Structure: silver/options_overview/v1/date=YYYY-MM-DD/ticker=TICKER/as_of=TIMESTAMP/data.parquet
    """
    df = transform_options_overview_to_silver(trade_date, ticker, latest_scraped_at, scrape_run_id)
    
    if df.empty:
        return None
    
    # Format as_of timestamp for path
    as_of_str = latest_scraped_at.strftime('%Y-%m-%dT%H-%M-%SZ')
    
    # Build S3 path
    s3_path = f"silver/options_overview/v1/date={trade_date}/ticker={ticker}/as_of={as_of_str}/data.parquet"
    
    # Save to temp file
    temp_file = TEMP_DIR / f"silver_overview_{trade_date}_{ticker}_{as_of_str}.parquet"
    df.to_parquet(temp_file, index=False, compression='snappy')
    
    # Upload to MinIO
    minio_client = get_minio_client()
    minio_client.upload_file(
        local_path=str(temp_file),
        object_name=s3_path
    )
    
    # Cleanup
    temp_file.unlink()
    
    logger.info(f"✅ Uploaded {len(df)} silver overview records to {s3_path}")
    return s3_path


def export_all_silver_to_parquet(limit_dates: int = None):
    """
    Export all silver data to partitioned parquet files in MinIO.
    
    Args:
        limit_dates: Optional limit on number of dates to process (for testing)
    """
    logger.info("="*60)
    logger.info("🚀 STARTING SILVER PARQUET EXPORT")
    logger.info("="*60)
    
    stats = {
        'options_chain_files': 0,
        'options_overview_files': 0,
        'total_contracts': 0,
        'errors': 0
    }
    
    # Get distinct trade dates from bronze
    trade_dates = get_trade_dates_from_bronze()
    
    if limit_dates:
        trade_dates = trade_dates[-limit_dates:]
        logger.info(f"⚠️  Limited to {limit_dates} most recent dates")
    
    logger.info(f"\n📅 Processing {len(trade_dates)} trade dates")
    
    # Export each trade date
    for i, (trade_date, ticker, latest_scraped_at) in enumerate(trade_dates, 1):
        try:
            logger.info(f"\n[{i}/{len(trade_dates)}] Processing {trade_date} | {ticker}")
            logger.info(f"   Latest scrape: {latest_scraped_at}")
            
            # Export overview
            overview_path = export_silver_options_overview(trade_date, ticker, latest_scraped_at)
            if overview_path:
                stats['options_overview_files'] += 1
            
            # Export options chain
            chain_path = export_silver_options_chain(trade_date, ticker, latest_scraped_at)
            if chain_path:
                stats['options_chain_files'] += 1
            
        except Exception as e:
            logger.error(f"❌ Error processing {trade_date} | {ticker}: {e}", exc_info=True)
            stats['errors'] += 1
            continue
    
    # Summary
    logger.info("\n" + "="*60)
    logger.info("🎉 SILVER PARQUET EXPORT COMPLETED")
    logger.info("="*60)
    logger.info(f"📊 Options chain files: {stats['options_chain_files']}")
    logger.info(f"📊 Options overview files: {stats['options_overview_files']}")
    logger.info(f"❌ Errors: {stats['errors']}")
    logger.info("\n✅ Data structure in MinIO:")
    logger.info("   silver/options_chain/v1/date=YYYY-MM-DD/ticker=TICKER/as_of=TIMESTAMP/data.parquet")
    logger.info("   silver/options_overview/v1/date=YYYY-MM-DD/ticker=TICKER/as_of=TIMESTAMP/data.parquet")


def main():
    """Main entry point"""
    import argparse
    
    parser = argparse.ArgumentParser(description='Export silver layer to partitioned parquet files in MinIO')
    parser.add_argument('--limit', type=int, help='Limit number of dates to process (for testing)', default=None)
    parser.add_argument('--date', type=str, help='Export specific date only (YYYY-MM-DD)', default=None)
    
    args = parser.parse_args()
    
    if args.date:
        # Export specific date
        logger.info(f"Exporting specific date: {args.date}")
        trade_dates = get_trade_dates_from_bronze()
        date_entries = [(d, t, s) for d, t, s in trade_dates if str(d) == args.date]
        
        for trade_date, ticker, latest_scraped_at in date_entries:
            export_silver_options_overview(trade_date, ticker, latest_scraped_at)
            export_silver_options_chain(trade_date, ticker, latest_scraped_at)
    else:
        # Export all
        export_all_silver_to_parquet(limit_dates=args.limit)


if __name__ == '__main__':
    main()
