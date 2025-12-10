#!/usr/bin/env python3
"""
Enrich historical migrated options data with Greeks and implied volatility.

Reads from:
- bronze_bd_options (source='mariadb_migration')
- bronze_bd_underlying (source_url='mariadb_migration')

Calculates:
- Black-Scholes Greeks (delta, gamma, theta, vega, rho)
- Implied volatility
- Mid price, days to expiry, moneyness

Writes to:
- silver_bd_options_enriched
"""

import argparse
import logging
import sys
from datetime import datetime, date
from typing import Optional, Dict

import psycopg2
from psycopg2.extras import RealDictCursor, execute_batch
from dotenv import load_dotenv
import os

# Add project root to path
sys.path.insert(0, '/Users/koenmarijt/Documents/Projects/ahold-options')

from src.analytics.black_scholes import calculate_option_metrics
from src.analytics.risk_free_rate import get_risk_free_rate_for_date

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


def connect_postgres():
    """Connect to PostgreSQL database."""
    load_dotenv('.env.migration')
    
    config = {
        "host": os.getenv("POSTGRES_HOST", "localhost"),
        "port": int(os.getenv("POSTGRES_PORT", 5432)),
        "user": os.getenv("POSTGRES_USER", "airflow"),
        "password": os.getenv("POSTGRES_PASSWORD", "airflow"),
        "database": os.getenv("POSTGRES_DB", "ahold_options"),
    }
    
    return psycopg2.connect(**config)


def fetch_bronze_options_with_underlying():
    """
    Fetch bronze options data joined with underlying prices.
    Only fetch migrated historical data.
    """
    conn = connect_postgres()
    cursor = conn.cursor(cursor_factory=RealDictCursor)
    
    query = """
    SELECT 
        o.ticker,
        o.trade_date,
        o.symbol_code,
        o.issue_id,
        o.option_type,
        o.strike,
        o.expiry_date,
        o.bid,
        o.ask,
        o.last_price,
        o.volume,
        o.last_timestamp,
        o.scraped_at,
        o.source_url,
        u.last_price as underlying_price,
        u.volume as underlying_volume
    FROM bronze_bd_options o
    LEFT JOIN bronze_bd_underlying u 
        ON o.ticker = u.ticker 
        AND o.trade_date = DATE(u.trade_date)
    WHERE o.source = 'mariadb_migration'
    ORDER BY o.trade_date, o.expiry_date, o.strike
    """
    
    cursor.execute(query)
    rows = cursor.fetchall()
    
    cursor.close()
    conn.close()
    
    return rows


def calculate_days_to_expiry(trade_date: date, expiry_date: date) -> int:
    """Calculate days until expiry."""
    return (expiry_date - trade_date).days


def determine_moneyness(S: float, K: float, option_type: str) -> str:
    """
    Determine if option is ITM, ATM, or OTM.
    
    Call: ITM if S > K, OTM if S < K
    Put:  ITM if S < K, OTM if S > K
    """
    if abs(S - K) / K < 0.02:  # Within 2% = ATM
        return 'ATM'
    
    if option_type == 'Call':
        return 'ITM' if S > K else 'OTM'
    else:  # Put
        return 'ITM' if S < K else 'OTM'


def enrich_option_with_greeks(option: dict) -> Optional[dict]:
    """
    Calculate Greeks and IV for a single option.
    """
    # Skip if missing critical data
    if not option['underlying_price'] or not option['bid'] or not option['ask']:
        logger.warning(f"Skipping {option['issue_id']} on {option['trade_date']}: missing price data")
        return None
    
    # Calculate mid price
    mid_price = (option['bid'] + option['ask']) / 2.0
    
    # Calculate days to expiry
    days_to_expiry = calculate_days_to_expiry(option['trade_date'], option['expiry_date'])
    
    if days_to_expiry < 0:
        logger.warning(f"Skipping expired option: {option['issue_id']}")
        return None
    
    # Get risk-free rate for this date
    risk_free_rate = get_risk_free_rate_for_date(option['trade_date'])
    
    # Calculate Greeks using Black-Scholes
    try:
        metrics = calculate_option_metrics(
            option_price=mid_price,
            underlying_price=option['underlying_price'],
            strike=option['strike'],
            days_to_expiry=days_to_expiry,
            option_type=option['option_type'],
            risk_free_rate=risk_free_rate
        )
        
        if not metrics:
            logger.warning(f"Greeks calculation failed for {option['issue_id']}")
            return None
        
        # Determine moneyness
        moneyness = determine_moneyness(
            option['underlying_price'],
            option['strike'],
            option['option_type']
        )
        
        # Build enriched record
        return {
            'ticker': option['ticker'],
            'trade_date': option['trade_date'],
            'option_type': option['option_type'],
            'strike': option['strike'],
            'expiry_date': option['expiry_date'],
            'symbol_code': option['symbol_code'],
            'issue_id': option['issue_id'],
            'bid': option['bid'],
            'ask': option['ask'],
            'mid_price': mid_price,
            'last_price': option['last_price'],
            'underlying_price': option['underlying_price'],
            'volume': option['volume'],
            'underlying_volume': option['underlying_volume'],
            'last_timestamp': option['last_timestamp'],
            'days_to_expiry': days_to_expiry,
            'moneyness': moneyness,
            'delta': metrics.get('delta'),
            'gamma': metrics.get('gamma'),
            'theta': metrics.get('theta'),
            'vega': metrics.get('vega'),
            'rho': metrics.get('rho'),
            'implied_volatility': metrics.get('implied_volatility'),
            'source_url': option['source_url'],
            'scraped_at': option['scraped_at'],
            'transformed_at': datetime.utcnow()
        }
        
    except Exception as e:
        logger.error(f"Error calculating Greeks for {option['issue_id']}: {e}")
        return None


def insert_silver_records(records: list, dry_run: bool = False) -> int:
    """Insert enriched records into silver_bd_options_enriched."""
    if dry_run:
        logger.info(f"DRY RUN - would insert {len(records)} records")
        if records:
            logger.info("Sample record:")
            for k, v in list(records[0].items())[:10]:
                logger.info(f"  {k}: {v}")
        return 0
    
    conn = connect_postgres()
    cursor = conn.cursor()
    
    insert_query = """
    INSERT INTO silver_bd_options_enriched (
        ticker, trade_date, option_type, strike, expiry_date,
        symbol_code, issue_id, bid, ask, mid_price, last_price,
        underlying_price, volume, underlying_volume, last_timestamp,
        days_to_expiry, moneyness, delta, gamma, theta, vega, rho,
        implied_volatility, source_url, scraped_at, transformed_at
    ) VALUES (
        %(ticker)s, %(trade_date)s, %(option_type)s, %(strike)s, %(expiry_date)s,
        %(symbol_code)s, %(issue_id)s, %(bid)s, %(ask)s, %(mid_price)s, %(last_price)s,
        %(underlying_price)s, %(volume)s, %(underlying_volume)s, %(last_timestamp)s,
        %(days_to_expiry)s, %(moneyness)s, %(delta)s, %(gamma)s, %(theta)s, %(vega)s, %(rho)s,
        %(implied_volatility)s, %(source_url)s, %(scraped_at)s, %(transformed_at)s
    )
    """
    
    try:
        execute_batch(cursor, insert_query, records, page_size=100)
        conn.commit()
        inserted = cursor.rowcount
        cursor.close()
        conn.close()
        return inserted
    except Exception as e:
        logger.error(f"❌ Insert failed: {e}")
        conn.rollback()
        cursor.close()
        conn.close()
        raise


def main():
    parser = argparse.ArgumentParser(
        description='Enrich historical BD options with Greeks and IV'
    )
    parser.add_argument('--dry-run', action='store_true',
                       help='Preview without inserting')
    parser.add_argument('--limit', type=int,
                       help='Limit number of records to process')
    
    args = parser.parse_args()
    
    logger.info("=" * 80)
    logger.info("HISTORICAL OPTIONS ENRICHMENT - BLACK-SCHOLES GREEKS")
    logger.info("=" * 80)
    logger.info(f"Source: bronze_bd_options + bronze_bd_underlying")
    logger.info(f"Target: silver_bd_options_enriched")
    logger.info(f"Dry run: {args.dry_run}")
    logger.info("=" * 80)
    
    try:
        # Step 1: Fetch bronze data
        logger.info("Fetching bronze options with underlying prices...")
        bronze_records = fetch_bronze_options_with_underlying()
        logger.info(f"✅ Fetched {len(bronze_records)} bronze records")
        
        if args.limit:
            bronze_records = bronze_records[:args.limit]
            logger.info(f"⚠️  Limited to {args.limit} records for testing")
        
        # Step 2: Enrich with Greeks
        logger.info("Calculating Greeks and implied volatility...")
        enriched = []
        skipped = 0
        
        for i, option in enumerate(bronze_records):
            if (i + 1) % 500 == 0:
                logger.info(f"  Processed {i + 1}/{len(bronze_records)} records...")
            
            enriched_option = enrich_option_with_greeks(option)
            if enriched_option:
                enriched.append(enriched_option)
            else:
                skipped += 1
        
        logger.info(f"✅ Enriched {len(enriched)} records (skipped {skipped})")
        
        # Show sample
        if enriched:
            logger.info("\nSample enriched record:")
            sample = enriched[0]
            logger.info(f"  Ticker: {sample['ticker']}, Date: {sample['trade_date']}")
            logger.info(f"  Strike: {sample['strike']}, Type: {sample['option_type']}")
            logger.info(f"  Underlying: {sample['underlying_price']}, Mid: {sample['mid_price']}")
            logger.info(f"  IV: {sample['implied_volatility']}, Delta: {sample['delta']}")
            logger.info(f"  Gamma: {sample['gamma']}, Theta: {sample['theta']}, Vega: {sample['vega']}")
        
        # Step 3: Insert to silver
        logger.info(f"\nInserting {len(enriched)} records into silver_bd_options_enriched...")
        inserted = insert_silver_records(enriched, dry_run=args.dry_run)
        
        if not args.dry_run:
            logger.info(f"✅ Inserted {inserted} records successfully")
        
        # Summary
        logger.info("\n" + "=" * 80)
        logger.info("✅ ENRICHMENT COMPLETE")
        logger.info("=" * 80)
        logger.info(f"Bronze records: {len(bronze_records)}")
        logger.info(f"Enriched: {len(enriched)}")
        logger.info(f"Skipped: {skipped}")
        logger.info(f"Inserted: {inserted}")
        logger.info("=" * 80)
        
    except Exception as e:
        logger.error(f"❌ Enrichment failed: {e}")
        import traceback
        traceback.print_exc()
        raise


if __name__ == '__main__':
    main()
