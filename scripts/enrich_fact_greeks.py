#!/usr/bin/env python3
"""
Enrich fact_option_timeseries with Greeks (delta, gamma, theta, vega, rho, IV).
Uses Black-Scholes model to calculate option Greeks from market prices.

Usage:
    python scripts/enrich_fact_greeks.py [--date YYYY-MM-DD] [--lookback-days N]
"""

import sys
import logging
from pathlib import Path
from datetime import datetime, date, timedelta
from typing import Optional
import argparse

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from sqlalchemy import create_engine, text
from sqlalchemy.orm import sessionmaker
from src.config import settings
from src.analytics.black_scholes import calculate_option_metrics
from src.analytics.risk_free_rate import get_risk_free_rate_for_date

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


def enrich_greeks_for_date(session, trade_date: date, risk_free_rate: float) -> dict:
    """
    Enrich Greeks for all options on a specific trade date.
    
    Returns dict with success/failure counts.
    """
    logger.info(f"Processing trade_date: {trade_date}, risk_free_rate: {risk_free_rate:.4f}")
    
    # Query all facts for this date that need Greeks calculation
    query = text("""
        SELECT 
            f.ts_id,
            f.underlying_price,
            f.mid_price,
            f.days_to_expiry,
            c.strike,
            c.call_put
        FROM fact_option_timeseries f
        JOIN dim_option_contract c ON f.option_id = c.option_id
        WHERE f.trade_date = :trade_date
            AND f.iv IS NULL  -- Only process rows without Greeks
            AND f.underlying_price IS NOT NULL
            AND f.mid_price IS NOT NULL
            AND f.mid_price > 0.01  -- Skip very cheap options (< 1 cent) - relaxed from 5 cents
            AND f.days_to_expiry > 0
            -- Relaxed moneyness filter: allow wider range (0.3 to 5.0 instead of 0.5 to 1.5)
            -- This includes deep ITM/OTM options which are still tradeable
            AND f.underlying_price / NULLIF(c.strike, 0) BETWEEN 0.3 AND 5.0
        ORDER BY c.strike, c.call_put
    """)
    
    results = session.execute(query, {"trade_date": trade_date}).fetchall()
    
    if not results:
        logger.info(f"No options to enrich for {trade_date}")
        return {"total": 0, "success": 0, "failed": 0, "skipped": 0}
    
    logger.info(f"Found {len(results)} options to enrich")
    
    success_count = 0
    failed_count = 0
    
    for row in results:
        ts_id = row[0]
        underlying_price = float(row[1])
        option_price = float(row[2])
        days_to_expiry = int(row[3])
        strike = int(row[4])
        call_put = row[5]
        option_type = 'call' if call_put == 'C' else 'put'
        
        try:
            # Calculate Greeks
            greeks = calculate_option_metrics(
                option_price=option_price,
                underlying_price=underlying_price,
                strike=strike,
                days_to_expiry=days_to_expiry,
                option_type=option_type,
                risk_free_rate=risk_free_rate
            )
            
            if greeks and greeks.get('implied_volatility') is not None:
                # Update the fact record
                update_query = text("""
                    UPDATE fact_option_timeseries
                    SET 
                        iv = :iv,
                        delta = :delta,
                        gamma = :gamma,
                        theta = :theta,
                        vega = :vega,
                        rho = :rho
                    WHERE ts_id = :ts_id
                """)
                
                session.execute(update_query, {
                    "ts_id": ts_id,
                    "iv": greeks['implied_volatility'],
                    "delta": greeks['delta'],
                    "gamma": greeks['gamma'],
                    "theta": greeks['theta'],
                    "vega": greeks['vega'],
                    "rho": greeks['rho']
                })
                
                success_count += 1
                
                if success_count % 100 == 0:
                    session.commit()
                    logger.info(f"Processed {success_count}/{len(results)} options...")
            else:
                if success_count + failed_count < 5:  # Log first few for debugging
                    logger.error(f"Greeks calculation returned None or no IV: greeks={greeks}, ts_id={ts_id}, underlying_price={underlying_price}, option_price={option_price}, strike={strike}, days_to_expiry={days_to_expiry}")
                failed_count += 1
                
        except Exception as e:
            logger.error(f"Failed to calculate Greeks for ts_id={ts_id}: {e}")
            failed_count += 1
    
    # Final commit
    session.commit()
    
    logger.info(f"Completed: {success_count} success, {failed_count} failed")
    
    return {
        "total": len(results),
        "success": success_count,
        "failed": failed_count,
        "skipped": 0
    }


def enrich_greeks(target_date: Optional[date] = None, lookback_days: int = 7) -> int:
    """
    Enrich Greeks for fact_option_timeseries.
    
    Args:
        target_date: Specific date to process (default: today)
        lookback_days: Number of days to look back if target_date is None
    
    Returns:
        0 if successful, 1 if failed
    """
    logger.info("=" * 60)
    logger.info("Starting Greeks Enrichment for fact_option_timeseries")
    logger.info("=" * 60)
    
    # Create database session
    engine = create_engine(settings.database_url, echo=False)
    Session = sessionmaker(bind=engine)
    session = Session()
    
    try:
        # Determine dates to process
        if target_date:
            dates_to_process = [target_date]
            logger.info(f"Processing single date: {target_date}")
        else:
            # Get last N days with data
            query = text("""
                SELECT DISTINCT trade_date 
                FROM fact_option_timeseries
                WHERE trade_date >= CURRENT_DATE - INTERVAL ':days days'
                    AND iv IS NULL
                ORDER BY trade_date DESC
            """)
            result = session.execute(query, {"days": lookback_days})
            dates_to_process = [row[0] for row in result.fetchall()]
            logger.info(f"Processing {len(dates_to_process)} dates from last {lookback_days} days")
        
        if not dates_to_process:
            logger.info("No dates to process")
            return 0
        
        # Process each date
        total_stats = {"total": 0, "success": 0, "failed": 0, "skipped": 0}
        
        for process_date in dates_to_process:
            # Get risk-free rate for this date
            risk_free_rate = get_risk_free_rate_for_date(process_date)
            
            # Enrich Greeks
            stats = enrich_greeks_for_date(session, process_date, risk_free_rate)
            
            # Accumulate stats
            for key in total_stats:
                total_stats[key] += stats[key]
        
        # Summary
        logger.info("=" * 60)
        logger.info("Greeks Enrichment Summary")
        logger.info("=" * 60)
        logger.info(f"Total options processed: {total_stats['total']}")
        logger.info(f"Successfully enriched: {total_stats['success']}")
        logger.info(f"Failed: {total_stats['failed']}")
        
        if total_stats['success'] > 0:
            success_rate = (total_stats['success'] / total_stats['total']) * 100
            logger.info(f"Success rate: {success_rate:.1f}%")
        
        return 0
        
    except Exception as e:
        logger.error(f"Error enriching Greeks: {e}", exc_info=True)
        session.rollback()
        return 1
        
    finally:
        session.close()


def main():
    parser = argparse.ArgumentParser(description="Enrich fact_option_timeseries with Greeks")
    parser.add_argument(
        "--date",
        type=str,
        help="Specific date to process (YYYY-MM-DD). If not provided, processes last N days"
    )
    parser.add_argument(
        "--lookback-days",
        type=int,
        default=7,
        help="Number of days to look back if --date not provided (default: 7)"
    )
    
    args = parser.parse_args()
    
    # Parse target date if provided
    target_date = None
    if args.date:
        try:
            target_date = datetime.strptime(args.date, "%Y-%m-%d").date()
            logger.info(f"Processing specific date: {target_date}")
        except ValueError:
            logger.error(f"Invalid date format: {args.date}. Use YYYY-MM-DD")
            return 1
    
    # Run enrichment
    return enrich_greeks(target_date=target_date, lookback_days=args.lookback_days)


if __name__ == "__main__":
    sys.exit(main())
