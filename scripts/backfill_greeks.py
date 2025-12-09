"""
Backfill Greeks for all historical options data.

This script:
1. Fetches historical risk-free rates for each date
2. Calculates implied volatility and Greeks for all historical options
3. Updates the Bronze layer with accurate Greeks data
4. Enables full historical analytics in Gold layer
"""

import sys
import logging
from datetime import datetime
from typing import Dict

sys.path.insert(0, '/opt/airflow')

from sqlalchemy import and_, func
from src.utils.db import get_db_session
from src.models.bronze import BronzeFDOptions, BronzeFDOverview
from src.analytics.risk_free_rate import get_rate_for_time_to_expiry
from src.analytics.black_scholes import calculate_option_metrics
from src.analytics.risk_free_rate import get_risk_free_rate_for_date, get_historical_rates_batch
from src.config import settings

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


def get_underlying_price_for_date(session, ticker: str, trade_date) -> float:
    """Get underlying price for a specific date from overview table."""
    overview = session.query(BronzeFDOverview).filter(
        and_(
            BronzeFDOverview.ticker == ticker,
            BronzeFDOverview.peildatum == trade_date
        )
    ).first()
    
    if overview and overview.koers:
        return overview.koers
    
    raise ValueError(f"No underlying price found for {ticker} on {trade_date}")


def backfill_greeks_for_date(
    session,
    ticker: str,
    trade_date,
    risk_free_rate: float
) -> Dict[str, int]:
    """
    Calculate and update Greeks for all options on a specific trade date.
    
    Args:
        session: Database session
        ticker: Stock ticker
        trade_date: The trade date to process
        risk_free_rate: Risk-free rate for this date
    
    Returns:
        Statistics dictionary
    """
    stats = {
        'processed': 0,
        'updated': 0,
        'skipped': 0,
        'errors': 0
    }
    
    # Get underlying price for this date
    try:
        underlying_price = get_underlying_price_for_date(session, ticker, trade_date)
    except ValueError as e:
        logger.error(f"Cannot process {trade_date}: {e}")
        return stats
    
    logger.info(f"Processing {trade_date}: underlying=${underlying_price:.2f}, r={risk_free_rate:.4%}")
    
    # Get all options for this date that don't have Greeks yet
    # For Fridays, we need to check SATURDAY scrapes (since Saturday morning = Friday data)
    # For other days, check both same day and next day scrapes
    from datetime import timedelta
    
    scrape_dates = [trade_date]
    if trade_date.weekday() == 4:  # Friday
        scrape_dates.append(trade_date + timedelta(days=1))  # Saturday scrape
    
    options = session.query(BronzeFDOptions).filter(
        and_(
            BronzeFDOptions.ticker == ticker,
            func.date(BronzeFDOptions.scraped_at).in_(scrape_dates),
            BronzeFDOptions.gamma.is_(None)  # Only process if Greeks missing
        )
    ).all()
    
    logger.info(f"Found {len(options)} options without Greeks for {trade_date}")
    
    for option in options:
        try:
            stats['processed'] += 1
            
            # Skip if missing required data
            if not option.strike or not option.expiry_date:
                stats['skipped'] += 1
                continue
            
            # Calculate time to expiry
            expiry = option.expiry_date
            if isinstance(trade_date, str):
                trade_dt = datetime.strptime(trade_date, '%Y-%m-%d').date()
            else:
                trade_dt = trade_date
            
            days_to_expiry = (expiry - trade_dt).days
            if days_to_expiry <= 0:
                stats['skipped'] += 1
                continue
            
            time_to_expiry = days_to_expiry / 365.0
            
            # Get term-matched risk-free rate for this option's maturity
            # This is critical for accuracy - longer-dated options need higher rates
            option_risk_free_rate = get_rate_for_time_to_expiry(time_to_expiry, trade_dt)
            
            # Determine option type
            is_call = option.option_type.lower() in ['call', 'c']
            
            # Skip if no market price available
            if not option.laatste or option.laatste <= 0:
                stats['skipped'] += 1
                continue
            
            # Calculate Greeks and IV using term-matched rate
            metrics = calculate_option_metrics(
                option_price=option.laatste,
                underlying_price=underlying_price,
                strike=option.strike,
                days_to_expiry=days_to_expiry,
                option_type='call' if is_call else 'put',
                risk_free_rate=option_risk_free_rate  # Use term-matched rate!
            )
            
            # Update the option record
            option.underlying_price = underlying_price
            option.implied_volatility = metrics.get('implied_volatility')
            option.delta = metrics.get('delta')
            option.gamma = metrics.get('gamma')
            option.theta = metrics.get('theta')
            option.vega = metrics.get('vega')
            
            stats['updated'] += 1
            
            # Commit in batches
            if stats['updated'] % 100 == 0:
                session.commit()
                logger.info(f"  Progress: {stats['updated']} options updated...")
            
        except Exception as e:
            logger.error(f"Error processing option {option.id}: {e}")
            stats['errors'] += 1
            continue
    
    # Final commit
    session.commit()
    
    return stats


def backfill_all_historical_greeks(ticker: str = None):
    """
    Backfill Greeks for all historical data.
    """
    ticker = ticker or settings.ahold_ticker
    
    logger.info("="*70)
    logger.info("🚀 STARTING GREEKS BACKFILL FOR HISTORICAL DATA")
    logger.info("="*70)
    
    with get_db_session() as session:
        # Get all distinct trade dates that have options without Greeks
        dates_query = session.query(
            func.date(BronzeFDOptions.scraped_at).label('trade_date')
        ).filter(
            and_(
                BronzeFDOptions.ticker == ticker,
                BronzeFDOptions.gamma.is_(None)
            )
        ).distinct().order_by('trade_date')
        
        trade_dates = [row.trade_date for row in dates_query.all()]
        
        if not trade_dates:
            logger.info("✅ No dates need Greeks backfill!")
            return
        
        logger.info(f"\n📅 Found {len(trade_dates)} dates needing Greeks calculation:")
        logger.info(f"   Range: {trade_dates[0]} to {trade_dates[-1]}")
        
        # Fetch risk-free rates for the entire period
        logger.info(f"\n📊 Fetching risk-free rates from ECB...")
        rates = get_historical_rates_batch(trade_dates[0], trade_dates[-1])
        
        # Fill in any missing dates with interpolation/fallback
        rate_lookup = {}
        last_known_rate = 0.035  # Fallback
        
        for trade_date in trade_dates:
            date_str = trade_date.strftime('%Y-%m-%d')
            if date_str in rates:
                rate_lookup[trade_date] = rates[date_str]
                last_known_rate = rates[date_str]
            else:
                # Use last known rate for weekends/holidays
                logger.warning(f"No rate for {date_str}, using last known: {last_known_rate:.4%}")
                rate_lookup[trade_date] = last_known_rate
        
        # Process each date
        total_stats = {
            'dates_processed': 0,
            'total_updated': 0,
            'total_skipped': 0,
            'total_errors': 0
        }
        
        logger.info(f"\n🔧 Processing {len(trade_dates)} dates...\n")
        
        for i, trade_date in enumerate(trade_dates, 1):
            risk_free_rate = rate_lookup.get(trade_date, 0.035)
            
            logger.info(f"\n[{i}/{len(trade_dates)}] Processing {trade_date}...")
            
            date_stats = backfill_greeks_for_date(
                session=session,
                ticker=ticker,
                trade_date=trade_date,
                risk_free_rate=risk_free_rate
            )
            
            total_stats['dates_processed'] += 1
            total_stats['total_updated'] += date_stats['updated']
            total_stats['total_skipped'] += date_stats['skipped']
            total_stats['total_errors'] += date_stats['errors']
            
            logger.info(f"✅ {trade_date}: Updated {date_stats['updated']}, "
                       f"Skipped {date_stats['skipped']}, Errors {date_stats['errors']}")
    
    # Summary
    logger.info("\n" + "="*70)
    logger.info("🎉 GREEKS BACKFILL COMPLETED")
    logger.info("="*70)
    logger.info(f"📊 Summary:")
    logger.info(f"   Dates processed: {total_stats['dates_processed']}")
    logger.info(f"   Options updated: {total_stats['total_updated']}")
    logger.info(f"   Skipped: {total_stats['total_skipped']}")
    logger.info(f"   Errors: {total_stats['total_errors']}")
    logger.info("="*70)
    
    return total_stats


if __name__ == '__main__':
    import argparse
    
    parser = argparse.ArgumentParser(description='Backfill Greeks for historical options data')
    parser.add_argument('--ticker', type=str, help='Stock ticker (default: from settings)')
    
    args = parser.parse_args()
    
    backfill_all_historical_greeks(ticker=args.ticker)
