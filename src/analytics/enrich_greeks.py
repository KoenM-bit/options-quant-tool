"""
Enrich bronze options data with Black-Scholes calculations.
Calculates implied volatility and Greeks for each option.
"""

import sys
import logging
from datetime import datetime
from typing import Optional

from sqlalchemy import and_

sys.path.insert(0, '/opt/airflow/dags/..')

from src.analytics.black_scholes import calculate_option_metrics
from src.utils.db import get_db_session
from src.models.bronze import BronzeFDOptions, BronzeFDOverview
from src.config import settings

logger = logging.getLogger(__name__)


def get_underlying_price(ticker: str, trade_date) -> Optional[float]:
    """Get underlying price for a given date.
    
    First tries to find an overview record for the exact date,
    then falls back to the most recent overview record for the ticker.
    """
    with get_db_session() as session:
        # Try exact date match first
        overview = session.query(BronzeFDOverview).filter(
            and_(
                BronzeFDOverview.ticker == ticker,
                BronzeFDOverview.peildatum == trade_date
            )
        ).order_by(BronzeFDOverview.scraped_at.desc()).first()
        
        if overview and overview.koers:
            return overview.koers
        
        # Fall back to most recent overview for this ticker
        overview = session.query(BronzeFDOverview).filter(
            BronzeFDOverview.ticker == ticker
        ).order_by(BronzeFDOverview.peildatum.desc()).first()
        
        if overview and overview.koers:
            return overview.koers
            
        return None


def enrich_options_with_greeks(
    ticker: str = None,
    risk_free_rate: float = 0.03,
    batch_size: int = 100
) -> dict:
    """
    Enrich options with Black-Scholes calculations.
    
    Args:
        ticker: Ticker to process (None = all)
        risk_free_rate: Annual risk-free rate (default 3%)
        batch_size: Number of options to process per batch
    
    Returns:
        Dictionary with processing statistics
    """
    ticker = ticker or settings.ahold_ticker
    
    logger.info(f"Starting Black-Scholes enrichment for {ticker}")
    
    stats = {
        'total_processed': 0,
        'iv_calculated': 0,
        'greeks_calculated': 0,
        'skipped': 0,
        'errors': 0,
    }
    
    with get_db_session() as session:
        # Get options that need enrichment (where IV is NULL)
        query = session.query(BronzeFDOptions).filter(
            and_(
                BronzeFDOptions.ticker == ticker,
                BronzeFDOptions.implied_volatility.is_(None),
                BronzeFDOptions.laatste.isnot(None),  # Must have a price
                BronzeFDOptions.laatste > 0
            )
        )
        
        total_options = query.count()
        logger.info(f"Found {total_options} options to enrich")
        
        # Process in batches
        offset = 0
        while True:
            batch = query.limit(batch_size).offset(offset).all()
            
            if not batch:
                break
            
            for option in batch:
                try:
                    stats['total_processed'] += 1
                    
                    # Get underlying price (from overview or from option record)
                    underlying_price = option.underlying_price
                    if not underlying_price:
                        # Try to get from overview table
                        underlying_price = get_underlying_price(
                            ticker,
                            option.scraped_at.date()
                        )
                    
                    if not underlying_price or underlying_price <= 0:
                        logger.warning(f"No underlying price for option {option.id}")
                        stats['skipped'] += 1
                        continue
                    
                    # Update underlying price if it was missing
                    if not option.underlying_price:
                        option.underlying_price = underlying_price
                    
                    # Calculate days to expiry
                    if not option.expiry_date:
                        stats['skipped'] += 1
                        continue
                    
                    trade_date = option.scraped_at.date()
                    days_to_expiry = (option.expiry_date - trade_date).days
                    
                    if days_to_expiry <= 0:
                        stats['skipped'] += 1
                        continue
                    
                    # Use mid price if available, otherwise last price
                    if option.bid and option.ask:
                        option_price = (option.bid + option.ask) / 2.0
                    else:
                        option_price = option.laatste
                    
                    if not option_price or option_price <= 0:
                        stats['skipped'] += 1
                        continue
                    
                    # Calculate metrics
                    metrics = calculate_option_metrics(
                        option_price=option_price,
                        underlying_price=underlying_price,
                        strike=option.strike,
                        days_to_expiry=days_to_expiry,
                        option_type=option.option_type,
                        risk_free_rate=risk_free_rate
                    )
                    
                    # Update option with calculated values
                    if metrics['implied_volatility'] is not None:
                        option.implied_volatility = metrics['implied_volatility']
                        stats['iv_calculated'] += 1
                    
                    if metrics['delta'] is not None:
                        option.delta = metrics['delta']
                        option.gamma = metrics['gamma']
                        option.vega = metrics['vega']
                        option.theta = metrics['theta']
                        stats['greeks_calculated'] += 1
                    
                except Exception as e:
                    logger.error(f"Error processing option {option.id}: {e}")
                    stats['errors'] += 1
                    continue
            
            # Commit batch
            session.commit()
            logger.info(f"Processed batch: {offset} - {offset + len(batch)}")
            
            offset += batch_size
    
    logger.info(f"✅ Enrichment complete: {stats}")
    return stats


if __name__ == "__main__":
    # For testing
    import sys
    ticker = sys.argv[1] if len(sys.argv) > 1 else "AD.AS"
    result = enrich_options_with_greeks(ticker=ticker)
    print(f"Results: {result}")
