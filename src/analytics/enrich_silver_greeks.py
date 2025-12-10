"""
Enrich SILVER layer options data with Black-Scholes calculations.
Calculates implied volatility and Greeks for deduplicated options.

This is the production-optimized version that runs AFTER dbt Silver transformation.
Bronze stays immutable (raw data only).
Silver gets enriched with calculated Greeks.
"""

import sys
import logging
from datetime import datetime
from typing import Optional

from sqlalchemy import and_, text

sys.path.insert(0, '/opt/airflow/dags/..')

from src.analytics.black_scholes import calculate_option_metrics
from src.analytics.risk_free_rate import get_risk_free_rate_for_date
from src.utils.db import get_db_session
from src.config import settings

logger = logging.getLogger(__name__)


def get_underlying_price_from_silver(ticker: str, trade_date) -> Optional[float]:
    """Get underlying price from Silver layer for a given trade date."""
    with get_db_session() as session:
        # Query silver_bd_options_enriched for underlying price on this trade date
        result = session.execute(
            text("""
                SELECT underlying_price
                FROM silver_bd_options_enriched
                WHERE ticker = :ticker
                  AND trade_date = :trade_date
                  AND underlying_price IS NOT NULL
                LIMIT 1
            """),
            {'ticker': ticker, 'trade_date': trade_date}
        ).fetchone()
        
        if result and result[0]:
            return result[0]
        
        return None


def enrich_silver_with_greeks(
    ticker: str = None,
    risk_free_rate: float = None,  # Will be fetched from ECB if None
    batch_size: int = 100
) -> dict:
    """
    Enrich Silver layer options with Black-Scholes calculations.
    
    This function operates on the SILVER table (deduplicated data), not Bronze.
    Uses actual ECB risk-free rates (€STR) for accurate pricing.
    
    Args:
        ticker: Ticker to process (None = all)
        risk_free_rate: Annual risk-free rate (None = fetch from ECB)
        batch_size: Number of options to process per batch
    
    Returns:
        Dictionary with processing statistics
    """
    ticker = ticker or settings.ahold_ticker
    
    logger.info(f"Starting Black-Scholes enrichment for Silver layer: {ticker}")
    
    stats = {
        'total_processed': 0,
        'iv_calculated': 0,
        'greeks_calculated': 0,
        'skipped': 0,
        'errors': 0,
        'api_calls': 0,
        'api_failures': 0,
    }
    
    with get_db_session() as session:
        # Query silver_options where Greeks need calculation
        # Silver has no implied_volatility column yet, so we check all records
        query_sql = text("""
            SELECT 
                ticker,
                strike,
                expiry_date,
                trade_date,
                option_type,
                mid_price,
                underlying_price,
                days_to_expiry
            FROM silver_bd_options_enriched
            WHERE ticker = :ticker
              AND mid_price IS NOT NULL
              AND mid_price > 0
              AND underlying_price IS NOT NULL
              AND underlying_price > 0
              AND days_to_expiry > 0
              AND days_to_expiry IS NOT NULL
            ORDER BY trade_date DESC, expiry_date ASC
        """)
        
        result = session.execute(query_sql, {'ticker': ticker})
        rows = result.fetchall()
        
        total_options = len(rows)
        logger.info(f"Found {total_options} options to enrich in Silver layer")
        
        if total_options == 0:
            logger.warning("No options found in Silver layer to enrich!")
            return stats
        
        # Cache for risk-free rates by trade date to avoid repeated ECB calls
        rate_cache = {}
        
        # Process in batches
        for i in range(0, len(rows), batch_size):
            batch = rows[i:i + batch_size]
            
            for row in batch:
                try:
                    stats['total_processed'] += 1
                    
                    # Composite key fields (matches SELECT order)
                    ticker_val = row[0]      # ticker
                    strike = float(row[1])   # strike
                    expiry_date = row[2]     # expiry_date
                    trade_date = row[3]      # trade_date
                    option_type = row[4]     # option_type
                    mid_price = float(row[5])          # mid_price
                    underlying_price = float(row[6])   # underlying_price
                    days_to_expiry = int(row[7])       # days_to_expiry
                    
                    # Convert days to years
                    T = days_to_expiry / 365.0
                    
                    if T <= 0:
                        stats['skipped'] += 1
                        continue
                    
                    # Get actual ECB risk-free rate for this trade date
                    if risk_free_rate is None:
                        if trade_date not in rate_cache:
                            try:
                                stats['api_calls'] += 1
                                fetched_rate = get_risk_free_rate_for_date(trade_date)
                                if fetched_rate is None:
                                    raise ValueError("No rate returned from ECB helper")
                                # Clamp to sane bounds 0% - 10%
                                clamped_rate = max(0.0, min(fetched_rate, 0.10))
                                rate_cache[trade_date] = clamped_rate
                                logger.info(f"Fetched ECB rate for {trade_date}: {clamped_rate*100:.2f}% (raw={fetched_rate*100:.2f}%)")
                            except Exception as e:
                                stats['api_failures'] += 1
                                # Fallback to conservative 2% if ECB API fails
                                logger.warning(f"ECB API failed for {trade_date}, using 2% fallback: {e}")
                                rate_cache[trade_date] = 0.020
                        actual_rate = rate_cache[trade_date]
                    else:
                        # Clamp user-provided rate as well
                        actual_rate = max(0.0, min(risk_free_rate, 0.10))
                    
                    # Calculate metrics
                    metrics = calculate_option_metrics(
                        option_price=mid_price,
                        underlying_price=underlying_price,
                        strike=strike,
                        days_to_expiry=days_to_expiry,
                        option_type=option_type,
                        risk_free_rate=actual_rate
                    )

                    has_greeks = metrics.get('implied_volatility') is not None

                    # Update Silver with calculated Greeks and metadata
                    update_sql = text("""
                        UPDATE silver_bd_options_enriched
                        SET 
                            implied_volatility = :iv,
                            delta = :delta,
                            gamma = :gamma,
                            vega = :vega,
                            theta = :theta,
                            rho = :rho,
                            transformed_at = CURRENT_TIMESTAMP
                        WHERE ticker = :ticker
                          AND option_type = :option_type
                          AND strike = :strike
                          AND expiry_date = :expiry_date
                          AND trade_date = :trade_date
                    """)

                    session.execute(update_sql, {
                        'ticker': ticker_val,
                        'option_type': option_type,
                        'strike': strike,
                        'expiry_date': expiry_date,
                        'trade_date': trade_date,
                        'iv': metrics.get('implied_volatility'),
                        'delta': metrics.get('delta'),
                        'gamma': metrics.get('gamma'),
                        'vega': metrics.get('vega'),
                        'theta': metrics.get('theta'),
                        'rho': metrics.get('rho'),
                    })

                    if has_greeks:
                        stats['iv_calculated'] += 1
                        stats['greeks_calculated'] += 1
                
                except Exception as e:
                    logger.error(f"Error processing option {row[0] if row else 'unknown'}: {e}")
                    stats['errors'] += 1
                    continue
            
            # Commit batch
            session.commit()
            logger.info(f"Processed batch: {i} - {i + len(batch)} ({stats['greeks_calculated']} Greeks calculated)")
    
    logger.info(f"✅ Silver enrichment complete: {stats}")
    return stats


if __name__ == "__main__":
    # For testing
    import sys
    ticker = sys.argv[1] if len(sys.argv) > 1 else "AD.AS"
    result = enrich_silver_with_greeks(ticker=ticker)
    print(f"Results: {result}")
