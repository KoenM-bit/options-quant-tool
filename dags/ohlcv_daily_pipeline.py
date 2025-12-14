"""
Daily OHLCV Data Pipeline

Fetches daily OHLCV (Open, High, Low, Close, Volume) stock data from Yahoo Finance
and loads it into the bronze_ohlcv table.

Schedule: Daily at 19:00 CET (after Amsterdam market close at 17:30)
Tickers: AD.AS (Ahold Delhaize), MT.AS (ArcelorMittal)
"""

from datetime import datetime, timedelta
from airflow import DAG
from airflow.operators.python import PythonOperator
from airflow.utils.dates import days_ago
import logging

logger = logging.getLogger(__name__)

# DAG default arguments
default_args = {
    'owner': 'airflow',
    'depends_on_past': False,
    'email_on_failure': False,
    'email_on_retry': False,
    'retries': 2,
    'retry_delay': timedelta(minutes=5),
    'start_date': datetime(2025, 10, 14),  # Start from when we have data
}

# Create DAG
dag = DAG(
    'ohlcv_daily_pipeline',
    default_args=default_args,
    description='Daily OHLCV data ingestion from Yahoo Finance',
    schedule_interval='0 19 * * 1-5',  # 19:00 CET, Monday-Friday only
    catchup=False,
    tags=['ohlcv', 'bronze', 'daily', 'yahoo-finance'],
)


def fetch_daily_ohlcv(**context):
    """
    Fetch today's OHLCV data for configured tickers.
    
    Uses execution_date to determine which date to fetch.
    For manual runs, fetches the current date.
    For scheduled runs, fetches the execution date.
    """
    import sys
    from pathlib import Path
    project_root = Path(__file__).parent.parent
    sys.path.insert(0, str(project_root))
    
    from datetime import date, timedelta
    from scripts.backfill_ohlcv_direct import OHLCVBackfill
    
    # Get execution date from context
    execution_date = context['execution_date']
    
    # For scheduled runs after market close, we want today's data
    # execution_date is in UTC, convert to CET
    target_date = execution_date.date()
    
    # If it's a weekend, skip (shouldn't happen with schedule, but safety check)
    if target_date.weekday() >= 5:  # Saturday = 5, Sunday = 6
        logger.info(f"⏭️  Skipping weekend date: {target_date}")
        return {
            'status': 'skipped',
            'reason': 'weekend',
            'date': str(target_date)
        }
    
    logger.info(f"📅 Fetching OHLCV data for: {target_date}")
    
    # Tickers to fetch
    tickers = ['AD.AS', 'MT.AS']
    
    # Create backfill instance
    backfill = OHLCVBackfill(tickers)
    
    # Ensure table exists
    backfill.create_table_if_not_exists()
    
    # Fetch data for target date
    # Note: We fetch target_date +1 day as end_date because Yahoo API is exclusive on end
    start_date = target_date
    end_date = target_date + timedelta(days=1)
    
    total_fetched = 0
    total_inserted = 0
    results = {}
    
    for ticker in tickers:
        logger.info(f"📊 Processing {ticker}...")
        
        try:
            fetched, inserted = backfill.backfill_ticker(ticker, start_date, end_date)
            total_fetched += fetched
            total_inserted += inserted
            
            results[ticker] = {
                'fetched': fetched,
                'inserted': inserted,
                'status': 'success'
            }
            
            logger.info(f"✅ {ticker}: Fetched {fetched} rows, Inserted {inserted} rows")
            
        except Exception as e:
            logger.error(f"❌ Error processing {ticker}: {e}")
            results[ticker] = {
                'fetched': 0,
                'inserted': 0,
                'status': 'failed',
                'error': str(e)
            }
    
    # Summary
    logger.info("=" * 80)
    logger.info(f"📊 DAILY OHLCV SUMMARY - {target_date}")
    logger.info("=" * 80)
    logger.info(f"Total: Fetched {total_fetched} rows, Inserted {total_inserted} rows")
    for ticker, result in results.items():
        status = result['status']
        if status == 'success':
            logger.info(f"  ✅ {ticker}: {result['fetched']} fetched, {result['inserted']} inserted")
        else:
            logger.info(f"  ❌ {ticker}: FAILED - {result.get('error', 'Unknown error')}")
    logger.info("=" * 80)
    
    # Return results for XCom
    return {
        'date': str(target_date),
        'total_fetched': total_fetched,
        'total_inserted': total_inserted,
        'tickers': results
    }


def calculate_indicators(**context):
    """
    Calculate technical indicators from OHLCV data.
    
    Calculates indicators for all configured tickers using available historical data.
    """
    import sys
    from pathlib import Path
    project_root = Path(__file__).parent.parent
    sys.path.insert(0, str(project_root))
    
    from scripts.calculate_technical_indicators import TechnicalIndicatorCalculator
    
    # Get execution date
    execution_date = context['execution_date']
    target_date = execution_date.date()
    
    logger.info(f"📊 Calculating technical indicators for {target_date}")
    
    # Tickers
    tickers = ['AD.AS', 'MT.AS']
    
    # Create calculator
    calculator = TechnicalIndicatorCalculator()
    
    # Ensure table exists
    calculator.create_table_if_not_exists()
    
    # Calculate indicators for each ticker
    # Use None as start_date to recalculate everything (ensures accuracy)
    total_rows = 0
    results = {}
    
    for ticker in tickers:
        try:
            rows = calculator.calculate_for_ticker(ticker, start_date=None, lookback_days=300)
            total_rows += rows
            results[ticker] = {
                'rows': rows,
                'status': 'success'
            }
            logger.info(f"✅ {ticker}: Calculated {rows} indicator rows")
        except Exception as e:
            logger.error(f"❌ Error calculating indicators for {ticker}: {e}")
            results[ticker] = {
                'rows': 0,
                'status': 'failed',
                'error': str(e)
            }
    
    logger.info(f"✅ Technical indicators calculated: {total_rows} total rows")
    
    return {
        'date': str(target_date),
        'total_rows': total_rows,
        'tickers': results
    }


def validate_data(**context):
    """
    Validate that data was successfully loaded.
    
    Checks:
    - At least one row was inserted or data already exists
    - OHLCV values are reasonable (not null, positive prices)
    """
    import sys
    from pathlib import Path
    project_root = Path(__file__).parent.parent
    sys.path.insert(0, str(project_root))
    
    from sqlalchemy import create_engine, text
    from src.config import settings
    
    # Get execution date
    execution_date = context['execution_date']
    target_date = execution_date.date()
    
    logger.info(f"🔍 Validating OHLCV data for {target_date}")
    
    engine = create_engine(settings.database_url)
    
    with engine.connect() as conn:
        # Check if data exists for target date
        result = conn.execute(
            text("""
                SELECT 
                    ticker,
                    COUNT(*) as row_count,
                    AVG(close) as avg_close,
                    SUM(volume) as total_volume
                FROM bronze_ohlcv
                WHERE trade_date = :target_date
                GROUP BY ticker
            """),
            {"target_date": target_date}
        )
        
        rows = result.fetchall()
        
        if not rows:
            logger.warning(f"⚠️  No data found for {target_date} (may be holiday or weekend)")
            return {
                'status': 'no_data',
                'date': str(target_date),
                'tickers': []
            }
        
        validation_results = {}
        all_valid = True
        
        for row in rows:
            ticker, count, avg_close, total_volume = row
            
            # Validation checks
            is_valid = True
            issues = []
            
            if count == 0:
                is_valid = False
                issues.append("No rows")
            
            if avg_close is None or float(avg_close) <= 0:
                is_valid = False
                issues.append(f"Invalid close price: {avg_close}")
            
            if total_volume is None or total_volume <= 0:
                is_valid = False
                issues.append(f"Invalid volume: {total_volume}")
            
            validation_results[ticker] = {
                'valid': is_valid,
                'row_count': count,
                'avg_close': float(avg_close) if avg_close else None,
                'total_volume': int(total_volume) if total_volume else None,
                'issues': issues
            }
            
            if is_valid:
                logger.info(f"✅ {ticker}: Valid - {count} rows, close €{avg_close:.2f}, volume {total_volume:,}")
            else:
                logger.error(f"❌ {ticker}: INVALID - {', '.join(issues)}")
                all_valid = False
        
        if not all_valid:
            raise ValueError(f"Data validation failed for {target_date}")
        
        logger.info(f"✅ All data validated successfully for {target_date}")
        
        return {
            'status': 'valid',
            'date': str(target_date),
            'tickers': validation_results
        }


def calculate_regimes(**context):
    """
    Calculate market regime classifications from technical indicators.
    
    Classifies market conditions into:
    - Trend regime (uptrend/downtrend/ranging)
    - Volatility regime (high/normal/low)
    - Market phase (accumulation/markup/distribution/markdown)
    - Recommended options strategy
    """
    import sys
    from pathlib import Path
    project_root = Path(__file__).parent.parent
    sys.path.insert(0, str(project_root))
    
    from scripts.calculate_market_regimes import MarketRegimeCalculator
    
    # Get execution date
    execution_date = context['execution_date']
    target_date = execution_date.date()
    
    logger.info(f"📊 Calculating market regimes for {target_date}")
    
    # Tickers
    tickers = ['AD.AS', 'MT.AS']
    
    # Create calculator
    calculator = MarketRegimeCalculator()
    
    # Ensure table exists
    calculator.ensure_table_exists()
    
    # Calculate regimes for each ticker
    # Use None as start_date to recalculate all available data
    total_rows = 0
    results = {}
    
    for ticker in tickers:
        try:
            rows = calculator.calculate_for_ticker(ticker, start_date=None)
            total_rows += rows
            results[ticker] = {
                'rows': rows,
                'status': 'success'
            }
            logger.info(f"✅ {ticker}: Calculated {rows} regime rows")
        except Exception as e:
            logger.error(f"❌ Error calculating regimes for {ticker}: {e}")
            results[ticker] = {
                'rows': 0,
                'status': 'failed',
                'error': str(e)
            }
    
    logger.info(f"✅ Market regimes calculated: {total_rows} total rows")
    
    return {
        'date': str(target_date),
        'total_rows': total_rows,
        'tickers': results
    }


def export_to_minio(**context):
    """
    Export OHLCV, indicators, and regime data to MinIO as Parquet files.
    
    Exports:
    - Bronze layer: bronze_ohlcv
    - Silver layer: fact_technical_indicators
    - Gold layer: fact_market_regime
    """
    import sys
    from pathlib import Path
    project_root = Path(__file__).parent.parent
    sys.path.insert(0, str(project_root))
    
    from scripts.export_parquet_simple import (
        export_bronze_ohlcv,
        export_technical_indicators,
        export_market_regime
    )
    
    # Get execution date
    execution_date = context['execution_date']
    target_date = execution_date.date()
    
    logger.info(f"📤 Exporting data to MinIO for {target_date}")
    
    tickers = ['AD.AS', 'MT.AS']
    results = {
        'bronze': {},
        'silver': {},
        'gold': {}
    }
    
    # Export bronze OHLCV
    for ticker in tickers:
        try:
            export_bronze_ohlcv(target_date, ticker)
            results['bronze'][ticker] = 'success'
            logger.info(f"✅ Bronze OHLCV exported for {ticker}")
        except Exception as e:
            logger.error(f"❌ Failed to export bronze OHLCV for {ticker}: {e}")
            results['bronze'][ticker] = f'failed: {e}'
    
    # Export silver indicators
    for ticker in tickers:
        try:
            export_technical_indicators(target_date, ticker)
            results['silver'][ticker] = 'success'
            logger.info(f"✅ Silver indicators exported for {ticker}")
        except Exception as e:
            logger.error(f"❌ Failed to export silver indicators for {ticker}: {e}")
            results['silver'][ticker] = f'failed: {e}'
    
    # Export gold regimes
    for ticker in tickers:
        try:
            export_market_regime(target_date, ticker)
            results['gold'][ticker] = 'success'
            logger.info(f"✅ Gold regimes exported for {ticker}")
        except Exception as e:
            logger.error(f"❌ Failed to export gold regimes for {ticker}: {e}")
            results['gold'][ticker] = f'failed: {e}'
    
    logger.info(f"✅ MinIO export completed for {target_date}")
    
    return {
        'date': str(target_date),
        'results': results
    }


# Define tasks
fetch_task = PythonOperator(
    task_id='fetch_daily_ohlcv',
    python_callable=fetch_daily_ohlcv,
    dag=dag,
)

validate_task = PythonOperator(
    task_id='validate_data',
    python_callable=validate_data,
    dag=dag,
)

calculate_task = PythonOperator(
    task_id='calculate_indicators',
    python_callable=calculate_indicators,
    dag=dag,
)

regime_task = PythonOperator(
    task_id='calculate_regimes',
    python_callable=calculate_regimes,
    dag=dag,
)

export_task = PythonOperator(
    task_id='export_to_minio',
    python_callable=export_to_minio,
    dag=dag,
)

# Set task dependencies
# Bronze -> Validate -> Silver -> Gold -> Export
fetch_task >> validate_task >> calculate_task >> regime_task >> export_task
