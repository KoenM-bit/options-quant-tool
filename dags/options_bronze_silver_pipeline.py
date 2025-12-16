"""
Bronze & Silver Data Pipeline DAG
==================================
Foundation pipeline for options data ingestion and processing.

Pipeline Flow:
1. Ensure bronze tables exist (idempotent table creation)
2. Scrape Beursduivel (BD) options + underlying → bronze_bd_options + bronze_bd_underlying
3. Scrape FD.nl options + overview → bronze_fd_options + bronze_fd_overview
4. Quality check bronze layer data (data validation)
5. DBT Silver - Build star schema → dim_underlying, dim_option_contract, fact_option_timeseries
6. Enrich fact table with Greeks - Calculate Black-Scholes Greeks and implied volatility
7. Export bronze & silver to MinIO - Save as partitioned parquet files
8. Sync ClickHouse - Update ClickHouse tables from MinIO parquet

This DAG provides the foundational data layer that gold analytics DAGs consume.
Runs Mon-Sat at 16:30 UTC (17:30 CET Amsterdam time).
BD scrapes live data for TODAY, FD scrapes delayed data (uses peildatum).

Schedule:
- Monday-Friday: Both BD and FD scrape (FD publishes Tue-Sat data)
- Saturday: Only FD scrapes (BD markets closed, FD publishes Friday data)
- Sunday: DAG does not run (no new data)
"""

from datetime import datetime, timedelta
from airflow import DAG
from airflow.operators.python import PythonOperator
from airflow.utils.trigger_rule import TriggerRule

import sys
import os
import subprocess

# Add project root to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from src.config import settings

# Default args
default_args = {
    'owner': 'data-engineering',
    'depends_on_past': False,
    'email_on_failure': True,
    'email_on_retry': False,
    'retries': 2,
    'retry_delay': timedelta(minutes=5),
    'execution_timeout': timedelta(hours=1),
}

# DAG definition
dag = DAG(
    'options_bronze_silver_pipeline',
    default_args=default_args,
    description='Foundation pipeline: Bronze ingestion + Silver processing + ClickHouse sync',
    schedule_interval='30 16 * * 1-6',  # 16:30 UTC = 17:30 CET (Amsterdam winter time), Mon-Sat
    start_date=datetime(2024, 1, 1),
    catchup=False,
    tags=['ahold', 'options', 'bronze', 'silver', 'ingestion', 'foundation'],
    max_active_runs=1,
)


def ensure_bronze_tables(**context):
    """
    Ensure all required bronze tables exist.
    Creates tables if they don't exist (idempotent).
    """
    import logging
    from src.utils.db import get_db_session
    from src.models.bronze import BronzeFDOptions, BronzeFDOverview
    from src.models.bronze_bd import BronzeBDOptions
    from src.models.bronze_bd_underlying import BronzeBDUnderlying
    from src.models.base import Base
    
    logger = logging.getLogger(__name__)
    logger.info("🔧 Ensuring bronze tables exist...")
    
    with get_db_session() as session:
        # Create tables if they don't exist
        Base.metadata.create_all(bind=session.get_bind(), checkfirst=True)
        logger.info("✅ All bronze tables verified/created")
    
    return True


def scrape_beursduivel(**context):
    """Scrape Beursduivel options + underlying data for ALL configured tickers."""
    import logging
    from src.scrapers.bd_options_scraper import scrape_all_options
    from src.utils.db import get_db_session
    from src.models.bronze_bd import BronzeBDOptions
    from src.models.bronze_bd_underlying import BronzeBDUnderlying
    from datetime import datetime
    
    logger = logging.getLogger(__name__)
    
    # Use ACTUAL run date (today), not execution_date (yesterday)
    # BD data is live scraped data for TODAY
    today = datetime.now().date()
    trade_date = today
    
    # Skip BD scraping on Saturday (only FD runs on Saturday)
    if today.weekday() == 5:  # 5=Saturday
        logger.info(f"⏭️  Skipping BD scrape on Saturday (only FD runs today)")
        context['ti'].xcom_push(key='bd_contracts', value=0)
        context['ti'].xcom_push(key='bd_underlying_price', value=None)
        return {'contracts': 0, 'underlying': None, 'skipped': True}
    
    logger.info(f"🚀 Scraping Beursduivel for {trade_date} (actual date) - MULTI-TICKER")
    
    # Loop through all configured tickers
    total_contracts = 0
    all_results = {}
    
    for ticker_config in settings.tickers:
        ticker = ticker_config['ticker']
        bd_url = ticker_config.get('bd_url')
        
        if not bd_url:
            logger.warning(f"⚠️  No BD URL configured for {ticker}, skipping...")
            continue
        
        logger.info(f"📊 Scraping {ticker} from {bd_url}")
        
        try:
            # Scrape data (fetch_live=True gets all strikes with live prices)
            options_data, underlying_data = scrape_all_options(
                ticker=ticker,
                url=bd_url,
                fetch_live=True
            )
            
            logger.info(f"Scraped {len(options_data)} options for {ticker}, underlying: {underlying_data}")
            
            # Load to database - skip duplicates (protects against holiday re-scrapes)
            with get_db_session() as session:
                inserted = 0
                skipped = 0
                
                # Options - map fields from scraper to database model
                for option in options_data:
                    # Map scraper field names to database column names
                    if 'type' in option:
                        option['option_type'] = option.pop('type')
                    if 'url' in option:
                        option['source_url'] = option.pop('url')
                    
                    option['trade_date'] = trade_date
                    option['ticker'] = ticker
                    
                    # Filter to only include fields that exist in the model
                    valid_fields = {
                        'ticker', 'symbol_code', 'issue_id', 'trade_date', 'option_type',
                        'expiry_date', 'expiry_text', 'strike', 'bid', 'ask', 'last_price',
                        'volume', 'last_timestamp', 'last_date_text', 'source', 'source_url', 'scraped_at'
                    }
                    filtered_option = {k: v for k, v in option.items() if k in valid_fields}
                    
                    # Check if record already exists - if so, SKIP (don't overwrite good data)
                    existing = session.query(BronzeBDOptions).filter_by(
                        ticker=filtered_option['ticker'],
                        trade_date=filtered_option['trade_date'],
                        option_type=filtered_option['option_type'],
                        strike=filtered_option['strike'],
                        expiry_date=filtered_option['expiry_date']
                    ).first()
                    
                    if existing:
                        # Skip - keep original data (likely better quality than holiday re-scrape)
                        skipped += 1
                    else:
                        # Insert new record
                        session.add(BronzeBDOptions(**filtered_option))
                        inserted += 1
                
                # Underlying
                if underlying_data:
                    underlying_data['trade_date'] = trade_date
                    underlying_data['ticker'] = ticker
                    session.add(BronzeBDUnderlying(**underlying_data))
                
                session.commit()
            
            if skipped > 0:
                logger.warning(f"⚠️  BD {ticker}: {inserted} new, {skipped} skipped (duplicates)")
            else:
                logger.info(f"✅ BD {ticker}: {inserted} new contracts, underlying €{underlying_data.get('last_price') if underlying_data else 'N/A'}")
            
            total_contracts += len(options_data)
            all_results[ticker] = {
                'contracts': len(options_data),
                'inserted': inserted,
                'underlying': underlying_data.get('last_price') if underlying_data else None
            }
        
        except Exception as e:
            logger.error(f"❌ BD scrape failed for {ticker}: {e}")
            all_results[ticker] = {'error': str(e)}
            # Continue with other tickers
    
    logger.info(f"🎉 BD scraping complete: {total_contracts} total contracts across {len(all_results)} tickers")
    
    context['ti'].xcom_push(key='bd_contracts', value=total_contracts)
    context['ti'].xcom_push(key='bd_results', value=all_results)
    
    return {'total_contracts': total_contracts, 'tickers': all_results}


def scrape_fd(**context):
    """Scrape FD.nl options + overview data for ALL configured tickers."""
    import logging
    from src.scrapers.fd_overview_scraper import scrape_fd_overview
    from src.scrapers.fd_options_scraper import scrape_fd_options
    from src.utils.db import get_db_session
    from src.models.bronze import BronzeFDOverview, BronzeFDOptions
    from datetime import datetime
    
    logger = logging.getLogger(__name__)
    
    # Use ACTUAL run date (today) for weekday check
    today = datetime.now().date()
    
    # Skip FD scraping on Monday (FD publishes with 1-day delay, no Saturday data exists)
    if today.weekday() == 0:  # 0=Monday
        logger.info(f"⏭️  Skipping FD scrape on Monday (no Saturday data available)")
        context['ti'].xcom_push(key='fd_contracts', value=0)
        context['ti'].xcom_push(key='fd_price', value=None)
        return {'contracts': 0, 'price': None, 'skipped': True}
    
    logger.info(f"🚀 Scraping FD.nl for today={today} - MULTI-TICKER")
    
    # Loop through all configured tickers
    total_contracts = 0
    all_results = {}
    
    for ticker_config in settings.tickers:
        ticker = ticker_config['ticker']
        symbol_code = ticker_config['symbol_code']
        
        logger.info(f"📊 Scraping {ticker} (symbol: {symbol_code})")
        
        try:
            # Scrape FD overview (price + timestamp) - IMPORTANT: Pass ticker and symbol_code!
            overview_data = scrape_fd_overview(ticker=ticker, symbol_code=symbol_code)
            trade_date = overview_data.get('peildatum')  # FD uses peildatum as trade_date
            overview_data['trade_date'] = trade_date  # Add trade_date to overview too
            logger.info(f"FD Overview {ticker}: €{overview_data.get('koers')} @ trade_date={trade_date} (peildatum)")
            
            # Scrape FD options chain
            options_data = scrape_fd_options(ticker=ticker, symbol_code=symbol_code)
            logger.info(f"FD Options {ticker}: {len(options_data)} contracts")
            
            # Add trade_date to each option (FD scraper doesn't include it)
            for option in options_data:
                option['trade_date'] = trade_date
            
            # Load to database
            with get_db_session() as session:
                # Overview
                session.add(BronzeFDOverview(**overview_data))
                
                # Options
                for option in options_data:
                    session.add(BronzeFDOptions(**option))
                
                session.commit()
            
            logger.info(f"✅ FD {ticker}: {len(options_data)} contracts, price €{overview_data.get('koers')}, trade_date={trade_date}")
            
            total_contracts += len(options_data)
            all_results[ticker] = {
                'contracts': len(options_data),
                'price': overview_data.get('koers'),
                'trade_date': trade_date
            }
        
        except Exception as e:
            logger.error(f"❌ FD scrape failed for {ticker}: {e}")
            all_results[ticker] = {'error': str(e)}
            # Continue with other tickers
    
    logger.info(f"🎉 FD scraping complete: {total_contracts} total contracts across {len(all_results)} tickers")
    
    context['ti'].xcom_push(key='fd_contracts', value=total_contracts)
    context['ti'].xcom_push(key='fd_results', value=all_results)
    
    return {'total_contracts': total_contracts, 'tickers': all_results}
    
    # Skip FD scraping on Monday (FD publishes with 1-day delay, no Saturday data exists)
    if today.weekday() == 0:  # 0=Monday
        logger.info(f"⏭️  Skipping FD scrape on Monday (no new data, FD publishes Tue-Sat)")
        context['ti'].xcom_push(key='fd_contracts', value=0)
        return {'contracts': 0, 'skipped': True}
    
    logger.info(f"🚀 Scraping FD.nl for today={today}")
    
    try:
        # Scrape overview first to get the peildatum (actual trading day)
        overview_data = scrape_fd_overview()
        peildatum = overview_data.get('peildatum')
        
        if not peildatum:
            raise ValueError("peildatum is missing from FD overview data!")
        
        # Use peildatum as the trade_date (the actual trading day this data represents)
        trade_date = peildatum
        logger.info(f"FD Overview: €{overview_data.get('koers')} @ trade_date={trade_date} (peildatum)")
        
        # Scrape options
        options_data = scrape_fd_options(ticker='AD.AS', symbol_code='AEX.AH/O')
        logger.info(f"FD Options: {len(options_data)} contracts")
        
        # Load to database
        with get_db_session() as session:
            # Overview - add trade_date (same as peildatum)
            overview_data['trade_date'] = trade_date
            session.add(BronzeFDOverview(**overview_data))
            
            # Options - add trade_date (same as peildatum from overview)
            for option in options_data:
                option['trade_date'] = trade_date
                session.add(BronzeFDOptions(**option))
            
            session.commit()
        
        logger.info(f"✅ FD: {len(options_data)} contracts, price €{overview_data.get('koers')}, trade_date={trade_date}")
        
        context['ti'].xcom_push(key='fd_contracts', value=len(options_data))
        context['ti'].xcom_push(key='fd_price', value=overview_data.get('koers'))
        
        return {'contracts': len(options_data), 'price': overview_data.get('koers')}
    
    except Exception as e:
        logger.error(f"❌ FD scrape failed: {e}")
        raise


def check_bronze_data_quality(**context):
    """
    Verify bronze layer data quality before proceeding.
    Checks both BD and FD data for today's trade date.
    """
    import logging
    from datetime import datetime
    from src.utils.db import get_db_session
    from sqlalchemy import text
    
    logger = logging.getLogger(__name__)
    
    # Use ACTUAL date (today) for quality check
    trade_date = datetime.now().date()
    
    logger.info(f"Checking bronze data quality for {trade_date} (today)")
    
    with get_db_session() as session:
        # Check BD data
        bd_query = text("""
            SELECT 
                COUNT(*) as bd_options_count,
                COUNT(DISTINCT strike) as bd_unique_strikes,
                SUM(CASE WHEN bid IS NOT NULL THEN 1 ELSE 0 END) as bd_with_bid,
                SUM(CASE WHEN ask IS NOT NULL THEN 1 ELSE 0 END) as bd_with_ask
            FROM bronze_bd_options
            WHERE trade_date = :trade_date
        """)
        bd_result = session.execute(bd_query, {'trade_date': trade_date}).fetchone()
        
        # Check FD data (use trade_date column)
        fd_query = text("""
            SELECT 
                COUNT(*) as fd_options_count,
                COUNT(DISTINCT strike) as fd_unique_strikes,
                SUM(CASE WHEN open_interest IS NOT NULL THEN 1 ELSE 0 END) as fd_with_oi
            FROM bronze_fd_options
            WHERE trade_date = :trade_date
        """)
        fd_result = session.execute(fd_query, {'trade_date': trade_date}).fetchone()
        
        # Check BD underlying
        bd_underlying_query = text("""
            SELECT COUNT(*), MAX(last_price) as underlying_price
            FROM bronze_bd_underlying
            WHERE trade_date = :trade_date
        """)
        bd_underlying = session.execute(bd_underlying_query, {'trade_date': trade_date}).fetchone()
        
        # Quality checks
        quality_issues = []
        
        if bd_result[0] < 100:  # Expect ~300+ contracts
            quality_issues.append(f"Low BD contract count: {bd_result[0]} (expected 300+)")
        
        if bd_result[1] < 20:  # Expect 30+ strikes
            quality_issues.append(f"Low BD strike count: {bd_result[1]} (expected 30+)")
        
        if bd_underlying[0] == 0:
            quality_issues.append("No BD underlying data found")
        
        if bd_underlying[1] is None or bd_underlying[1] <= 0:
            quality_issues.append(f"Invalid underlying price: {bd_underlying[1]}")
        
        # Calculate data quality score
        bd_bid_coverage = (bd_result[2] / bd_result[0] * 100) if bd_result[0] > 0 else 0
        bd_ask_coverage = (bd_result[3] / bd_result[0] * 100) if bd_result[0] > 0 else 0
        
        stats = {
            'trade_date': str(trade_date),
            'bd_contracts': bd_result[0],
            'bd_strikes': bd_result[1],
            'bd_bid_coverage': round(bd_bid_coverage, 1),
            'bd_ask_coverage': round(bd_ask_coverage, 1),
            'fd_contracts': fd_result[0],
            'fd_strikes': fd_result[1],
            'fd_oi_coverage': round((fd_result[2] / fd_result[0] * 100) if fd_result[0] > 0 else 0, 1),
            'underlying_price': float(bd_underlying[1]) if bd_underlying[1] else None,
            'quality_issues': quality_issues,
            'quality_score': 'PASS' if len(quality_issues) == 0 else 'WARN'
        }
        
        logger.info(f"Bronze data quality: {stats}")
        
        # Push to XCom
        context['ti'].xcom_push(key='bronze_quality', value=stats)
        
        if len(quality_issues) > 0:
            logger.warning(f"Quality issues detected: {quality_issues}")
        
        return stats


def run_dbt_silver(**context):
    """Run DBT silver models to build star schema."""
    import logging
    logger = logging.getLogger(__name__)
    
    logger.info("Running DBT Silver transformation (star schema)")
    
    # First, ensure dbt dependencies are installed
    logger.info("Installing DBT dependencies...")
    deps_result = subprocess.run(
        ['dbt', 'deps', '--profiles-dir', '/opt/airflow/dbt'],
        cwd='/opt/airflow/dbt/ahold_options',
        capture_output=True,
        text=True
    )
    
    if deps_result.returncode != 0:
        logger.error(f"DBT deps failed:\n{deps_result.stdout}\n{deps_result.stderr}")
        raise Exception(f"DBT deps failed with return code {deps_result.returncode}")
    
    logger.info("DBT dependencies installed successfully")
    
    # Run dimensions first, then fact table
    result = subprocess.run(
        ['dbt', 'run', '--models', 'tag:silver', '--profiles-dir', '/opt/airflow/dbt'],
        cwd='/opt/airflow/dbt/ahold_options',
        capture_output=True,
        text=True
    )
    
    logger.info(f"DBT stdout:\n{result.stdout}")
    if result.stderr:
        logger.warning(f"DBT stderr:\n{result.stderr}")
    
    if result.returncode != 0:
        raise Exception(f"DBT Silver failed with return code {result.returncode}")
    
    # Parse results - count fact table records
    if "fact_option_timeseries" in result.stdout:
        import re
        # Look for pattern like "SELECT 5869" after fact_option_timeseries
        match = re.search(r'fact_option_timeseries.*?SELECT (\d+)', result.stdout, re.DOTALL)
        if match:
            records_count = int(match.group(1))
            logger.info(f"✅ DBT Silver completed: {records_count} fact records")
            context['ti'].xcom_push(key='silver_records', value=records_count)
    
    logger.info("✅ DBT Silver star schema completed successfully")
    return result.returncode


def enrich_fact_greeks(**context):
    """Calculate Greeks and implied volatility for fact_option_timeseries."""
    import logging
    logger = logging.getLogger(__name__)
    
    logger.info("Enriching fact_option_timeseries with Greeks and implied volatility")
    
    try:
        # Run enrichment script for today's data
        from datetime import datetime
        today = datetime.now().date()
        
        result = subprocess.run(
            ['python', '/opt/airflow/scripts/enrich_fact_greeks.py', '--date', today.strftime('%Y-%m-%d')],
            cwd='/opt/airflow',
            capture_output=True,
            text=True,
            env={**os.environ, 'POSTGRES_HOST': 'postgres'}
        )
        
        logger.info(f"Greeks enrichment stdout:\n{result.stdout}")
        if result.stderr:
            logger.warning(f"Greeks enrichment stderr:\n{result.stderr}")
        
        if result.returncode != 0:
            raise Exception(f"Greeks enrichment failed with return code {result.returncode}")
        
        # Parse stats from output
        import re
        success_match = re.search(r'Successfully enriched: (\d+)', result.stdout)
        failed_match = re.search(r'Failed: (\d+)', result.stdout)
        
        success_count = int(success_match.group(1)) if success_match else 0
        failed_count = int(failed_match.group(1)) if failed_match else 0
        total = success_count + failed_count
        
        stats = {
            'total_processed': total,
            'greeks_calculated': success_count,
            'failed': failed_count,
            'success_rate': round(success_count / total * 100, 1) if total > 0 else 0
        }
        
        logger.info(f"""
✅ Greeks Enrichment Complete:
- Total Processed: {stats['total_processed']}
- Greeks Calculated: {stats['greeks_calculated']}
- Failed: {stats['failed']}
- Success Rate: {stats['success_rate']}%
        """)
        
        context['ti'].xcom_push(key='greeks_stats', value=stats)
        return stats
        
    except Exception as e:
        logger.error(f"Greeks enrichment failed: {str(e)}")
        raise
        
        context['ti'].xcom_push(key='greeks_calculated', value=stats['greeks_calculated'])
        
        return stats
        
    except Exception as e:
        logger.error(f"❌ Greeks enrichment failed: {e}")
        raise


def export_bronze_silver_to_minio(**context):
    """Export bronze and silver layers to MinIO as partitioned parquet files."""
    import logging
    from datetime import datetime
    logger = logging.getLogger(__name__)
    
    # Use ACTUAL date (today) for export
    trade_date = datetime.now().strftime('%Y-%m-%d')
    
    logger.info(f"Exporting bronze + silver data to MinIO for {trade_date} (today) - MULTI-TICKER")
    
    total_records_all = 0
    
    # Loop through all configured tickers
    for ticker_config in settings.tickers:
        ticker = ticker_config['ticker']
        logger.info(f"📦 Exporting data for {ticker}...")
        
        # Export bronze layer
        logger.info(f"📦 Exporting BRONZE layer for {ticker}...")
        result_bronze = subprocess.run(
            ['python', '/opt/airflow/scripts/export_parquet_simple.py', 
             '--date', trade_date, 
             '--ticker', ticker,
             '--layer', 'bronze'],
            capture_output=True,
            text=True
        )
        
        logger.info(f"Bronze export stdout ({ticker}):\n{result_bronze.stdout}")
        if result_bronze.stderr:
            logger.warning(f"Bronze export stderr ({ticker}):\n{result_bronze.stderr}")
        
        if result_bronze.returncode != 0:
            logger.error(f"❌ Bronze export failed for {ticker} with return code {result_bronze.returncode}")
            continue  # Continue with other tickers
        
        # Export silver layer
        logger.info(f"📦 Exporting SILVER layer for {ticker}...")
        result_silver = subprocess.run(
            ['python', '/opt/airflow/scripts/export_parquet_simple.py', 
             '--date', trade_date, 
             '--ticker', ticker,
             '--layer', 'silver'],
            capture_output=True,
            text=True
        )
        
        logger.info(f"Silver export stdout ({ticker}):\n{result_silver.stdout}")
        if result_silver.stderr:
            logger.warning(f"Silver export stderr ({ticker}):\n{result_silver.stderr}")
        
        if result_silver.returncode != 0:
            logger.error(f"❌ Silver export failed for {ticker} with return code {result_silver.returncode}")
            continue  # Continue with other tickers
        
        # Parse export stats for this ticker
        if "Total records:" in result_bronze.stdout:
            import re
            match = re.search(r'Total records: (\d+)', result_bronze.stdout)
            if match:
                total_records_all += int(match.group(1))
        
        if "Total records:" in result_silver.stdout:
            import re
            match = re.search(r'Total records: (\d+)', result_silver.stdout)
            if match:
                total_records_all += int(match.group(1))
        
        logger.info(f"✅ Exported {ticker} data to MinIO")
    
    logger.info(f"✅ Exported {total_records_all} total records to MinIO (all tickers, bronze + silver)")
    context['ti'].xcom_push(key='minio_records', value=total_records_all)
    
    return total_records_all


def _deprecated_export_bronze_silver_to_minio_old(**context):
    """OLD VERSION - kept for reference, not used"""
    import logging
    from datetime import datetime
    logger = logging.getLogger(__name__)
    
    # Use ACTUAL date (today) for export
    trade_date = datetime.now().strftime('%Y-%m-%d')
    
    logger.info(f"Exporting bronze + silver data to MinIO for {trade_date} (today)")
    
    # Export bronze layer
    logger.info("📦 Exporting BRONZE layer...")
    result_bronze = subprocess.run(
        ['python', '/opt/airflow/scripts/export_parquet_simple.py', 
         '--date', trade_date, 
         '--ticker', 'AD.AS',
         '--layer', 'bronze'],
        capture_output=True,
        text=True
    )
    
    logger.info(f"Bronze export stdout:\n{result_bronze.stdout}")
    if result_bronze.stderr:
        logger.warning(f"Bronze export stderr:\n{result_bronze.stderr}")
    
    if result_bronze.returncode != 0:
        raise Exception(f"Bronze export failed with return code {result_bronze.returncode}")
    
    # Export silver layer
    logger.info("📦 Exporting SILVER layer...")
    result_silver = subprocess.run(
        ['python', '/opt/airflow/scripts/export_parquet_simple.py', 
         '--date', trade_date, 
         '--ticker', 'AD.AS',
         '--layer', 'silver'],
        capture_output=True,
        text=True
    )
    
    logger.info(f"Silver export stdout:\n{result_silver.stdout}")
    if result_silver.stderr:
        logger.warning(f"Silver export stderr:\n{result_silver.stderr}")
    
    if result_silver.returncode != 0:
        raise Exception(f"Silver export failed with return code {result_silver.returncode}")
    
    # Parse export stats
    total_records = 0
    if "Total records:" in result_bronze.stdout:
        import re
        match = re.search(r'Total records: (\d+)', result_bronze.stdout)
        if match:
            total_records += int(match.group(1))
    
    if "Total records:" in result_silver.stdout:
        import re
        match = re.search(r'Total records: (\d+)', result_silver.stdout)
        if match:
            total_records += int(match.group(1))
    
    logger.info(f"✅ Exported {total_records} total records to MinIO (bronze + silver)")
    context['ti'].xcom_push(key='minio_records', value=total_records)
    
    return total_records


def sync_clickhouse(**context):
    """Sync ClickHouse tables with MinIO parquet files (bronze + silver only)."""
    import logging
    logger = logging.getLogger(__name__)
    
    logger.info("🔄 Syncing ClickHouse with MinIO parquet files (bronze + silver)...")
    
    try:
        # Run setup/refresh script
        result = subprocess.run(
            ['python', '/opt/airflow/scripts/setup_clickhouse.py', 'refresh'],
            capture_output=True,
            text=True
        )
        
        logger.info(f"ClickHouse sync stdout:\n{result.stdout}")
        if result.stderr:
            logger.warning(f"ClickHouse sync stderr:\n{result.stderr}")
        
        if result.returncode != 0:
            # If refresh fails, try full setup (first run)
            logger.info("Refresh failed, trying full setup...")
            result = subprocess.run(
                ['python', '/opt/airflow/scripts/setup_clickhouse.py'],
                capture_output=True,
                text=True
            )
            
            logger.info(f"ClickHouse setup stdout:\n{result.stdout}")
            if result.stderr:
                logger.warning(f"ClickHouse setup stderr:\n{result.stderr}")
            
            if result.returncode != 0:
                raise Exception(f"ClickHouse sync failed with return code {result.returncode}")
        
        logger.info("✅ ClickHouse synced successfully (bronze + silver layers)")
        context['ti'].xcom_push(key='clickhouse_synced', value=True)
        
        return result.returncode
        
    except Exception as e:
        logger.error(f"❌ ClickHouse sync failed: {e}")
        # Don't fail the pipeline if ClickHouse sync fails
        logger.warning("Pipeline will continue despite ClickHouse sync failure")
        return None


def send_pipeline_summary(**context):
    """Send pipeline completion summary with stats."""
    import logging
    logger = logging.getLogger(__name__)
    
    execution_date = context['execution_date']
    
    # Gather stats from XCom
    bronze_quality = context['ti'].xcom_pull(key='bronze_quality', task_ids='check_bronze_quality')
    silver_records = context['ti'].xcom_pull(key='silver_records', task_ids='run_dbt_silver')
    greeks_stats = context['ti'].xcom_pull(key='greeks_calculated', task_ids='enrich_silver_greeks')
    minio_records = context['ti'].xcom_pull(key='minio_records', task_ids='export_bronze_silver_to_minio')
    clickhouse_synced = context['ti'].xcom_pull(key='clickhouse_synced', task_ids='sync_clickhouse')
    
    message = f"""
✅ Bronze & Silver Pipeline Completed Successfully
==================================================
Date: {execution_date.strftime('%Y-%m-%d')}
Ticker: AD.AS

Bronze Layer:
- BD Contracts: {bronze_quality.get('bd_contracts', 'N/A')}
- FD Contracts: {bronze_quality.get('fd_contracts', 'N/A')}
- Underlying: €{bronze_quality.get('underlying_price', 'N/A')}
- Data Quality: {bronze_quality.get('quality_score', 'N/A')}

Silver Layer:
- Enriched Records: {silver_records or 'N/A'}
- Greeks Calculated: {greeks_stats or 'N/A'}

MinIO Export:
- Bronze + Silver Records: {minio_records or 'N/A'}
- Location: s3://options-data/

ClickHouse:
- Sync Status: {'✅ Synced' if clickhouse_synced else '⚠️ Not synced'}
- Query at: http://localhost:8123

Pipeline Status: ✅ SUCCESS
Foundation data ready for gold analytics consumption.
==================================================
    """
    
    logger.info(message)
    
    # TODO: Send to Slack/Email if configured
    # if settings.enable_slack_alerts:
    #     send_slack_message(message)
    
    return message


def send_failure_notification(**context):
    """Send failure notification with error details."""
    import logging
    logger = logging.getLogger(__name__)
    
    execution_date = context['execution_date']
    exception = context.get('exception', 'Unknown error')
    
    message = f"""
❌ Bronze & Silver Pipeline FAILED
===================================
Date: {execution_date.strftime('%Y-%m-%d')}
Error: {exception}

Please check Airflow logs for details.
===================================
    """
    
    logger.error(message)
    
    # TODO: Send to Slack/Email
    return message


# ============================================================================
# TASK DEFINITIONS
# ============================================================================

# Step 0: Initialize tables
ensure_tables_task = PythonOperator(
    task_id='ensure_bronze_tables',
    python_callable=ensure_bronze_tables,
    dag=dag,
)

# Step 1: Scrape Beursduivel
scrape_bd_task = PythonOperator(
    task_id='scrape_beursduivel',
    python_callable=scrape_beursduivel,
    dag=dag,
)

# Step 2: Scrape FD
scrape_fd_task = PythonOperator(
    task_id='scrape_fd',
    python_callable=scrape_fd,
    dag=dag,
)

# Step 3: Quality check
check_bronze_quality_task = PythonOperator(
    task_id='check_bronze_quality',
    python_callable=check_bronze_data_quality,
    dag=dag,
)

# Step 4: DBT Silver transformation
run_dbt_silver_task = PythonOperator(
    task_id='run_dbt_silver',
    python_callable=run_dbt_silver,
    dag=dag,
)

# Step 5: Enrich fact table with Greeks
enrich_greeks_task = PythonOperator(
    task_id='enrich_fact_greeks',
    python_callable=enrich_fact_greeks,
    dag=dag,
)

# Step 6: Export bronze + silver to MinIO
export_to_minio_task = PythonOperator(
    task_id='export_bronze_silver_to_minio',
    python_callable=export_bronze_silver_to_minio,
    dag=dag,
)

# Step 7: Sync ClickHouse with MinIO
sync_clickhouse_task = PythonOperator(
    task_id='sync_clickhouse',
    python_callable=sync_clickhouse,
    trigger_rule=TriggerRule.ALL_SUCCESS,  # Only run if export succeeded
    dag=dag,
)

# Step 8: Send success summary
send_summary_task = PythonOperator(
    task_id='send_pipeline_summary',
    python_callable=send_pipeline_summary,
    trigger_rule=TriggerRule.ALL_SUCCESS,
    dag=dag,
)

# Step 9: Send failure notification
send_failure_task = PythonOperator(
    task_id='send_failure_notification',
    python_callable=send_failure_notification,
    trigger_rule=TriggerRule.ONE_FAILED,
    dag=dag,
)

# ============================================================================
# PIPELINE FLOW
# ============================================================================

# Foundation data pipeline:
# 0. Ensure bronze tables exist (creates if missing)
# 1. Scrape both sources in parallel (BD + FD)
# 2. Quality check bronze data
# 3. Transform to silver star schema (dim_underlying, dim_option_contract, fact_option_timeseries)
# 4. Enrich fact table with Greeks and implied volatility
# 5. Export bronze + silver to MinIO (partitioned parquet)
# 6. Sync ClickHouse with MinIO parquet files
# 7. Send success summary

ensure_tables_task >> [scrape_bd_task, scrape_fd_task] >> check_bronze_quality_task >> run_dbt_silver_task >> enrich_greeks_task >> export_to_minio_task >> sync_clickhouse_task >> send_summary_task

# Failure handling - any task failure triggers notification
[ensure_tables_task, scrape_bd_task, scrape_fd_task, check_bronze_quality_task, run_dbt_silver_task, enrich_greeks_task, export_to_minio_task, sync_clickhouse_task] >> send_failure_task
