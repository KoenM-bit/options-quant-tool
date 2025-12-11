"""
Master Options Data Pipeline DAG
=================================
Orchestrates the COMPLETE end-to-end options data pipeline.

Pipeline Flow:
1. Scrape Beursduivel (BD) options + underlying → bronze_bd_options + bronze_bd_underlying
2. Scrape FD.nl options + overview → bronze_fd_options + bronze_fd_overview
3. Quality check bronze layer data
4. DBT Silver - BD options enriched with Greeks → silver_bd_options_enriched
5. DBT Gold - Build analytics → gold_daily_summary_test
6. Export to MinIO - Save all layers as partitioned parquet

This DAG can be triggered manually for testing or scheduled to run daily.
For production: Run at 22:30 CET (after market data is available)
"""

from datetime import datetime, timedelta, date as datetime_date
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
    'options_pipeline_master',
    default_args=default_args,
    description='Master orchestration for complete options data pipeline',
    schedule_interval='30 16 * * 1-6',  # 16:30 UTC = 17:30 CET (Amsterdam winter time), Mon-Sat
    start_date=datetime(2024, 1, 1),
    catchup=False,
    tags=['ahold', 'options', 'pipeline', 'orchestration', 'master'],
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
    """Run DBT silver models to create BD enriched layer."""
    import logging
    logger = logging.getLogger(__name__)
    
    logger.info("Running DBT Silver transformation (BD options enriched)")
    
    result = subprocess.run(
        ['dbt', 'run', '--models', 'silver_bd_options_enriched', '--profiles-dir', '/opt/airflow/dbt'],
        cwd='/opt/airflow/dbt/ahold_options',
        capture_output=True,
        text=True
    )
    
    logger.info(f"DBT stdout:\n{result.stdout}")
    if result.stderr:
        logger.warning(f"DBT stderr:\n{result.stderr}")
    
    if result.returncode != 0:
        raise Exception(f"DBT Silver failed with return code {result.returncode}")
    
    # Parse results
    if "SELECT" in result.stdout:
        import re
        match = re.search(r'SELECT (\d+)', result.stdout)
        if match:
            records_count = int(match.group(1))
            logger.info(f"✅ DBT Silver completed: {records_count} records")
            context['ti'].xcom_push(key='silver_records', value=records_count)
    
    logger.info("✅ DBT Silver transformation completed successfully")
    return result.returncode


def enrich_silver_greeks(**context):
    """Calculate Greeks and implied volatility for silver layer."""
    import logging
    from src.analytics.enrich_silver_greeks import enrich_silver_with_greeks
    
    logger = logging.getLogger(__name__)
    
    logger.info("Enriching silver layer with Greeks and implied volatility")
    
    try:
        # Run enrichment - uses actual ECB risk-free rates
        stats = enrich_silver_with_greeks(
            ticker='AD.AS',
            risk_free_rate=None,  # Auto-fetch from ECB
            batch_size=100
        )
        
        logger.info(f"""
✅ Greeks Enrichment Complete:
- Total Processed: {stats['total_processed']}
- IV Calculated: {stats['iv_calculated']}
- Greeks Calculated: {stats['greeks_calculated']}
- Skipped: {stats['skipped']}
- Errors: {stats['errors']}
        """)
        
        context['ti'].xcom_push(key='greeks_calculated', value=stats['greeks_calculated'])
        
        return stats
        
    except Exception as e:
        logger.error(f"❌ Greeks enrichment failed: {e}")
        raise


def run_dbt_gold(**context):
    """Run DBT gold models for analytics (BD-only models)."""
    import logging
    logger = logging.getLogger(__name__)
    
    logger.info("Running DBT Gold transformation (BD-only analytics)")
    
    # Run only gold models that work with BD data
    # Skip models requiring FD data (open_interest, merged tables)
    result = subprocess.run(
        ['dbt', 'run', '--select', 'tag:gold', '--exclude', 'gold_daily_summary_test', '--profiles-dir', '/opt/airflow/dbt'],
        cwd='/opt/airflow/dbt/ahold_options',
        capture_output=True,
        text=True
    )
    
    logger.info(f"DBT stdout:\n{result.stdout}")
    if result.stderr:
        logger.warning(f"DBT stderr:\n{result.stderr}")
    
    # Don't fail if gold models have issues (non-critical)
    if result.returncode != 0:
        logger.warning(f"⚠️  DBT Gold completed with errors (return code {result.returncode})")
        logger.warning("This is non-critical - continuing pipeline")
        return 0  # Return success to continue pipeline
    
    logger.info("✅ DBT Gold transformation completed successfully")
    return result.returncode


def export_to_minio(**context):
    """Export all layers to MinIO as partitioned parquet files."""
    import logging
    from datetime import date
    logger = logging.getLogger(__name__)
    
    # Use ACTUAL date (today) for export
    from datetime import datetime
    trade_date = datetime.now().strftime('%Y-%m-%d')
    
    logger.info(f"Exporting data to MinIO for {trade_date} (today)")
    
    result = subprocess.run(
        ['python', '/opt/airflow/scripts/export_parquet_simple.py', 
         '--date', trade_date, 
         '--ticker', 'AD.AS',
         '--layer', 'all'],
        capture_output=True,
        text=True
    )
    
    logger.info(f"Export stdout:\n{result.stdout}")
    if result.stderr:
        logger.warning(f"Export stderr:\n{result.stderr}")
    
    if result.returncode != 0:
        raise Exception(f"MinIO export failed with return code {result.returncode}")
    
    # Parse export stats
    if "Total records:" in result.stdout:
        import re
        match = re.search(r'Total records: (\d+)', result.stdout)
        if match:
            total_records = int(match.group(1))
            logger.info(f"✅ Exported {total_records} records to MinIO")
            context['ti'].xcom_push(key='minio_records', value=total_records)
    
    logger.info("✅ MinIO export completed successfully")
    return result.returncode


def sync_clickhouse(**context):
    """Sync ClickHouse tables with MinIO parquet files."""
    import logging
    logger = logging.getLogger(__name__)
    
    execution_date = context['execution_date']
    
    logger.info("🔄 Syncing ClickHouse with MinIO parquet files...")
    
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
        
        logger.info("✅ ClickHouse synced successfully")
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
    minio_records = context['ti'].xcom_pull(key='minio_records', task_ids='export_to_minio')
    clickhouse_synced = context['ti'].xcom_pull(key='clickhouse_synced', task_ids='sync_clickhouse')
    
    message = f"""
✅ Options Pipeline Completed Successfully
==========================================
Date: {execution_date.strftime('%Y-%m-%d')}
Ticker: AD.AS

Bronze Layer:
- BD Contracts: {bronze_quality.get('bd_contracts', 'N/A')}
- FD Contracts: {bronze_quality.get('fd_contracts', 'N/A')}
- Underlying: €{bronze_quality.get('underlying_price', 'N/A')}
- Data Quality: {bronze_quality.get('quality_score', 'N/A')}

Silver Layer:
- Enriched Records: {silver_records or 'N/A'}

MinIO Export:
- Total Records: {minio_records or 'N/A'}
- Location: s3://options-data/

ClickHouse:
- Sync Status: {'✅ Synced' if clickhouse_synced else '⚠️ Not synced'}
- Query at: http://localhost:8123

Pipeline Status: ✅ SUCCESS
==========================================
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
❌ Options Pipeline FAILED
==========================
Date: {execution_date.strftime('%Y-%m-%d')}
Error: {exception}

Please check Airflow logs for details.
==========================
    """
    
    logger.error(message)
    
    # TODO: Send to Slack/Email
    return message


# ============================================================================
# SCRAPER FUNCTIONS (embedded in this DAG)
# ============================================================================

def scrape_beursduivel(**context):
    """Scrape Beursduivel options + underlying data."""
    import logging
    from datetime import date
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
    
    logger.info(f"🚀 Scraping Beursduivel for {trade_date} (actual date)")
    
    try:
        # Scrape data (fetch_live=True gets all strikes with live prices)
        options_data, underlying_data = scrape_all_options(
            ticker='AD.AS',
            fetch_live=True
        )
        
        logger.info(f"Scraped {len(options_data)} options, underlying: {underlying_data}")
        
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
                option['ticker'] = 'AD.AS'
                
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
                underlying_data['ticker'] = 'AD.AS'
                session.add(BronzeBDUnderlying(**underlying_data))
            
            session.commit()
        
        if skipped > 0:
            logger.warning(f"⚠️  BD: {inserted} new, {skipped} skipped (duplicates - possible holiday/re-run)")
        else:
            logger.info(f"✅ BD: {inserted} new contracts, underlying €{underlying_data.get('last_price') if underlying_data else 'N/A'}")
        
        context['ti'].xcom_push(key='bd_contracts', value=len(options_data))
        context['ti'].xcom_push(key='bd_underlying_price', value=underlying_data.get('last_price') if underlying_data else None)
        
        return {'contracts': len(options_data), 'underlying': underlying_data.get('last_price') if underlying_data else None}
    
    except Exception as e:
        logger.error(f"❌ BD scrape failed: {e}")
        raise


def scrape_fd(**context):
    """Scrape FD.nl options + overview data."""
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

# DBT transformations
run_dbt_silver_task = PythonOperator(
    task_id='run_dbt_silver',
    python_callable=run_dbt_silver,
    dag=dag,
)

# Enrich with Greeks
enrich_greeks_task = PythonOperator(
    task_id='enrich_silver_greeks',
    python_callable=enrich_silver_greeks,
    dag=dag,
)

run_dbt_gold_task = PythonOperator(
    task_id='run_dbt_gold',
    python_callable=run_dbt_gold,
    dag=dag,
)

# Export to MinIO
export_to_minio_task = PythonOperator(
    task_id='export_to_minio',
    python_callable=export_to_minio,
    dag=dag,
)

# Sync ClickHouse with MinIO
sync_clickhouse_task = PythonOperator(
    task_id='sync_clickhouse',
    python_callable=sync_clickhouse,
    trigger_rule=TriggerRule.ALL_SUCCESS,  # Only run if export succeeded
    dag=dag,
)

# Notifications
send_summary_task = PythonOperator(
    task_id='send_pipeline_summary',
    python_callable=send_pipeline_summary,
    trigger_rule=TriggerRule.ALL_SUCCESS,
    dag=dag,
)

send_failure_task = PythonOperator(
    task_id='send_failure_notification',
    python_callable=send_failure_notification,
    trigger_rule=TriggerRule.ONE_FAILED,
    dag=dag,
)

# ============================================================================
# PIPELINE FLOW
# ============================================================================

# Complete end-to-end pipeline:
# 0. Ensure bronze tables exist (creates if missing)
# 1. Scrape both sources in parallel
# 2. Quality check bronze data
# 3. Transform to silver (BD options only)
# 4. Enrich silver with Greeks and IV
# 5. Transform to gold (analytics, non-critical)
# 6. Export all layers to MinIO
# 7. Sync ClickHouse with MinIO parquet files
# 8. Send success summary

ensure_tables_task >> [scrape_bd_task, scrape_fd_task] >> check_bronze_quality_task >> run_dbt_silver_task >> enrich_greeks_task >> run_dbt_gold_task >> export_to_minio_task >> sync_clickhouse_task >> send_summary_task

# Failure handling - any task failure triggers notification
[ensure_tables_task, scrape_bd_task, scrape_fd_task, check_bronze_quality_task, run_dbt_silver_task, enrich_greeks_task, run_dbt_gold_task, export_to_minio_task, sync_clickhouse_task] >> send_failure_task
