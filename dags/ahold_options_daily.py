"""
Daily Ahold Options Scraping DAG
Scrapes FD.nl options data daily after market close.
"""

from datetime import datetime, timedelta
from airflow import DAG
from airflow.operators.python import PythonOperator
from airflow.providers.postgres.operators.postgres import PostgresOperator
from airflow.providers.postgres.hooks.postgres import PostgresHook
from airflow.utils.trigger_rule import TriggerRule

import sys
import os

# Add project root to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from src.scrapers.fd_overview_scraper import scrape_fd_overview
from src.scrapers.fd_options_scraper import scrape_fd_options
from src.analytics.enrich_silver_greeks import enrich_silver_with_greeks
from src.utils.db import get_db_session
from src.models.bronze import BronzeFDOverview, BronzeFDOptions
from src.config import settings

# Default args for all tasks
default_args = {
    'owner': 'data-engineering',
    'depends_on_past': False,
    'email_on_failure': True,
    'email_on_retry': False,
    'retries': 3,
    'retry_delay': timedelta(minutes=5),
    'execution_timeout': timedelta(minutes=30),
}

# DAG definition
dag = DAG(
    'ahold_options_daily',
    default_args=default_args,
    description='Daily scrape of Ahold options data from FD.nl',
    schedule_interval='0 22 * * 1-5',  # 22:00 CET, Monday-Friday (after market close)
    start_date=datetime(2024, 1, 1),
    catchup=False,
    tags=['ahold', 'options', 'scraping', 'bronze'],
    max_active_runs=1,
)


def scrape_and_load_overview(**context):
    """
    Scrape overview data and load to Bronze layer.
    """
    import logging
    logger = logging.getLogger(__name__)
    
    execution_date = context['execution_date']
    logger.info(f"Scraping overview for execution date: {execution_date}")
    
    try:
        # Scrape data
        overview_data = scrape_fd_overview()
        logger.info(f"Scraped overview data: {overview_data}")
        
        # Load to database
        with get_db_session() as session:
            bronze_record = BronzeFDOverview(**overview_data)
            session.add(bronze_record)
            session.commit()
            
            record_id = bronze_record.id
            logger.info(f"✅ Loaded overview to Bronze layer: ID={record_id}")
            
            # Push record ID to XCom for downstream tasks
            context['ti'].xcom_push(key='overview_id', value=record_id)
            
            return {
                'status': 'success',
                'record_id': record_id,
                'ticker': overview_data.get('ticker'),
                'koers': overview_data.get('koers'),
            }
    
    except Exception as e:
        logger.error(f"Failed to scrape and load overview: {e}")
        raise


def scrape_and_load_options(**context):
    """
    Scrape options chain data and load to Bronze layer.
    """
    import logging
    logger = logging.getLogger(__name__)
    
    execution_date = context['execution_date']
    logger.info(f"Scraping options chain for execution date: {execution_date}")
    
    try:
        # Scrape options data
        options_data = scrape_fd_options(
            ticker=settings.ahold_ticker,
            symbol_code=settings.ahold_symbol_code
        )
        logger.info(f"Scraped {len(options_data)} option contracts")
        
        if not options_data:
            logger.warning("No options data scraped")
            return {'status': 'no_data', 'count': 0}
        
        # Load to database
        with get_db_session() as session:
            records_added = 0
            for option in options_data:
                bronze_record = BronzeFDOptions(**option)
                session.add(bronze_record)
                records_added += 1
            
            session.commit()
            logger.info(f"✅ Loaded {records_added} option contracts to Bronze layer")
            
            # Push count to XCom
            context['ti'].xcom_push(key='options_count', value=records_added)
            
            return {
                'status': 'success',
                'count': records_added,
            }
    
    except Exception as e:
        logger.error(f"Failed to scrape and load options: {e}")
        raise


def calculate_greeks(**context):
    """
    Calculate implied volatility and Greeks for options in SILVER layer.
    This runs AFTER dbt transforms Bronze → Silver (deduplication).
    Greeks are stored in Silver, not Bronze, for proper data architecture.
    """
    import logging
    logger = logging.getLogger(__name__)
    
    execution_date = context['execution_date']
    logger.info(f"Calculating Greeks for execution date: {execution_date}")
    
    try:
        # Enrich SILVER with Black-Scholes calculations (not Bronze!)
        stats = enrich_silver_with_greeks(ticker=settings.ahold_ticker)
        
        logger.info(f"✅ Greeks calculation complete: {stats}")
        
        # Push stats to XCom
        context['ti'].xcom_push(key='greeks_stats', value=stats)
        
        return stats
    
    except Exception as e:
        logger.error(f"Failed to calculate Greeks: {e}")
        raise


def validate_overview_data(**context):
    """
    Validate scraped overview data for data quality.
    """
    import logging
    logger = logging.getLogger(__name__)
    
    overview_id = context['ti'].xcom_pull(key='overview_id', task_ids='scrape_overview')
    
    if not overview_id:
        raise ValueError("No overview_id found in XCom")
    
    with get_db_session() as session:
        record = session.query(BronzeFDOverview).filter_by(id=overview_id).first()
        
        if not record:
            raise ValueError(f"Record {overview_id} not found in database")
        
        # Data quality checks
        issues = []
        
        if record.koers is None or record.koers <= 0:
            issues.append("Invalid koers (price)")
        
        if record.totaal_volume is not None and record.totaal_volume < 0:
            issues.append("Negative total volume")
        
        if record.totaal_oi is not None and record.totaal_oi < 0:
            issues.append("Negative open interest")
        
        if issues:
            logger.warning(f"Data quality issues found: {issues}")
            return {'status': 'warning', 'issues': issues}
        
        logger.info(f"✅ Data quality validation passed for record {overview_id}")
        return {'status': 'success'}


def send_success_notification(**context):
    """
    Send success notification (Slack/Email).
    """
    import logging
    logger = logging.getLogger(__name__)
    
    execution_date = context['execution_date']
    overview_data = context['ti'].xcom_pull(task_ids='scrape_overview')
    
    message = (
        f"✅ Ahold Options Daily Scrape Complete\n"
        f"Date: {execution_date.strftime('%Y-%m-%d')}\n"
        f"Ticker: {overview_data.get('ticker')}\n"
        f"Price: €{overview_data.get('koers')}\n"
    )
    
    logger.info(message)
    
    # TODO: Implement Slack/Email notifications based on config
    # if settings.enable_slack_alerts:
    #     send_slack_alert(message)


def send_failure_notification(**context):
    """
    Send failure notification.
    """
    import logging
    logger = logging.getLogger(__name__)
    
    execution_date = context['execution_date']
    exception = context.get('exception')
    
    message = (
        f"❌ Ahold Options Daily Scrape FAILED\n"
        f"Date: {execution_date.strftime('%Y-%m-%d')}\n"
        f"Error: {exception}\n"
    )
    
    logger.error(message)


# Task definitions
scrape_overview_task = PythonOperator(
    task_id='scrape_overview',
    python_callable=scrape_and_load_overview,
    dag=dag,
)

scrape_options_task = PythonOperator(
    task_id='scrape_options',
    python_callable=scrape_and_load_options,
    dag=dag,
)

validate_data_task = PythonOperator(
    task_id='validate_data',
    python_callable=validate_overview_data,
    dag=dag,
)

# Greeks calculation task - runs AFTER dbt Silver (not before!)
calculate_greeks_task = PythonOperator(
    task_id='calculate_greeks',
    python_callable=calculate_greeks,
    dag=dag,
)

# Clean old bronze data (retention policy)
# NOTE: Commented out - this should run separately (weekly/monthly), not on every daily run
# cleanup_bronze_task = PostgresOperator(
#     task_id='cleanup_old_bronze_data',
#     postgres_conn_id='postgres_default',
#     sql=f"""
#         DELETE FROM bronze_fd_overview
#         WHERE scraped_at < NOW() - INTERVAL '{settings.bronze_retention_days} days';
#     """,
#     dag=dag,
# )

success_notification_task = PythonOperator(
    task_id='send_success_notification',
    python_callable=send_success_notification,
    trigger_rule=TriggerRule.ALL_SUCCESS,
    dag=dag,
)

failure_notification_task = PythonOperator(
    task_id='send_failure_notification',
    python_callable=send_failure_notification,
    trigger_rule=TriggerRule.ONE_FAILED,
    dag=dag,
)

# DBT transformations - Silver layer only (Bronze → Silver deduplication)
def run_dbt_silver(**context):
    """Run DBT silver models with proper logging."""
    import subprocess
    import logging
    logger = logging.getLogger(__name__)
    
    logger.info("Starting DBT Silver transformation...")
    
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
        raise Exception(f"DBT failed with return code {result.returncode}")
    
    logger.info("✅ DBT Silver transformation completed successfully")
    return result.returncode

run_dbt_silver_task = PythonOperator(
    task_id='run_dbt_silver',
    python_callable=run_dbt_silver,
    dag=dag,
)

# DBT Gold layer - run AFTER Greeks enrichment
def run_dbt_gold(**context):
    """Run DBT gold models with proper logging."""
    import subprocess
    import logging
    logger = logging.getLogger(__name__)
    
    logger.info("Starting DBT Gold transformation...")
    
    result = subprocess.run(
        ['dbt', 'run', '--models', 'tag:gold', '--profiles-dir', '/opt/airflow/dbt'],
        cwd='/opt/airflow/dbt/ahold_options',
        capture_output=True,
        text=True
    )
    
    logger.info(f"DBT stdout:\n{result.stdout}")
    if result.stderr:
        logger.warning(f"DBT stderr:\n{result.stderr}")
    
    if result.returncode != 0:
        raise Exception(f"DBT failed with return code {result.returncode}")
    
    logger.info("✅ DBT Gold transformation completed successfully")
    return result.returncode

run_dbt_gold_task = PythonOperator(
    task_id='run_dbt_gold',
    python_callable=run_dbt_gold,
    dag=dag,
)

# Export to Parquet and upload to MinIO
export_parquet_task = PythonOperator(
    task_id='export_to_parquet',
    python_callable=lambda: os.system(
        'python /opt/airflow/scripts/export_to_parquet.py'
    ),
    dag=dag,
)

# Sync from MinIO to ClickHouse for analytics
sync_clickhouse_task = PythonOperator(
    task_id='sync_to_clickhouse',
    python_callable=lambda: os.system(
        'python /opt/airflow/scripts/sync_to_clickhouse.py'
    ),
    dag=dag,
)

# Task dependencies
# Enhanced pipeline: Bronze → Silver → Greeks → Gold → MinIO → ClickHouse → Power BI
# 1. Scrape raw data into Bronze
# 2. DBT Silver: Deduplicate and standardize
# 3. Calculate Greeks: Enrich Silver with validated Black-Scholes Greeks
# 4. DBT Gold: Build analytics (GEX, max pain, vol surface, etc.)
# 5. Export to Parquet and upload to MinIO S3-compatible storage (data lake)
# 6. Sync from MinIO to ClickHouse for fast analytics queries
# 7. Power BI connects to ClickHouse via DirectQuery

[scrape_overview_task, scrape_options_task] >> validate_data_task
validate_data_task >> run_dbt_silver_task >> calculate_greeks_task
calculate_greeks_task >> run_dbt_gold_task >> export_parquet_task
export_parquet_task >> sync_clickhouse_task >> success_notification_task

# Failure handling
[scrape_overview_task, scrape_options_task, run_dbt_silver_task, calculate_greeks_task, run_dbt_gold_task, export_parquet_task, sync_clickhouse_task] >> failure_notification_task
