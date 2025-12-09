"""
Data Quality Monitoring DAG
Monitors data freshness, completeness, and anomalies.
"""

from datetime import datetime, timedelta
from airflow import DAG
from airflow.operators.python import PythonOperator, BranchPythonOperator
from airflow.providers.postgres.hooks.postgres import PostgresHook
from airflow.utils.trigger_rule import TriggerRule

import sys
import os

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from src.config import settings

default_args = {
    'owner': 'data-engineering',
    'depends_on_past': False,
    'email_on_failure': True,
    'retries': 1,
    'retry_delay': timedelta(minutes=2),
}

dag = DAG(
    'ahold_data_quality_checks',
    default_args=default_args,
    description='Data quality monitoring for Ahold options data',
    schedule_interval='0 */4 * * *',  # Every 4 hours
    start_date=datetime(2024, 1, 1),
    catchup=False,
    tags=['ahold', 'data-quality', 'monitoring'],
)


def check_data_freshness(**context):
    """
    Check if data is fresh (scraped recently).
    """
    import logging
    logger = logging.getLogger(__name__)
    
    postgres_hook = PostgresHook(postgres_conn_id='postgres_default')
    
    # Check latest scrape time
    query = """
        SELECT 
            MAX(scraped_at) as last_scrape,
            EXTRACT(EPOCH FROM (NOW() - MAX(scraped_at)))/3600 as hours_since_scrape
        FROM bronze_fd_overview
    """
    
    result = postgres_hook.get_first(query)
    
    if not result or not result[0]:
        logger.error("❌ No data found in bronze_fd_overview")
        raise ValueError("No data found")
    
    last_scrape, hours_since = result
    logger.info(f"Last scrape: {last_scrape} ({hours_since:.1f} hours ago)")
    
    # Alert if data is older than 24 hours
    if hours_since > 24:
        logger.warning(f"⚠️ Data is stale: {hours_since:.1f} hours old")
        return {'status': 'warning', 'hours_since': hours_since}
    
    logger.info("✅ Data freshness check passed")
    return {'status': 'success', 'hours_since': hours_since}


def check_data_completeness(**context):
    """
    Check data completeness (no missing fields).
    """
    import logging
    logger = logging.getLogger(__name__)
    
    postgres_hook = PostgresHook(postgres_conn_id='postgres_default')
    
    # Check for records with missing critical fields
    query = """
        SELECT COUNT(*) as incomplete_records
        FROM bronze_fd_overview
        WHERE 
            scraped_at >= NOW() - INTERVAL '7 days'
            AND (koers IS NULL OR totaal_volume IS NULL OR totaal_oi IS NULL)
    """
    
    result = postgres_hook.get_first(query)
    incomplete_count = result[0] if result else 0
    
    if incomplete_count > 0:
        logger.warning(f"⚠️ Found {incomplete_count} incomplete records in last 7 days")
        return {'status': 'warning', 'incomplete_records': incomplete_count}
    
    logger.info("✅ Data completeness check passed")
    return {'status': 'success'}


def check_for_anomalies(**context):
    """
    Check for data anomalies (outliers, unusual patterns).
    """
    import logging
    logger = logging.getLogger(__name__)
    
    postgres_hook = PostgresHook(postgres_conn_id='postgres_default')
    
    # Check for unusual price movements
    query = """
        SELECT 
            koers,
            delta_pct,
            scraped_at
        FROM bronze_fd_overview
        WHERE scraped_at >= NOW() - INTERVAL '1 day'
        ORDER BY scraped_at DESC
        LIMIT 1
    """
    
    result = postgres_hook.get_first(query)
    
    if result:
        koers, delta_pct, scraped_at = result
        
        # Alert on large price movements (>5%)
        if delta_pct and abs(delta_pct) > 5:
            logger.warning(f"⚠️ Large price movement detected: {delta_pct:.2f}%")
            return {'status': 'warning', 'delta_pct': delta_pct}
    
    logger.info("✅ Anomaly check passed")
    return {'status': 'success'}


def aggregate_quality_results(**context):
    """
    Aggregate all quality check results.
    """
    import logging
    logger = logging.getLogger(__name__)
    
    ti = context['ti']
    
    freshness = ti.xcom_pull(task_ids='check_freshness')
    completeness = ti.xcom_pull(task_ids='check_completeness')
    anomalies = ti.xcom_pull(task_ids='check_anomalies')
    
    all_checks = {
        'freshness': freshness,
        'completeness': completeness,
        'anomalies': anomalies,
    }
    
    warnings = [k for k, v in all_checks.items() if v and v.get('status') == 'warning']
    
    if warnings:
        logger.warning(f"⚠️ Data quality warnings in: {', '.join(warnings)}")
        return {'status': 'warning', 'warnings': warnings}
    
    logger.info("✅ All data quality checks passed")
    return {'status': 'success'}


# Task definitions
check_freshness_task = PythonOperator(
    task_id='check_freshness',
    python_callable=check_data_freshness,
    dag=dag,
)

check_completeness_task = PythonOperator(
    task_id='check_completeness',
    python_callable=check_data_completeness,
    dag=dag,
)

check_anomalies_task = PythonOperator(
    task_id='check_anomalies',
    python_callable=check_for_anomalies,
    dag=dag,
)

aggregate_results_task = PythonOperator(
    task_id='aggregate_results',
    python_callable=aggregate_quality_results,
    trigger_rule=TriggerRule.ALL_DONE,
    dag=dag,
)

# Task dependencies
[check_freshness_task, check_completeness_task, check_anomalies_task] >> aggregate_results_task
