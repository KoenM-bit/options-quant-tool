"""
Gold Daily Analytics DAG
========================
Daily analytics layer consuming silver BD data with Greeks.

Pipeline Flow:
1. Check silver data availability (verify today's silver_bd_options_enriched exists)
2. Run dbt gold daily models (volatility, gamma, put/call metrics, key levels)
3. Export gold layer to MinIO (partitioned parquet)
4. Sync ClickHouse with gold data

Consumes: silver_bd_options_enriched (BD data with Greeks from foundation pipeline)
Produces: Gold analytics tables for daily trading insights

Schedule: Mon-Fri 17:00 UTC (18:00 CET) - 30 minutes after foundation pipeline
Runs after options_bronze_silver_pipeline completes.

Note: This DAG does NOT scrape new data - it transforms existing silver data.
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
    'owner': 'data-analytics',
    'depends_on_past': False,
    'email_on_failure': True,
    'email_on_retry': False,
    'retries': 2,
    'retry_delay': timedelta(minutes=5),
    'execution_timeout': timedelta(minutes=30),
}

# DAG definition
dag = DAG(
    'gold_daily_analytics',
    default_args=default_args,
    description='Daily gold analytics from silver BD data with Greeks',
    schedule_interval='0 17 * * 1-5',  # 17:00 UTC = 18:00 CET, Mon-Fri only (no Saturday analytics)
    start_date=datetime(2024, 1, 1),
    catchup=False,
    tags=['ahold', 'options', 'gold', 'analytics', 'daily'],
    max_active_runs=1,
)


def check_silver_data_available(**context):
    """
    Check if silver data is available for today.
    Ensures foundation pipeline has completed before running analytics.
    """
    import logging
    from datetime import datetime
    from src.utils.db import get_db_session
    from sqlalchemy import text
    
    logger = logging.getLogger(__name__)
    
    # Check for today's silver data
    trade_date = datetime.now().date()
    
    logger.info(f"Checking if silver data exists for {trade_date}")
    
    with get_db_session() as session:
        # Check silver_bd_options_enriched
        silver_query = text("""
            SELECT 
                COUNT(*) as total_records,
                COUNT(CASE WHEN delta IS NOT NULL THEN 1 END) as with_greeks,
                MIN(trade_date) as min_date,
                MAX(trade_date) as max_date
            FROM silver_bd_options_enriched
            WHERE trade_date = :trade_date
        """)
        result = session.execute(silver_query, {'trade_date': trade_date}).fetchone()
        
        if result[0] == 0:
            raise Exception(f"No silver data found for {trade_date}. Foundation pipeline may not have run yet.")
        
        if result[1] == 0:
            raise Exception(f"Silver data exists but no Greeks calculated for {trade_date}. Greeks enrichment may have failed.")
        
        stats = {
            'trade_date': str(trade_date),
            'total_records': result[0],
            'records_with_greeks': result[1],
            'greeks_coverage': round((result[1] / result[0] * 100), 1) if result[0] > 0 else 0
        }
        
        logger.info(f"✅ Silver data available: {stats}")
        
        # Push to XCom
        context['ti'].xcom_push(key='silver_stats', value=stats)
        
        return stats


def run_dbt_gold_daily(**context):
    """Run dbt gold models tagged as 'daily' (BD-based analytics)."""
    import logging
    logger = logging.getLogger(__name__)
    
    logger.info("Running dbt Gold Daily transformations (BD analytics)")
    
    # Run gold models tagged as 'daily'
    # These models use only silver_bd_options_enriched (no FD data required)
    result = subprocess.run(
        ['dbt', 'run', '--select', 'tag:gold,tag:daily', '--profiles-dir', '/opt/airflow/dbt'],
        cwd='/opt/airflow/dbt/ahold_options',
        capture_output=True,
        text=True
    )
    
    logger.info(f"dbt stdout:\n{result.stdout}")
    if result.stderr:
        logger.warning(f"dbt stderr:\n{result.stderr}")
    
    if result.returncode != 0:
        raise Exception(f"dbt Gold Daily failed with return code {result.returncode}")
    
    # Parse results
    models_run = 0
    if "Completed successfully" in result.stdout:
        import re
        # Count successful models
        matches = re.findall(r'OK created', result.stdout)
        models_run = len(matches)
        logger.info(f"✅ dbt Gold Daily completed: {models_run} models")
        context['ti'].xcom_push(key='models_run', value=models_run)
    
    logger.info("✅ dbt Gold Daily transformation completed successfully")
    return result.returncode


def export_gold_to_minio(**context):
    """Export gold layer to MinIO as partitioned parquet files."""
    import logging
    from datetime import datetime
    logger = logging.getLogger(__name__)
    
    # Use ACTUAL date (today) for export
    trade_date = datetime.now().strftime('%Y-%m-%d')
    
    logger.info(f"Exporting GOLD layer to MinIO for {trade_date}")
    
    result = subprocess.run(
        ['python', '/opt/airflow/scripts/export_parquet_simple.py', 
         '--date', trade_date, 
         '--ticker', 'AD.AS',
         '--layer', 'gold'],
        capture_output=True,
        text=True
    )
    
    logger.info(f"Gold export stdout:\n{result.stdout}")
    if result.stderr:
        logger.warning(f"Gold export stderr:\n{result.stderr}")
    
    if result.returncode != 0:
        raise Exception(f"Gold export failed with return code {result.returncode}")
    
    # Parse export stats
    if "Total records:" in result.stdout:
        import re
        match = re.search(r'Total records: (\d+)', result.stdout)
        if match:
            total_records = int(match.group(1))
            logger.info(f"✅ Exported {total_records} gold records to MinIO")
            context['ti'].xcom_push(key='gold_records', value=total_records)
    
    logger.info("✅ Gold export completed successfully")
    return result.returncode


def sync_clickhouse_gold(**context):
    """Sync ClickHouse gold tables with MinIO parquet files."""
    import logging
    logger = logging.getLogger(__name__)
    
    logger.info("🔄 Syncing ClickHouse gold layer with MinIO...")
    
    try:
        # Run refresh to update gold tables only
        result = subprocess.run(
            ['python', '/opt/airflow/scripts/setup_clickhouse.py', 'refresh'],
            capture_output=True,
            text=True
        )
        
        logger.info(f"ClickHouse sync stdout:\n{result.stdout}")
        if result.stderr:
            logger.warning(f"ClickHouse sync stderr:\n{result.stderr}")
        
        if result.returncode != 0:
            # Non-critical - log warning but don't fail
            logger.warning(f"⚠️  ClickHouse sync had issues (return code {result.returncode})")
            logger.warning("Analytics pipeline will continue")
            return None
        
        logger.info("✅ ClickHouse gold layer synced successfully")
        context['ti'].xcom_push(key='clickhouse_synced', value=True)
        
        return result.returncode
        
    except Exception as e:
        logger.error(f"❌ ClickHouse sync failed: {e}")
        # Don't fail the pipeline if ClickHouse sync fails
        logger.warning("Analytics pipeline will continue despite ClickHouse sync failure")
        return None


def send_analytics_summary(**context):
    """Send analytics completion summary with stats."""
    import logging
    logger = logging.getLogger(__name__)
    
    execution_date = context['execution_date']
    
    # Gather stats from XCom
    silver_stats = context['ti'].xcom_pull(key='silver_stats', task_ids='check_silver_available')
    models_run = context['ti'].xcom_pull(key='models_run', task_ids='run_dbt_gold_daily')
    gold_records = context['ti'].xcom_pull(key='gold_records', task_ids='export_gold_to_minio')
    clickhouse_synced = context['ti'].xcom_pull(key='clickhouse_synced', task_ids='sync_clickhouse_gold')
    
    message = f"""
✅ Gold Daily Analytics Completed Successfully
==============================================
Date: {execution_date.strftime('%Y-%m-%d')}
Ticker: AD.AS

Silver Input:
- Total Records: {silver_stats.get('total_records', 'N/A')}
- Records with Greeks: {silver_stats.get('records_with_greeks', 'N/A')}
- Greeks Coverage: {silver_stats.get('greeks_coverage', 'N/A')}%

Gold Analytics:
- Models Run: {models_run or 'N/A'}
- Gold Records: {gold_records or 'N/A'}

ClickHouse:
- Sync Status: {'✅ Synced' if clickhouse_synced else '⚠️ Not synced'}
- Query at: http://localhost:8123

Pipeline Status: ✅ SUCCESS
Daily analytics ready for consumption.
==============================================
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
❌ Gold Daily Analytics FAILED
===============================
Date: {execution_date.strftime('%Y-%m-%d')}
Error: {exception}

Possible causes:
- Foundation pipeline not completed yet
- No silver data for today
- dbt model errors
- Export script issues

Please check Airflow logs for details.
===============================
    """
    
    logger.error(message)
    
    # TODO: Send to Slack/Email
    return message


# ============================================================================
# TASK DEFINITIONS
# ============================================================================

# Step 1: Check silver data availability
check_silver_task = PythonOperator(
    task_id='check_silver_available',
    python_callable=check_silver_data_available,
    dag=dag,
)

# Step 2: Run dbt gold daily models
run_dbt_gold_daily_task = PythonOperator(
    task_id='run_dbt_gold_daily',
    python_callable=run_dbt_gold_daily,
    dag=dag,
)

# Step 3: Export gold to MinIO
export_gold_task = PythonOperator(
    task_id='export_gold_to_minio',
    python_callable=export_gold_to_minio,
    dag=dag,
)

# Step 4: Sync ClickHouse gold layer
sync_clickhouse_task = PythonOperator(
    task_id='sync_clickhouse_gold',
    python_callable=sync_clickhouse_gold,
    trigger_rule=TriggerRule.ALL_SUCCESS,
    dag=dag,
)

# Step 5: Send success summary
send_summary_task = PythonOperator(
    task_id='send_analytics_summary',
    python_callable=send_analytics_summary,
    trigger_rule=TriggerRule.ALL_SUCCESS,
    dag=dag,
)

# Step 6: Send failure notification
send_failure_task = PythonOperator(
    task_id='send_failure_notification',
    python_callable=send_failure_notification,
    trigger_rule=TriggerRule.ONE_FAILED,
    dag=dag,
)

# ============================================================================
# PIPELINE FLOW
# ============================================================================

# Daily analytics pipeline:
# 1. Check silver data exists (foundation pipeline completed)
# 2. Run dbt gold daily models (volatility, gamma, metrics)
# 3. Export gold to MinIO
# 4. Sync ClickHouse with gold data
# 5. Send success summary

check_silver_task >> run_dbt_gold_daily_task >> export_gold_task >> sync_clickhouse_task >> send_summary_task

# Failure handling - any task failure triggers notification
[check_silver_task, run_dbt_gold_daily_task, export_gold_task, sync_clickhouse_task] >> send_failure_task
