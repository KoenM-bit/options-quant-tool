"""
Gold Weekly Analytics DAG
=========================
Weekly analytics layer consuming FD data for open interest analysis.

Pipeline Flow:
1. Check silver FD data availability (verify week's silver_fd_options exists)
2. Run dbt gold weekly models (max pain, OI flow, GEX positioning trends)
3. Export gold weekly layer to MinIO (partitioned parquet)
4. Sync ClickHouse with weekly gold data

Consumes: silver_fd_options (FD data with open_interest from foundation pipeline)
Produces: Gold weekly analytics tables for OI-based insights

Schedule: Saturday 17:00 UTC (18:00 CET) - 30 minutes after foundation pipeline
Runs after options_bronze_silver_pipeline completes on Saturday.

Note: This DAG does NOT scrape new data - it transforms existing silver FD data.
      Requires silver_fd_options table to be created (currently only BD silver exists).
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
    'gold_weekly_analytics',
    default_args=default_args,
    description='Weekly gold analytics from silver FD data (open interest analysis)',
    schedule_interval='0 17 * * 6',  # 17:00 UTC = 18:00 CET, Saturday only
    start_date=datetime(2024, 1, 1),
    catchup=False,
    tags=['ahold', 'options', 'gold', 'analytics', 'weekly', 'open-interest'],
    max_active_runs=1,
)


def check_silver_fd_data_available(**context):
    """
    Check if silver FD data is available for the week.
    Ensures foundation pipeline has collected FD data throughout the week.
    """
    import logging
    from datetime import datetime, timedelta
    from src.utils.db import get_db_session
    from sqlalchemy import text
    
    logger = logging.getLogger(__name__)
    
    # Check for this week's FD data (last 7 days)
    today = datetime.now().date()
    week_start = today - timedelta(days=7)
    
    logger.info(f"Checking if silver FD data exists for week {week_start} to {today}")
    
    with get_db_session() as session:
        # Check if silver_fd_options table exists
        table_check = text("""
            SELECT EXISTS (
                SELECT FROM information_schema.tables 
                WHERE table_schema = 'public' 
                AND table_name = 'silver_fd_options'
            )
        """)
        table_exists = session.execute(table_check).scalar()
        
        if not table_exists:
            logger.warning("⚠️  silver_fd_options table does not exist yet")
            logger.warning("Weekly analytics requires FD silver layer to be created first")
            raise Exception(
                "silver_fd_options table not found. "
                "Create dbt silver model for FD data before running weekly analytics."
            )
        
        # Check silver_fd_options for this week
        silver_query = text("""
            SELECT 
                COUNT(*) as total_records,
                COUNT(DISTINCT trade_date) as trading_days,
                MIN(trade_date) as earliest_date,
                MAX(trade_date) as latest_date,
                SUM(CASE WHEN open_interest IS NOT NULL THEN 1 ELSE 0 END) as with_oi
            FROM silver_fd_options
            WHERE trade_date >= :week_start AND trade_date <= :today
        """)
        result = session.execute(silver_query, {
            'week_start': week_start, 
            'today': today
        }).fetchone()
        
        if result[0] == 0:
            raise Exception(
                f"No silver FD data found for week {week_start} to {today}. "
                "Foundation pipeline may not have collected FD data."
            )
        
        if result[1] < 4:
            logger.warning(
                f"⚠️  Only {result[1]} trading days of FD data found (expected ~5). "
                "Weekly analytics may be incomplete."
            )
        
        stats = {
            'week_start': str(week_start),
            'week_end': str(today),
            'total_records': result[0],
            'trading_days': result[1],
            'earliest_date': str(result[2]) if result[2] else None,
            'latest_date': str(result[3]) if result[3] else None,
            'records_with_oi': result[4],
            'oi_coverage': round((result[4] / result[0] * 100), 1) if result[0] > 0 else 0
        }
        
        logger.info(f"✅ Silver FD data available: {stats}")
        
        # Push to XCom
        context['ti'].xcom_push(key='silver_fd_stats', value=stats)
        
        return stats


def run_dbt_gold_weekly(**context):
    """Run dbt gold models tagged as 'weekly' (FD OI-based analytics)."""
    import logging
    logger = logging.getLogger(__name__)
    
    logger.info("Running dbt Gold Weekly transformations (FD OI analytics)")
    
    # Run gold models tagged as 'weekly'
    # These models use silver_fd_options (open_interest data)
    result = subprocess.run(
        ['dbt', 'run', '--select', 'tag:gold,tag:weekly', '--profiles-dir', '/opt/airflow/dbt'],
        cwd='/opt/airflow/dbt/ahold_options',
        capture_output=True,
        text=True
    )
    
    logger.info(f"dbt stdout:\n{result.stdout}")
    if result.stderr:
        logger.warning(f"dbt stderr:\n{result.stderr}")
    
    if result.returncode != 0:
        raise Exception(f"dbt Gold Weekly failed with return code {result.returncode}")
    
    # Parse results
    models_run = 0
    if "Completed successfully" in result.stdout:
        import re
        # Count successful models
        matches = re.findall(r'OK created', result.stdout)
        models_run = len(matches)
        logger.info(f"✅ dbt Gold Weekly completed: {models_run} models")
        context['ti'].xcom_push(key='models_run', value=models_run)
    
    logger.info("✅ dbt Gold Weekly transformation completed successfully")
    return result.returncode


def export_gold_weekly_to_minio(**context):
    """Export gold weekly layer to MinIO as partitioned parquet files."""
    import logging
    from datetime import datetime
    logger = logging.getLogger(__name__)
    
    # Use ACTUAL date (today/Saturday) for export
    trade_date = datetime.now().strftime('%Y-%m-%d')
    
    logger.info(f"Exporting GOLD WEEKLY layer to MinIO for week ending {trade_date}")
    
    result = subprocess.run(
        ['python', '/opt/airflow/scripts/export_parquet_simple.py', 
         '--date', trade_date, 
         '--ticker', 'AD.AS',
         '--layer', 'gold'],
        capture_output=True,
        text=True
    )
    
    logger.info(f"Gold weekly export stdout:\n{result.stdout}")
    if result.stderr:
        logger.warning(f"Gold weekly export stderr:\n{result.stderr}")
    
    if result.returncode != 0:
        raise Exception(f"Gold weekly export failed with return code {result.returncode}")
    
    # Parse export stats
    if "Total records:" in result.stdout:
        import re
        match = re.search(r'Total records: (\d+)', result.stdout)
        if match:
            total_records = int(match.group(1))
            logger.info(f"✅ Exported {total_records} gold weekly records to MinIO")
            context['ti'].xcom_push(key='gold_weekly_records', value=total_records)
    
    logger.info("✅ Gold weekly export completed successfully")
    return result.returncode


def sync_clickhouse_gold_weekly(**context):
    """Sync ClickHouse gold weekly tables with MinIO parquet files."""
    import logging
    logger = logging.getLogger(__name__)
    
    logger.info("🔄 Syncing ClickHouse gold weekly layer with MinIO...")
    
    try:
        # Run refresh to update gold tables
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
            logger.warning("Weekly analytics pipeline will continue")
            return None
        
        logger.info("✅ ClickHouse gold weekly layer synced successfully")
        context['ti'].xcom_push(key='clickhouse_synced', value=True)
        
        return result.returncode
        
    except Exception as e:
        logger.error(f"❌ ClickHouse sync failed: {e}")
        # Don't fail the pipeline if ClickHouse sync fails
        logger.warning("Weekly analytics pipeline will continue despite ClickHouse sync failure")
        return None


def send_weekly_analytics_summary(**context):
    """Send weekly analytics completion summary with stats."""
    import logging
    logger = logging.getLogger(__name__)
    
    execution_date = context['execution_date']
    
    # Gather stats from XCom
    silver_fd_stats = context['ti'].xcom_pull(key='silver_fd_stats', task_ids='check_silver_fd_available')
    models_run = context['ti'].xcom_pull(key='models_run', task_ids='run_dbt_gold_weekly')
    gold_weekly_records = context['ti'].xcom_pull(key='gold_weekly_records', task_ids='export_gold_weekly_to_minio')
    clickhouse_synced = context['ti'].xcom_pull(key='clickhouse_synced', task_ids='sync_clickhouse_gold_weekly')
    
    message = f"""
✅ Gold Weekly Analytics Completed Successfully
===============================================
Week Ending: {execution_date.strftime('%Y-%m-%d')}
Ticker: AD.AS

Silver FD Input:
- Total Records: {silver_fd_stats.get('total_records', 'N/A')}
- Trading Days: {silver_fd_stats.get('trading_days', 'N/A')}
- Records with OI: {silver_fd_stats.get('records_with_oi', 'N/A')}
- OI Coverage: {silver_fd_stats.get('oi_coverage', 'N/A')}%
- Date Range: {silver_fd_stats.get('earliest_date', 'N/A')} to {silver_fd_stats.get('latest_date', 'N/A')}

Gold Weekly Analytics:
- Models Run: {models_run or 'N/A'}
- Gold Records: {gold_weekly_records or 'N/A'}

ClickHouse:
- Sync Status: {'✅ Synced' if clickhouse_synced else '⚠️ Not synced'}
- Query at: http://localhost:8123

Pipeline Status: ✅ SUCCESS
Weekly OI analytics ready for consumption.
===============================================
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
❌ Gold Weekly Analytics FAILED
================================
Week Ending: {execution_date.strftime('%Y-%m-%d')}
Error: {exception}

Possible causes:
- silver_fd_options table not created yet (requires dbt model)
- Foundation pipeline not collecting FD data
- No FD data for this week
- dbt model errors
- Export script issues

Please check Airflow logs for details.
================================
    """
    
    logger.error(message)
    
    # TODO: Send to Slack/Email
    return message


# ============================================================================
# TASK DEFINITIONS
# ============================================================================

# Step 1: Check silver FD data availability
check_silver_fd_task = PythonOperator(
    task_id='check_silver_fd_available',
    python_callable=check_silver_fd_data_available,
    dag=dag,
)

# Step 2: Run dbt gold weekly models
run_dbt_gold_weekly_task = PythonOperator(
    task_id='run_dbt_gold_weekly',
    python_callable=run_dbt_gold_weekly,
    dag=dag,
)

# Step 3: Export gold weekly to MinIO
export_gold_weekly_task = PythonOperator(
    task_id='export_gold_weekly_to_minio',
    python_callable=export_gold_weekly_to_minio,
    dag=dag,
)

# Step 4: Sync ClickHouse gold weekly layer
sync_clickhouse_weekly_task = PythonOperator(
    task_id='sync_clickhouse_gold_weekly',
    python_callable=sync_clickhouse_gold_weekly,
    trigger_rule=TriggerRule.ALL_SUCCESS,
    dag=dag,
)

# Step 5: Send success summary
send_summary_task = PythonOperator(
    task_id='send_weekly_analytics_summary',
    python_callable=send_weekly_analytics_summary,
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

# Weekly analytics pipeline:
# 1. Check silver FD data exists (foundation pipeline collected FD all week)
# 2. Run dbt gold weekly models (max pain, OI flow, GEX trends)
# 3. Export gold weekly to MinIO
# 4. Sync ClickHouse with weekly gold data
# 5. Send success summary

check_silver_fd_task >> run_dbt_gold_weekly_task >> export_gold_weekly_task >> sync_clickhouse_weekly_task >> send_summary_task

# Failure handling - any task failure triggers notification
[check_silver_fd_task, run_dbt_gold_weekly_task, export_gold_weekly_task, sync_clickhouse_weekly_task] >> send_failure_task
