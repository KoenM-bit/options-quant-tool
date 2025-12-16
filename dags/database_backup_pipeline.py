"""
Database Backup Pipeline

Creates daily compressed backups of the PostgreSQL database and uploads to MinIO.
Implements automatic cleanup of backups older than 7 days.

Schedule: Daily at 23:00 CET (after all data pipelines complete)
Retention: 7 days (configurable)
Storage: MinIO bucket under backups/postgres/

Backup naming: ahold_options_YYYY-MM-DD_HHMMSS.sql.gz
"""

from datetime import datetime, timedelta
from airflow import DAG
from airflow.operators.python import PythonOperator
import logging

logger = logging.getLogger(__name__)

# DAG default arguments
default_args = {
    'owner': 'data-engineering',
    'depends_on_past': False,
    'email_on_failure': True,
    'email_on_retry': False,
    'retries': 2,
    'retry_delay': timedelta(minutes=5),
    'start_date': datetime(2025, 12, 16),
}

# Create DAG
dag = DAG(
    'database_backup_pipeline',
    default_args=default_args,
    description='Daily database backup to MinIO with 7-day retention',
    schedule_interval='0 23 * * *',  # 23:00 CET (22:00 UTC), daily
    catchup=False,
    tags=['backup', 'database', 'minio', 'maintenance'],
)


def create_and_upload_backup(**context):
    """
    Create a compressed database backup and upload to MinIO.
    Automatically cleans up backups older than retention period.
    """
    import sys
    from pathlib import Path
    project_root = Path(__file__).parent.parent
    sys.path.insert(0, str(project_root))
    
    from scripts.backup_database_to_minio import DatabaseBackup
    
    # Get execution date
    execution_date = context['execution_date']
    backup_date = execution_date.date()
    
    logger.info(f"📦 Starting database backup for {backup_date}")
    
    # Create backup with 7-day retention
    backup = DatabaseBackup(retention_days=7)
    
    try:
        result = backup.run_backup(backup_date)
        
        logger.info(f"✅ Backup successful: {result['backup_filename']}")
        logger.info(f"📍 Location: {result['s3_path']}")
        
        # Store result in XCom for monitoring
        context['ti'].xcom_push(key='backup_result', value=result)
        
        return result
        
    except Exception as e:
        logger.error(f"❌ Backup failed: {e}")
        raise


def verify_backup(**context):
    """
    Verify that the backup was created successfully.
    Checks that the backup file exists in MinIO and has reasonable size.
    """
    import sys
    from pathlib import Path
    project_root = Path(__file__).parent.parent
    sys.path.insert(0, str(project_root))
    
    from src.utils.minio_client import get_minio_client
    from minio import Minio
    
    # Get backup result from previous task
    backup_result = context['ti'].xcom_pull(
        key='backup_result',
        task_ids='create_backup'
    )
    
    if not backup_result:
        raise ValueError("No backup result found from previous task")
    
    s3_path = backup_result['s3_path']
    
    logger.info(f"🔍 Verifying backup: {s3_path}")
    
    # Get MinIO client
    minio_client = get_minio_client()
    
    # Create Minio client for stat operations
    minio = Minio(
        minio_client.endpoint,
        access_key=minio_client.access_key,
        secret_key=minio_client.secret_key,
        secure=False
    )
    
    try:
        # Check if object exists and get stats
        stat = minio.stat_object(minio_client.bucket, s3_path)
        
        file_size_mb = stat.size / (1024 * 1024)
        
        # Verify size is reasonable (should be at least 1 MB for our database)
        if stat.size < 1024 * 1024:  # Less than 1 MB
            raise ValueError(f"Backup file too small: {file_size_mb:.2f} MB")
        
        logger.info(f"✅ Backup verified: {file_size_mb:.2f} MB")
        logger.info(f"📅 Last modified: {stat.last_modified}")
        
        return {
            'status': 'verified',
            'size_mb': file_size_mb,
            'last_modified': str(stat.last_modified)
        }
        
    except Exception as e:
        logger.error(f"❌ Backup verification failed: {e}")
        raise


def send_backup_notification(**context):
    """
    Send notification about backup status.
    (Optional: Can be extended to send email/slack notifications)
    """
    # Get results from previous tasks
    backup_result = context['ti'].xcom_pull(
        key='backup_result',
        task_ids='create_backup'
    )
    
    verify_result = context['ti'].xcom_pull(
        task_ids='verify_backup'
    )
    
    execution_date = context['execution_date']
    
    logger.info("=" * 80)
    logger.info("📊 DAILY BACKUP SUMMARY")
    logger.info("=" * 80)
    logger.info(f"Date: {execution_date.date()}")
    logger.info(f"Backup: {backup_result['backup_filename']}")
    logger.info(f"Size: {verify_result['size_mb']:.2f} MB")
    logger.info(f"Location: s3://{backup_result['s3_path']}")
    logger.info(f"Status: ✅ SUCCESS")
    logger.info("=" * 80)
    
    # TODO: Add email/Slack notification here if needed
    
    return {
        'date': str(execution_date.date()),
        'status': 'success',
        'backup': backup_result['backup_filename'],
        'size_mb': verify_result['size_mb']
    }


# Define tasks
create_backup_task = PythonOperator(
    task_id='create_backup',
    python_callable=create_and_upload_backup,
    dag=dag,
)

verify_backup_task = PythonOperator(
    task_id='verify_backup',
    python_callable=verify_backup,
    dag=dag,
)

notify_task = PythonOperator(
    task_id='send_notification',
    python_callable=send_backup_notification,
    dag=dag,
)

# Set task dependencies
create_backup_task >> verify_backup_task >> notify_task
