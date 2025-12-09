"""
DBT Transformation DAG
Runs DBT models to transform data through Bronze → Silver → Gold layers.
"""

from datetime import datetime, timedelta
from airflow import DAG
from airflow.operators.bash import BashOperator
from airflow.operators.python import PythonOperator
from airflow.utils.trigger_rule import TriggerRule

import sys
import os

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from src.config import settings

default_args = {
    'owner': 'data-engineering',
    'depends_on_past': False,
    'email_on_failure': True,
    'email_on_retry': False,
    'retries': 2,
    'retry_delay': timedelta(minutes=5),
}

dag = DAG(
    'ahold_dbt_transform',
    default_args=default_args,
    description='DBT transformations for Ahold options data',
    schedule_interval=None,  # Triggered by ahold_options_daily DAG
    start_date=datetime(2024, 1, 1),
    catchup=False,
    tags=['ahold', 'dbt', 'transformation', 'silver', 'gold'],
)


def log_dbt_start(**context):
    """Log start of DBT run."""
    import logging
    logger = logging.getLogger(__name__)
    logger.info("Starting DBT transformation pipeline...")


def log_dbt_success(**context):
    """Log successful DBT run."""
    import logging
    logger = logging.getLogger(__name__)
    logger.info("✅ DBT transformation pipeline completed successfully")


# Task definitions
dbt_debug = BashOperator(
    task_id='dbt_debug',
    bash_command=f'cd {settings.dbt_project_dir} && dbt debug',
    dag=dag,
)

dbt_deps = BashOperator(
    task_id='dbt_deps',
    bash_command=f'cd {settings.dbt_project_dir} && dbt deps',
    dag=dag,
)

# Bronze → Silver transformation
dbt_run_silver = BashOperator(
    task_id='dbt_run_silver',
    bash_command=f'cd {settings.dbt_project_dir} && dbt run --models tag:silver',
    dag=dag,
)

# Silver → Gold transformation
dbt_run_gold = BashOperator(
    task_id='dbt_run_gold',
    bash_command=f'cd {settings.dbt_project_dir} && dbt run --models tag:gold',
    dag=dag,
)

# Run DBT tests
dbt_test = BashOperator(
    task_id='dbt_test',
    bash_command=f'cd {settings.dbt_project_dir} && dbt test',
    dag=dag,
)

# Generate DBT documentation
dbt_docs_generate = BashOperator(
    task_id='dbt_docs_generate',
    bash_command=f'cd {settings.dbt_project_dir} && dbt docs generate',
    trigger_rule=TriggerRule.ALL_SUCCESS,
    dag=dag,
)

log_start_task = PythonOperator(
    task_id='log_start',
    python_callable=log_dbt_start,
    dag=dag,
)

log_success_task = PythonOperator(
    task_id='log_success',
    python_callable=log_dbt_success,
    trigger_rule=TriggerRule.ALL_SUCCESS,
    dag=dag,
)

# Task dependencies
log_start_task >> dbt_debug >> dbt_deps
dbt_deps >> dbt_run_silver >> dbt_run_gold
dbt_run_gold >> dbt_test >> dbt_docs_generate >> log_success_task
