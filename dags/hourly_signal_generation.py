"""
Airflow DAG: Hourly Signal Generation

Replicates the 'make daily' pipeline but runs hourly:
1. Backfill latest OHLCV to PostgreSQL (same as backfill_ohlcv_hourly.py)
2. Rebuild events parquet from database (same as build_accum_distrib_events.py)
3. Generate signals from parquet (same as live_tracker_breakouts.py)

This ensures we use the SAME proven pipeline that achieved 80%+ win rates,
just automated to run hourly instead of manually.

Schedule: Every hour during market hours (9 AM - 5 PM)
Database: Same PostgreSQL as backfill scripts (settings.database_url)
Output: Signals saved to data/signals/ with timestamps
"""

from airflow import DAG
from airflow.operators.python import PythonOperator
from airflow.operators.bash import BashOperator
from airflow.utils.dates import days_ago
from datetime import datetime, timedelta
import subprocess
import sys
from pathlib import Path

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

# Ticker lists (same as Makefile)
US_TICKERS = [
    'AAPL', 'MSFT', 'GOOGL', 'GOOG', 'AMZN', 'NVDA', 'META', 'TSLA',
    'SNOW', 'CRWD', 'ZS', 'DDOG', 'NET', 'SHOP', 'COIN', 'PLTR', 'RBLX', 'U',
    'MRVL', 'WDAY', 'TEAM', 'FTNT', 'MNST', 'ZM', 'DOCU', 'OKTA',
    'UBER', 'LYFT', 'DASH', 'ABNB', 'RIVN', 'LCID', 'NIO', 'XPEV', 'LI',
    'SOFI', 'HOOD', 'NU', 'MRNA', 'BNTX', 'ENPH', 'FSLR',
    'BRK.B', 'V', 'JPM', 'WMT', 'LLY', 'UNH', 'XOM', 'MA', 'AVGO', 'JNJ',
    'PG', 'HD', 'COST', 'ABBV', 'ORCL', 'NFLX', 'CVX', 'MRK', 'KO', 'BAC',
    'AMD', 'PEP', 'TMO', 'ADBE', 'CSCO', 'ACN', 'CRM', 'MCD', 'LIN', 'ABT',
    'INTC', 'WFC', 'DHR', 'NKE', 'CMCSA', 'TXN', 'QCOM', 'DIS', 'VZ', 'PM',
    'IBM', 'INTU', 'UNP', 'AMGN', 'CAT', 'GE', 'RTX', 'SPGI', 'LOW', 'HON',
    'NEE', 'UPS', 'AMAT', 'PFE', 'BLK', 'ISRG', 'SYK', 'BA', 'T', 'ELV',
    'AXP', 'DE', 'BKNG', 'ADI', 'GILD', 'MS', 'LMT', 'TJX', 'CI', 'PLD',
    'VRTX', 'SBUX', 'MDLZ', 'MMC', 'GS', 'ADP', 'TMUS', 'BMY', 'C', 'NOW',
    'REGN', 'ZTS', 'SO', 'SCHW', 'MO', 'AMT', 'ETN', 'BDX', 'CB', 'PANW',
    'DUK', 'SLB', 'BSX', 'COP', 'AON', 'MMM', 'PNC', 'MU', 'ITW', 'USB'
]

NL_TICKERS = [
    'ABN.AS', 'AD.AS', 'AGN.AS', 'AKZA.AS', 'ALFEN.AS', 'ALLFG.AS',
    'ASML.AS', 'ASRNL.AS', 'BAMNB.AS', 'BESI.AS', 'HEIA.AS', 'INGA.AS',
    'INPST.AS', 'KPN.AS', 'LIGHT.AS', 'NN.AS', 'OCI.AS', 'PHIA.AS',
    'RAND.AS', 'REN.AS', 'SBMO.AS', 'SHELL.AS', 'TOM2.AS', 'UNA.AS', 'WKL.AS'
]



def init_database():
    """
    Step 0: Initialize database tables.
    Ensures bronze_ohlcv_intraday table exists before backfilling.
    This prevents race conditions when US and NL backfill run in parallel.
    """
    print(f"🔧 Initializing database tables...")
    
    cmd = [
        "python",
        f"{project_root}/scripts/init_db.py"
    ]
    
    result = subprocess.run(cmd, capture_output=True, text=True)
    
    if result.returncode != 0:
        print(f"❌ Database init failed: {result.stderr}")
        raise RuntimeError(f"Database init failed: {result.stderr}")
    
    print(result.stdout)
    print(f"✅ Database initialized!")
    return "Database initialization successful"

def run_backfill_us():
    """
    Step 1a: Backfill US market data to PostgreSQL.
    Uses existing backfill_ohlcv_hourly.py script (same as make backfill-us).
    """
    print(f"🔄 Backfilling US market data ({len(US_TICKERS)} tickers)...")
    print(f"   This will take ~3-4 minutes. Watch for progress below:")
    print("-" * 60)
    
    cmd = [
        "python",
        f"{project_root}/scripts/backfill_ohlcv_hourly.py",
        "--tickers", *US_TICKERS,
        "--days", "5"  # Use --days instead of --period to avoid Yahoo API issues
    ]
    
    # Stream output in real-time
    process = subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True, bufsize=1)
    
    for line in process.stdout:
        print(line.rstrip())
    
    process.wait()
    
    if process.returncode != 0:
        raise RuntimeError(f"US Backfill failed with exit code {process.returncode}")
    
    print("-" * 60)
    print(f"✅ US backfill complete!")
    return "US backfill successful"


def run_backfill_nl():
    """
    Step 1b: Backfill NL market data to PostgreSQL.
    Uses existing backfill_ohlcv_hourly.py script (same as make backfill-nl).
    """
    print(f"🔄 Backfilling NL market data ({len(NL_TICKERS)} tickers)...")
    print(f"   This will take ~1 minute. Watch for progress below:")
    print("-" * 60)
    
    cmd = [
        "python",
        f"{project_root}/scripts/backfill_ohlcv_hourly.py",
        "--tickers", *NL_TICKERS,
        "--days", "5"  # Use --days instead of --period to avoid Yahoo API issues
    ]
    
    # Stream output in real-time
    process = subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True, bufsize=1)
    
    for line in process.stdout:
        print(line.rstrip())
    
    process.wait()
    
    if process.returncode != 0:
        raise RuntimeError(f"NL Backfill failed with exit code {process.returncode}")
    
    print("-" * 60)
    print(f"✅ NL backfill complete!")
    return "NL backfill successful"


def rebuild_events_parquet():
    """
    Step 2: Rebuild events parquet from PostgreSQL database.
    Uses existing build_accum_distrib_events.py (same as make rebuild-events).
    
    This scans the database for consolidation patterns and creates the
    accum_distrib_events.parquet file that signals are generated from.
    """
    print(f"� Rebuilding events parquet from database...")
    
    output_path = f"{project_root}/data/ml_datasets/accum_distrib_events.parquet"
    
    cmd = [
        "python",
        f"{project_root}/scripts/build_accum_distrib_events.py",
        "--output", output_path
    ]
    
    result = subprocess.run(cmd, capture_output=True, text=True)
    
    if result.returncode != 0:
        print(f"❌ Rebuild parquet failed: {result.stderr}")
        raise RuntimeError(f"Rebuild failed: {result.stderr}")
    
    print(result.stdout)
    print(f"✅ Parquet file rebuilt!")
    print(f"📁 Saved to: {output_path}")
    return "Parquet rebuild successful"


def generate_signals():
    """
    Step 3: Generate trading signals from parquet file.
    Uses existing live_tracker_breakouts.py (same as make signals).
    
    This applies the calibrated ML model to detect fresh breakout signals
    from the latest consolidation events.
    """
    print(f"🎯 Generating signals from parquet...")
    
    # Check if parquet file exists
    parquet_path = Path(f"{project_root}/data/ml_datasets/accum_distrib_events.parquet")
    if not parquet_path.exists():
        print(f"⚠️  Parquet file not found: {parquet_path}")
        print("   Skipping signal generation (this is normal for new datasets)")
        return "Skipped - no events file"
    
    timestamp = datetime.now()
    date_str = timestamp.strftime('%Y-%m-%d')
    asof_str = timestamp.strftime('%Y-%m-%d %H:00:00')  # Round to hour
    
    output_csv = f"{project_root}/data/signals/breakout_signals_{date_str}_{timestamp.strftime('%H%M')}.csv"
    summary_json = f"{project_root}/data/signals/signal_summary_{date_str}_{timestamp.strftime('%H%M')}.json"
    
    # Ensure signals directory exists
    signals_dir = Path(f"{project_root}/data/signals")
    signals_dir.mkdir(parents=True, exist_ok=True)
    
    cmd = [
        "python",
        f"{project_root}/scripts/live_tracker_breakouts.py",
        "--config", f"{project_root}/config/live_universe.json",
        "--events", f"{project_root}/data/ml_datasets/accum_distrib_events.parquet",
        "--asof", asof_str,
        "--lookback_hours", "24",
        "--output", output_csv,
        "--summary_out", summary_json
    ]
    
    result = subprocess.run(cmd, capture_output=True, text=True)
    
    if result.returncode != 0:
        print(f"❌ Signal generation failed: {result.stderr}")
        raise RuntimeError(f"Signal generation failed: {result.stderr}")
    
    print(result.stdout)
    print(f"✅ Signal generation complete!")
    print(f"📁 Signals: {output_csv}")
    print(f"📁 Summary: {summary_json}")
    
    # Display signals if any were generated
    if Path(output_csv).exists():
        display_cmd = [
            "python",
            f"{project_root}/scripts/display_signals.py",
            output_csv
        ]
        subprocess.run(display_cmd)
    
    return f"Signals saved to {output_csv}"


def send_slack_notification(**context):
    """
    Send Slack notification with pipeline summary.
    
    Reads the generated signal summary and sends a formatted
    message to Slack with key metrics and signals.
    """
    print(f"📱 Sending Slack notification...")
    
    try:
        from src.utils.alerts import AlertManager
        import json
        
        # Find latest signal summary
        signals_dir = Path(f"{project_root}/data/signals")
        summary_files = sorted(signals_dir.glob("signal_summary_*.json"), reverse=True)
        
        if not summary_files:
            print("⚠️  No signal summary found")
            return "No summary to send"
        
        latest_summary = summary_files[0]
        
        # Read summary
        with open(latest_summary, 'r') as f:
            summary = json.load(f)
        
        # Build message
        timestamp = datetime.now().strftime('%Y-%m-%d %H:%M')
        num_signals = summary.get('num_signals', 0)
        tickers = summary.get('tickers', [])
        
        if num_signals > 0:
            message = f"🚀 *{num_signals} New Trading Signals Generated*\n\n"
            message += f"*Tickers:* {', '.join(tickers)}\n"
            message += f"*Time:* {timestamp}\n"
            message += f"\n📊 Check `{latest_summary.name}` for details"
            level = "info"
        else:
            message = f"✅ *Pipeline Complete - No New Signals*\n\n"
            message += f"*Time:* {timestamp}\n"
            message += "All markets scanned, no breakout opportunities detected."
            level = "info"
        
        # Send alert
        alert_manager = AlertManager()
        success = alert_manager.send_alert(
            message=message,
            level=level,
            context={
                "Pipeline": "Hourly Signal Generation",
                "Markets": "US (143 tickers) + NL (25 tickers)",
                "Status": "✅ Success"
            }
        )
        
        if success:
            print(f"✅ Slack notification sent!")
        else:
            print(f"⚠️  Slack notification failed (may be disabled in config)")
        
        return "Notification sent"
        
    except Exception as e:
        print(f"⚠️  Slack notification error: {e}")
        # Don't fail the DAG if notification fails
        return f"Notification failed: {e}"


def send_failure_notification(context):
    """
    Send Slack notification when DAG fails.
    
    Args:
        context: Airflow context dictionary
    """
    try:
        from src.utils.alerts import AlertManager
        
        task_instance = context.get('task_instance')
        dag_id = context.get('dag').dag_id
        task_id = task_instance.task_id
        execution_date = context.get('execution_date')
        exception = context.get('exception')
        
        message = f"🚨 *DAG Failure Alert*\n\n"
        message += f"*DAG:* {dag_id}\n"
        message += f"*Task:* {task_id}\n"
        message += f"*Execution Date:* {execution_date}\n"
        message += f"*Error:* {exception}"
        
        alert_manager = AlertManager()
        alert_manager.send_alert(
            message=message,
            level="error",
            context={
                "Pipeline": "Hourly Signal Generation",
                "Status": "❌ Failed"
            }
        )
        
    except Exception as e:
        print(f"⚠️  Failed to send failure notification: {e}")


# DAG definition
default_args = {
    'owner': 'options_trader',
    'depends_on_past': False,
    'email_on_failure': False,
    'email_on_retry': False,
    'retries': 2,
    'retry_delay': timedelta(minutes=5),
    'on_failure_callback': send_failure_notification,
}

# Run every hour during market hours
# NL: 9 AM - 5:30 PM CET (08:00-16:30 UTC)
# US: 9:30 AM - 4 PM EST (14:30-21:00 UTC)
# Run hourly 08:00-21:00 UTC to cover both markets
dag = DAG(
    'hourly_signal_generation',
    default_args=default_args,
    description='Automated version of "make daily" pipeline - runs hourly',
    schedule_interval='0 8-21 * * 1-5',  # Every hour 8-21 UTC, Mon-Fri
    start_date=days_ago(1),
    catchup=False,
    tags=['trading', 'signals', 'production', 'automated'],
)

# Task 0: Initialize database
init_task = PythonOperator(
    task_id='init_database',
    python_callable=init_database,
    dag=dag,
)

# Task 1: Backfill US market data to PostgreSQL
backfill_us_task = PythonOperator(
    task_id='backfill_us',
    python_callable=run_backfill_us,
    dag=dag,
)

# Task 2: Backfill NL market data to PostgreSQL  
backfill_nl_task = PythonOperator(
    task_id='backfill_nl',
    python_callable=run_backfill_nl,
    dag=dag,
)

# Task 3: Rebuild events parquet from PostgreSQL
rebuild_task = PythonOperator(
    task_id='rebuild_events',
    python_callable=rebuild_events_parquet,
    dag=dag,
)

# Task 4: Generate signals from parquet
signals_task = PythonOperator(
    task_id='generate_signals',
    python_callable=generate_signals,
    dag=dag,
)

# Task 5: Send Slack notification
slack_task = PythonOperator(
    task_id='send_slack_notification',
    python_callable=send_slack_notification,
    provide_context=True,
    dag=dag,
)

# Task dependencies
# Init DB first, then backfill both markets in parallel, then rebuild parquet, then generate signals, then notify
init_task >> [backfill_us_task, backfill_nl_task] >> rebuild_task >> signals_task >> slack_task
