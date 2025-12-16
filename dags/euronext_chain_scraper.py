"""
Airflow DAG for scraping Euronext options chain data.

This DAG scrapes the entire options chain (all strikes, all expirations) from
the settlement prices page on live.euronext.com. This is much more efficient
than scraping individual option pages - it gets all data in one page load.

Features:
- Scrapes 400+ options in ~10 seconds (vs 20-30 minutes for individual pages)
- Gets open interest, volume, prices, and settlement data
- Automatically identifies expiration dates from table headers
- Filters out summary rows
- Saves to PostgreSQL bronze layer

Schedule: Tuesday-Saturday at 18:15 CET (scrapes previous trading day data)
"""

from airflow import DAG
from airflow.operators.python import PythonOperator
from airflow.utils.dates import days_ago
from datetime import datetime, timedelta
import logging
import sys
import os

# Add project root to path
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
if project_root not in sys.path:
    sys.path.insert(0, project_root)

logger = logging.getLogger(__name__)

# Default arguments
default_args = {
    'owner': 'airflow',
    'depends_on_past': False,
    'email_on_failure': False,
    'email_on_retry': False,
    'retries': 2,
    'retry_delay': timedelta(minutes=5),
    'execution_timeout': timedelta(minutes=30),  # Should complete in ~10 seconds
}


def scrape_chain_task(**context):
    """
    Scrape the full options chain for the previous trading day.
    
    This function:
    1. Calculates the previous trading day
    2. Opens the settlement prices page
    3. Selects the target date
    4. Clicks "Toepassen" to load data
    5. Scrapes all tables with options data
    6. Extracts expiration dates from headers
    7. Returns DataFrame with all options
    """
    from scripts.scrape_options_chain import scrape_options_chain
    
    # Calculate previous trading day
    today = datetime.now()
    
    # If today is Monday, go back 3 days (to Friday)
    # If today is Sunday, go back 2 days (to Friday)
    # Otherwise, go back 1 day
    if today.weekday() == 0:  # Monday
        days_back = 3
    elif today.weekday() == 6:  # Sunday
        days_back = 2
    else:
        days_back = 1
    
    previous_day = today - timedelta(days=days_back)
    
    # Format as "15 december 2025" (Dutch format)
    month_names_nl = {
        1: 'januari', 2: 'februari', 3: 'maart', 4: 'april',
        5: 'mei', 6: 'juni', 7: 'juli', 8: 'augustus',
        9: 'september', 10: 'oktober', 11: 'november', 12: 'december'
    }
    
    target_date_str = f"{previous_day.day} {month_names_nl[previous_day.month]} {previous_day.year}"
    
    logger.info(f"Scraping options chain for: {target_date_str}")
    
    # Scrape the chain
    df = scrape_options_chain(target_date_str=target_date_str, headless=True)
    
    if df.empty:
        logger.error("No data scraped!")
        raise ValueError("Scraping returned empty DataFrame")
    
    logger.info(f"✅ Scraped {len(df)} options")
    logger.info(f"Expirations: {sorted(df['expiration_date'].dropna().unique())}")
    logger.info(f"Strikes: {df['strike'].nunique()} unique")
    
    # Push data to XCom for next task
    context['task_instance'].xcom_push(key='options_data', value=df.to_dict('records'))
    context['task_instance'].xcom_push(key='row_count', value=len(df))
    
    return len(df)


def save_to_database_task(**context):
    """
    Save scraped options chain data to PostgreSQL bronze layer.
    
    This function:
    1. Retrieves data from previous task via XCom
    2. Connects to database
    3. Creates table if it doesn't exist
    4. Checks if record exists for this trade_date + option combination
    5. Updates existing records or inserts new ones (no duplicates)
    """
    import pandas as pd
    from src.utils.db import SessionLocal
    from src.models.base import Base
    from src.models.bronze_euronext import BronzeEuronextOptions
    from sqlalchemy import text
    
    # Get data from previous task
    task_instance = context['task_instance']
    options_data = task_instance.xcom_pull(task_ids='scrape_chain', key='options_data')
    
    if not options_data:
        logger.error("No data received from scrape task")
        raise ValueError("No options data to save")
    
    df = pd.DataFrame(options_data)
    logger.info(f"Saving {len(df)} options to database...")
    
    # Connect to database
    session = SessionLocal()
    
    try:
        # Create table if it doesn't exist
        Base.metadata.create_all(bind=session.get_bind(), tables=[BronzeEuronextOptions.__table__], checkfirst=True)
        logger.info("✅ Table check/creation complete")
        
        saved_count = 0
        updated_count = 0
        
        for _, row in df.iterrows():
            # Convert values
            strike = float(row['strike']) if pd.notna(row['strike']) else None
            volume = int(row['volume']) if pd.notna(row['volume']) and row['volume'] != '' and row['volume'] != 'N/A' else None
            
            # Parse open interest (could be number or '-')
            oi_value = row.get('open_interest', '-')
            if oi_value and oi_value != '-':
                try:
                    open_interest = int(oi_value)
                except (ValueError, TypeError):
                    open_interest = None
            else:
                open_interest = None
            
            # Parse prices
            def safe_float(val):
                if pd.isna(val) or val == '' or val == 'N/A' or val == '-':
                    return None
                try:
                    return float(val)
                except (ValueError, TypeError):
                    return None
            
            opening_price = safe_float(row.get('open'))
            high_price = safe_float(row.get('high'))
            low_price = safe_float(row.get('low'))
            last_price = safe_float(row.get('last'))
            settlement_price = safe_float(row.get('settle'))
            
            # Parse actual expiration date
            actual_expiration_date = row.get('expiration_date')
            if pd.notna(actual_expiration_date):
                try:
                    actual_expiration_date = datetime.strptime(str(actual_expiration_date), '%Y-%m-%d').date()
                except:
                    actual_expiration_date = None
            else:
                actual_expiration_date = None
            
            # Parse trade date from data_date string
            data_date_str = row.get('data_date', '')
            if data_date_str:
                try:
                    month_map = {
                        'januari': 1, 'februari': 2, 'maart': 3, 'april': 4,
                        'mei': 5, 'juni': 6, 'juli': 7, 'augustus': 8,
                        'september': 9, 'oktober': 10, 'november': 11, 'december': 12
                    }
                    parts = data_date_str.split()
                    if len(parts) == 3:
                        day = int(parts[0])
                        month = month_map.get(parts[1].lower())
                        year = int(parts[2])
                        if month:
                            trade_date = datetime(year, month, day).date()
                        else:
                            trade_date = datetime.now().date()
                    else:
                        trade_date = datetime.now().date()
                except:
                    trade_date = datetime.now().date()
            else:
                trade_date = datetime.now().date()
            
            # Create expiration string
            expiration_str = actual_expiration_date.strftime('%m-%Y') if actual_expiration_date else ''
            
            # Check if record exists for this trade_date + option
            existing = session.query(BronzeEuronextOptions).filter(
                BronzeEuronextOptions.ticker == 'AH',
                BronzeEuronextOptions.option_type == row['option_type'],
                BronzeEuronextOptions.strike == strike,
                BronzeEuronextOptions.actual_expiration_date == actual_expiration_date,
                BronzeEuronextOptions.trade_date == trade_date
            ).first()
            
            if existing:
                # Update existing record
                existing.expiration_date = expiration_str
                existing.open_interest = open_interest
                existing.volume = volume
                existing.opening_price = opening_price
                existing.day_high = high_price
                existing.day_low = low_price
                existing.settlement_price = settlement_price
                existing.scraped_at = datetime.now()
                updated_count += 1
            else:
                # Create new record
                option_record = BronzeEuronextOptions(
                    ticker='AH',
                    option_type=row['option_type'],
                    strike=strike,
                    expiration_date=expiration_str,
                    actual_expiration_date=actual_expiration_date,
                    open_interest=open_interest,
                    volume=volume,
                    opening_price=opening_price,
                    day_high=high_price,
                    day_low=low_price,
                    settlement_price=settlement_price,
                    scraped_at=datetime.now(),
                    trade_date=trade_date,
                )
                session.add(option_record)
            
            saved_count += 1
        
        # Commit all changes
        session.commit()
        logger.info(f"✅ Successfully processed {saved_count} options ({updated_count} updated, {saved_count - updated_count} new)")
        
        return saved_count
        
    except Exception as e:
        session.rollback()
        logger.error(f"Error saving to database: {e}")
        raise
    finally:
        session.close()


def cleanup_old_data_task(**context):
    """
    Clean up old duplicate data from the database.
    Keeps only the most recent record for each (ticker, option_type, strike, expiration_date, data_date).
    """
    from src.utils.db import SessionLocal
    from sqlalchemy import text
    
    session = SessionLocal()
    
    try:
        # Delete duplicate records, keeping only the most recent
        cleanup_query = text("""
            DELETE FROM bronze_euronext_options
            WHERE id NOT IN (
                SELECT MAX(id)
                FROM bronze_euronext_options
                GROUP BY ticker, option_type, strike, actual_expiration_date, trade_date
            )
        """)
        
        result = session.execute(cleanup_query)
        deleted_count = result.rowcount
        session.commit()
        
        logger.info(f"✅ Cleaned up {deleted_count} duplicate records")
        return deleted_count
        
    except Exception as e:
        session.rollback()
        logger.error(f"Error during cleanup: {e}")
        raise
    finally:
        session.close()


# Define the DAG
with DAG(
    dag_id='euronext_chain_scraper',
    default_args=default_args,
    description='Scrape entire Euronext options chain from settlement prices page',
    schedule_interval='15 18 * * 2-6',  # 18:15 CET Tuesday-Saturday (scrapes Mon-Fri data)
    start_date=days_ago(1),
    catchup=False,
    tags=['euronext', 'options', 'scraper', 'bronze'],
) as dag:
    
    # Task 1: Scrape the options chain
    scrape_task = PythonOperator(
        task_id='scrape_chain',
        python_callable=scrape_chain_task,
        provide_context=True,
    )
    
    # Task 2: Save to database
    save_task = PythonOperator(
        task_id='save_to_database',
        python_callable=save_to_database_task,
        provide_context=True,
    )
    
    # Task 3: Cleanup duplicates
    cleanup_task = PythonOperator(
        task_id='cleanup_duplicates',
        python_callable=cleanup_old_data_task,
        provide_context=True,
    )
    
    # Define task dependencies
    scrape_task >> save_task >> cleanup_task
