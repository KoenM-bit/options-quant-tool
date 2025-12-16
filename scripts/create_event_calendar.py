"""
Create earnings and dividend calendar tables for AD.AS
"""

import pandas as pd
from sqlalchemy import create_engine, text
from datetime import datetime

# Database connection
db_url = "postgresql://airflow:airflow@192.168.1.201:5433/ahold_options"
engine = create_engine(db_url)

# Earnings dates for AD.AS (Ahold Delhaize) - from user
earnings_dates = [
    "2020-02-12", "2020-05-07", "2020-08-05", "2020-11-04",
    "2021-02-17", "2021-05-12", "2021-08-11", "2021-11-10",
    "2022-02-16", "2022-05-11", "2022-08-10", "2022-11-09",
    "2023-02-15", "2023-05-10", "2023-08-09", "2023-11-08",
    "2024-02-14", "2024-05-08", "2024-08-07", "2024-11-06",
    "2025-02-25", "2025-05-06", "2025-08-06", "2025-11-05",
]

# Ex-dividend dates for AD.AS - from user
# Format: (ex_date, payment_date, amount_estimate)
dividend_dates = [
    # 2020
    ("2020-04-14", "2020-04-23", None),  # Interim
    ("2020-08-07", "2020-08-27", None),  # Final
    # 2021
    ("2021-04-16", "2021-04-29", None),  # Interim
    ("2021-08-13", "2021-09-02", None),  # Final
    # 2022
    ("2022-04-19", "2022-04-28", None),  # Interim
    ("2022-08-12", "2022-09-01", None),  # Final
    # 2023
    ("2023-04-14", "2023-04-27", None),  # Interim
    ("2023-08-11", "2023-08-31", None),  # Final
    # 2024
    ("2024-04-12", "2024-04-25", None),  # Interim
    ("2024-08-09", "2024-08-29", None),  # Final
    # 2025
    ("2025-04-11", "2025-04-24", None),  # Interim
    ("2025-08-08", "2025-08-28", None),  # Final
]

def create_tables():
    """Create earnings and dividend calendar tables"""
    
    # Create earnings table
    create_earnings = """
    CREATE TABLE IF NOT EXISTS calendar_earnings (
        ticker VARCHAR(10) NOT NULL,
        earnings_date DATE NOT NULL,
        quarter VARCHAR(2),  -- Q1, Q2, Q3, Q4
        year INTEGER,
        fiscal_year INTEGER,
        created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
        PRIMARY KEY (ticker, earnings_date)
    );
    
    CREATE INDEX IF NOT EXISTS idx_earnings_ticker_date 
        ON calendar_earnings(ticker, earnings_date);
    """
    
    # Create dividends table
    create_dividends = """
    CREATE TABLE IF NOT EXISTS calendar_dividends (
        ticker VARCHAR(10) NOT NULL,
        ex_dividend_date DATE NOT NULL,
        payment_date DATE,
        record_date DATE,
        amount FLOAT,
        currency VARCHAR(3) DEFAULT 'EUR',
        dividend_type VARCHAR(20),  -- interim, final, special
        created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
        PRIMARY KEY (ticker, ex_dividend_date)
    );
    
    CREATE INDEX IF NOT EXISTS idx_dividends_ticker_date 
        ON calendar_dividends(ticker, ex_dividend_date);
    """
    
    with engine.begin() as conn:
        conn.execute(text(create_earnings))
        conn.execute(text(create_dividends))
    
    print("✓ Created calendar tables")


def populate_earnings():
    """Populate earnings calendar for AD.AS"""
    
    data = []
    for date_str in earnings_dates:
        date = pd.to_datetime(date_str)
        
        # Determine quarter based on month
        month = date.month
        if month <= 2:
            quarter = "Q4"  # Feb = Q4 results (prev year)
            fiscal_year = date.year - 1
        elif month <= 5:
            quarter = "Q1"  # May = Q1 results
            fiscal_year = date.year
        elif month <= 8:
            quarter = "Q2"  # Aug = Q2 results
            fiscal_year = date.year
        else:
            quarter = "Q3"  # Nov = Q3 results
            fiscal_year = date.year
        
        data.append({
            "ticker": "AD.AS",
            "earnings_date": date,
            "quarter": quarter,
            "year": date.year,
            "fiscal_year": fiscal_year
        })
    
    df = pd.DataFrame(data)
    df.to_sql("calendar_earnings", engine, if_exists="append", index=False)
    
    print(f"✓ Inserted {len(df)} earnings dates")


def populate_dividends():
    """Populate dividend calendar for AD.AS"""
    
    data = []
    for ex_date_str, pay_date_str, amount in dividend_dates:
        ex_date = pd.to_datetime(ex_date_str)
        pay_date = pd.to_datetime(pay_date_str)
        
        # Determine dividend type based on month
        month = ex_date.month
        if month <= 5:
            div_type = "interim"
        else:
            div_type = "final"
        
        data.append({
            "ticker": "AD.AS",
            "ex_dividend_date": ex_date,
            "payment_date": pay_date,
            "amount": amount,  # TODO: Fill in actual amounts
            "dividend_type": div_type
        })
    
    df = pd.DataFrame(data)
    df.to_sql("calendar_dividends", engine, if_exists="append", index=False)
    
    print(f"✓ Inserted {len(df)} dividend dates")


def main():
    print("="*80)
    print("CREATING EVENT CALENDAR TABLES")
    print("="*80)
    
    create_tables()
    populate_earnings()
    populate_dividends()
    
    print("\n" + "="*80)
    print("VERIFICATION")
    print("="*80)
    
    # Verify earnings
    earnings = pd.read_sql(
        "SELECT * FROM calendar_earnings WHERE ticker='AD.AS' ORDER BY earnings_date",
        engine
    )
    print(f"\nEarnings records: {len(earnings)}")
    print(earnings.tail(5).to_string(index=False))
    
    # Verify dividends
    dividends = pd.read_sql(
        "SELECT * FROM calendar_dividends WHERE ticker='AD.AS' ORDER BY ex_dividend_date",
        engine
    )
    print(f"\nDividend records: {len(dividends)}")
    print(dividends.tail(5).to_string(index=False))
    
    print("\n✓ Event calendar tables created successfully")
    print("\nNOTE: Dividend amounts need to be filled in manually")
    print("      Check https://www.aholddelhaize.com/en/investors/")


if __name__ == "__main__":
    main()
