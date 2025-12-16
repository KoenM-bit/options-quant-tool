"""
Calculate event-based features for options trading ML models
- Earnings proximity and flags
- Dividend proximity and flags  
- OPEX (monthly options expiration) proximity
- Calendar effects (month-end, quarter-end)
"""

import pandas as pd
import numpy as np
from sqlalchemy import create_engine, text
from datetime import datetime, timedelta


def get_third_friday(year: int, month: int) -> pd.Timestamp:
    """Get 3rd Friday of the month (monthly OPEX)"""
    first_day = pd.Timestamp(year, month, 1)
    first_friday = first_day + pd.Timedelta(days=(4 - first_day.dayofweek) % 7)
    third_friday = first_friday + pd.Timedelta(days=14)
    return third_friday


def calculate_event_features(df: pd.DataFrame, 
                              earnings_df: pd.DataFrame,
                              dividends_df: pd.DataFrame) -> pd.DataFrame:
    """
    Calculate all event-based features
    
    Features created:
    1. Earnings features (15 features)
    2. Dividend features (12 features)
    3. OPEX features (8 features)
    4. Calendar features (6 features)
    
    Total: ~41 new features
    """
    X = df.copy()
    X["trade_date"] = pd.to_datetime(X["trade_date"])
    
    # ========================================================================
    # 1. EARNINGS FEATURES
    # ========================================================================
    
    # Create mapping of dates to next/prev earnings
    earnings_dates = pd.to_datetime(earnings_df["earnings_date"].unique())
    
    # For each trading day, find next and previous earnings
    next_earnings = []
    prev_earnings = []
    
    for trade_date in X["trade_date"]:
        future = earnings_dates[earnings_dates >= trade_date]
        past = earnings_dates[earnings_dates < trade_date]
        
        if len(future) > 0:
            next_earnings.append(future.min())
        else:
            next_earnings.append(pd.NaT)
        
        if len(past) > 0:
            prev_earnings.append(past.max())
        else:
            prev_earnings.append(pd.NaT)
    
    X["next_earnings_date"] = next_earnings
    X["prev_earnings_date"] = prev_earnings
    
    # Days to/from earnings
    X["days_to_earnings"] = (X["next_earnings_date"] - X["trade_date"]).dt.days
    X["days_since_earnings"] = (X["trade_date"] - X["prev_earnings_date"]).dt.days
    
    # Clamp days_to_earnings at 90 (beyond 90 days = not relevant)
    X["days_to_earnings"] = X["days_to_earnings"].clip(upper=90)
    
    # Binary flags for earnings proximity
    X["is_earnings_week"] = (X["days_to_earnings"] >= 0) & (X["days_to_earnings"] <= 5)
    X["is_earnings_tomorrow"] = X["days_to_earnings"] == 1
    X["is_earnings_today"] = X["days_to_earnings"] == 0
    X["is_post_earnings_1d"] = X["days_since_earnings"] == 1
    X["is_post_earnings_5d"] = (X["days_since_earnings"] >= 1) & (X["days_since_earnings"] <= 5)
    
    # Earnings bucket (categorical binned)
    def earnings_bucket(days):
        if pd.isna(days):
            return "unknown"
        elif days < 0:
            return "past"
        elif days == 0:
            return "today"
        elif days <= 5:
            return "0-5d"
        elif days <= 14:
            return "6-14d"
        elif days <= 30:
            return "15-30d"
        else:
            return ">30d"
    
    X["earnings_bucket"] = X["days_to_earnings"].apply(earnings_bucket)
    
    # Earnings quarter (merge from earnings table)
    X = X.merge(
        earnings_df[["earnings_date", "quarter"]].rename(columns={"earnings_date": "next_earnings_date"}),
        on="next_earnings_date",
        how="left"
    )
    X = X.rename(columns={"quarter": "next_earnings_quarter"})
    
    # One-hot encode quarter
    X["earnings_q1"] = (X["next_earnings_quarter"] == "Q1").astype(int)
    X["earnings_q2"] = (X["next_earnings_quarter"] == "Q2").astype(int)
    X["earnings_q3"] = (X["next_earnings_quarter"] == "Q3").astype(int)
    X["earnings_q4"] = (X["next_earnings_quarter"] == "Q4").astype(int)
    
    # ========================================================================
    # 2. DIVIDEND FEATURES
    # ========================================================================
    
    div_dates = pd.to_datetime(dividends_df["ex_dividend_date"].unique())
    
    # For each trading day, find next/prev ex-dividend date
    next_exdiv = []
    prev_exdiv = []
    
    for trade_date in X["trade_date"]:
        future = div_dates[div_dates >= trade_date]
        past = div_dates[div_dates < trade_date]
        
        if len(future) > 0:
            next_exdiv.append(future.min())
        else:
            next_exdiv.append(pd.NaT)
        
        if len(past) > 0:
            prev_exdiv.append(past.max())
        else:
            prev_exdiv.append(pd.NaT)
    
    X["next_exdiv_date"] = next_exdiv
    X["prev_exdiv_date"] = prev_exdiv
    
    # Days to/from ex-dividend
    X["days_to_exdiv"] = (X["next_exdiv_date"] - X["trade_date"]).dt.days
    X["days_since_exdiv"] = (X["trade_date"] - X["prev_exdiv_date"]).dt.days
    
    X["days_to_exdiv"] = X["days_to_exdiv"].clip(upper=90)
    
    # Binary flags
    X["is_exdiv_week"] = (X["days_to_exdiv"] >= 0) & (X["days_to_exdiv"] <= 5)
    X["is_exdiv_tomorrow"] = X["days_to_exdiv"] == 1
    X["is_exdiv_today"] = X["days_to_exdiv"] == 0
    X["is_post_exdiv_1d"] = X["days_since_exdiv"] == 1
    
    # Dividend amount (if available - merge from dividends table)
    X = X.merge(
        dividends_df[["ex_dividend_date", "amount", "dividend_type"]].rename(
            columns={"ex_dividend_date": "next_exdiv_date"}
        ),
        on="next_exdiv_date",
        how="left"
    )
    X = X.rename(columns={"amount": "next_div_amount", "dividend_type": "next_div_type"})
    
    # Dividend yield (annualized - assuming 2 divs per year)
    X["div_yield_annual"] = (X["next_div_amount"] * 2 / X["close"]).fillna(0)
    
    # Dividend as fraction of ATR (super useful!)
    if "atr_14" in X.columns:
        X["div_as_atr"] = (X["next_div_amount"] / X["atr_14"]).fillna(0)
    
    # Dividend seasonality
    X["is_dividend_interim"] = (X["next_div_type"] == "interim").astype(int)
    X["is_dividend_final"] = (X["next_div_type"] == "final").astype(int)
    
    # ========================================================================
    # 3. OPEX (MONTHLY OPTIONS EXPIRATION) FEATURES
    # ========================================================================
    
    # For each trading day, calculate next OPEX (3rd Friday)
    def get_next_opex(trade_date):
        year = trade_date.year
        month = trade_date.month
        
        # Try current month
        opex = get_third_friday(year, month)
        if opex >= trade_date:
            return opex
        
        # Next month
        month += 1
        if month > 12:
            month = 1
            year += 1
        
        return get_third_friday(year, month)
    
    X["next_opex_date"] = X["trade_date"].apply(get_next_opex)
    
    # Days to OPEX
    X["days_to_opex"] = (X["next_opex_date"] - X["trade_date"]).dt.days
    
    # OPEX flags
    X["is_opex_week"] = (X["days_to_opex"] >= 0) & (X["days_to_opex"] <= 5)
    X["is_opex_friday"] = X["days_to_opex"] == 0
    X["is_post_opex_week"] = (X["days_to_opex"] < 0) & (X["days_to_opex"] >= -5)
    
    # OPEX bucket
    def opex_bucket(days):
        if days < -5:
            return "post_opex"
        elif days < 0:
            return "opex_+1to5"
        elif days == 0:
            return "opex_today"
        elif days <= 5:
            return "opex_0-5d"
        elif days <= 10:
            return "opex_6-10d"
        else:
            return "opex_>10d"
    
    X["opex_bucket"] = X["days_to_opex"].apply(opex_bucket)
    
    # ========================================================================
    # 4. CALENDAR EFFECTS (Month-end, Quarter-end)
    # ========================================================================
    
    # Days to month end
    X["days_to_month_end"] = X["trade_date"].apply(
        lambda d: (pd.Timestamp(d.year, d.month, 1) + pd.DateOffset(months=1) - pd.Timedelta(days=1) - d).days
    )
    
    X["is_month_end_week"] = (X["days_to_month_end"] >= 0) & (X["days_to_month_end"] <= 5)
    
    # Days to quarter end
    def days_to_quarter_end(trade_date):
        quarter_ends = {
            1: pd.Timestamp(trade_date.year, 3, 31),
            2: pd.Timestamp(trade_date.year, 6, 30),
            3: pd.Timestamp(trade_date.year, 9, 30),
            4: pd.Timestamp(trade_date.year, 12, 31)
        }
        
        quarter = (trade_date.month - 1) // 3 + 1
        qtr_end = quarter_ends[quarter]
        
        if qtr_end >= trade_date:
            return (qtr_end - trade_date).days
        else:
            # Next quarter
            next_q = quarter + 1 if quarter < 4 else 1
            next_year = trade_date.year if quarter < 4 else trade_date.year + 1
            next_qtr_end = quarter_ends[next_q].replace(year=next_year)
            return (next_qtr_end - trade_date).days
    
    X["days_to_quarter_end"] = X["trade_date"].apply(days_to_quarter_end)
    X["is_quarter_end_week"] = (X["days_to_quarter_end"] >= 0) & (X["days_to_quarter_end"] <= 5)
    
    # Quarter indicator
    X["quarter"] = X["trade_date"].dt.quarter
    
    # ========================================================================
    # 5. INTERACTION FEATURES (conditional event effects)
    # ========================================================================
    
    # These capture "how does market behave DURING events"
    # Only using past data (no leakage)
    
    if "atr_14" in X.columns:
        X["earnings_week_x_atr"] = X["is_earnings_week"].astype(int) * X["atr_14"]
    
    if "adx_14" in X.columns:
        X["opex_week_x_adx"] = X["is_opex_week"].astype(int) * X["adx_14"]
    
    if "realized_volatility_20" in X.columns:
        X["earnings_week_x_rv"] = X["is_earnings_week"].astype(int) * X["realized_volatility_20"]
    
    return X


def main():
    """Test feature generation"""
    db_url = "postgresql://airflow:airflow@192.168.1.201:5433/ahold_options"
    engine = create_engine(db_url)
    
    print("="*80)
    print("EVENT FEATURE ENGINEERING TEST")
    print("="*80)
    
    # Load data
    print("\nLoading data...")
    ohlcv = pd.read_sql(
        "SELECT * FROM bronze_ohlcv WHERE ticker='AD.AS' AND trade_date >= '2020-01-01' ORDER BY trade_date",
        engine
    )
    
    # Load technical indicators (for interaction features)
    tech = pd.read_sql(
        "SELECT trade_date, ticker, atr_14, adx_14, realized_volatility_20 "
        "FROM fact_technical_indicators WHERE ticker='AD.AS' AND trade_date >= '2020-01-01'",
        engine
    )
    
    ohlcv = ohlcv.merge(tech, on=["trade_date", "ticker"], how="left")
    
    earnings = pd.read_sql(
        "SELECT * FROM calendar_earnings WHERE ticker='AD.AS'",
        engine
    )
    
    dividends = pd.read_sql(
        "SELECT * FROM calendar_dividends WHERE ticker='AD.AS'",
        engine
    )
    
    print(f"✓ Loaded {len(ohlcv)} trading days")
    print(f"✓ Loaded {len(earnings)} earnings dates")
    print(f"✓ Loaded {len(dividends)} dividend dates")
    
    # Calculate features
    print("\nCalculating event features...")
    result = calculate_event_features(ohlcv, earnings, dividends)
    
    # Show summary
    event_features = [col for col in result.columns if any(x in col for x in [
        "earnings", "exdiv", "dividend", "opex", "month_end", "quarter"
    ])]
    
    print(f"\n✓ Created {len(event_features)} event features:")
    for feat in sorted(event_features):
        print(f"  - {feat}")
    
    # Show recent data
    print(f"\n{'='*80}")
    print("RECENT DATA (last 5 days)")
    print(f"{'='*80}")
    
    display_cols = [
        "trade_date", "close",
        "days_to_earnings", "is_earnings_week",
        "days_to_exdiv", "is_exdiv_week",
        "days_to_opex", "is_opex_week"
    ]
    
    print(result[display_cols].tail(10).to_string(index=False))
    
    # Show statistics
    print(f"\n{'='*80}")
    print("FEATURE STATISTICS")
    print(f"{'='*80}")
    
    print(f"\nEarnings proximity:")
    print(f"  Days to earnings (mean): {result['days_to_earnings'].mean():.1f}")
    print(f"  Earnings week days: {result['is_earnings_week'].sum()}")
    print(f"  Post-earnings 5d: {result['is_post_earnings_5d'].sum()}")
    
    print(f"\nDividend proximity:")
    print(f"  Days to ex-div (mean): {result['days_to_exdiv'].mean():.1f}")
    print(f"  Ex-div week days: {result['is_exdiv_week'].sum()}")
    
    print(f"\nOPEX proximity:")
    print(f"  Days to OPEX (mean): {result['days_to_opex'].mean():.1f}")
    print(f"  OPEX week days: {result['is_opex_week'].sum()}")
    print(f"  OPEX Friday count: {result['is_opex_friday'].sum()}")
    
    print(f"\n✓ Event features ready for ML training!")


if __name__ == "__main__":
    main()
