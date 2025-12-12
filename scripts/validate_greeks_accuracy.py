#!/usr/bin/env python3
"""
Validate Greeks Accuracy
========================
Checks a sample of options (ATM, ITM, OTM) to verify Greeks calculations are reasonable.

Tests include:
- Delta: Should be 0.5 for ATM calls, close to 1 for deep ITM, close to 0 for OTM
- Gamma: Should be highest for ATM options
- Theta: Should be negative (time decay)
- Vega: Should be highest for ATM options
- IV: Should be reasonable (typically 15-50% for equity options)
"""

import sys
from pathlib import Path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

import pandas as pd
from sqlalchemy import text
from src.utils.db import get_db_session
from datetime import date

def validate_greeks(trade_date: date = None):
    """Validate Greeks for ATM, ITM, and OTM options."""
    
    if trade_date is None:
        trade_date = date.today()
    
    print("=" * 80)
    print(f"GREEKS VALIDATION REPORT - {trade_date}")
    print("=" * 80)
    
    with get_db_session() as session:
        # Get sample options from each moneyness bucket
        query = text("""
            WITH moneyness_buckets AS (
                SELECT 
                    f.ts_id,
                    c.ticker,
                    c.strike,
                    c.call_put,
                    c.expiration_date,
                    f.underlying_price,
                    f.mid_price,
                    f.days_to_expiry,
                    f.moneyness,
                    f.iv,
                    f.delta,
                    f.gamma,
                    f.theta,
                    f.vega,
                    f.rho,
                    f.intrinsic_value,
                    f.time_value,
                    CASE 
                        WHEN f.moneyness BETWEEN 0.98 AND 1.02 THEN 'ATM'
                        WHEN (c.call_put = 'C' AND f.moneyness > 1.02) OR 
                             (c.call_put = 'P' AND f.moneyness < 0.98) THEN 'ITM'
                        WHEN (c.call_put = 'C' AND f.moneyness < 0.98) OR 
                             (c.call_put = 'P' AND f.moneyness > 1.02) THEN 'OTM'
                    END as moneyness_bucket
                FROM fact_option_timeseries f
                JOIN dim_option_contract c ON f.option_id = c.option_id
                WHERE f.trade_date = :trade_date
                    AND f.iv IS NOT NULL
                    AND f.days_to_expiry BETWEEN 30 AND 90  -- Focus on 1-3 month options
            )
            SELECT *
            FROM moneyness_buckets
            WHERE moneyness_bucket IS NOT NULL
            ORDER BY ticker, call_put, moneyness_bucket, ABS(moneyness - 1.0)
        """)
        
        df = pd.read_sql(query, session.connection(), params={'trade_date': trade_date})
        
        if df.empty:
            print("❌ No options with Greeks found for validation")
            return
        
        print(f"\n📊 Total options with Greeks: {len(df)}")
        print(f"   Tickers: {', '.join(df['ticker'].unique())}")
        print(f"   Date range: {df['days_to_expiry'].min()}-{df['days_to_expiry'].max()} days to expiry")
        
        # Analyze by moneyness bucket
        for bucket in ['ATM', 'ITM', 'OTM']:
            bucket_df = df[df['moneyness_bucket'] == bucket]
            if bucket_df.empty:
                continue
            
            print("\n" + "=" * 80)
            print(f"📈 {bucket} OPTIONS ({len(bucket_df)} total)")
            print("=" * 80)
            
            # Split by call/put
            for option_type in ['C', 'P']:
                type_df = bucket_df[bucket_df['call_put'] == option_type]
                if type_df.empty:
                    continue
                
                option_name = "CALLS" if option_type == 'C' else "PUTS"
                print(f"\n🔹 {option_name} ({len(type_df)} options)")
                print("-" * 80)
                
                # Statistics
                stats = {
                    'IV': type_df['iv'].describe(),
                    'Delta': type_df['delta'].describe(),
                    'Gamma': type_df['gamma'].describe(),
                    'Theta': type_df['theta'].describe(),
                    'Vega': type_df['vega'].describe()
                }
                
                # Print summary stats
                print(f"{'Metric':<15} {'Mean':<12} {'Std':<12} {'Min':<12} {'Max':<12}")
                print("-" * 80)
                for metric, stat in stats.items():
                    print(f"{metric:<15} {stat['mean']:>11.4f} {stat['std']:>11.4f} "
                          f"{stat['min']:>11.4f} {stat['max']:>11.4f}")
                
                # Sample 3 options for detailed view
                if len(type_df) > 0:
                    type_df = type_df.copy()
                    type_df['moneyness_diff'] = abs(type_df['moneyness'] - 1.0)
                    sample = type_df.nsmallest(min(3, len(type_df)), 'moneyness_diff')
                
                if not sample.empty:
                    print(f"\n📋 Sample {option_name} (closest to {'ATM' if bucket == 'ATM' else bucket}):")
                    print("-" * 80)
                    
                    for _, opt in sample.iterrows():
                        print(f"\nTicker: {opt['ticker']}, Strike: {opt['strike']:.0f}, "
                              f"Expiry: {opt['expiration_date']}, DTE: {opt['days_to_expiry']}")
                        print(f"  Underlying: €{opt['underlying_price']:.2f}, "
                              f"Option Price: €{opt['mid_price']:.2f}, "
                              f"Moneyness: {opt['moneyness']:.3f}")
                        print(f"  IV: {opt['iv']*100:.2f}%, "
                              f"Delta: {opt['delta']:.4f}, "
                              f"Gamma: {opt['gamma']:.4f}")
                        print(f"  Theta: {opt['theta']:.4f}, "
                              f"Vega: {opt['vega']:.4f}, "
                              f"Rho: {opt['rho']:.4f}")
                        print(f"  Intrinsic: €{opt['intrinsic_value']:.2f}, "
                              f"Time Value: €{opt['time_value']:.2f}")
        
        # Validation checks
        print("\n" + "=" * 80)
        print("✅ VALIDATION CHECKS")
        print("=" * 80)
        
        checks = []
        
        # Check 1: ATM call delta should be around 0.5
        atm_calls = df[(df['moneyness_bucket'] == 'ATM') & (df['call_put'] == 'C')]
        if not atm_calls.empty:
            avg_delta = atm_calls['delta'].mean()
            check_pass = 0.45 <= avg_delta <= 0.55
            checks.append(('ATM Call Delta ≈ 0.5', avg_delta, check_pass))
        
        # Check 2: ATM put delta should be around -0.5
        atm_puts = df[(df['moneyness_bucket'] == 'ATM') & (df['call_put'] == 'P')]
        if not atm_puts.empty:
            avg_delta = atm_puts['delta'].mean()
            check_pass = -0.55 <= avg_delta <= -0.45
            checks.append(('ATM Put Delta ≈ -0.5', avg_delta, check_pass))
        
        # Check 3: Gamma should be highest for ATM
        if not df.empty:
            avg_gamma_atm = df[df['moneyness_bucket'] == 'ATM']['gamma'].mean() if len(df[df['moneyness_bucket'] == 'ATM']) > 0 else 0
            avg_gamma_itm = df[df['moneyness_bucket'] == 'ITM']['gamma'].mean() if len(df[df['moneyness_bucket'] == 'ITM']) > 0 else 0
            avg_gamma_otm = df[df['moneyness_bucket'] == 'OTM']['gamma'].mean() if len(df[df['moneyness_bucket'] == 'OTM']) > 0 else 0
            check_pass = avg_gamma_atm > avg_gamma_itm and avg_gamma_atm > avg_gamma_otm
            checks.append(('Gamma highest for ATM', f"ATM:{avg_gamma_atm:.4f} > ITM:{avg_gamma_itm:.4f}, OTM:{avg_gamma_otm:.4f}", check_pass))
        
        # Check 4: Theta should be negative (time decay)
        avg_theta = df['theta'].mean()
        check_pass = avg_theta < 0
        checks.append(('Theta < 0 (time decay)', avg_theta, check_pass))
        
        # Check 5: IV should be reasonable (10-80%)
        avg_iv = df['iv'].mean() * 100
        check_pass = 10 <= avg_iv <= 80
        checks.append(('IV in reasonable range (10-80%)', f"{avg_iv:.2f}%", check_pass))
        
        # Check 6: ITM calls should have delta > ATM calls
        itm_calls = df[(df['moneyness_bucket'] == 'ITM') & (df['call_put'] == 'C')]
        if not itm_calls.empty and not atm_calls.empty:
            avg_delta_itm = itm_calls['delta'].mean()
            avg_delta_atm = atm_calls['delta'].mean()
            check_pass = avg_delta_itm > avg_delta_atm
            checks.append(('ITM Call Delta > ATM Call Delta', f"ITM:{avg_delta_itm:.3f} > ATM:{avg_delta_atm:.3f}", check_pass))
        
        # Print validation results
        print(f"\n{'Check':<40} {'Value':<30} {'Status'}")
        print("-" * 80)
        for check_name, value, passed in checks:
            status = "✅ PASS" if passed else "❌ FAIL"
            print(f"{check_name:<40} {str(value):<30} {status}")
        
        # Overall summary
        passed_checks = sum(1 for _, _, p in checks if p)
        total_checks = len(checks)
        
        print("\n" + "=" * 80)
        print(f"SUMMARY: {passed_checks}/{total_checks} validation checks passed")
        print("=" * 80)
        
        if passed_checks == total_checks:
            print("\n🎉 All validation checks passed! Greeks calculations look accurate.")
        elif passed_checks >= total_checks * 0.8:
            print("\n✅ Most validation checks passed. Greeks are generally accurate.")
        else:
            print("\n⚠️  Some validation checks failed. Review Greeks calculations.")
        
        return df


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description='Validate Greeks accuracy')
    parser.add_argument('--date', type=str, help='Trade date (YYYY-MM-DD)', default=None)
    args = parser.parse_args()
    
    if args.date:
        from datetime import datetime
        trade_date = datetime.strptime(args.date, '%Y-%m-%d').date()
    else:
        trade_date = None
    
    validate_greeks(trade_date)
