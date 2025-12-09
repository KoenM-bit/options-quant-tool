"""
Interactive Black-Scholes Greeks Validator
Run this to manually test and validate Greeks calculations
"""

import sys
sys.path.insert(0, '/opt/airflow/dags/..')

from src.analytics.black_scholes import BlackScholes, calculate_option_metrics
from src.utils.db import get_db_session
from src.models.bronze import BronzeFDOptions, BronzeFDOverview
from sqlalchemy import and_, func
from datetime import date

print("=" * 70)
print("BLACK-SCHOLES GREEKS INTERACTIVE VALIDATOR")
print("=" * 70)
print()

# Test 1: Known test case (should match textbook values)
print("📚 Test 1: Known Textbook Case")
print("-" * 70)
S, K, T, r, sigma = 100.0, 100.0, 1.0, 0.05, 0.20
print(f"Parameters: S={S}, K={K}, T={T}, r={r}, σ={sigma}")
print()

call = BlackScholes.call_price(S, K, T, r, sigma)
put = BlackScholes.put_price(S, K, T, r, sigma)
print(f"Call Price:  ${call:.4f}  (Expected: ~$10.45)")
print(f"Put Price:   ${put:.4f}   (Expected: ~$5.57)")
print()

greeks = BlackScholes.calculate_all_greeks(S, K, T, r, sigma, 'call')
print("Greeks (Call):")
print(f"  Delta:  {greeks['delta']:.4f}  (Expected: ~0.6368)")
print(f"  Gamma:  {greeks['gamma']:.6f}  (Expected: ~0.0199)")
print(f"  Vega:   {greeks['vega']:.4f}  (Expected: ~0.3969)")
print(f"  Theta:  {greeks['theta']:.4f}  (Expected: ~-0.0183)")
print(f"  Rho:    {greeks['rho']:.4f}  (Expected: ~0.5327)")
print()

# Test 2: Your actual data
print("📊 Test 2: Your Ahold Options Data")
print("-" * 70)

with get_db_session() as session:
    # Get underlying price
    overview = session.query(BronzeFDOverview).order_by(
        BronzeFDOverview.scraped_at.desc()
    ).first()
    
    if overview:
        print(f"Underlying: {overview.ticker} @ €{overview.koers:.2f}")
        print(f"Date: {overview.peildatum}")
        print()
    
    # Get some sample options
    options = session.query(BronzeFDOptions).filter(
        BronzeFDOptions.laatste.isnot(None),
        BronzeFDOptions.laatste > 0
    ).order_by(
        func.abs(BronzeFDOptions.strike - (overview.koers if overview else 35))
    ).limit(5).all()
    
    if not options:
        print("❌ No options found with prices")
        sys.exit(1)
    
    print(f"Found {len(options)} sample options near ATM")
    print()
    
    for i, opt in enumerate(options, 1):
        print(f"Option {i}: {opt.option_type} Strike €{opt.strike} Exp {opt.expiry_date}")
        print(f"  Market Price: €{opt.laatste:.4f}")
        
        # Get underlying price
        underlying = opt.underlying_price or (overview.koers if overview else 35.0)
        days_to_expiry = (opt.expiry_date - opt.scraped_at.date()).days
        
        # Calculate Greeks
        metrics = calculate_option_metrics(
            option_price=opt.laatste,
            underlying_price=underlying,
            strike=opt.strike,
            days_to_expiry=days_to_expiry,
            option_type=opt.option_type,
            risk_free_rate=0.03
        )
        
        print(f"  Days to Expiry: {days_to_expiry}")
        print(f"  Underlying: €{underlying:.2f}")
        print(f"  Moneyness: {underlying/opt.strike:.3f}")
        
        if metrics['implied_volatility']:
            print(f"  📈 Implied Vol: {metrics['implied_volatility']*100:.2f}%")
            print(f"  📊 Delta: {metrics['delta']:.4f}")
            print(f"  📊 Gamma: {metrics['gamma']:.6f}")
            print(f"  📊 Vega:  {metrics['vega']:.4f}")
            print(f"  📊 Theta: {metrics['theta']:.4f} (per day)")
            print(f"  📊 Rho:   {metrics['rho']:.4f}")
        else:
            print(f"  ⚠️  IV could not be calculated (may be at intrinsic value)")
        
        print()

# Test 3: Database statistics
print("📊 Test 3: Database Statistics")
print("-" * 70)

with get_db_session() as session:
    total = session.query(func.count(BronzeFDOptions.id)).scalar()
    with_iv = session.query(func.count(BronzeFDOptions.id)).filter(
        BronzeFDOptions.implied_volatility.isnot(None)
    ).scalar()
    
    print(f"Total Options: {total}")
    print(f"With IV: {with_iv} ({with_iv/total*100:.1f}%)")
    print()
    
    # IV statistics
    if with_iv > 0:
        iv_stats = session.query(
            func.min(BronzeFDOptions.implied_volatility),
            func.max(BronzeFDOptions.implied_volatility),
            func.avg(BronzeFDOptions.implied_volatility)
        ).filter(
            BronzeFDOptions.implied_volatility.isnot(None)
        ).first()
        
        print("Implied Volatility Distribution:")
        print(f"  Min: {iv_stats[0]*100:.2f}%")
        print(f"  Max: {iv_stats[1]*100:.2f}%")
        print(f"  Avg: {iv_stats[2]*100:.2f}%")
        print()
        
        # Greeks by option type
        for opt_type in ['Call', 'Put']:
            delta_avg = session.query(
                func.avg(BronzeFDOptions.delta)
            ).filter(
                BronzeFDOptions.option_type == opt_type,
                BronzeFDOptions.delta.isnot(None)
            ).scalar()
            
            gamma_avg = session.query(
                func.avg(BronzeFDOptions.gamma)
            ).filter(
                BronzeFDOptions.option_type == opt_type,
                BronzeFDOptions.gamma.isnot(None)
            ).scalar()
            
            if delta_avg and gamma_avg:
                print(f"{opt_type}s:")
                print(f"  Avg Delta: {delta_avg:.4f}")
                print(f"  Avg Gamma: {gamma_avg:.6f}")

print()
print("=" * 70)
print("✅ VALIDATION COMPLETE")
print("=" * 70)
print()
print("Key things to check:")
print("  1. Call deltas should be between 0 and 1")
print("  2. Put deltas should be between -1 and 0")
print("  3. Gamma should always be positive")
print("  4. Vega should always be positive")
print("  5. IV should be reasonable (15-50% for typical equities)")
print("  6. ATM options should have highest gamma and vega")
print()
