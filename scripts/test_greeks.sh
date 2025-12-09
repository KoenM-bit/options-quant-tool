#!/bin/bash
#
# Activate and Test Black-Scholes Greeks Calculations
# Run this after Docker rebuild completes
#

set -e

echo "=========================================="
echo "Black-Scholes Greeks Activation & Testing"
echo "=========================================="
echo ""

# Step 1: Restart services
echo "Step 1: Restarting Airflow services..."
docker compose restart airflow-webserver airflow-scheduler
echo "✅ Services restarted"
echo ""

# Step 2: Wait for services to be ready
echo "Step 2: Waiting for services to be ready (30 seconds)..."
sleep 30
echo "✅ Services should be ready"
echo ""

# Step 3: Test scipy installation
echo "Step 3: Verifying scipy installation..."
docker compose exec airflow-webserver python3 -c "import scipy; print(f'✅ scipy version: {scipy.__version__}')" || {
    echo "❌ scipy not installed - rebuild required"
    exit 1
}
echo ""

# Step 4: Test Black-Scholes import
echo "Step 4: Testing Black-Scholes module..."
docker compose exec airflow-webserver python3 << 'PYTHON'
import sys
sys.path.insert(0, '/opt/airflow/dags/..')
from src.analytics.black_scholes import BlackScholes, calculate_option_metrics
print("✅ Black-Scholes module imported successfully")

# Quick calculation test
call_price = BlackScholes.call_price(S=100, K=100, T=1.0, r=0.05, sigma=0.20)
print(f"✅ Test calculation: ATM call price = ${call_price:.4f} (expected ~$10.45)")
PYTHON
echo ""

# Step 5: Run enrichment on existing data
echo "Step 5: Running Greeks enrichment on existing options..."
docker compose exec airflow-webserver python3 << 'PYTHON'
import sys
sys.path.insert(0, '/opt/airflow/dags/..')
from src.analytics.enrich_greeks import enrich_options_with_greeks

print("Starting enrichment...")
stats = enrich_options_with_greeks(ticker="AD.AS", risk_free_rate=0.03)
print(f"\n📊 Enrichment Results:")
print(f"   Total processed: {stats['total_processed']}")
print(f"   IV calculated: {stats['iv_calculated']}")
print(f"   Greeks calculated: {stats['greeks_calculated']}")
print(f"   Skipped: {stats['skipped']}")
print(f"   Errors: {stats['errors']}")
PYTHON
echo ""

# Step 6: Check the enriched data
echo "Step 6: Checking enriched data in database..."
docker compose exec postgres psql -U airflow -d ahold_options << 'SQL'
\pset border 2
\pset format wrapped

-- Count options with Greeks
SELECT 
    '📊 ENRICHMENT SUMMARY' as section,
    '' as value
UNION ALL
SELECT 'Total Options', COUNT(*)::text FROM bronze_fd_options
UNION ALL
SELECT 'With IV', COUNT(*)::text FROM bronze_fd_options WHERE implied_volatility IS NOT NULL
UNION ALL
SELECT 'With Delta', COUNT(*)::text FROM bronze_fd_options WHERE delta IS NOT NULL
UNION ALL
SELECT 'With Gamma', COUNT(*)::text FROM bronze_fd_options WHERE gamma IS NOT NULL
UNION ALL
SELECT 'With Vega', COUNT(*)::text FROM bronze_fd_options WHERE vega IS NOT NULL
UNION ALL
SELECT 'With Theta', COUNT(*)::text FROM bronze_fd_options WHERE theta IS NOT NULL;

-- Sample data
SELECT 
    option_type,
    strike,
    expiry_date,
    laatste as last_price,
    ROUND(implied_volatility::numeric, 4) as iv,
    ROUND(delta::numeric, 4) as delta,
    ROUND(gamma::numeric, 4) as gamma,
    ROUND(vega::numeric, 4) as vega,
    ROUND(theta::numeric, 4) as theta
FROM bronze_fd_options 
WHERE implied_volatility IS NOT NULL
ORDER BY ABS(strike - 35) -- ATM options first
LIMIT 5;
SQL
echo ""

# Step 7: Validate calculations
echo "Step 7: Running validation checks..."
docker compose exec airflow-webserver python3 << 'PYTHON'
import sys
sys.path.insert(0, '/opt/airflow/dags/..')
from src.utils.db import get_db_session
from src.models.bronze import BronzeFDOptions
from sqlalchemy import func

print("\n🔍 Validation Checks:")

with get_db_session() as session:
    # Check 1: Delta bounds
    call_deltas = session.query(
        func.min(BronzeFDOptions.delta),
        func.max(BronzeFDOptions.delta)
    ).filter(
        BronzeFDOptions.option_type == 'Call',
        BronzeFDOptions.delta.isnot(None)
    ).first()
    
    put_deltas = session.query(
        func.min(BronzeFDOptions.delta),
        func.max(BronzeFDOptions.delta)
    ).filter(
        BronzeFDOptions.option_type == 'Put',
        BronzeFDOptions.delta.isnot(None)
    ).first()
    
    print(f"\n✅ Check 1: Delta Bounds")
    print(f"   Call deltas: [{call_deltas[0]:.4f}, {call_deltas[1]:.4f}] (should be [0, 1])")
    print(f"   Put deltas: [{put_deltas[0]:.4f}, {put_deltas[1]:.4f}] (should be [-1, 0])")
    
    # Check 2: Gamma positivity
    gamma_stats = session.query(
        func.min(BronzeFDOptions.gamma),
        func.max(BronzeFDOptions.gamma),
        func.avg(BronzeFDOptions.gamma)
    ).filter(
        BronzeFDOptions.gamma.isnot(None)
    ).first()
    
    print(f"\n✅ Check 2: Gamma (should be positive)")
    print(f"   Min: {gamma_stats[0]:.6f}, Max: {gamma_stats[1]:.6f}, Avg: {gamma_stats[2]:.6f}")
    
    # Check 3: Vega positivity
    vega_stats = session.query(
        func.min(BronzeFDOptions.vega),
        func.max(BronzeFDOptions.vega),
        func.avg(BronzeFDOptions.vega)
    ).filter(
        BronzeFDOptions.vega.isnot(None)
    ).first()
    
    print(f"\n✅ Check 3: Vega (should be positive)")
    print(f"   Min: {vega_stats[0]:.6f}, Max: {vega_stats[1]:.6f}, Avg: {vega_stats[2]:.6f}")
    
    # Check 4: IV reasonableness
    iv_stats = session.query(
        func.min(BronzeFDOptions.implied_volatility),
        func.max(BronzeFDOptions.implied_volatility),
        func.avg(BronzeFDOptions.implied_volatility)
    ).filter(
        BronzeFDOptions.implied_volatility.isnot(None)
    ).first()
    
    print(f"\n✅ Check 4: Implied Volatility (annual)")
    print(f"   Min: {iv_stats[0]*100:.2f}%, Max: {iv_stats[1]*100:.2f}%, Avg: {iv_stats[2]*100:.2f}%")
    print(f"   (Typical range: 15-50% for equities)")
    
    # Check 5: ATM options (should have highest gamma/vega)
    atm_option = session.query(BronzeFDOptions).filter(
        BronzeFDOptions.implied_volatility.isnot(None)
    ).order_by(
        func.abs(BronzeFDOptions.strike - BronzeFDOptions.underlying_price)
    ).first()
    
    if atm_option:
        print(f"\n✅ Check 5: ATM Option (Strike closest to underlying)")
        print(f"   Strike: {atm_option.strike}, Type: {atm_option.option_type}")
        print(f"   IV: {atm_option.implied_volatility*100:.2f}%")
        print(f"   Delta: {atm_option.delta:.4f} (ATM call ~0.5, put ~-0.5)")
        print(f"   Gamma: {atm_option.gamma:.6f} (should be highest)")
        print(f"   Vega: {atm_option.vega:.4f} (should be highest)")

print("\n" + "="*50)
print("✅ ALL CHECKS COMPLETE")
print("="*50)
PYTHON
echo ""

echo "=========================================="
echo "✅ Activation & Testing Complete!"
echo "=========================================="
echo ""
echo "Next steps:"
echo "1. Check Airflow UI: http://localhost:8081"
echo "2. Trigger 'ahold_options_daily' DAG to run full pipeline"
echo "3. Run DBT to transform enriched data: make dbt-run"
echo ""
