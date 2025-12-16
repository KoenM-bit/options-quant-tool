"""
Verify OHLCV data in ClickHouse.

Checks that bronze_ohlcv, fact_technical_indicators, and fact_market_regime
tables are populated correctly.
"""

import clickhouse_connect

# Connect to ClickHouse
print("Connecting to ClickHouse at 192.168.1.201:8123...")
client = clickhouse_connect.get_client(
    host='192.168.1.201',
    port=8123,
    username='default',
    password=''
)

print("\n" + "="*80)
print("📊 CLICKHOUSE OHLCV DATA VERIFICATION")
print("="*80)

# Check bronze_ohlcv
print("\n🥉 BRONZE LAYER - bronze_ohlcv:")
try:
    result = client.query("""
        SELECT 
            ticker,
            COUNT(*) as total_rows,
            MIN(trade_date) as first_date,
            MAX(trade_date) as last_date
        FROM bronze_ohlcv
        GROUP BY ticker
        ORDER BY ticker
    """)
    
    if result.result_rows:
        for row in result.result_rows:
            print(f"  ✅ {row[0]}: {row[1]:,} rows ({row[2]} to {row[3]})")
    else:
        print("  ⚠️  No data found")
except Exception as e:
    print(f"  ❌ Error: {e}")

# Check fact_technical_indicators
print("\n🥈 SILVER LAYER - fact_technical_indicators:")
try:
    result = client.query("""
        SELECT 
            ticker,
            COUNT(*) as total_rows,
            MIN(trade_date) as first_date,
            MAX(trade_date) as last_date,
            COUNT(CASE WHEN sma_200 IS NOT NULL THEN 1 END) as sma200_count
        FROM fact_technical_indicators
        GROUP BY ticker
        ORDER BY ticker
    """)
    
    if result.result_rows:
        for row in result.result_rows:
            coverage = (row[4] / row[1] * 100) if row[1] > 0 else 0
            print(f"  ✅ {row[0]}: {row[1]:,} rows ({row[2]} to {row[3]})")
            print(f"     SMA200 coverage: {coverage:.1f}%")
    else:
        print("  ⚠️  No data found")
except Exception as e:
    print(f"  ❌ Error: {e}")

# Check fact_market_regime
print("\n🥇 GOLD LAYER - fact_market_regime:")
try:
    result = client.query("""
        SELECT 
            ticker,
            COUNT(*) as total_rows,
            MIN(trade_date) as first_date,
            MAX(trade_date) as last_date
        FROM fact_market_regime
        GROUP BY ticker
        ORDER BY ticker
    """)
    
    if result.result_rows:
        for row in result.result_rows:
            print(f"  ✅ {row[0]}: {row[1]:,} rows ({row[2]} to {row[3]})")
    else:
        print("  ⚠️  No data found")
except Exception as e:
    print(f"  ❌ Error: {e}")

# Show regime distribution
print("\n📈 REGIME DISTRIBUTION:")
try:
    result = client.query("""
        SELECT 
            ticker,
            trend_regime,
            COUNT(*) as days,
            ROUND(AVG(trend_strength), 1) as avg_strength
        FROM fact_market_regime
        GROUP BY ticker, trend_regime
        ORDER BY ticker, days DESC
    """)
    
    if result.result_rows:
        current_ticker = None
        for row in result.result_rows:
            if row[0] != current_ticker:
                current_ticker = row[0]
                print(f"\n  {row[0]}:")
            print(f"    {row[1].upper()}: {row[2]} days (avg strength: {row[3]})")
    else:
        print("  ⚠️  No data found")
except Exception as e:
    print(f"  ❌ Error: {e}")

# Show latest regime for each ticker
print("\n🎯 LATEST MARKET REGIME:")
try:
    result = client.query("""
        SELECT 
            ticker,
            trade_date,
            trend_regime,
            trend_strength,
            volatility_regime,
            volatility_percentile,
            market_phase,
            phase_confidence,
            recommended_strategy,
            days_in_regime
        FROM fact_market_regime
        WHERE (ticker, trade_date) IN (
            SELECT ticker, MAX(trade_date)
            FROM fact_market_regime
            GROUP BY ticker
        )
        ORDER BY ticker
    """)
    
    if result.result_rows:
        for row in result.result_rows:
            print(f"\n  {row[0]} ({row[1]}):")
            print(f"    Trend: {row[2].upper()} (strength: {row[3]:.1f}, {row[9]} days)")
            print(f"    Volatility: {row[4].upper()} ({row[5]:.1f}th percentile)")
            print(f"    Phase: {row[6].upper()} (confidence: {row[7]:.1f})")
            print(f"    Strategy: {row[8]}")
    else:
        print("  ⚠️  No data found")
except Exception as e:
    print(f"  ❌ Error: {e}")

# Sample technical indicators
print("\n📊 SAMPLE TECHNICAL INDICATORS (Latest):")
try:
    result = client.query("""
        SELECT 
            ticker,
            trade_date,
            close,
            sma_20,
            sma_50,
            sma_200,
            rsi_14,
            obv,
            adx_14
        FROM fact_technical_indicators
        WHERE (ticker, trade_date) IN (
            SELECT ticker, MAX(trade_date)
            FROM fact_technical_indicators
            GROUP BY ticker
        )
        ORDER BY ticker
    """)
    
    if result.result_rows:
        for row in result.result_rows:
            print(f"\n  {row[0]} ({row[1]}):")
            print(f"    Close: €{row[2]:.2f}")
            print(f"    SMA20: €{row[3]:.2f}, SMA50: €{row[4]:.2f}, SMA200: €{row[5]:.2f}")
            print(f"    RSI: {row[6]:.2f}, ADX: {row[8]:.2f}")
            print(f"    OBV: {row[7]:,}")
    else:
        print("  ⚠️  No data found")
except Exception as e:
    print(f"  ❌ Error: {e}")

print("\n" + "="*80)
print("✅ VERIFICATION COMPLETE")
print("="*80)
