import sys
sys.path.insert(0, '/opt/airflow')

from src.analytics.black_scholes import calculate_option_metrics

# Strike 34 Call data from Friday 2025-12-05
option_price = 1.29
underlying_price = 35.01
strike = 34.0
days_to_expiry = 14  # Dec 19 - Dec 5 = 14 days
option_type = 'Call'
risk_free_rate = 0.02

print('📊 MANUAL GREEKS CALCULATION - Strike 34 Call')
print('='*80)
print(f'Option Price:     €{option_price}')
print(f'Underlying:       €{underlying_price}')
print(f'Strike:           €{strike}')
print(f'Days to Expiry:   {days_to_expiry}')
print(f'Option Type:      {option_type}')
print(f'Risk-Free Rate:   {risk_free_rate:.2%}')
print()

# Calculate intrinsic and time value
intrinsic = max(underlying_price - strike, 0)
time_value = option_price - intrinsic
moneyness = underlying_price / strike

print(f'Intrinsic Value:  €{intrinsic:.2f}')
print(f'Time Value:       €{time_value:.2f}')
print(f'Moneyness (S/K):  {moneyness:.4f}')
print()

# Try to calculate Greeks
result = calculate_option_metrics(
    option_price=option_price,
    underlying_price=underlying_price,
    strike=strike,
    days_to_expiry=days_to_expiry,
    option_type=option_type,
    risk_free_rate=risk_free_rate
)

print('RESULT:')
print('-'*80)
if result['gamma'] is not None:
    print(f'✅ IMPLIED VOLATILITY: {result["implied_volatility"]:.2%}')
    print(f'✅ Delta:               {result["delta"]:.6f}')
    print(f'✅ Gamma:               {result["gamma"]:.6f}')
    print(f'✅ Vega:                {result["vega"]:.6f}')
    print(f'✅ Theta:               {result["theta"]:.6f}')
else:
    print('❌ FAILED - All Greeks returned as NULL')
    print(f'   This means one of the quality checks failed')
    print()
    print('   Checking quality criteria:')
    print(f'   1. Time value >= -0.05: {time_value >= -0.05} (time_value={time_value:.4f})')
    print(f'   2. Moneyness 0.1-10:    {0.1 <= moneyness <= 10} (moneyness={moneyness:.4f})')
    print(f'   3. Price > 0:           {option_price > 0}')
    print(f'   4. Days > 0:            {days_to_expiry > 0}')
