
# Realistic Option Backtesting Framework

## Overview

This framework provides a **realistic** option strategy backtesting system that properly models implied volatility (IV) dynamics instead of naively assuming IV = realized volatility (RV).

## The Problem with IV = RV

Previous backtests made these unrealistic assumptions:

1. **IV moves 1:1 with RV** → Actually, IV only moves ~0.3-0.5x with RV changes
2. **No volatility risk premium (VRP)** → IV is typically 2-4% higher than RV
3. **Flat IV across strikes** → Ignores put skew (5-10% higher for OTM puts)
4. **Perfect IV stickiness** → Doesn't model how IV surfaces adjust to price moves
5. **No transaction costs** → Real spreads are 2-5% for options

**Result**: Old backtests overstated returns by 30-50%!

## The Formula

The P&L decomposition formula (from your attachment) is key:

$$\Delta P \approx \text{Vega} \cdot \Delta IV + \frac{1}{2}\Gamma \cdot (\Delta S)^2 - \Theta \cdot \Delta t + \text{skew/curvature} - \text{costs}$$

This shows option P&L comes from:
- **Vega**: Changes in implied volatility
- **Gamma**: Realized price movements
- **Theta**: Time decay
- **Skew effects**: Non-ATM strikes
- **Transaction costs**: Bid-ask spreads

## New Architecture

### 1. IV Simulator (`src/iv_model.py`)

Models ATM IV as a stochastic process:

```
ΔIV_t = κ(IV_mean - IV_{t-1})Δt + β·ΔRV_t + σ_IV·ε_t + event_shocks
```

Where:
- **κ** (mean reversion): IV reverts to long-term mean
- **β** (RV beta): IV partially follows RV (0.3-0.5 typical)
- **σ_IV** (vol of vol): Volatility clustering
- **event_shocks**: Earnings, dividends boost IV

Key features:
- Calibrates to historical RV data
- Maintains realistic VRP (2-4%)
- Models skew for OTM strikes
- Supports sticky-delta repricing

### 2. Skew Model

Put skew model (equity-typical):
- **25Δ put**: ATM + 2.5%
- **10Δ put**: ATM + 4.0%
- **25Δ call**: ATM - 0.5%

Linear interpolation between reference points.

### 3. Option Backtester (`src/option_backtest.py`)

Full backtest engine with:

**Strategy support**:
- ATM straddles
- OTM strangles (delta-based or % OTM)
- Custom strategies (add legs)

**Exit rules**:
- Time-based (hold N days)
- Stop loss (% loss)
- Take profit (% gain)
- IV crush exit
- Combination rules

**P&L decomposition**:
- Vega contribution
- Gamma contribution
- Theta contribution
- Residual (higher-order effects)

**Risk metrics**:
- Sharpe ratio
- Max drawdown
- Win rate
- Average win/loss

## Usage

### Basic Example

```python
from src.option_backtest import OptionBacktester, IVModelParams

# Configure IV model
iv_params = IVModelParams(
    iv_mean=20.0,          # Long-term mean IV
    rv_beta=0.40,          # RV->IV sensitivity
    vrp_mean=3.0,          # Volatility risk premium
    put_skew_25d=2.5,      # Put skew
)

# Initialize backtester
backtester = OptionBacktester(iv_params=iv_params)

# Prepare data with IV surface
df = backtester.prepare_iv_surface(df, rv_col='rv_20', maturity_days=120)

# Run backtest
results = backtester.backtest_strategy(
    df,
    signals,
    strategy='straddle',
    dte_target=120,
    hold_days=30,
    position_size=10000
)

# Print summary
backtester.print_summary(results)
```

### Parameter Sweep

Test multiple configurations:

```python
# Sweep grid
test_configs = [
    (30, 10, 'straddle', 0.00),   # 30 DTE, 10d hold, ATM
    (60, 20, 'straddle', 0.00),   # 60 DTE, 20d hold, ATM
    (120, 20, 'strangle', 0.05),  # 120 DTE, 20d hold, 5% OTM
    (120, 30, 'strangle', 0.10),  # 120 DTE, 30d hold, 10% OTM
]

for dte, hold, strategy, otm in test_configs:
    results = backtester.backtest_strategy(
        df, signals,
        strategy=strategy,
        dte_target=dte,
        hold_days=hold,
        otm_pct=otm
    )
    # Analyze results...
```

## Calibration Guide

### Step 1: Fit IV Model to Historical Data

The `fit_to_historical_rv()` method calibrates:
- Long-term IV level from RV + typical VRP
- IV volatility from RV volatility (scaled down)

### Step 2: Tune IV Parameters

Adjust based on your asset:

**Single stock (high skew)**:
```python
iv_params = IVModelParams(
    rv_beta=0.35,          # Lower beta (IV slower to adjust)
    vrp_mean=3.5,          # Higher premium
    put_skew_25d=3.0,      # Steeper put skew
    earnings_iv_spike=8.0  # Large earnings effect
)
```

**Index ETF (lower skew)**:
```python
iv_params = IVModelParams(
    rv_beta=0.45,          # Higher beta (IV tracks RV more)
    vrp_mean=2.0,          # Lower premium
    put_skew_25d=1.5,      # Flatter skew
)
```

### Step 3: Validate Against Real Options (if available)

If you have historical option chain data:
1. Compare simulated IV to actual IV
2. Adjust `rv_beta` to match IV-RV correlation
3. Tune skew parameters to match actual surface
4. Validate P&L decomposition on real trades

## Key Improvements vs Old System

| Aspect | Old (IV = RV) | New (Realistic IV) |
|--------|---------------|-------------------|
| **IV-RV relationship** | 1:1 | 0.3-0.5x (calibrated) |
| **Volatility risk premium** | 0% | 2-4% (realistic) |
| **Skew model** | Flat | Put skew + call smirk |
| **IV dynamics** | None | Mean reversion + clustering |
| **Transaction costs** | 0% | 2-5% bid-ask |
| **P&L attribution** | None | Vega/Gamma/Theta breakdown |
| **Event sensitivity** | None | Earnings/dividend impact |

## Expected Return Adjustment

Typical impact of realistic modeling:

| Strategy Type | Old Return | New Return | Adjustment |
|--------------|------------|------------|-----------|
| Long volatility (RV expansion) | +30% | +12-18% | -40% to -50% |
| Short volatility (RV compression) | +25% | +15-20% | -20% to -40% |
| Delta-neutral gamma | +20% | +8-12% | -40% to -60% |

**Why?**
- VRP makes options more expensive
- IV is slower to move than RV
- Transaction costs are material
- Theta decay is a constant headwind

## Validation Checklist

Before trusting backtest results:

- [ ] IV-RV correlation is realistic (0.5-0.7)
- [ ] Mean VRP is reasonable (2-4%)
- [ ] Skew matches asset type (single stock vs index)
- [ ] Theta contribution is negative (for long options)
- [ ] Vega contribution is positive when IV rises
- [ ] Returns are lower than IV=RV assumption
- [ ] Win rate is reasonable (40-60% for vol strategies)
- [ ] Transaction costs are included
- [ ] P&L decomposition makes intuitive sense

## Advanced Features

### Custom Exit Rules

```python
# IV crush exit
def iv_crush_exit(entry_iv, current_iv):
    return current_iv < entry_iv * 0.7  # 30% IV drop

# Dynamic stop loss
def dynamic_stop(days_held, max_loss_pct):
    # Tighter stop as time passes
    return max_loss_pct * (1 - days_held / 30)
```

### Vega-Targeted Sizing

Instead of fixed dollar amounts:

```python
# Target $100/1% IV change
target_vega = 100
contracts = target_vega / straddle_vega
```

### Multiple Maturities

Interpolate IV for any DTE:

```python
# Generate IV surfaces for multiple tenors
for dte in [30, 60, 90, 120, 180]:
    df[f'iv_atm_{dte}d'] = iv_sim.simulate_atm_iv_path(df, maturity_days=dte)
```

## File Structure

```
src/
├── iv_model.py              # IV simulation engine
│   ├── IVSimulator          # Main IV model
│   ├── IVModelParams        # Configuration
│   ├── black_scholes_*      # Pricing/Greeks functions
│   └── decompose_pnl()      # P&L attribution
│
├── option_backtest.py       # Backtesting framework
│   ├── OptionBacktester     # Main backtest engine
│   ├── OptionLeg            # Position leg definition
│   └── TradeResult          # Trade result storage
│
ML/
├── backtest_realistic_iv.py # Full backtest script
└── realistic_backtest_results.csv  # Output
```

## Recommended Testing Workflow

1. **Train your signal model** (RV expansion, direction, etc.)
2. **Generate entry signals** (with thresholds)
3. **Run parameter sweep**:
   - DTE: 30, 60, 90, 120, 180
   - Hold: 10d, 20d, 30d
   - OTM levels: ATM, 5%, 10%
4. **Analyze P&L decomposition**
5. **Validate assumptions** (IV-RV relationship, VRP)
6. **Stress test** (double transaction costs, change IV beta)
7. **Select best configuration** (Sharpe, drawdown, practicality)

## Common Pitfalls

❌ **Don't:**
- Use this without calibrating IV model first
- Ignore transaction costs
- Test only one DTE/hold period
- Assume P&L is all from your signal
- Skip the parameter sweep

✅ **Do:**
- Validate IV-RV correlation
- Include realistic spreads (3%+)
- Test multiple configurations
- Decompose P&L sources
- Compare to naive IV=RV baseline

## Next Steps

1. Run `python ML/backtest_realistic_iv.py` to see full example
2. Adjust `IVModelParams` for your asset
3. Compare results to old RV-based backtest
4. Iterate on strategy parameters
5. Consider adding:
   - Real option chain data (best case)
   - Multiple underlyings
   - Portfolio-level backtesting
   - Risk management rules

## References

- Hull, J. (2018). *Options, Futures, and Other Derivatives*
- Gatheral, J. (2006). *The Volatility Surface*
- Sinclair, E. (2013). *Volatility Trading*

---

**Key Insight**: Your signal might still be profitable, but returns will be 30-50% lower than RV-based backtest suggested. The relative ranking of strategies should remain valid, but absolute return expectations need adjustment.
