# RV Compress Q30 H=30 Clean Model

**Validated on:** December 16, 2025  
**Model Type:** Binary Classification (XGBoost)  
**Target:** Predict volatility compression over next 30 days (bottom 30% of RV change)

## Model Summary

Predicts whether realized volatility will compress (decrease) over the next 30 days. This is useful for:
- **Medium-term premium selling** (sell when vol will compress)
- **Vega positioning** (avoid long vol when compression expected)
- **Options strategy selection** (theta strategies vs vega strategies)

## Performance Metrics

| Metric | Value | Notes |
|--------|-------|-------|
| **Test AUC (Clean)** | 80.2% | WITHOUT volatility features (legitimate) |
| **Test AUC (Original)** | 93.6% | WITH volatility features (LEAKAGE - DO NOT USE) |
| **Leakage Amount** | 13.4 pp | Difference due to current vol features |
| **Validation Status** | ✅ PASSED | Shift test and walk-forward validated |
| **Base Rate** | ~16% | Only 16% of periods have vol compression |
| **Feature Count** | 21 | NO volatility features (clean) |

## Critical Finding: Feature Leakage Detected & Fixed

### 🚨 Original Model Had Leakage
- **Problem:** Used current volatility (rv_21d, rv_5d, atr_pct, etc.) to predict future volatility
- **Why it's leakage:** Current vol → future vol via mean reversion (trivial relationship)
- **Impact:** Inflated 93.6% AUC (87% of performance was from leakage)
- **Detection method:** Walk-forward validation + feature importance analysis + shift tests

### ✅ Clean Model Validated
- **Removed:** ALL volatility features (rv_5d, rv_21d, atr, atr_pct, bb_width, realized_volatility_20, etc.)
- **Kept:** Price position, momentum, trend, volume, breadth (21 features)
- **True AUC:** 80.2% (legitimate predictive power)
- **Shift test:** ✅ PASSED - forward shifts degrade performance (no hidden leakage)

## Validation Tests

### ✅ Walk-Forward Validation
- Mean test AUC: 76.0% ± 20.9%
- Multiple rolling train/val/test windows
- Performance degrades with old data (as expected)

### ✅ Feature Importance Analysis
- Original: rv_21d was #1 (24.2%) - **LEAKAGE**
- Clean: pct_from_low_20d is #1 (10.9%) - **LEGITIMATE**
- No single feature dominates (healthy distribution)

### ✅ Shift Test (Temporal Integrity)
- **Baseline (no shift):** 83.1% AUC
- **Forward shift (future data):** Degrades to 65.3% - ✅ GOOD
- **Backward shift (stale data):** Degrades to 73.2% - ✅ GOOD
- **Verdict:** No hidden temporal leakage

### ✅ Permutation Test
- Real labels: 88.0% AUC
- Shuffled labels: 51.2% AUC
- Model learns real patterns (not noise)

## Usage

```bash
# Get prediction for latest trading day
POSTGRES_HOST=192.168.1.201 POSTGRES_PORT=5433 \
POSTGRES_DB=ahold_options POSTGRES_USER=airflow \
POSTGRES_PASSWORD=airflow \
python predict.py
```

## Trading Signals

| Classification | Probability | Expected Hit | Lift | Strategy |
|----------------|-------------|--------------|------|----------|
| **TOP 5%** | > 87.6% | 76.9% | 4.83x | Strong: Sell straddles/strangles (high conviction) |
| **TOP 10%** | > 63.3% | 52.0% | 3.27x | Good: Sell credit spreads |
| **TOP 25%** | > 31.0% | 37.1% | 2.33x | Moderate: Consider iron condors |
| **BELOW** | < 31.0% | ~16% | 1.0x | Weak: Avoid selling premium (vol may expand) |

## Latest Predictions (Last 3 Days)

| Date | Probability | Signal | Interpretation |
|------|-------------|--------|----------------|
| Dec 11, 2025 | 20.3% | ❌ WEAK | Vol may expand or stay stable |
| Dec 12, 2025 | 19.6% | ❌ WEAK | Vol may expand or stay stable |
| Dec 15, 2025 | 4.9% | ❌ WEAK | Vol may expand or stay stable |

**Current assessment:** NOT expecting vol compression. Avoid premium selling based on this model.

## Model Features (Clean Set - NO Volatility)

### Returns
- ret_1d, ret_5d, ret_21d

### Price Position
- px_vs_sma20, px_vs_sma50

### Momentum
- rsi_14, stochastic_k, stochastic_d
- macd, macd_signal, macd_histogram
- roc_20, cci, willr, mfi

### Trend
- adx_14, di_diff (plus_di - minus_di)

### Volume
- obv_norm, vol_ma_ratio

### Breadth
- pct_from_high_20d, pct_from_low_20d
- pct_from_high_52w, pct_from_low_52w

## Comparison with Other Models

| Model | Horizon | Target | Current Signal |
|-------|---------|--------|----------------|
| **low_range_atr H=3** | 3 days | Tight range | ✅ 84.8% (STRONG) |
| **rv_compress_q30 H=30** | 30 days | Vol compression | ❌ 4.9% (WEAK) |

**Interpretation:** Short-term tight range expected, but long-term vol may expand. Good for short-dated premium selling, but be cautious with longer-dated positions.

## Known Limitations

1. **Low base rate (16%):** Vol compression is rare, so most predictions will be negative
2. **Long horizon (30 days):** Market regime can change significantly
3. **No vol features:** Cannot directly measure if current vol is elevated (by design - prevents leakage)
4. **Mean reversion bias:** Model may miss structural vol regime changes
5. **Single ticker:** AD.AS specific, needs retraining for other tickers

## Leakage Prevention Checklist

When retraining or adapting this model:
- ❌ **DO NOT USE:** rv_5d, rv_21d, atr, atr_pct, bb_width, bb_width_pct, realized_volatility_20, parkinson_volatility_20, rv20_logret
- ✅ **DO USE:** Returns, price position, momentum, trend, volume, breadth
- ✅ **ALWAYS RUN:** Shift test to verify temporal integrity
- ✅ **CHECK:** Feature importance - no single feature > 15%
- ✅ **VALIDATE:** Walk-forward performance, not just train/test split

## Maintenance Notes

- **Last validation:** December 16, 2025
- **Data through:** December 15, 2025
- **Leakage detection:** December 16, 2025
- **Recommended re-validation:** Monthly (check for drift)
- **Feature monitoring:** Watch for regime changes in price patterns
- **Re-run leakage tests:** Quarterly (shift test + walk-forward)
