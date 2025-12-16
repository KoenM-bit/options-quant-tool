# Low Range ATR H=3 Model

**Validated on:** December 16, 2025  
**Model Type:** Binary Classification (XGBoost)  
**Target:** Predict tight 3-day trading range (range/ATR ≤ 1.0)

## Model Summary

Predicts whether the next 3 days will have a tight trading range relative to ATR. This is useful for:
- **Short-term premium selling** (sell options when tight range expected)
- **Avoiding premium buying** when breakout unlikely
- **Position sizing** based on expected volatility

## Performance Metrics

| Metric | Value | Notes |
|--------|-------|-------|
| **Test AUC** | ~83% | Strong predictive power |
| **Validation Status** | ✅ PASSED | Stable across time periods |
| **Base Rate** | ~68% | Most 3-day periods have tight ranges |
| **Feature Count** | ~35 | Full feature set (price, momentum, trend, vol, volume, breadth) |

## Validation Tests

### ✅ Cross-Validation
- Multiple train/val/test splits tested
- Performance stable across different time periods
- No degradation on recent data (Dec 2024 - Dec 2025)

### ✅ Latest Prediction (Dec 15, 2025)
- **Probability:** 84.8% (tight range expected)
- **Classification:** TOP 25% (1.25x lift vs base rate)
- **Expected Hit Rate:** 82%
- **Signal Strength:** MODERATE

## Usage

```bash
# Get prediction for latest trading day
POSTGRES_HOST=192.168.1.201 POSTGRES_PORT=5433 \
POSTGRES_DB=ahold_options POSTGRES_USER=airflow \
POSTGRES_PASSWORD=airflow \
python predict.py
```

## Trading Signals

| Classification | Probability | Expected Hit | Strategy |
|----------------|-------------|--------------|----------|
| **TOP 10%** | > 86.2% | 92.9% | Strong: Sell straddles/strangles (delta 0.20-0.30) |
| **TOP 25%** | > 79.0% | 82.0% | Moderate: Sell credit spreads (delta 0.15-0.25) |
| **TOP 50%** | > 66.8% | 72.2% | Weak: Consider iron condors |
| **BELOW** | < 66.8% | < 70% | Avoid: No premium selling |

## Model Features

Uses comprehensive feature set:
- **Price Position:** SMA ratios (20/50/200), price vs moving averages
- **Momentum:** RSI, Stochastic, MACD, ROC
- **Trend:** ADX, DI+/-, trend strength
- **Volatility:** ATR, Bollinger Bands, Realized Vol
- **Volume:** Volume ratio, OBV
- **Breadth:** Distance from highs/lows (20d, 52w)

## Known Limitations

1. **Short horizon (3 days):** Very short-term, complements longer-term models
2. **High base rate (68%):** Most periods are tight, model identifies extra-tight periods
3. **Market regime dependency:** Performance may vary in extreme volatility regimes
4. **Single ticker trained:** AD.AS specific, may need retraining for other tickers

## Maintenance Notes

- **Last validation:** December 16, 2025
- **Data through:** December 15, 2025
- **Recommended re-validation:** Monthly
- **Feature drift monitoring:** Check RSI, ATR, volume patterns quarterly
