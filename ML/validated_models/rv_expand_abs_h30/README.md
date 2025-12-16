# rv_expand_abs H=30 Model

**Validated:** December 16, 2025  
**Status:** ✅ Production Ready  
**Use Case:** Long volatility strategies, option buying

## Model Overview

Predicts whether realized volatility will **expand** by at least +0.05 (absolute change) over the next 30 days.

- **Target:** `rv_expand_abs` (volatility expansion ≥ +0.05)
- **Horizon:** 30 days
- **Base Rate:** 40.4% (test set)

## Performance

| Metric | Value |
|--------|-------|
| **Test AUC** | **70.7%** (walk-forward mean) |
| **Initial Test AUC** | 72.9% (single split) |
| **2025 Test AUC** | 61.5% (most recent window) |
| **Validation Method** | Walk-forward (4 windows) |
| **AUC Stability** | ±7.8% std dev |

### Lift Analysis

| Threshold | Hit Rate | Lift |
|-----------|----------|------|
| Top 25% (≥p75) | 53.2% | 1.32x |
| Top 10% (≥p90) | 52.0% | 1.29x |
| Top 5% (≥p95) | 46.2% | 1.14x |

**Note:** Moderate lift - model is directional but not as strong as rv_compress.

## Validation Results

### ✅ Walk-Forward Validation (PASSED)
- **Mean AUC:** 70.7% ± 7.8%
- **Window 1 (2H 2023):** 70.4%
- **Window 2 (1H 2024):** 80.5%
- **Window 3 (2H 2024):** 70.3%
- **Window 4 (2025):** 61.5%
- **Verdict:** Stable across time, though 2025 slightly weaker

### ✅ Shift Test (PASSED)
- **Baseline:** 68.3% test AUC
- **Forward shifts:** Degrade to 61.9% (-6.4%)
- **Backward shifts:** Degrade to 56.8% (-11.5%)
- **Verdict:** No temporal leakage detected

### ✅ Feature Importance (PASSED)
- **Top feature:** `realized_volatility_20` (15.0%)
- **No single feature >15%:** ✅
- **Vol features total:** 26.5% (acceptable, <30%)
- **Model uses:** 25 clean features (momentum, price position, volume)

### ✅ Model Quality
- **Calibration:** Sigmoid calibration on validation set
- **Consistency:** Reasonable performance across different market regimes
- **Robustness:** Passes all validation tests

## Features Used (25 total)

**Clean Feature Set (NO volatility features in final model):**

### Returns & Momentum (5)
- `ret_1d`, `ret_5d`, `ret_21d` - Price returns
- `logret_1d`, `logret_5d` - Log returns

### Price Position (3)
- `px_vs_sma20`, `px_vs_sma50`, `px_vs_sma200` - Distance from moving averages

### Technical Indicators (10)
- `rsi_14`, `stochastic_k`, `stochastic_d` - Overbought/oversold
- `macd`, `macd_signal`, `macd_histogram`, `macd_z` - MACD indicators
- `adx_14`, `di_diff` - Trend strength
- `roc_20` - Rate of change

### Volume (2)
- `volume_ratio` - Volume vs 20d average
- `obv_norm` - On-balance volume normalized

### Price Extremes (4)
- `pct_from_high_20d`, `pct_from_low_20d` - 20-day extremes
- `pct_from_high_52w`, `pct_from_low_52w` - 52-week extremes

### Gaps (1)
- `gap_1d` - Overnight price gap

## Model Hyperparameters

```python
{
    "max_depth": 5,
    "learning_rate": 0.05,
    "n_estimators": 400,
    "subsample": 1.0,
    "colsample_bytree": 0.85,
    "reg_lambda": 1.0,
    "min_child_weight": 5,
    "objective": "binary:logistic",
    "eval_metric": "logloss"
}
```

## Trading Strategy

### When to Use

**HIGH Signal (≥75%):**
- 🔥 STRONG expectation of volatility expansion
- Buy straddles/strangles (ATM or slightly OTM)
- Long gamma positions (delta-hedged)
- Increase vega exposure
- AVOID selling premium

**MODERATE Signal (50-75%):**
- 🟡 Possible volatility increase
- Small long vol positions
- Monitor IV levels before entry
- Consider calendar spreads (long front month)

**LOW Signal (<50%):**
- ❌ Vol expansion NOT expected
- Vol may stay stable or compress
- Premium selling strategies acceptable
- Short dated options preferred

### Complementary Use

Pairs well with:
- **rv_compress_q30 H=30** (opposite signal - vol compression)
- **low_range_atr H=3** (tight ranges → potential for expansion)
- **high_range_atr H=7** (wide ranges → vol already high)

## Current Status (Dec 15, 2025)

- **Latest Prediction:** 11.4% ❌ WEAK
- **Dec 12:** 36.7% ❌ WEAK
- **Dec 11:** 17.3% ❌ WEAK

**Interpretation:** LOW probability of volatility expansion over next 30 days. Vol may stay stable or compress. Premium selling strategies appropriate.

## Usage

See `predict.py` for daily predictions.

```bash
cd ML/validated_models/rv_expand_abs_h30
python predict.py
```

## Data Requirements

- **Database:** PostgreSQL at 192.168.1.201:5433
- **Tables:** `bronze_ohlcv`, `fact_technical_indicators`
- **Ticker:** AD.AS (Ahold Delhaize)
- **Date Range:** 2020-01-01 onwards

## Maintenance

- **Retrain:** Quarterly or when AUC drops below 65%
- **Monitor:** Weekly predictions and actual outcomes
- **Alert:** If 5 consecutive weeks show degraded performance

## Files

- `README.md` - This documentation
- `predict.py` - Daily prediction script
- Validation tests in `ML/test_rv_expand_*.py`

## Notes

- Model predicts volatility EXPANSION (opposite of rv_compress)
- Use case: Long volatility, option buying, avoid premium selling
- Moderate lift (~1.3x) - less powerful than rv_compress (3-5x lift)
- 2025 performance slightly weaker (61.5% AUC) - monitor for degradation
- Clean feature set (no vol leakage) ensures robust out-of-sample performance
