# Quick Start Guide - Validated Models

## TL;DR - Get Today's Predictions

```bash
# From project root
cd /Users/koenmarijt/Documents/Projects/ahold-options

# Model 1: Low Range ATR H=3 (tight 3-day range)
POSTGRES_HOST=192.168.1.201 POSTGRES_PORT=5433 \
POSTGRES_DB=ahold_options POSTGRES_USER=airflow \
POSTGRES_PASSWORD=airflow \
python ML/validated_models/low_range_atr_h3/predict.py

# Model 2: RV Compress Q30 H=30 (30-day vol compression)
POSTGRES_HOST=192.168.1.201 POSTGRES_PORT=5433 \
POSTGRES_DB=ahold_options POSTGRES_USER=airflow \
POSTGRES_PASSWORD=airflow \
python ML/validated_models/rv_compress_q30_h30_clean/predict.py
```

## Current Signals (Dec 15, 2025)

| Model | Signal | Strength | Recommendation |
|-------|--------|----------|----------------|
| **low_range_atr_h3** | 84.8% ✅ | MODERATE | Sell short-dated premium (3-7 DTE) |
| **rv_compress_q30_h30** | 4.9% ❌ | WEAK | Avoid long-dated premium (30+ DTE) |

**Combined strategy:**
- ✅ Sell weekly options with tight strikes
- ❌ Avoid monthly options
- Target delta: 0.15-0.25 (conservative)

## Folder Structure

```
ML/validated_models/
├── README.md                          # Overview of all validated models
├── VALIDATION_TESTS.md                # Explanation of validation methodology
├── QUICK_START.md                     # This file
├── low_range_atr_h3/
│   ├── README.md                      # Model documentation
│   └── predict.py                     # Prediction script
└── rv_compress_q30_h30_clean/
    ├── README.md                      # Model documentation
    └── predict.py                     # Prediction script
```

## What Each Model Predicts

### Low Range ATR H=3
- **Horizon:** 3 days
- **Target:** Trading range ≤ 1.0x ATR
- **Use for:** Short-term premium selling
- **Current:** 84.8% (TOP 25% - good for selling)

### RV Compress Q30 H=30 Clean
- **Horizon:** 30 days
- **Target:** Volatility will compress (bottom 30%)
- **Use for:** Medium-term vol positioning
- **Current:** 4.9% (WEAK - vol may expand)

## Trading Decision Matrix

| low_range_atr_h3 | rv_compress_q30_h30 | Strategy |
|------------------|---------------------|----------|
| ✅ HIGH (>80%) | ✅ HIGH (>60%) | **STRONG:** Sell across all maturities |
| ✅ HIGH (>80%) | ❌ LOW (<30%) | **MODERATE:** Sell short-dated only ← **YOU ARE HERE** |
| ❌ LOW (<70%) | ✅ HIGH (>60%) | **SELECTIVE:** Sell long-dated only |
| ❌ LOW (<70%) | ❌ LOW (<30%) | **AVOID:** No premium selling |

## Validation Status

Both models have passed:
- ✅ Walk-forward validation
- ✅ Shift test (temporal integrity)
- ✅ Feature importance analysis
- ✅ Permutation test
- ✅ Lift analysis

See `VALIDATION_TESTS.md` for details.

## Next Steps

1. **Read model READMEs** for detailed documentation
2. **Run predictions daily** to get latest signals
3. **Monitor performance** weekly
4. **Re-validate quarterly** (walk-forward + shift tests)

## Support

For questions or issues:
- Check model-specific README.md files
- Review VALIDATION_TESTS.md for methodology
- See main README.md for model comparison

**Last Updated:** December 16, 2025
