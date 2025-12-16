# Validated ML Models

This folder contains production-ready ML models that have passed rigorous validation testing.

## Validation Criteria

Before a model is added to this folder, it must pass:

1. ✅ **Performance validation:** AUC > 0.75 on test set
2. ✅ **Temporal validation:** Walk-forward or rolling window testing
3. ✅ **Leakage detection:** Shift tests, permutation tests, feature importance analysis
4. ✅ **Stability check:** Performance consistent across multiple time periods
5. ✅ **Documentation:** Clear explanation of what it predicts and how to use it

## Current Validated Models

### 1. Low Range ATR H=3
**Path:** `low_range_atr_h3/`  
**Status:** ✅ VALIDATED  
**Last Updated:** December 16, 2025

**What it predicts:** Tight 3-day trading range (range/ATR ≤ 1.0)

**Performance:**
- Test AUC: ~83%
- Base rate: 68%
- Horizon: 3 days

**Use case:** Short-term premium selling when tight range expected

**Latest signal (Dec 15, 2025):** 84.8% probability (TOP 25% - MODERATE)

---

### 2. RV Compress Q30 H=30 Clean
**Path:** `rv_compress_q30_h30_clean/`  
**Status:** ✅ VALIDATED (after leakage cleanup)  
**Last Updated:** December 16, 2025

**What it predicts:** Volatility compression over next 30 days (bottom 30% of RV change)

**Performance:**
- Test AUC: 80.2% (clean, without vol features)
- Original AUC: 93.6% (LEAKAGE - do not use)
- Base rate: 16%
- Horizon: 30 days

**Use case:** Medium-term premium selling when vol compression expected

**Latest signal (Dec 15, 2025):** 4.9% probability (BELOW threshold - WEAK)

**Critical note:** This model had severe feature leakage (current vol → future vol). Clean version excludes ALL volatility features. Always use the clean version.

---

### 3. RV Expand Abs H=30
**Path:** `rv_expand_abs_h30/`  
**Status:** ✅ VALIDATED  
**Last Updated:** December 16, 2025

**What it predicts:** Volatility expansion over next 30 days (absolute RV change ≥ +0.05)

**Performance:**
- Test AUC: 70.7% (walk-forward mean)
- Initial AUC: 72.9% (single split)
- Base rate: 40.4%
- Horizon: 30 days

**Use case:** Long volatility strategies - BUY options when expansion expected

**Latest signal (Dec 15, 2025):** 11.4% probability (BELOW threshold - WEAK)

**Critical note:** This is the OPPOSITE of rv_compress - use for identifying when to GO LONG volatility (buy straddles/strangles) rather than selling premium. Moderate lift (~1.3x) makes it directional but not as strong as rv_compress.

---

## Quick Start

### Run Low Range ATR H=3
```bash
cd low_range_atr_h3
POSTGRES_HOST=192.168.1.201 POSTGRES_PORT=5433 \
POSTGRES_DB=ahold_options POSTGRES_USER=airflow \
POSTGRES_PASSWORD=airflow \
python predict.py
```

### Run RV Compress Q30 H=30 Clean
```bash
cd rv_compress_q30_h30_clean
POSTGRES_HOST=192.168.1.201 POSTGRES_PORT=5433 \
POSTGRES_DB=ahold_options POSTGRES_USER=airflow \
POSTGRES_PASSWORD=airflow \
python predict.py
```

### Run RV Expand Abs H=30
```bash
cd rv_expand_abs_h30
python predict.py
```

## Model Comparison

| Model | Horizon | Target | Current Signal | Strength |
|-------|---------|--------|----------------|----------|
| **low_range_atr_h3** | 3 days | Tight range | 84.8% ✅ | MODERATE |
| **rv_compress_q30_h30** | 30 days | Vol compression | 4.9% ❌ | WEAK |
| **rv_expand_abs_h30** | 30 days | Vol expansion | 11.4% ❌ | WEAK |

**Current interpretation (Dec 15, 2025):**
- ✅ **Short-term (3d):** Tight range expected → Good for short-dated premium selling
- ❌ **Medium-term (30d):** Vol compression NOT expected, expansion also unlikely → Vol may stay stable
- 💡 **Strategy:** Neutral vol regime - neither strong premium selling nor long vol opportunity

## Trading Signal Integration

### Decision Matrix (3 Models)

| low_range_atr_h3 | rv_compress_q30_h30 | rv_expand_abs_h30 | Strategy |
|------------------|---------------------|-------------------|----------|
| ✅ HIGH | ✅ HIGH | ❌ LOW | **STRONG SELL PREMIUM:** All maturities |
| ✅ HIGH | ❌ LOW | ❌ LOW | **MODERATE SELL:** Short-dated only (< 7 DTE) |
| ❌ LOW | ✅ HIGH | ❌ LOW | **SELECTIVE SELL:** Longer-dated (> 21 DTE) |
| ❌ LOW | ❌ LOW | ✅ HIGH | **LONG VOL:** Buy straddles/strangles |
| ✅ HIGH | ❌ LOW | ✅ HIGH | **CONFLICTING:** Monitor, wait for confirmation |
| ❌ LOW | ❌ LOW | ❌ LOW | **NEUTRAL:** No strong edge either direction |

**Current state (Dec 15):** ✅ HIGH + ❌ LOW + ❌ LOW = **MODERATE** signal (neutral vol regime)

**Recommended strategy:**
- Sell short-dated options (3-7 DTE) with tight strikes - leverage tight range signal
- AVOID selling long-dated options (30+ DTE) - vol regime unclear
- NO strong long vol opportunity - expansion signal too weak
- Consider ratio spreads or iron condors (defined risk premium selling)
- Monitor daily for changes in vol expansion signal

## Validation Test Results

### Low Range ATR H=3
- ✅ Cross-validation: Stable across time periods
- ✅ Recent data: Performance maintained through Dec 2025
- ✅ Feature importance: No single feature > 10%
- ✅ Lift analysis: Top 5% has 92.9% hit rate (1.42x lift)

### RV Compress Q30 H=30 Clean
- ✅ Walk-forward: 76% mean AUC (proper temporal validation)
- ✅ Shift test: Forward shift degrades to 65.3% (no hidden leakage)
- ✅ Feature importance: pct_from_low_20d is #1 at 10.9% (legitimate)
- ✅ Lift analysis: Top 5% has 76.9% hit rate (4.83x lift)
- 🚨 Leakage detected: Original 93.6% AUC → 80.2% after cleanup (13.4pp from leakage)

### RV Expand Abs H=30
- ✅ Walk-forward: 70.7% mean AUC ± 7.8% (stable across time)
- ✅ Shift test: Forward shift degrades to 61.9% (-6.4%, no leakage)
- ✅ Feature importance: realized_volatility_20 is #1 at 15.0% (acceptable)
- ✅ Lift analysis: Top 10% has 52.0% hit rate (1.29x lift)
- ⚠️ Performance note: 2025 window shows 61.5% AUC (weakest period, monitor)

## Failed/Rejected Models

These models were tested but did NOT pass validation:

### ❌ rv_compress_q30_h30 (original with vol features)
- **Reason:** Severe feature leakage (current vol → future vol)
- **Detection:** Shift test, walk-forward validation, feature importance
- **Impact:** 87% of performance was from trivial mean-reversion relationship
- **Action:** Rebuilt as "clean" version without vol features

## Maintenance Schedule

| Task | Frequency | Last Done | Next Due |
|------|-----------|-----------|----------|
| Re-run predictions | Daily | Dec 15, 2025 | Dec 16, 2025 |
| Performance monitoring | Weekly | Dec 15, 2025 | Dec 22, 2025 |
| Walk-forward validation | Monthly | Dec 16, 2025 | Jan 16, 2026 |
| Leakage testing | Quarterly | Dec 16, 2025 | Mar 16, 2026 |
| Feature drift analysis | Quarterly | Dec 16, 2025 | Mar 16, 2026 |
| Full retraining | Semi-annually | N/A | Jun 16, 2026 |

## Adding New Models

To add a new model to this folder:

1. **Create validation report:**
   - Performance metrics (AUC, Brier, calibration)
   - Temporal validation (walk-forward or rolling)
   - Leakage tests (shift test, permutation, feature importance)
   - Stability analysis (multiple time periods)

2. **Pass all validation gates:**
   - Test AUC > 0.75
   - Walk-forward AUC within 5% of test AUC
   - Shift test: forward shift degrades performance
   - No single feature > 15% importance
   - Performance stable across time periods

3. **Create model folder:**
   ```
   model_name/
   ├── README.md          # Model documentation
   ├── predict.py         # Prediction script
   └── validation_tests/  # Optional: test scripts
   ```

4. **Update this README:**
   - Add to "Current Validated Models" section
   - Add to "Model Comparison" table
   - Update "Decision Matrix" if needed

## Contact

**Maintainer:** Koen Marijt  
**Last Updated:** December 16, 2025  
**Repository:** ahold-options
