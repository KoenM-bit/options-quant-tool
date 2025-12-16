# Validation Test Reference

This document explains the validation tests used to verify model quality.

## Test Suite Overview

| Test | Purpose | Pass Criteria | Tools Used |
|------|---------|---------------|------------|
| **Walk-Forward Validation** | Temporal integrity | AUC within 5% of static test | Rolling train/val/test windows |
| **Shift Test** | Hidden leakage detection | Forward shift degrades performance | Feature time-shifting |
| **Feature Importance** | Feature leakage detection | No feature > 15%, sensible top features | XGBoost feature_importances_ |
| **Permutation Test** | Noise vs signal | Shuffled AUC ≈ 0.5 | Label shuffling |
| **Lift Analysis** | Practical utility | Top deciles show clear lift | Percentile binning |

---

## 1. Walk-Forward Validation

**Purpose:** Test if model performance is stable over time (temporal generalization)

**How it works:**
1. Create multiple rolling train/val/test windows
2. Train on each window and evaluate on its test set
3. Calculate mean and std of test AUC across all windows

**Example (rv_compress_q30_h30):**
```
Window 1: Train [2020-2021] → Val [2022] → Test [2023]  → AUC: 0.85
Window 2: Train [2021-2022] → Val [2023] → Test [2024]  → AUC: 0.72
Window 3: Train [2022-2023] → Val [2024] → Test [2025]  → AUC: 0.68
...
Mean: 0.76 ± 0.21
```

**Pass criteria:**
- Mean walk-forward AUC within 5-10% of static test AUC
- Standard deviation < 0.20 (low variance)

**Red flags:**
- High variance (σ > 0.25): Model unstable across time
- Mean much lower than test AUC: Static test set was lucky

---

## 2. Shift Test

**Purpose:** Detect hidden temporal leakage by shifting features in time

**How it works:**
1. **Baseline (shift=0):** Normal features, get baseline AUC
2. **Forward shift (shift=+N):** Use features from N days in FUTURE
3. **Backward shift (shift=-N):** Use features from N days in PAST
4. Compare: If future data improves performance = LEAKAGE

**Example (rv_compress_q30_h30_clean):**
```
Shift   | Test AUC | Change  | Interpretation
--------|----------|---------|----------------
T+10    | 0.653    | -17.9%  | ✅ Future data degrades (good!)
T+5     | 0.729    | -10.2%  | ✅ Future data degrades
T+3     | 0.788    | -4.4%   | ✅ Future data degrades
T+1     | 0.826    | -0.5%   | ✅ Future data degrades
T+0     | 0.831    | (base)  | Baseline
T-1     | 0.821    | -1.0%   | ✅ Stale data degrades
T-3     | 0.762    | -7.0%   | ✅ Stale data degrades
T-5     | 0.732    | -9.9%   | ✅ Stale data degrades
```

**Pass criteria:**
- Forward shifts: Performance degrades (or stays flat)
- Backward shifts: Performance degrades slightly
- No forward shift should improve AUC by > 2%

**Red flags:**
- Forward shift IMPROVES performance = HIDDEN LEAKAGE
- Performance improves with stale data = Something wrong

---

## 3. Feature Importance Analysis

**Purpose:** Identify feature leakage and ensure sensible feature usage

**How it works:**
1. Train XGBoost model
2. Extract feature_importances_ (Gain metric)
3. Analyze top 10-20 features
4. Check if features make domain sense

**Example (rv_compress_q30_h30):**

**BEFORE cleanup (LEAKAGE):**
```
Rank | Feature        | Importance | Issue
-----|----------------|------------|---------------------------
  1  | rv_21d         | 24.2%      | 🚨 Current vol → future vol
  2  | pct_from_low   | 15.1%      | ✅ Legitimate price pattern
 10  | rv_5d          | 4.2%       | 🚨 Current vol → future vol
```

**AFTER cleanup (CLEAN):**
```
Rank | Feature        | Importance | Valid?
-----|----------------|------------|-------
  1  | pct_from_low   | 10.9%      | ✅ Price position
  2  | pct_from_high  | 9.0%       | ✅ Price position
  3  | pct_from_52w   | 7.1%       | ✅ Breadth
  4  | cci            | 6.9%       | ✅ Momentum
```

**Pass criteria:**
- No single feature > 15% importance
- Top features make domain sense
- No features mechanically related to target

**Red flags:**
- One feature > 20%: Model relies too heavily on single feature
- Current vol features for vol prediction: Circular logic
- Future-looking features in top 10: Direct leakage

---

## 4. Permutation Test

**Purpose:** Verify model learns real patterns vs just noise

**How it works:**
1. Train model on real labels → Get real AUC
2. Shuffle labels randomly → Train on shuffled → Get shuffled AUC
3. Repeat 10-20 times → Calculate mean shuffled AUC
4. Compare: Real AUC should be much higher than shuffled

**Example (rv_compress_q30_h30):**
```
Real labels:     AUC = 0.880
Shuffled (run 1): AUC = 0.523
Shuffled (run 2): AUC = 0.498
Shuffled (run 3): AUC = 0.512
...
Shuffled mean:   AUC = 0.512 ± 0.038
```

**Pass criteria:**
- Real AUC > Shuffled AUC + 3σ
- Shuffled AUC ≈ 0.5 (random guessing)

**Red flags:**
- Shuffled AUC > 0.6: Features leak information
- Real AUC ≈ Shuffled AUC: Model learns nothing

**Note:** This test has limitations - it can't detect trivial relationships like vol mean-reversion. Use in combination with other tests.

---

## 5. Lift Analysis

**Purpose:** Measure practical utility by percentile

**How it works:**
1. Bin predictions into percentiles (p50, p75, p90, p95)
2. Calculate hit rate for each bin
3. Compare to base rate → Calculate lift

**Example (rv_compress_q30_h30_clean):**
```
Percentile | Threshold | N   | Hit Rate | Lift
-----------|-----------|-----|----------|------
≥ p95      | 0.8764    | 13  | 76.9%    | 4.83x
≥ p90      | 0.6331    | 25  | 52.0%    | 3.27x
≥ p75      | 0.3098    | 62  | 37.1%    | 2.33x
Base rate  | -         | 245 | 15.9%    | 1.00x
```

**Pass criteria:**
- Clear monotonic relationship (higher percentile = higher hit rate)
- Top 10% shows at least 1.5x lift vs base rate
- Top 5% shows at least 2.0x lift vs base rate

**Red flags:**
- No lift in top percentiles: Model not separating classes
- Non-monotonic lift: Model predictions poorly calibrated

---

## Leakage Detection Decision Tree

```
Is model performance suspiciously high (AUC > 0.90)?
├─ YES → Run leakage tests
│  ├─ Walk-forward test
│  │  └─ Mean AUC << Test AUC? → LEAKAGE DETECTED
│  ├─ Shift test
│  │  └─ Forward shift improves? → LEAKAGE DETECTED
│  └─ Feature importance
│     └─ Top feature mechanically related to target? → LEAKAGE DETECTED
└─ NO → Still run validation tests (good practice)
```

---

## Test Scripts Location

All validation test scripts are in `ML/`:

- **Walk-forward:** `test_rv_compress_leakage.py` (Test 1)
- **Shift test:** `test_rv_compress_shift.py`
- **Feature importance:** `test_rv_compress_leakage.py` (Test 2)
- **Permutation:** `test_rv_compress_leakage.py` (Test 4)
- **Clean model:** `test_rv_compress_clean.py`

---

## When to Re-run Tests

| Scenario | Tests to Run |
|----------|--------------|
| **New model** | ALL tests (full validation) |
| **Model retraining** | Walk-forward + Feature importance |
| **Adding new features** | Feature importance + Shift test |
| **Poor recent performance** | Walk-forward (check for drift) |
| **Quarterly check** | Shift test + Lift analysis |
| **Before production deployment** | ALL tests (full validation) |

---

## Common Leakage Patterns in Finance

### ❌ Target Leakage
- Using future returns to predict future returns
- Using future vol to predict future vol
- Using end-of-period data for beginning-of-period prediction

### ❌ Feature Leakage
- **Vol mean reversion:** Current vol → future vol (trivial)
- **Price momentum:** Last return → next return (some edge but mostly noise)
- **Correlation leakage:** Feature highly correlated with target

### ❌ Temporal Leakage
- Training on future data
- Using forward-filled data without proper lags
- Not accounting for weekends/holidays

### ✅ Legitimate Patterns
- Price patterns → future range
- Momentum indicators → trend persistence
- Volume patterns → volatility changes
- Technical indicators → regime shifts

---

## Validation Checklist

Before deploying a model:

- [ ] Test AUC > 0.75
- [ ] Walk-forward AUC within 10% of test AUC
- [ ] Shift test passed (forward degrades)
- [ ] No feature > 15% importance
- [ ] Top features make domain sense
- [ ] Permutation test: shuffled AUC ≈ 0.5
- [ ] Lift analysis: top 10% > 1.5x base rate
- [ ] Performance stable across time periods
- [ ] Documentation complete
- [ ] Prediction script tested

---

**Last Updated:** December 16, 2025  
**Validated Models:** 2 (low_range_atr_h3, rv_compress_q30_h30_clean)
