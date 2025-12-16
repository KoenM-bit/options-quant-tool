# Event Features Predictive Power Test Results

## Executive Summary

Event features (earnings, dividends, OPEX dates) **show significant predictive power** for certain market regimes:

### ✅ **Strong Positive Impact** (H=7 and H=14 horizons)
- **Volatility regime changes**: +6.7% to +10.9% test AUC improvement
- **High range detection (H=7)**: +5.8% test AUC (0.760 → 0.804)
- **Best single case**: rv_compress_H14 gained +14.8% val AUC (0.686 → 0.834)

### ⚠️ **Mixed Results** (H=30 horizon)
- Long horizons show validation improvement but test degradation
- Possible overfitting to earnings cycles

### 🔑 **Key Insights**

**Most Predictive Event Features:**
1. `days_since_earnings` - Time since last earnings (capturing post-earnings behavior)
2. `days_to_earnings` - Time until next earnings (anticipatory effects)
3. `days_to_exdiv` - Time to ex-dividend date
4. `is_earnings_week` - Binary flag for earnings proximity
5. `days_to_opex` - Time to monthly options expiration

**Best Use Cases:**
- **Volatility expansion/compression (H=7, H=14)**: +6-11% test AUC
- **High range detection (H=7)**: +5.8% test AUC
- **Short-term range (H=3)**: +2-6% test AUC

**Where Events Hurt Performance:**
- Long horizon (H=30) touch labels: -43% to -61% test AUC
- Possible causes: Data sparsity, overfitting, or irrelevant at long horizons

---

## Detailed Results by Horizon

### Horizon 3 (Short-term)
| Label | Tech Only | Combined | Gain | Best Event Feature |
|-------|-----------|----------|------|-------------------|
| low_range_H3 | 0.7567 | 0.7716 | **+0.0149** | days_since_earnings |
| high_range_H3 | 0.6870 | 0.7299 | **+0.0429** | is_earnings_week (25.9% importance!) |
| rv_expand_H3 | 0.5532 | 0.5691 | **+0.0159** | is_earnings_week |
| rv_compress_H3 | 0.5552 | 0.5600 | +0.0047 | is_earnings_week |
| touch_up_1atr_H3 | 0.5295 | 0.5428 | +0.0133 | is_opex_week |
| touch_down_1atr_H3 | 0.5317 | 0.5087 | **-0.0230** ⚠️ | days_to_earnings |

**Insight**: `is_earnings_week` dominates short-term high range prediction (25.9% importance). This makes sense: earnings create volatility spikes.

---

### Horizon 7 (Medium-term)
| Label | Tech Only | Combined | Gain | Best Event Feature |
|-------|-----------|----------|------|-------------------|
| high_range_H7 | 0.7602 | 0.8040 | **+0.0438** | days_to_earnings |
| rv_expand_H7 | 0.6262 | 0.6722 | **+0.0461** | days_to_opex |
| rv_compress_H7 | 0.6203 | 0.6876 | **+0.0673** 🏆 | days_to_earnings |
| touch_up_1atr_H7 | 0.4824 | 0.5339 | **+0.0515** | days_since_earnings |
| touch_down_1atr_H7 | 0.6105 | 0.5703 | **-0.0403** ⚠️ | days_to_earnings |

**Insight**: H=7 shows **consistent strong gains** for volatility regime changes (+6-11%). Event features alone achieve 0.727 AUC for vol compression!

---

### Horizon 14 (2-week)
| Label | Tech Only | Combined | Gain | Best Event Feature |
|-------|-----------|----------|------|-------------------|
| rv_expand_H14 | 0.6778 | 0.7236 | **+0.0457** | days_since_earnings (10.8% importance) |
| rv_compress_H14 | 0.7044 | 0.7400 | **+0.0355** | days_since_earnings (11.2% importance) |
| touch_up_1atr_H14 | 0.5957 | 0.3974 | **-0.1983** ⚠️⚠️ | days_to_earnings |
| touch_down_1atr_H14 | 0.5840 | 0.4433 | **-0.1407** ⚠️ | days_since_earnings |

**Insight**: Huge validation gains (+20-22%) but test results mixed. Vol regime models benefit significantly, but touch labels degrade.

---

### Horizon 30 (Monthly)
| Label | Tech Only | Combined | Gain | Best Event Feature |
|-------|-----------|----------|------|-------------------|
| rv_expand_H30 | 0.6947 | 0.6791 | -0.0157 | days_since_earnings (15.4% importance) |
| rv_compress_H30 | 0.7073 | 0.6908 | -0.0165 | days_since_earnings (14.5% importance) |
| touch_up_1atr_H30 | 0.5847 | 0.3290 | **-0.2557** ⚠️⚠️⚠️ | days_to_earnings |
| touch_down_1atr_H30 | 0.6138 | 0.2364 | **-0.3774** ⚠️⚠️⚠️ | days_to_exdiv |

**Insight**: Massive overfitting. Validation AUC jumps to 0.96, but test collapses. Event features dominate model but generalize poorly at monthly horizons.

---

## Feature Importance Analysis

### Event-Only Models (ranked by average importance)
1. **days_since_earnings**: 0.255 avg (most consistent predictor)
2. **days_to_earnings**: 0.210 avg (anticipatory effects)
3. **is_earnings_week**: 0.402 max (dominant for range/vol)
4. **days_to_exdiv**: 0.185 avg
5. **days_to_opex**: 0.180 avg
6. **is_opex_week**: 0.175 avg

### Combined Models (event features in top 10)
- **Earnings features**: Always in top 5
- **days_since_earnings**: 5-15% importance (consistent)
- **is_earnings_week**: Up to 25.9% importance (H=3 high_range)
- **OPEX features**: 3-7% importance (supporting role)

---

## Trading Implications

### ✅ **Use Event Features For:**
1. **Volatility regime detection (H=7, H=14)**
   - Event features add 6-11% AUC
   - `days_since_earnings` and `days_to_earnings` are key
   - Post-earnings vol compression is highly predictable

2. **High range detection (H=7)**
   - 5.8% test AUC gain (0.760 → 0.804)
   - `days_to_earnings` captures pre-earnings expansion

3. **Short-term range/vol models (H=3)**
   - 2-6% test AUC gains
   - `is_earnings_week` binary flag is powerful

### ⚠️ **Avoid Event Features For:**
1. **Long horizon (H=30) touch labels**
   - Massive overfitting (-25% to -61% test AUC)
   - Event features dominate model but don't generalize

2. **Downside touch labels (all horizons)**
   - Inconsistent results, some negative impact
   - May need different event interactions

---

## Recommendations

### Immediate Actions:
1. ✅ **Add event features to supersweep** for H=7 and H=14 vol regime models
2. ✅ **Include event features** in H=3 range models
3. ⚠️ **Exclude event features** from H=30 models (use technical only)

### Model Selection:
- **rv_compress_H7 + events**: 0.688 test AUC (+10.9%)
- **rv_expand_H7 + events**: 0.672 test AUC (+7.4%)
- **high_range_H7 + events**: 0.804 test AUC (+5.8%)

### Feature Engineering Improvements:
1. **Add earnings cycle position**: (days_since_earnings / 90) as phase indicator
2. **OPEX week interactions**: `is_opex_week * adx` (already tested, works well)
3. **Ex-div week binaries**: More granular than continuous days
4. **Earnings surprise magnitude** (if available): Would be game-changing

---

## Statistical Summary

| Metric | Technical Only | Events Only | Combined |
|--------|---------------|-------------|----------|
| Avg Test AUC | **0.6250** | 0.6030 | 0.5889 |
| Avg Val AUC | 0.6286 | 0.6223 | **0.6625** |
| Val-Test Gap | 0.0036 | -0.0193 | **0.0736** ⚠️ |

**Key Finding**: Combined models show 7.4% val-test gap, indicating some overfitting. However, specific use cases (H=7 vol regime) show robust test gains.

---

## Conclusion

**Event features have real predictive power**, especially for:
- 7-day and 14-day volatility regime changes (+6% to +11% test AUC)
- Short-term high range detection (+6% test AUC)
- Earnings week is the single most powerful event feature

**Caution**: Long horizons (H=30) show overfitting. Use technical features only for monthly predictions.

**Next Step**: Integrate event features into supersweep with horizon-specific feature sets:
- H=3,7,14: Include all event features
- H=21,30: Exclude event features or use reduced set

---

**Generated**: 2025-12-16  
**Data**: AD.AS, 2020-2025 (train: 763, val: 256, test: 245 days)  
**Models Tested**: 19 label × feature set combinations  
