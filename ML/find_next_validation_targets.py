"""
Find best unvalidated models from strong families (range_atr and vol_regime)
"""

import pandas as pd

# Load results
df = pd.read_csv("ML/supersweep_staged_results.csv")

print("="*80)
print("STRONG FAMILIES - DETAILED ANALYSIS")
print("="*80)

# Focus on range_atr and vol_regime
strong_families = df[df['label_family'].isin(['range_atr', 'vol_regime'])].copy()

print(f"\nTotal models in strong families: {len(strong_families)}")

# Group by label and horizon to see all variants
print("\n" + "="*80)
print("RANGE_ATR FAMILY")
print("="*80)

range_atr = strong_families[strong_families['label_family'] == 'range_atr']
range_summary = (range_atr.groupby(['label', 'horizon'])
                 .agg({
                     'test_auc': ['max', 'mean', 'count'],
                     'test_base_rate': 'mean'
                 })
                 .round(4)
                 .sort_values(('test_auc', 'max'), ascending=False))

print("\n", range_summary)

print("\nTop models in range_atr:")
top_range = range_atr.nlargest(5, 'test_auc')
for idx, row in top_range.iterrows():
    validated = "✅ VALIDATED" if "low_range" in row['label'] and row['horizon'] == 3 else "⏳ TODO"
    print(f"{validated} {row['label']:20s} H={row['horizon']:<3} AUC={row['test_auc']:.4f} Base={row['test_base_rate']:5.1%}")

print("\n" + "="*80)
print("VOL_REGIME FAMILY")
print("="*80)

vol_regime = strong_families[strong_families['label_family'] == 'vol_regime']
vol_summary = (vol_regime.groupby(['label', 'horizon'])
               .agg({
                   'test_auc': ['max', 'mean', 'count'],
                   'test_base_rate': 'mean'
               })
               .round(4)
               .sort_values(('test_auc', 'max'), ascending=False))

print("\n", vol_summary)

print("\nTop models in vol_regime:")
top_vol = vol_regime.nlargest(10, 'test_auc')
for idx, row in top_vol.iterrows():
    validated = "✅ VALIDATED" if "rv_compress_q30" in row['label'] and row['horizon'] == 30 else "⏳ TODO"
    print(f"{validated} {row['label']:30s} H={row['horizon']:<3} AUC={row['test_auc']:.4f} Base={row['test_base_rate']:5.1%}")

print("\n" + "="*80)
print("RECOMMENDATIONS FOR NEXT VALIDATION")
print("="*80)

# Find best unvalidated models
print("\n🎯 PRIORITY 1 - Best unvalidated from each family:")

# range_atr: Find high_range_atr (opposite of low_range)
high_range = range_atr[range_atr['label'] == 'high_range_atr'].nlargest(1, 'test_auc')
if len(high_range) > 0:
    hr = high_range.iloc[0]
    print(f"\n  range_atr: high_range_atr H={hr['horizon']} (AUC: {hr['test_auc']:.4f}, Base: {hr['test_base_rate']:.1%})")
else:
    print(f"\n  range_atr: No high_range_atr models found")

# vol_regime: Find rv_expand or rv_compress_abs (different from rv_compress_q30)
rv_expand = vol_regime[vol_regime['label'].str.contains('expand')].nlargest(1, 'test_auc')
if len(rv_expand) > 0:
    rve = rv_expand.iloc[0]
    print(f"  vol_regime: {rve['label']} H={rve['horizon']} (AUC: {rve['test_auc']:.4f}, Base: {rve['test_base_rate']:.1%})")
else:
    print(f"  vol_regime: No rv_expand models found")

print("\n🎯 PRIORITY 2 - Different horizons of validated models:")

# low_range_atr at different horizons
low_range_other_h = range_atr[(range_atr['label'] == 'low_range_atr') & (range_atr['horizon'] != 3)].nlargest(2, 'test_auc')
if len(low_range_other_h) > 0:
    print(f"\n  low_range_atr:")
    for idx, row in low_range_other_h.iterrows():
        print(f"    H={row['horizon']:<3} AUC={row['test_auc']:.4f} Base={row['test_base_rate']:5.1%}")

# rv_compress_q30 at different horizons
rv_compress_other_h = vol_regime[(vol_regime['label'] == 'rv_compress_q30') & (vol_regime['horizon'] != 30)].nlargest(2, 'test_auc')
if len(rv_compress_other_h) > 0:
    print(f"\n  rv_compress_q30:")
    for idx, row in rv_compress_other_h.iterrows():
        print(f"    H={row['horizon']:<3} AUC={row['test_auc']:.4f} Base={row['test_base_rate']:5.1%}")

print("\n" + "="*80)
print("FINAL RECOMMENDATION")
print("="*80)
print("""
Focus on validating:

1. high_range_atr (best horizon from range_atr family)
   - Predicts WIDE ranges (opposite of low_range)
   - Useful for volatility expansion strategies
   - Complements low_range_atr model

2. rv_expand_q70 or rv_expand_abs (best from vol_regime family)
   - Predicts volatility EXPANSION (opposite of rv_compress)
   - Useful for long volatility strategies
   - Complements rv_compress model

These give you a complete toolkit:
- Tight range + Vol compression = Sell premium
- Wide range + Vol expansion = Buy premium/delta
""")
