"""
Analyze ALL label families, not just those >= 0.70 AUC
"""

import pandas as pd
import numpy as np

# Load results
df = pd.read_csv("ML/supersweep_staged_results.csv")

print("="*80)
print("COMPREHENSIVE ANALYSIS - ALL LABEL FAMILIES")
print("="*80)

# Get top 3 per family regardless of AUC
print("\nTOP 3 MODELS PER LABEL FAMILY (sorted by test_auc)")
print("="*80)

top_by_family = (df
                 .sort_values(['label_family', 'test_auc'], ascending=[True, False])
                 .groupby('label_family')
                 .head(3))

for family in sorted(top_by_family['label_family'].unique()):
    family_models = top_by_family[top_by_family['label_family'] == family]
    best_auc = family_models['test_auc'].max()
    
    status = "✅ STRONG" if best_auc >= 0.75 else "⚠️  MODERATE" if best_auc >= 0.65 else "❌ WEAK"
    
    print(f"\n{family.upper()} - {status} (best: {best_auc:.4f})")
    print("-" * 80)
    
    for idx, row in family_models.iterrows():
        print(f"  {row['label']:30s} H={row['horizon']:<3} | AUC: {row['test_auc']:.4f} | Base: {row['test_base_rate']:5.1%} | {row['featureset']}")

# Summary statistics per family
print("\n" + "="*80)
print("SUMMARY BY FAMILY")
print("="*80)

summary = (df.groupby('label_family')
           .agg({
               'test_auc': ['count', 'mean', 'max', 'std'],
               'test_base_rate': ['mean', 'std']
           })
           .round(4))

print(summary)

# Recommend which families to focus on
print("\n" + "="*80)
print("RECOMMENDATIONS")
print("="*80)

strong_families = df.groupby('label_family')['test_auc'].max() >= 0.75
moderate_families = (df.groupby('label_family')['test_auc'].max() >= 0.65) & (df.groupby('label_family')['test_auc'].max() < 0.75)
weak_families = df.groupby('label_family')['test_auc'].max() < 0.65

print("\n✅ STRONG FAMILIES (max AUC >= 0.75) - Focus here:")
for fam in strong_families[strong_families].index:
    max_auc = df[df['label_family'] == fam]['test_auc'].max()
    print(f"  {fam:20s} (max AUC: {max_auc:.4f})")

print("\n⚠️  MODERATE FAMILIES (max AUC 0.65-0.75) - Consider if needed:")
for fam in moderate_families[moderate_families].index:
    max_auc = df[df['label_family'] == fam]['test_auc'].max()
    print(f"  {fam:20s} (max AUC: {max_auc:.4f})")

print("\n❌ WEAK FAMILIES (max AUC < 0.65) - Skip for now:")
for fam in weak_families[weak_families].index:
    max_auc = df[df['label_family'] == fam]['test_auc'].max()
    print(f"  {fam:20s} (max AUC: {max_auc:.4f})")

# Find diverse models to validate (different families, different horizons)
print("\n" + "="*80)
print("DIVERSE MODEL PORTFOLIO FOR VALIDATION")
print("="*80)

# Get best model from each family
best_per_family = (df.sort_values('test_auc', ascending=False)
                   .groupby('label_family')
                   .first()
                   .reset_index()
                   .sort_values('test_auc', ascending=False))

print("\nBest model from each family:")
for idx, row in best_per_family.iterrows():
    status = "✅" if row['test_auc'] >= 0.75 else "⚠️ " if row['test_auc'] >= 0.65 else "❌"
    validated = "✓ VALIDATED" if (row['label_family'] == 'range_atr' and 'low_range' in row['label']) or (row['label_family'] == 'vol_regime' and 'rv_compress' in row['label']) else ""
    print(f"{status} {row['label_family']:20s} {row['label']:30s} H={row['horizon']:<3} AUC={row['test_auc']:.4f} {validated}")

print("\n" + "="*80)
print("NEXT ACTIONS")
print("="*80)
print("""
Based on this analysis:

1. ✅ ALREADY VALIDATED:
   - low_range_atr H=3 (range_atr)
   - rv_compress_q30 H=30 (vol_regime) 

2. 🎯 NO OTHER FAMILIES MEET THRESHOLD:
   - Only range_atr and vol_regime have models with AUC >= 0.75
   - Other families have weaker performance

3. 💡 OPTIONS:
   
   Option A - STICK WITH WINNERS:
   - Focus on improving/variants of range_atr and vol_regime
   - Look at different horizons within these families
   - These are proven to work well
   
   Option B - EXPLORE MODERATE FAMILIES:
   - Validate best models from 0.65-0.75 AUC families
   - May find useful signal even if weaker
   - Useful for ensemble/diversification
   
   Option C - INVESTIGATE WHY OTHERS FAILED:
   - Check if those labels are just hard to predict
   - Try different feature sets
   - May need more sophisticated features

RECOMMENDATION: Option A - Stick with winners
- Look at high_range_atr (opposite of low_range_atr)
- Look at rv_expand (opposite of rv_compress)
- These leverage our proven predictive families
""")
