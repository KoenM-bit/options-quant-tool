"""
Analyze supersweep results and find top 2-3 models per label family
"""

import pandas as pd
import numpy as np

# Load results
df = pd.read_csv("ML/supersweep_staged_results.csv")

print("="*80)
print("SUPERSWEEP RESULTS ANALYSIS - TOP MODELS BY FAMILY")
print("="*80)
print(f"\nTotal models: {len(df)}")
print(f"Unique label families: {df['label_family'].nunique()}")
print(f"Label families: {sorted(df['label_family'].unique())}")

# Filter for good models (test AUC >= 0.70)
good_models = df[df['test_auc'] >= 0.70].copy()
print(f"\nModels with test AUC >= 0.70: {len(good_models)}")

# Get top 3 per family by test AUC
print("\n" + "="*80)
print("TOP 3 MODELS PER LABEL FAMILY (by test_auc)")
print("="*80)

top_by_family = (good_models
                 .sort_values(['label_family', 'test_auc'], ascending=[True, False])
                 .groupby('label_family')
                 .head(3))

for family in sorted(top_by_family['label_family'].unique()):
    family_models = top_by_family[top_by_family['label_family'] == family]
    
    print(f"\n{family.upper()} (n={len(family_models)})")
    print("-" * 80)
    
    for idx, row in family_models.iterrows():
        print(f"\n  {row['label']} H={row['horizon']}")
        print(f"    Test AUC: {row['test_auc']:.4f} | Val AUC: {row['val_auc']:.4f}")
        print(f"    Test Brier: {row['test_brier']:.4f} | Val Brier: {row['val_brier']:.4f}")
        print(f"    Base rate: {row['test_base_rate']:.1%}")
        print(f"    Feature set: {row['featureset']}")
        print(f"    Params: {row['best_params']}")

# Summary table
print("\n" + "="*80)
print("SUMMARY TABLE")
print("="*80)

summary = (top_by_family
           .groupby('label_family')
           .agg({
               'test_auc': ['count', 'mean', 'max'],
               'test_base_rate': 'mean'
           })
           .round(4))

print(summary)

# Recommend top candidates for validation
print("\n" + "="*80)
print("RECOMMENDED CANDIDATES FOR VALIDATION")
print("="*80)

# Criteria: test AUC >= 0.75, reasonable base rate (15-85%)
candidates = top_by_family[
    (top_by_family['test_auc'] >= 0.75) & 
    (top_by_family['test_base_rate'] >= 0.15) &
    (top_by_family['test_base_rate'] <= 0.85)
].sort_values('test_auc', ascending=False)

print(f"\nFound {len(candidates)} strong candidates:")
print("\n" + candidates[['label_family', 'label', 'horizon', 'test_auc', 'test_base_rate', 'featureset']].to_string(index=False))

# Export top candidates
candidates.to_csv("ML/top_candidates_for_validation.csv", index=False)
print(f"\n✓ Saved top candidates to: ML/top_candidates_for_validation.csv")

# Show what we already validated
print("\n" + "="*80)
print("ALREADY VALIDATED")
print("="*80)
print("  ✅ low_range_atr H=3 (range_atr family)")
print("  ✅ rv_compress_q30 H=30 (vol_regime family)")

print("\n" + "="*80)
print("NEXT STEPS")
print("="*80)
print("\nRecommend validating top 1-2 from each family:")

for family in sorted(candidates['label_family'].unique()):
    family_top = candidates[candidates['label_family'] == family].head(2)
    print(f"\n{family.upper()}:")
    for idx, row in family_top.iterrows():
        status = "✅ VALIDATED" if (family == "range_atr" and "low_range" in row['label']) or (family == "vol_regime" and "rv_compress" in row['label']) else "⏳ TODO"
        print(f"  {status} {row['label']} H={row['horizon']} (AUC: {row['test_auc']:.4f})")
