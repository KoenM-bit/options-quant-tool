"""
Analyze supersweep results to find best models
"""
import pandas as pd
import numpy as np

# Load results
df = pd.read_csv("ML/supersweep_staged_results.csv")

print("="*80)
print("SUPERSWEEP RESULTS ANALYSIS")
print("="*80)
print(f"\nTotal models trained: {len(df)}")
print(f"Unique horizons: {sorted(df['horizon'].unique())}")
print(f"Unique label families: {sorted(df['label_family'].unique())}")

print("\n" + "="*80)
print("TOP 20 MODELS BY TEST AUC")
print("="*80)

top20 = df.sort_values("test_auc", ascending=False).head(20)
print(top20[["horizon", "label_family", "label", "featureset", 
             "val_auc", "test_auc", "test_base_rate", "p_spread"]].to_string(index=False))

print("\n" + "="*80)
print("TOP 20 MODELS BY VAL AUC (no test peeking)")
print("="*80)

top20_val = df.sort_values("val_auc", ascending=False).head(20)
print(top20_val[["horizon", "label_family", "label", "featureset", 
                 "val_auc", "test_auc", "test_base_rate", "p_spread"]].to_string(index=False))

print("\n" + "="*80)
print("BEST MODEL PER (HORIZON, LABEL_FAMILY)")
print("="*80)

# Best per family/horizon by VAL AUC (proper selection)
best_per_group = df.loc[df.groupby(["horizon", "label_family"])["val_auc"].idxmax()]
pivot = best_per_group.pivot_table(
    index="horizon", 
    columns="label_family", 
    values="test_auc", 
    aggfunc="first"
)
print("\nTest AUC matrix:")
print(pivot.round(4).to_string())

print("\n" + "="*80)
print("MODELS WITH TEST AUC >= 0.80")
print("="*80)

high_performers = df[df["test_auc"] >= 0.80].sort_values("test_auc", ascending=False)
if len(high_performers) > 0:
    print(high_performers[["horizon", "label_family", "label", "featureset", 
                           "val_auc", "test_auc", "test_base_rate", "p_spread"]].to_string(index=False))
else:
    print("No models with test AUC >= 0.80")

print("\n" + "="*80)
print("MODELS WITH TEST AUC >= 0.75")
print("="*80)

good_performers = df[df["test_auc"] >= 0.75].sort_values("test_auc", ascending=False)
print(f"\nFound {len(good_performers)} models")
print(good_performers[["horizon", "label_family", "label", "featureset", 
                       "val_auc", "test_auc", "test_base_rate", "p_spread"]].to_string(index=False))

print("\n" + "="*80)
print("LABEL FAMILY PERFORMANCE SUMMARY")
print("="*80)

family_stats = df.groupby("label_family").agg({
    "test_auc": ["mean", "max", "count"],
    "val_auc": ["mean", "max"]
}).round(4)
print(family_stats)

print("\n" + "="*80)
print("HORIZON PERFORMANCE SUMMARY")
print("="*80)

horizon_stats = df.groupby("horizon").agg({
    "test_auc": ["mean", "max", "count"],
    "val_auc": ["mean", "max"]
}).round(4)
print(horizon_stats)

print("\n" + "="*80)
print("RECOMMENDATIONS FOR VALIDATION")
print("="*80)

# Find top 5 candidates for validation
candidates = df[df["test_auc"] >= 0.70].sort_values("val_auc", ascending=False).head(10)

print("\nTop 10 candidates (Test AUC >= 0.70, sorted by Val AUC):\n")
for i, (idx, row) in enumerate(candidates.iterrows(), 1):
    print(f"{i}. H={row['horizon']} | {row['label_family']:20s} | {row['label']:30s}")
    print(f"   Val AUC: {row['val_auc']:.4f} | Test AUC: {row['test_auc']:.4f} | Base: {row['test_base_rate']:.1%}")
    print(f"   Featureset: {row['featureset']}")
    
    # Assess if worth validating
    if row['test_auc'] >= 0.80:
        status = "🌟 EXCELLENT - High priority for validation"
    elif row['test_auc'] >= 0.75:
        status = "✅ GOOD - Worth validating"
    elif row['test_auc'] >= 0.70:
        status = "⚠️  MODERATE - Consider validating"
    else:
        status = "❌ WEAK - Skip"
    print(f"   {status}")
    print()

print("\n" + "="*80)
print("NEXT STEPS")
print("="*80)
print("""
Based on results, should focus validation on:
1. Models with Test AUC >= 0.75
2. Diverse label families (not just one type)
3. Different horizons (short, medium, long-term)
4. Good Val/Test AUC consistency (no overfitting)

Run leakage tests on top candidates before deploying!
""")
