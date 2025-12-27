#!/usr/bin/env python3
"""
Ablation Test: Train without close_pos_end to see if other features contribute.

This will show if close_pos_end is doing all the work or if other features help.
"""

import sys
from pathlib import Path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

import pandas as pd
from datetime import date
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.pipeline import Pipeline
from sklearn.metrics import accuracy_score, f1_score, roc_auc_score

# Load data
df = pd.read_parquet('data/ml_datasets/accum_distrib_events.parquet')

# Filter to valid events
df_valid = df[df['label_valid'] == 1].copy()
df_valid = df_valid[df_valid['label_generic'].isin(['UP_RESOLVE', 'DOWN_RESOLVE'])]
df_valid['target'] = (df_valid['label_generic'] == 'UP_RESOLVE').astype(int)
df_valid['t_end_date'] = pd.to_datetime(df_valid['t_end']).dt.date

# Time-based splits
train_cutoff = date(2024, 12, 31)
val_cutoff = date(2025, 8, 31)
test_start = date(2025, 9, 1)

train = df_valid[df_valid['t_end_date'] <= train_cutoff]
val = df_valid[(df_valid['t_end_date'] > train_cutoff) & (df_valid['t_end_date'] <= val_cutoff)]
test = df_valid[df_valid['t_end_date'] >= test_start]

# Features WITHOUT close_pos_end
features_without = [
    'range_width_pct', 'atr_pct_mean', 'atr_pct_last', 'vol_ratio', 'vol_compression',
    'rel_vol_mean', 'rel_vol_slope', 'up_vol_share', 'effort_vs_result',
    'clv_mean',  # Removed: close_pos_end
    'pct_top_quartile', 'pct_bottom_quartile',
    'max_consecutive_up', 'max_consecutive_down',
    'prior_dir', 'R_pre'
]

# Features WITH close_pos_end
features_with = [
    'range_width_pct', 'atr_pct_mean', 'atr_pct_last', 'vol_ratio', 'vol_compression',
    'rel_vol_mean', 'rel_vol_slope', 'up_vol_share', 'effort_vs_result',
    'clv_mean', 'close_pos_end',
    'pct_top_quartile', 'pct_bottom_quartile',
    'max_consecutive_up', 'max_consecutive_down',
    'prior_dir', 'R_pre'
]

def train_and_eval(features, name):
    """Train and evaluate model with given features."""
    print(f"\n{'='*80}")
    print(f"🧪 {name} ({len(features)} features)")
    print(f"{'='*80}")
    
    # Prepare data
    X_train = train[features].fillna(0)
    y_train = train['target']
    X_val = val[features].fillna(0)
    y_val = val['target']
    X_test = test[features].fillna(0)
    y_test = test['target']
    
    # Train model
    model = HistGradientBoostingClassifier(
        random_state=42,
        max_iter=300,
        learning_rate=0.05,
        max_depth=5,
        early_stopping=False,
        verbose=0
    )
    
    print(f"⏳ Training...")
    model.fit(X_train, y_train)
    
    # Evaluate on all splits
    for split_name, X, y in [('Train', X_train, y_train), 
                              ('Val', X_val, y_val), 
                              ('Test', X_test, y_test)]:
        y_pred = model.predict(X)
        y_proba = model.predict_proba(X)[:, 1]
        
        acc = accuracy_score(y, y_pred)
        f1 = f1_score(y, y_pred)
        auc = roc_auc_score(y, y_proba)
        
        print(f"  {split_name:5s}: Acc={acc:.4f}, F1={f1:.4f}, AUC={auc:.4f}")
    
    return model

# Run both experiments
print("\n" + "="*80)
print("🔬 ABLATION TEST: Impact of close_pos_end")
print("="*80)

model_without = train_and_eval(features_without, "WITHOUT close_pos_end")
model_with = train_and_eval(features_with, "WITH close_pos_end")

print("\n" + "="*80)
print("📊 SUMMARY")
print("="*80)
print("If performance drops significantly without close_pos_end:")
print("  → It's doing most of the work (position at breakout is key)")
print("\nIf performance stays similar:")
print("  → Other features compensate (pattern matters beyond just position)")
