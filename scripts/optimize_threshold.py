#!/usr/bin/env python3
"""
Optimize Decision Threshold

Sweeps probability thresholds on validation set to find optimal cutoff
for trading signals based on precision, coverage, and expected value.

Usage:
    python scripts/optimize_threshold.py --bundle ML/production/v20251223_212702
"""

import sys
from pathlib import Path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

import argparse
import json
import joblib
import numpy as np
import pandas as pd
from datetime import date
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix


def optimize_threshold(bundle_dir: str, use_calibrated: bool = True):
    """
    Sweep thresholds on validation set to optimize decision rule.
    
    Args:
        bundle_dir: Path to production bundle
        use_calibrated: Whether to use calibrated probabilities
    """
    bundle_dir = Path(bundle_dir)
    
    print(f"🎯 Optimizing decision threshold")
    print(f"   Bundle: {bundle_dir.name}")
    print(f"   Using: {'calibrated' if use_calibrated else 'uncalibrated'} probabilities")
    
    # Load model
    if use_calibrated and (bundle_dir / 'model_calibrated.pkl').exists():
        cal_bundle = joblib.load(bundle_dir / 'model_calibrated.pkl')
        model = cal_bundle['model']
        calibrator = cal_bundle['calibrator']
        print(f"✅ Loaded calibrated model")
    else:
        model = joblib.load(bundle_dir / 'model.pkl')
        calibrator = None
        print(f"✅ Loaded uncalibrated model")
    
    with open(bundle_dir / 'features.json', 'r') as f:
        features = json.load(f)['feature_names']
    
    # Load dataset
    dataset_path = project_root / 'data' / 'ml_datasets' / 'accum_distrib_events.parquet'
    df = pd.read_parquet(dataset_path)
    
    # Filter and split
    df_valid = df[df['label_valid'] == 1].copy()
    df_valid = df_valid[df_valid['label_generic'].isin(['UP_RESOLVE', 'DOWN_RESOLVE'])]
    df_valid['target'] = (df_valid['label_generic'] == 'UP_RESOLVE').astype(int)
    df_valid['t_end_date'] = pd.to_datetime(df_valid['t_end']).dt.date
    
    train_cutoff = date(2024, 12, 31)
    val_cutoff = date(2025, 8, 31)
    test_start = date(2025, 9, 1)
    
    val = df_valid[(df_valid['t_end_date'] > train_cutoff) & (df_valid['t_end_date'] <= val_cutoff)]
    test = df_valid[df_valid['t_end_date'] >= test_start]
    
    X_val = val[features].fillna(0)
    y_val = val['target']
    X_test = test[features].fillna(0)
    y_test = test['target']
    
    # Get probabilities
    y_val_proba = model.predict_proba(X_val)[:, 1]
    y_test_proba = model.predict_proba(X_test)[:, 1]
    
    if calibrator is not None:
        y_val_proba = calibrator.predict(y_val_proba)
        y_test_proba = calibrator.predict(y_test_proba)
    
    print(f"✅ Loaded data: {len(val)} val, {len(test)} test")
    
    # ===== Sweep thresholds on validation =====
    thresholds = np.arange(0.50, 0.91, 0.01)  # 50% to 90% in 1% steps
    
    results = []
    
    print(f"\n🔍 Sweeping thresholds from 0.50 to 0.90...")
    
    baseline_accuracy = y_val.mean()  # Random accuracy
    
    for thr in thresholds:
        # Long signals: P(UP) >= thr
        # Short signals: P(UP) <= (1 - thr)
        # No-trade: thr < P(UP) < (1 - thr)
        
        long_mask = y_val_proba >= thr
        short_mask = y_val_proba <= (1 - thr)
        trade_mask = long_mask | short_mask
        
        if trade_mask.sum() == 0:
            continue
        
        # Predictions: 1 for long, 0 for short
        y_pred = np.zeros(len(y_val))
        y_pred[long_mask] = 1
        y_pred[short_mask] = 0
        
        # Metrics on traded samples only
        y_true_traded = y_val[trade_mask]
        y_pred_traded = y_pred[trade_mask]
        
        coverage = trade_mask.mean()  # % signals taken
        accuracy = accuracy_score(y_true_traded, y_pred_traded)
        precision_up = precision_score(y_true_traded, y_pred_traded, zero_division=0)
        recall_up = recall_score(y_true_traded, y_pred_traded, zero_division=0)
        
        # Confusion matrix
        cm = confusion_matrix(y_true_traded, y_pred_traded)
        tn, fp, fn, tp = cm.ravel() if cm.size == 4 else (0, 0, 0, 0)
        
        # Win rates
        long_win_rate = tp / (tp + fp) if (tp + fp) > 0 else 0
        short_win_rate = tn / (tn + fn) if (tn + fn) > 0 else 0
        
        # Simple expected value proxy (assumes 1:1 R:R)
        # E = P(win) * 1R - P(loss) * 1R = 2*P(win) - 1
        ev_long = 2 * long_win_rate - 1 if (tp + fp) > 0 else 0
        ev_short = 2 * short_win_rate - 1 if (tn + fn) > 0 else 0
        ev_overall = coverage * (long_mask[trade_mask].mean() * ev_long + short_mask[trade_mask].mean() * ev_short)
        
        results.append({
            'threshold': thr,
            'coverage': coverage,
            'n_trades': int(trade_mask.sum()),
            'n_long': int(long_mask.sum()),
            'n_short': int(short_mask.sum()),
            'accuracy': accuracy,
            'precision_up': precision_up,
            'recall_up': recall_up,
            'long_win_rate': long_win_rate,
            'short_win_rate': short_win_rate,
            'ev_long': ev_long,
            'ev_short': ev_short,
            'ev_overall': ev_overall,
            'tp': int(tp),
            'fp': int(fp),
            'tn': int(tn),
            'fn': int(fn)
        })
    
    results_df = pd.DataFrame(results)
    
    # ===== Find optimal thresholds =====
    print(f"\n📊 Top 5 thresholds by metric:")
    
    # By coverage (most trades)
    print(f"\n  By Coverage (most trades):")
    top_coverage = results_df.nlargest(5, 'coverage')[['threshold', 'coverage', 'accuracy', 'ev_overall']]
    for _, row in top_coverage.iterrows():
        print(f"    thr={row['threshold']:.2f}: coverage={row['coverage']:.1%}, acc={row['accuracy']:.1%}, EV={row['ev_overall']:+.3f}")
    
    # By accuracy
    print(f"\n  By Accuracy (win rate):")
    top_acc = results_df.nlargest(5, 'accuracy')[['threshold', 'coverage', 'accuracy', 'ev_overall']]
    for _, row in top_acc.iterrows():
        print(f"    thr={row['threshold']:.2f}: coverage={row['coverage']:.1%}, acc={row['accuracy']:.1%}, EV={row['ev_overall']:+.3f}")
    
    # By expected value
    print(f"\n  By Expected Value (1:1 R:R assumption):")
    top_ev = results_df.nlargest(5, 'ev_overall')[['threshold', 'coverage', 'accuracy', 'ev_overall']]
    for _, row in top_ev.iterrows():
        print(f"    thr={row['threshold']:.2f}: coverage={row['coverage']:.1%}, acc={row['accuracy']:.1%}, EV={row['ev_overall']:+.3f}")
    
    # Recommended threshold (balance coverage + accuracy)
    # Find threshold with accuracy > 65% and highest coverage
    viable = results_df[results_df['accuracy'] >= 0.65]
    if len(viable) > 0:
        recommended_idx = viable['coverage'].idxmax()
        recommended = results_df.loc[recommended_idx]
    else:
        # Fallback: highest EV
        recommended_idx = results_df['ev_overall'].idxmax()
        recommended = results_df.loc[recommended_idx]
    
    print(f"\n⭐ RECOMMENDED THRESHOLD: {recommended['threshold']:.2f}")
    print(f"   Coverage:   {recommended['coverage']:.1%} ({recommended['n_trades']} trades)")
    print(f"   Accuracy:   {recommended['accuracy']:.1%}")
    print(f"   Long signals:  {recommended['n_long']} (win rate: {recommended['long_win_rate']:.1%})")
    print(f"   Short signals: {recommended['n_short']} (win rate: {recommended['short_win_rate']:.1%})")
    print(f"   EV (1:1 R:R):  {recommended['ev_overall']:+.3f}")
    
    # ===== Apply to test set =====
    print(f"\n🧪 Applying threshold={recommended['threshold']:.2f} to TEST set:")
    
    thr_opt = recommended['threshold']
    long_mask_test = y_test_proba >= thr_opt
    short_mask_test = y_test_proba <= (1 - thr_opt)
    trade_mask_test = long_mask_test | short_mask_test
    
    y_pred_test = np.zeros(len(y_test))
    y_pred_test[long_mask_test] = 1
    y_pred_test[short_mask_test] = 0
    
    y_true_traded_test = y_test[trade_mask_test]
    y_pred_traded_test = y_pred_test[trade_mask_test]
    
    test_coverage = trade_mask_test.mean()
    test_accuracy = accuracy_score(y_true_traded_test, y_pred_traded_test)
    
    cm_test = confusion_matrix(y_true_traded_test, y_pred_traded_test)
    tn_t, fp_t, fn_t, tp_t = cm_test.ravel()
    
    test_long_wr = tp_t / (tp_t + fp_t) if (tp_t + fp_t) > 0 else 0
    test_short_wr = tn_t / (tn_t + fn_t) if (tn_t + fn_t) > 0 else 0
    
    print(f"   Coverage:   {test_coverage:.1%} ({trade_mask_test.sum()} trades)")
    print(f"   Accuracy:   {test_accuracy:.1%}")
    print(f"   Long:  {long_mask_test.sum()} trades, {test_long_wr:.1%} win rate")
    print(f"   Short: {short_mask_test.sum()} trades, {test_short_wr:.1%} win rate")
    print(f"   Confusion matrix:")
    print(f"     {cm_test[0].tolist()}  <- DOWN_RESOLVE")
    print(f"     {cm_test[1].tolist()}  <- UP_RESOLVE")
    
    # ===== Save threshold config =====
    threshold_config = {
        "recommended_threshold": float(recommended['threshold']),
        "decision_rule": {
            "long": f"P(UP) >= {recommended['threshold']:.2f}",
            "short": f"P(UP) <= {1 - recommended['threshold']:.2f}",
            "no_trade": f"{1 - recommended['threshold']:.2f} < P(UP) < {recommended['threshold']:.2f}"
        },
        "validation_performance": {
            "coverage": float(recommended['coverage']),
            "n_trades": int(recommended['n_trades']),
            "accuracy": float(recommended['accuracy']),
            "long_win_rate": float(recommended['long_win_rate']),
            "short_win_rate": float(recommended['short_win_rate']),
            "expected_value_1R": float(recommended['ev_overall'])
        },
        "test_performance": {
            "coverage": float(test_coverage),
            "n_trades": int(trade_mask_test.sum()),
            "accuracy": float(test_accuracy),
            "long_win_rate": float(test_long_wr),
            "short_win_rate": float(test_short_wr)
        },
        "sweep_results": results_df.to_dict('records'),
        "notes": [
            "Threshold optimized on validation set",
            "EV assumes 1:1 risk:reward ratio (simplified)",
            "Actual EV depends on entry/exit rules and slippage",
            "Use backtest to validate with realistic trading costs"
        ]
    }
    
    threshold_path = bundle_dir / 'threshold_config.json'
    with open(threshold_path, 'w') as f:
        json.dump(threshold_config, f, indent=2)
    print(f"\n💾 Saved threshold config: {threshold_path.name}")
    
    # ===== Save sweep results CSV =====
    results_csv = bundle_dir / 'threshold_sweep.csv'
    results_df.to_csv(results_csv, index=False)
    print(f"💾 Saved sweep results: {results_csv.name}")
    
    print(f"\n✅ Threshold optimization complete!")


def main():
    parser = argparse.ArgumentParser(
        description="Optimize decision threshold on validation set"
    )
    
    parser.add_argument(
        '--bundle',
        type=str,
        required=True,
        help='Path to production bundle directory'
    )
    
    parser.add_argument(
        '--uncalibrated',
        action='store_true',
        help='Use uncalibrated probabilities'
    )
    
    args = parser.parse_args()
    
    optimize_threshold(args.bundle, use_calibrated=not args.uncalibrated)


if __name__ == "__main__":
    main()
