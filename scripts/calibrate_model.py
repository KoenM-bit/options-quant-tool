#!/usr/bin/env python3
"""
Calibrate Model Probabilities

Fits a calibrator on validation set to ensure predicted probabilities
match observed frequencies (e.g., when model says 80%, it's actually 80% likely).

Usage:
    python scripts/calibrate_model.py --bundle ML/production/v20251223_212702
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
from sklearn.calibration import calibration_curve
from sklearn.isotonic import IsotonicRegression
from sklearn.metrics import brier_score_loss, roc_auc_score, accuracy_score


def compute_calibration_curve(y_true, y_prob, n_bins=10):
    """Compute calibration curve (reliability diagram)."""
    bins = np.linspace(0, 1, n_bins + 1)
    bin_indices = np.digitize(y_prob, bins) - 1
    bin_indices = np.clip(bin_indices, 0, n_bins - 1)
    
    bin_sums = np.zeros(n_bins)
    bin_counts = np.zeros(n_bins)
    bin_true = np.zeros(n_bins)
    
    for i in range(len(y_true)):
        bin_idx = bin_indices[i]
        bin_sums[bin_idx] += y_prob[i]
        bin_counts[bin_idx] += 1
        bin_true[bin_idx] += y_true[i]
    
    # Avoid division by zero
    mask = bin_counts > 0
    bin_pred = np.zeros(n_bins)
    bin_pred[mask] = bin_sums[mask] / bin_counts[mask]
    
    bin_observed = np.zeros(n_bins)
    bin_observed[mask] = bin_true[mask] / bin_counts[mask]
    
    return bin_pred, bin_observed, bin_counts


def calibrate_model(bundle_dir: str):
    """
    Calibrate model probabilities using validation set.
    
    Args:
        bundle_dir: Path to production bundle directory
    """
    bundle_dir = Path(bundle_dir)
    
    if not bundle_dir.exists():
        raise FileNotFoundError(f"Bundle not found: {bundle_dir}")
    
    print(f"📊 Calibrating model from {bundle_dir.name}")
    
    # Load bundle artifacts
    model = joblib.load(bundle_dir / 'model.pkl')
    
    with open(bundle_dir / 'features.json', 'r') as f:
        features_schema = json.load(f)
    
    features = features_schema['feature_names']
    
    # Load dataset
    dataset_path = project_root / 'data' / 'ml_datasets' / 'accum_distrib_events.parquet'
    df = pd.read_parquet(dataset_path)
    
    # Filter to valid events
    df_valid = df[df['label_valid'] == 1].copy()
    df_valid = df_valid[df_valid['label_generic'].isin(['UP_RESOLVE', 'DOWN_RESOLVE'])]
    df_valid['target'] = (df_valid['label_generic'] == 'UP_RESOLVE').astype(int)
    df_valid['t_end_date'] = pd.to_datetime(df_valid['t_end']).dt.date
    
    # Split data
    train_cutoff = date(2024, 12, 31)
    val_cutoff = date(2025, 8, 31)
    test_start = date(2025, 9, 1)
    
    train = df_valid[df_valid['t_end_date'] <= train_cutoff]
    val = df_valid[(df_valid['t_end_date'] > train_cutoff) & (df_valid['t_end_date'] <= val_cutoff)]
    test = df_valid[df_valid['t_end_date'] >= test_start]
    
    X_train = train[features].fillna(0)
    y_train = train['target']
    X_val = val[features].fillna(0)
    y_val = val['target']
    X_test = test[features].fillna(0)
    y_test = test['target']
    
    print(f"✅ Loaded data: {len(train)} train, {len(val)} val, {len(test)} test")
    
    # ===== Get uncalibrated predictions =====
    y_val_proba_uncal = model.predict_proba(X_val)[:, 1]
    y_test_proba_uncal = model.predict_proba(X_test)[:, 1]
    
    uncal_val_brier = brier_score_loss(y_val, y_val_proba_uncal)
    uncal_test_brier = brier_score_loss(y_test, y_test_proba_uncal)
    uncal_val_auc = roc_auc_score(y_val, y_val_proba_uncal)
    uncal_test_auc = roc_auc_score(y_test, y_test_proba_uncal)
    
    print(f"\n📈 Uncalibrated Performance:")
    print(f"   Val:  Brier={uncal_val_brier:.4f}, AUC={uncal_val_auc:.4f}")
    print(f"   Test: Brier={uncal_test_brier:.4f}, AUC={uncal_test_auc:.4f}")
    
    # ===== Calibrate using validation set =====
    print(f"\n🔧 Fitting calibrator on validation set...")
    
    # Extract the base model from pipeline (if it's a pipeline)
    if hasattr(model, 'named_steps'):
        base_model = model  # Keep the whole pipeline
    else:
        base_model = model
    
    # Use isotonic regression (non-parametric, best for tree models)
    # Since we're using cv='prefit', we need to use the correct API
    from sklearn.calibration import calibration_curve
    
    # Manual calibration: fit isotonic regression on val predictions
    from sklearn.isotonic import IsotonicRegression
    
    calibrator = IsotonicRegression(out_of_bounds='clip')
    calibrator.fit(y_val_proba_uncal, y_val)
    
    print(f"✅ Isotonic calibrator fitted on {len(y_val)} validation samples")
    
    # ===== Get calibrated predictions =====
    # Apply calibrator to uncalibrated probabilities
    y_val_proba_cal = calibrator.predict(y_val_proba_uncal)
    y_test_proba_cal = calibrator.predict(y_test_proba_uncal)
    
    cal_val_brier = brier_score_loss(y_val, y_val_proba_cal)
    cal_test_brier = brier_score_loss(y_test, y_test_proba_cal)
    cal_val_auc = roc_auc_score(y_val, y_val_proba_cal)
    cal_test_auc = roc_auc_score(y_test, y_test_proba_cal)
    
    print(f"\n📈 Calibrated Performance:")
    print(f"   Val:  Brier={cal_val_brier:.4f}, AUC={cal_val_auc:.4f}")
    print(f"   Test: Brier={cal_test_brier:.4f}, AUC={cal_test_auc:.4f}")
    
    print(f"\n📊 Improvement:")
    print(f"   Val Brier:  {uncal_val_brier:.4f} → {cal_val_brier:.4f} ({cal_val_brier - uncal_val_brier:+.4f})")
    print(f"   Test Brier: {uncal_test_brier:.4f} → {cal_test_brier:.4f} ({cal_test_brier - uncal_test_brier:+.4f})")
    print(f"   (Lower Brier is better)")
    
    # ===== Compute calibration curves =====
    print(f"\n📐 Computing reliability curves...")
    
    val_pred_uncal, val_obs_uncal, val_counts_uncal = compute_calibration_curve(y_val.values, y_val_proba_uncal, n_bins=10)
    val_pred_cal, val_obs_cal, val_counts_cal = compute_calibration_curve(y_val.values, y_val_proba_cal, n_bins=10)
    
    test_pred_uncal, test_obs_uncal, test_counts_uncal = compute_calibration_curve(y_test.values, y_test_proba_uncal, n_bins=10)
    test_pred_cal, test_obs_cal, test_counts_cal = compute_calibration_curve(y_test.values, y_test_proba_cal, n_bins=10)
    
    # ===== Save calibrated model (model + calibrator) =====
    calibrated_bundle = {
        'model': model,
        'calibrator': calibrator,
        'method': 'isotonic'
    }
    
    calibrated_model_path = bundle_dir / 'model_calibrated.pkl'
    joblib.dump(calibrated_bundle, calibrated_model_path)
    print(f"\n💾 Saved calibrated model bundle: {calibrated_model_path.name}")
    
    # ===== Save calibration metrics =====
    calibration_metrics = {
        "calibration_method": "isotonic",
        "calibrator_fitted_on": "validation_set",
        "uncalibrated": {
            "val": {
                "brier_score": float(uncal_val_brier),
                "roc_auc": float(uncal_val_auc)
            },
            "test": {
                "brier_score": float(uncal_test_brier),
                "roc_auc": float(uncal_test_auc)
            }
        },
        "calibrated": {
            "val": {
                "brier_score": float(cal_val_brier),
                "roc_auc": float(cal_val_auc)
            },
            "test": {
                "brier_score": float(cal_test_brier),
                "roc_auc": float(cal_test_auc)
            }
        },
        "improvement": {
            "val_brier_delta": float(cal_val_brier - uncal_val_brier),
            "test_brier_delta": float(cal_test_brier - uncal_test_brier)
        },
        "reliability_curves": {
            "val_uncalibrated": {
                "predicted": val_pred_uncal.tolist(),
                "observed": val_obs_uncal.tolist(),
                "counts": val_counts_uncal.tolist()
            },
            "val_calibrated": {
                "predicted": val_pred_cal.tolist(),
                "observed": val_obs_cal.tolist(),
                "counts": val_counts_cal.tolist()
            },
            "test_uncalibrated": {
                "predicted": test_pred_uncal.tolist(),
                "observed": test_obs_uncal.tolist(),
                "counts": test_counts_uncal.tolist()
            },
            "test_calibrated": {
                "predicted": test_pred_cal.tolist(),
                "observed": test_obs_cal.tolist(),
                "counts": test_counts_cal.tolist()
            }
        },
        "interpretation": {
            "brier_score": "Measures calibration quality (0=perfect, lower is better)",
            "reliability_curve": "Predicted proba vs observed frequency (should lie on diagonal)",
            "note": "AUC unchanged by calibration (discrimination unaffected)"
        }
    }
    
    calibration_path = bundle_dir / 'calibration_metrics.json'
    with open(calibration_path, 'w') as f:
        json.dump(calibration_metrics, f, indent=2)
    print(f"💾 Saved calibration metrics: {calibration_path.name}")
    
    # ===== Print calibration table =====
    print(f"\n📊 Test Set Reliability (10 bins):")
    print(f"{'Predicted':<12} {'Observed':<12} {'Count':<8} {'Error':<10}")
    print("-" * 45)
    
    for pred, obs, count in zip(test_pred_cal, test_obs_cal, test_counts_cal):
        if count > 0:
            error = abs(pred - obs)
            print(f"{pred:>10.2%}  {obs:>10.2%}  {int(count):>6}  {error:>8.2%}")
    
    print(f"\n✅ Calibration complete!")
    print(f"\n📝 Usage:")
    print(f"   # Load calibrated model bundle")
    print(f"   bundle = joblib.load('ML/production/v20251223_212702/model_calibrated.pkl')")
    print(f"   model = bundle['model']")
    print(f"   calibrator = bundle['calibrator']")
    print(f"   ")
    print(f"   # Make calibrated predictions")
    print(f"   y_proba_uncal = model.predict_proba(X)[:, 1]")
    print(f"   y_proba_cal = calibrator.predict(y_proba_uncal)  # Calibrated!")


def main():
    parser = argparse.ArgumentParser(
        description="Calibrate model probabilities on validation set"
    )
    
    parser.add_argument(
        '--bundle',
        type=str,
        required=True,
        help='Path to production bundle directory'
    )
    
    args = parser.parse_args()
    
    calibrate_model(args.bundle)


if __name__ == "__main__":
    main()
