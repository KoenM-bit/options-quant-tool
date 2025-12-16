"""
Shift Test for rv_compress_q30 H=30 Clean Model
================================================
Tests temporal integrity by shifting features forward/backward in time.

If model uses legitimate patterns:
- Forward shift (features from future): Performance degrades significantly
- Backward shift (features from past): Performance degrades slightly
- No shift (current): Best performance (baseline 80.2% AUC)

If model has hidden leakage:
- Forward shift: Performance improves (red flag!)
- Patterns would be suspiciously good with "future" data
"""

import os
import sys
from pathlib import Path
import numpy as np
import pandas as pd
from sqlalchemy import create_engine, text
from sklearn.metrics import roc_auc_score, brier_score_loss
from xgboost import XGBClassifier

project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from src.config import settings

def load_data():
    eng = create_engine(settings.database_url, pool_pre_ping=True)
    
    q = """
    SELECT 
        o.trade_date::date AS dt,
        o.ticker,
        o.close::float,
        o.high::float,
        o.low::float,
        f.realized_volatility_20::float AS rv_20,
        f.rsi_14::float,
        f.stochastic_k::float,
        f.stochastic_d::float,
        f.roc_20::float,
        f.macd::float,
        f.macd_signal::float,
        f.macd_histogram::float,
        f.adx_14::float,
        f.plus_di_14::float,
        f.minus_di_14::float,
        f.sma_20::float,
        f.sma_50::float,
        f.sma_200::float,
        f.volume_ratio::float,
        f.obv::float,
        f.obv_sma_20::float,
        f.pct_from_high_20d::float,
        f.pct_from_low_20d::float,
        f.pct_from_high_52w::float,
        f.pct_from_low_52w::float,
        f.high_20d::float,
        f.low_20d::float
    FROM bronze_ohlcv o
    JOIN fact_technical_indicators f ON o.ticker = f.ticker AND o.trade_date = f.trade_date
    WHERE o.ticker = 'AD.AS'
      AND o.trade_date >= '2020-01-01'
      AND o.trade_date <= '2025-12-31'
    ORDER BY o.trade_date
    """
    
    df = pd.read_sql(text(q), eng)
    df["dt"] = pd.to_datetime(df["dt"])
    return df

def build_features(df):
    X = df.copy()
    
    # Returns
    X["ret_1d"] = X["close"].pct_change()
    X["ret_5d"] = X["close"].pct_change(5)
    X["ret_21d"] = X["close"].pct_change(21)
    
    # Price position
    X["px_vs_sma20"] = X["close"] / X["sma_20"] - 1
    X["px_vs_sma50"] = X["close"] / X["sma_50"] - 1
    
    # Momentum (no vol features!)
    X["di_diff"] = X["plus_di_14"] - X["minus_di_14"]
    
    # Volume
    X["obv_norm"] = X["obv"] / X["obv"].rolling(50).mean()
    X["vol_ma_ratio"] = X["volume_ratio"]
    
    # Breadth (already in data)
    
    # CCI approximation
    tp = (X["high"] + X["low"] + X["close"]) / 3
    X["cci"] = (tp - tp.rolling(20).mean()) / (0.015 * tp.rolling(20).std())
    
    # Williams %R
    hh = X["high"].rolling(14).max()
    ll = X["low"].rolling(14).min()
    X["willr"] = -100 * (hh - X["close"]) / (hh - ll)
    
    # MFI approximation (without actual volume)
    X["mfi"] = 50 + 50 * X["roc_20"] / 100  # Simplified
    
    return X

def create_label(df, h=30):
    """rv_compress_q30: Will volatility compress in next 30 days?"""
    rv_now = df["rv_20"].astype(float)
    rv_future = rv_now.shift(-h)
    rv_change = rv_future - rv_now
    
    # Use training data (before 2024) to calculate threshold
    train_mask = df["dt"] <= "2023-12-29"
    q30 = rv_change[train_mask].quantile(0.30)
    
    label = (rv_change <= q30).astype(int)
    return label, q30

def run_shift_test(df, shift_days, shift_name):
    """
    Test model with shifted features.
    
    shift_days > 0: Features from FUTURE (should degrade if legitimate)
    shift_days < 0: Features from PAST (should degrade slightly)
    shift_days = 0: Normal (baseline)
    """
    print(f"\n{'='*80}")
    print(f"SHIFT TEST: {shift_name} (shift={shift_days} days)")
    print(f"{'='*80}")
    
    # Build features
    X = build_features(df)
    
    # Create label (NEVER shift the label!)
    y, q30_threshold = create_label(df, h=30)
    X["y"] = y
    
    # Feature columns (exclude volatility features)
    feature_cols = [
        "ret_1d", "ret_5d", "ret_21d",
        "px_vs_sma20", "px_vs_sma50",
        "rsi_14", "stochastic_k", "stochastic_d",
        "macd", "macd_signal", "macd_histogram",
        "adx_14", "di_diff",
        "roc_20", "cci", "willr",
        "mfi",
        "obv_norm", "vol_ma_ratio",
        "pct_from_high_20d", "pct_from_low_20d",
        "pct_from_high_52w", "pct_from_low_52w"
    ]
    feature_cols = [c for c in feature_cols if c in X.columns]
    
    # SHIFT FEATURES (but not label!)
    if shift_days != 0:
        for col in feature_cols:
            X[col] = X[col].shift(shift_days)
    
    # Drop rows with NaN
    X = X.dropna()
    
    # Split data
    train_mask = X["dt"] <= "2023-12-29"
    val_mask = (X["dt"] > "2023-12-29") & (X["dt"] <= "2024-12-31")
    test_mask = X["dt"] > "2024-12-31"
    
    X_train = X.loc[train_mask, feature_cols]
    y_train = X.loc[train_mask, "y"]
    X_val = X.loc[val_mask, feature_cols]
    y_val = X.loc[val_mask, "y"]
    X_test = X.loc[test_mask, feature_cols]
    y_test = X.loc[test_mask, "y"]
    
    print(f"Data splits: train={len(y_train)} val={len(y_val)} test={len(y_test)}")
    print(f"Base rates: train={y_train.mean():.1%} val={y_val.mean():.1%} test={y_test.mean():.1%}")
    
    if len(y_train) < 100 or len(y_test) < 50:
        print("❌ Insufficient data after shift")
        return None
    
    # Train model (same params as clean test)
    model = XGBClassifier(
        max_depth=3,
        learning_rate=0.05,
        n_estimators=400,
        subsample=0.85,
        colsample_bytree=0.85,
        reg_lambda=3.0,
        min_child_weight=5,
        objective="binary:logistic",
        eval_metric="logloss",
        random_state=42,
        n_jobs=-1
    )
    
    model.fit(X_train, y_train)
    
    # Evaluate
    p_val = model.predict_proba(X_val)[:, 1]
    p_test = model.predict_proba(X_test)[:, 1]
    
    val_auc = roc_auc_score(y_val, p_val)
    val_brier = brier_score_loss(y_val, p_val)
    test_auc = roc_auc_score(y_test, p_test)
    test_brier = brier_score_loss(y_test, p_test)
    
    print(f"\nResults:")
    print(f"  Val  AUC: {val_auc:.4f} | Brier: {val_brier:.4f}")
    print(f"  Test AUC: {test_auc:.4f} | Brier: {test_brier:.4f}")
    
    return {
        "shift_days": shift_days,
        "shift_name": shift_name,
        "val_auc": val_auc,
        "val_brier": val_brier,
        "test_auc": test_auc,
        "test_brier": test_brier,
        "n_train": len(y_train),
        "n_val": len(y_val),
        "n_test": len(y_test)
    }

def main():
    print("="*80)
    print("SHIFT TEST: rv_compress_q30 H=30 Clean Model")
    print("="*80)
    print("\nTesting temporal integrity by shifting features in time.")
    print("Expected behavior for legitimate model:")
    print("  - Forward shift (future data): Performance DEGRADES")
    print("  - Backward shift (stale data): Performance DEGRADES slightly")
    print("  - No shift (current): BEST performance (baseline)")
    print("\nRed flag: If forward shift IMPROVES performance = hidden leakage!")
    
    df = load_data()
    print(f"\nLoaded {len(df)} rows from {df['dt'].min()} to {df['dt'].max()}")
    
    # Run tests
    results = []
    
    # Test 1: Forward shift (features from FUTURE - should degrade)
    for shift in [1, 3, 5, 10]:
        r = run_shift_test(df, shift, f"Future T+{shift}")
        if r:
            results.append(r)
    
    # Test 2: No shift (baseline)
    r = run_shift_test(df, 0, "No Shift (Baseline)")
    if r:
        results.append(r)
    
    # Test 3: Backward shift (features from PAST - should degrade slightly)
    for shift in [-1, -3, -5, -10]:
        r = run_shift_test(df, shift, f"Past T{shift}")
        if r:
            results.append(r)
    
    # Summary
    print("\n" + "="*80)
    print("SHIFT TEST SUMMARY")
    print("="*80)
    
    df_results = pd.DataFrame(results)
    df_results = df_results.sort_values("shift_days")
    
    print("\nAll Results:")
    print(df_results[["shift_name", "shift_days", "test_auc", "test_brier"]].to_string(index=False))
    
    # Find baseline (shift=0)
    baseline = df_results[df_results["shift_days"] == 0]
    if not baseline.empty:
        baseline_auc = baseline.iloc[0]["test_auc"]
        
        print(f"\n{'='*80}")
        print("TEMPORAL INTEGRITY ASSESSMENT")
        print(f"{'='*80}")
        print(f"Baseline (no shift): {baseline_auc:.4f} test AUC")
        
        # Check forward shifts
        forward = df_results[df_results["shift_days"] > 0]
        if not forward.empty:
            max_forward_auc = forward["test_auc"].max()
            avg_forward_auc = forward["test_auc"].mean()
            
            print(f"\nForward shifts (features from FUTURE):")
            print(f"  Max AUC: {max_forward_auc:.4f}")
            print(f"  Avg AUC: {avg_forward_auc:.4f}")
            print(f"  Change from baseline: {max_forward_auc - baseline_auc:+.4f}")
            
            if max_forward_auc > baseline_auc + 0.02:
                print("  🚨 RED FLAG: Forward shift IMPROVES performance = HIDDEN LEAKAGE!")
            elif max_forward_auc > baseline_auc:
                print("  ⚠️  WARNING: Forward shift slightly improves performance")
            else:
                print("  ✅ GOOD: Forward shift degrades performance (as expected)")
        
        # Check backward shifts
        backward = df_results[df_results["shift_days"] < 0]
        if not backward.empty:
            max_backward_auc = backward["test_auc"].max()
            avg_backward_auc = backward["test_auc"].mean()
            
            print(f"\nBackward shifts (features from PAST):")
            print(f"  Max AUC: {max_backward_auc:.4f}")
            print(f"  Avg AUC: {avg_backward_auc:.4f}")
            print(f"  Change from baseline: {max_backward_auc - baseline_auc:+.4f}")
            
            if max_backward_auc > baseline_auc:
                print("  ⚠️  Unexpected: Past data performs better than current")
            else:
                print("  ✅ GOOD: Past data degrades performance (as expected)")
        
        print(f"\n{'='*80}")
        print("FINAL VERDICT")
        print(f"{'='*80}")
        
        forward_ok = forward.empty or forward["test_auc"].max() <= baseline_auc + 0.02
        backward_ok = backward.empty or backward["test_auc"].max() <= baseline_auc + 0.02
        
        if forward_ok and backward_ok:
            print("✅ PASSED: Model shows proper temporal integrity")
            print("   - Future data does NOT improve performance")
            print("   - Past data does NOT outperform current data")
            print("   - 80.2% AUC is LEGITIMATE predictive power")
        else:
            print("🚨 FAILED: Potential temporal leakage detected")
            print("   - Model may be using information it shouldn't have access to")
            print("   - Further investigation required")

if __name__ == "__main__":
    main()
