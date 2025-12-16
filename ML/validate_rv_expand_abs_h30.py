"""
Complete validation pipeline for rv_expand_abs H=30
Predicts volatility expansion (absolute change >= +0.05)
"""

import os
import sys
from pathlib import Path
import numpy as np
import pandas as pd
from sqlalchemy import create_engine, text
from sklearn.metrics import roc_auc_score, brier_score_loss
from xgboost import XGBClassifier
from sklearn.calibration import CalibratedClassifierCV

project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from src.config import settings


def load_data():
    # Use correct database connection
    db_url = "postgresql://airflow:airflow@192.168.1.201:5433/ahold_options"
    eng = create_engine(db_url, pool_pre_ping=True)
    
    q = """
    SELECT 
        o.trade_date::date AS dt,
        o.ticker,
        o.open::float,
        o.high::float,
        o.low::float,
        o.close::float,
        o.volume::float,
        f.sma_20::float,
        f.sma_50::float,
        f.sma_200::float,
        f.ema_12::float,
        f.ema_26::float,
        f.macd::float,
        f.macd_signal::float,
        f.macd_histogram::float,
        f.rsi_14::float,
        f.stochastic_k::float,
        f.stochastic_d::float,
        f.roc_20::float,
        f.atr_14::float,
        f.bollinger_width::float,
        f.realized_volatility_20::float,
        f.parkinson_volatility_20::float,
        f.pct_from_high_20d::float,
        f.pct_from_low_20d::float,
        f.pct_from_high_52w::float,
        f.pct_from_low_52w::float,
        f.volume_ratio::float,
        f.adx_14::float,
        f.plus_di_14::float,
        f.minus_di_14::float,
        f.obv::float,
        f.obv_sma_20::float
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
    
    # Log returns
    X["logret_1d"] = np.log(X["close"] / X["close"].shift(1))
    X["logret_5d"] = X["logret_1d"].rolling(5).sum()
    
    # Price position
    X["px_vs_sma20"] = X["close"] / X["sma_20"] - 1
    X["px_vs_sma50"] = X["close"] / X["sma_50"] - 1
    X["px_vs_sma200"] = X["close"] / X["sma_200"] - 1
    
    # Volatility
    X["atr_pct"] = X["atr_14"] / X["close"]
    X["bb_width_pct"] = X["bollinger_width"] / X["close"]
    X["rv20_logret"] = X["logret_1d"].rolling(20).std() * np.sqrt(252)
    
    # Momentum
    X["di_diff"] = X["plus_di_14"] - X["minus_di_14"]
    X["macd_z"] = (X["macd"] - X["macd"].rolling(60).mean()) / (X["macd"].rolling(60).std().replace(0, np.nan))
    
    # Volume
    X["obv_norm"] = X["obv"] / X["obv"].rolling(50).mean()
    
    # Gap
    X["gap_1d"] = (X["open"] - X["close"].shift(1)) / X["close"].shift(1)
    
    return X


def create_label(df, h=30, threshold=0.05):
    """
    rv_expand_abs: Will volatility expand by at least +0.05 over next H days?
    Label = 1 if (rv_future - rv_now) >= threshold
    
    NOTE: Using ALL volatility features to check for leakage
    """
    rv = df["realized_volatility_20"]
    if rv.isna().all():
        rv = df["rv20_logret"]
    rv = rv.astype(float)
    
    rv_future = rv.shift(-h)
    rv_change = rv_future - rv
    
    label = (rv_change >= threshold).astype(int)
    return label


def train_model(X, y):
    """Train with best params from supersweep"""
    
    # Split data
    train_mask = X["dt"] <= "2023-12-29"
    val_mask = (X["dt"] > "2023-12-29") & (X["dt"] <= "2024-12-31")
    test_mask = X["dt"] > "2024-12-31"
    
    # Feature columns (INCLUDE vol features for initial test - will test without later)
    feature_cols = [
        "ret_1d", "ret_5d", "ret_21d",
        "logret_1d", "logret_5d",
        "px_vs_sma20", "px_vs_sma50", "px_vs_sma200",
        "atr_pct", "bb_width_pct", "rv20_logret",
        "realized_volatility_20", "parkinson_volatility_20",
        "rsi_14", "stochastic_k", "stochastic_d",
        "macd", "macd_signal", "macd_histogram", "macd_z",
        "adx_14", "di_diff", "roc_20",
        "volume_ratio", "obv_norm",
        "pct_from_high_20d", "pct_from_low_20d",
        "pct_from_high_52w", "pct_from_low_52w",
        "gap_1d"
    ]
    feature_cols = [c for c in feature_cols if c in X.columns]
    
    # Prepare data
    X_prep = X[feature_cols + ["dt"]].copy()
    X_prep["y"] = y
    X_prep = X_prep.dropna()
    
    train = X_prep[train_mask.loc[X_prep.index]]
    val = X_prep[val_mask.loc[X_prep.index]]
    test = X_prep[test_mask.loc[X_prep.index]]
    
    X_train, y_train = train[feature_cols], train["y"]
    X_val, y_val = val[feature_cols], val["y"]
    X_test, y_test = test[feature_cols], test["y"]
    
    print(f"\n{'='*80}")
    print("DATA SPLITS")
    print(f"{'='*80}")
    print(f"Train: {len(y_train)} samples, base rate: {y_train.mean():.1%}")
    print(f"Val:   {len(y_val)} samples, base rate: {y_val.mean():.1%}")
    print(f"Test:  {len(y_test)} samples, base rate: {y_test.mean():.1%}")
    
    # Best params from supersweep (read from results)
    params = {
        "max_depth": 5,
        "learning_rate": 0.05,
        "n_estimators": 400,
        "subsample": 1.0,
        "colsample_bytree": 0.85,
        "reg_lambda": 1.0,
        "min_child_weight": 5
    }
    
    print(f"\n{'='*80}")
    print("TRAINING MODEL (with vol features)")
    print(f"{'='*80}")
    print(f"Using {len(feature_cols)} features")
    
    base_model = XGBClassifier(
        objective="binary:logistic",
        eval_metric="logloss",
        random_state=42,
        n_jobs=-1,
        **params
    )
    
    base_model.fit(X_train, y_train)
    
    # Calibrate
    calib_model = CalibratedClassifierCV(base_model, method="sigmoid", cv="prefit")
    calib_model.fit(X_val, y_val)
    
    # Evaluate
    val_pred = calib_model.predict_proba(X_val)[:, 1]
    test_pred = calib_model.predict_proba(X_test)[:, 1]
    
    val_auc = roc_auc_score(y_val, val_pred)
    val_brier = brier_score_loss(y_val, val_pred)
    test_auc = roc_auc_score(y_test, test_pred)
    test_brier = brier_score_loss(y_test, test_pred)
    
    print(f"\n{'='*80}")
    print("INITIAL RESULTS (with vol features)")
    print(f"{'='*80}")
    print(f"Val  AUC: {val_auc:.4f} | Brier: {val_brier:.4f}")
    print(f"Test AUC: {test_auc:.4f} | Brier: {test_brier:.4f}")
    
    # Feature importance
    feature_imp = pd.DataFrame({
        "feature": feature_cols,
        "importance": base_model.feature_importances_
    }).sort_values("importance", ascending=False)
    
    print(f"\nTop 10 Features:")
    for i, row in feature_imp.head(10).iterrows():
        flag = "🚨" if any(v in row['feature'] for v in ['rv', 'atr', 'bb_width', 'parkinson']) else "✅"
        print(f"  {flag} {row['feature']:30s} {row['importance']:.4f}")
    
    # Check for leakage
    vol_features = [f for f in feature_cols if any(v in f for v in ['rv', 'atr', 'bb_width', 'parkinson'])]
    vol_importance = feature_imp[feature_imp['feature'].isin(vol_features)]['importance'].sum()
    
    print(f"\n🔍 Leakage Check:")
    print(f"  Volatility features: {len(vol_features)}/{len(feature_cols)}")
    print(f"  Vol feature importance: {vol_importance:.1%}")
    
    if vol_importance > 0.3:
        print(f"  🚨 WARNING: Likely feature leakage (vol features > 30%)")
    else:
        print(f"  ✅ OK: Vol features < 30% importance")
    
    # Lift analysis
    test_df = test.copy()
    test_df["pred"] = test_pred
    
    p75 = np.percentile(test_pred, 75)
    p90 = np.percentile(test_pred, 90)
    p95 = np.percentile(test_pred, 95)
    
    base_rate = y_test.mean()
    top75_rate = test_df[test_df["pred"] >= p75]["y"].mean()
    top90_rate = test_df[test_df["pred"] >= p90]["y"].mean()
    top95_rate = test_df[test_df["pred"] >= p95]["y"].mean()
    
    print(f"\n{'='*80}")
    print("LIFT ANALYSIS")
    print(f"{'='*80}")
    print(f"Base rate: {base_rate:.1%}")
    print(f"  ≥ p75 (>{p75:.2f}): {top75_rate:.1%} ({top75_rate/base_rate:.2f}x lift)")
    print(f"  ≥ p90 (>{p90:.2f}): {top90_rate:.1%} ({top90_rate/base_rate:.2f}x lift)")
    print(f"  ≥ p95 (>{p95:.2f}): {top95_rate:.1%} ({top95_rate/base_rate:.2f}x lift)")
    
    return {
        "model": calib_model,
        "base_model": base_model,
        "val_auc": val_auc,
        "test_auc": test_auc,
        "feature_importance": feature_imp,
        "p75": p75,
        "p90": p90,
        "p95": p95,
        "test_df": test_df,
        "X_train": X_train,
        "y_train": y_train,
        "X_val": X_val,
        "y_val": y_val,
        "X_test": X_test,
        "y_test": y_test
    }


def main():
    print("="*80)
    print("RV_EXPAND_ABS H=30 - COMPLETE VALIDATION PIPELINE")
    print("="*80)
    print("\nTarget: Volatility expansion (rv_future - rv_now >= +0.05)")
    print("Use case: Long volatility strategies, buy options")
    
    df = load_data()
    print(f"\nLoaded {len(df)} rows from {df['dt'].min()} to {df['dt'].max()}")
    
    X = build_features(df)
    y = create_label(X, h=30, threshold=0.05)
    
    print(f"Label base rate (full data): {y.mean():.1%}")
    
    # Train initial model (with vol features)
    results = train_model(X, y)
    
    print(f"\n{'='*80}")
    print("NEXT STEPS")
    print(f"{'='*80}")
    print("\n1. Run leakage tests:")
    print("   - Walk-forward validation")
    print("   - Shift test")
    print("   - Clean model (without vol features)")
    print("\n2. If passes validation:")
    print("   - Get predictions for recent days")
    print("   - Create validated model folder")
    print("   - Add to production portfolio")


if __name__ == "__main__":
    main()
