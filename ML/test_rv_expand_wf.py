"""
Walk-Forward Validation for rv_expand_abs H=30
Tests temporal stability across multiple time windows
"""

import numpy as np
import pandas as pd
from sqlalchemy import create_engine, text
from sklearn.metrics import roc_auc_score
from xgboost import XGBClassifier
from sklearn.calibration import CalibratedClassifierCV


def load_data():
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
    rv = df["realized_volatility_20"]
    if rv.isna().all():
        rv = df["rv20_logret"]
    rv = rv.astype(float)
    
    rv_future = rv.shift(-h)
    rv_change = rv_future - rv
    
    label = (rv_change >= threshold).astype(int)
    return label


def walk_forward_validation(X, y):
    """
    Rolling window validation
    Train on expanding window, test on next period
    """
    
    # Feature columns (WITHOUT vol features for clean test)
    feature_cols = [
        "ret_1d", "ret_5d", "ret_21d",
        "logret_1d", "logret_5d",
        "px_vs_sma20", "px_vs_sma50", "px_vs_sma200",
        "rsi_14", "stochastic_k", "stochastic_d",
        "macd", "macd_signal", "macd_histogram", "macd_z",
        "adx_14", "di_diff", "roc_20",
        "volume_ratio", "obv_norm",
        "pct_from_high_20d", "pct_from_low_20d",
        "pct_from_high_52w", "pct_from_low_52w",
        "gap_1d"
    ]
    feature_cols = [c for c in feature_cols if c in X.columns]
    
    X_prep = X[feature_cols + ["dt"]].copy()
    X_prep["y"] = y
    X_prep = X_prep.dropna()
    
    # Define windows (6-month test windows)
    windows = [
        ("2021-01-01", "2023-06-30", "2023-07-01", "2023-12-31"),
        ("2021-01-01", "2023-12-31", "2024-01-01", "2024-06-30"),
        ("2021-01-01", "2024-06-30", "2024-07-01", "2024-12-31"),
        ("2021-01-01", "2024-12-31", "2025-01-01", "2025-12-31"),
    ]
    
    results = []
    
    print(f"\n{'='*80}")
    print("WALK-FORWARD VALIDATION (Clean Model - No Vol Features)")
    print(f"{'='*80}")
    print(f"Using {len(feature_cols)} features (excluding volatility)")
    
    for i, (train_start, train_end, test_start, test_end) in enumerate(windows, 1):
        train_mask = (X_prep["dt"] >= train_start) & (X_prep["dt"] <= train_end)
        test_mask = (X_prep["dt"] >= test_start) & (X_prep["dt"] <= test_end)
        
        train = X_prep[train_mask]
        test = X_prep[test_mask]
        
        if len(test) < 50:
            continue
        
        X_train, y_train = train[feature_cols], train["y"]
        X_test, y_test = test[feature_cols], test["y"]
        
        # Train model
        model = XGBClassifier(
            max_depth=5,
            learning_rate=0.05,
            n_estimators=400,
            subsample=1.0,
            colsample_bytree=0.85,
            reg_lambda=1.0,
            min_child_weight=5,
            objective="binary:logistic",
            eval_metric="logloss",
            random_state=42,
            n_jobs=-1
        )
        
        model.fit(X_train, y_train)
        
        # Calibrate
        calib_model = CalibratedClassifierCV(model, method="sigmoid", cv=3)
        calib_model.fit(X_train, y_train)
        
        # Predict
        test_pred = calib_model.predict_proba(X_test)[:, 1]
        test_auc = roc_auc_score(y_test, test_pred)
        
        results.append({
            "window": i,
            "train_start": train_start,
            "train_end": train_end,
            "test_start": test_start,
            "test_end": test_end,
            "train_size": len(X_train),
            "test_size": len(X_test),
            "train_base_rate": y_train.mean(),
            "test_base_rate": y_test.mean(),
            "test_auc": test_auc
        })
        
        print(f"\nWindow {i}:")
        print(f"  Train: {train_start} to {train_end} (n={len(X_train)}, base={y_train.mean():.1%})")
        print(f"  Test:  {test_start} to {test_end} (n={len(X_test)}, base={y_test.mean():.1%})")
        print(f"  Test AUC: {test_auc:.4f}")
    
    results_df = pd.DataFrame(results)
    
    print(f"\n{'='*80}")
    print("SUMMARY")
    print(f"{'='*80}")
    print(f"Mean Test AUC: {results_df['test_auc'].mean():.4f}")
    print(f"Std Test AUC:  {results_df['test_auc'].std():.4f}")
    print(f"Min Test AUC:  {results_df['test_auc'].min():.4f}")
    print(f"Max Test AUC:  {results_df['test_auc'].max():.4f}")
    
    if results_df['test_auc'].std() > 0.10:
        print("\n🚨 WARNING: High AUC variance (>0.10) - model may be unstable")
    else:
        print("\n✅ PASS: Consistent performance across time windows")
    
    if results_df['test_auc'].min() < 0.65:
        print("🚨 WARNING: Some windows have AUC < 0.65 - model may be unreliable")
    else:
        print("✅ PASS: All windows have AUC >= 0.65")
    
    return results_df


def main():
    print("="*80)
    print("RV_EXPAND_ABS H=30 - WALK-FORWARD VALIDATION")
    print("="*80)
    
    df = load_data()
    print(f"\nLoaded {len(df)} rows from {df['dt'].min()} to {df['dt'].max()}")
    
    X = build_features(df)
    y = create_label(X, h=30, threshold=0.05)
    
    results = walk_forward_validation(X, y)
    
    print(f"\n{'='*80}")
    print("CONCLUSION")
    print(f"{'='*80}")
    mean_auc = results['test_auc'].mean()
    std_auc = results['test_auc'].std()
    
    if mean_auc >= 0.70 and std_auc <= 0.10:
        print(f"✅ PASSED: Mean AUC {mean_auc:.4f} ± {std_auc:.4f}")
        print("   Model shows stable predictive power across time windows")
    else:
        print(f"❌ FAILED: Mean AUC {mean_auc:.4f} ± {std_auc:.4f}")
        print("   Model may be unreliable or overfitted")


if __name__ == "__main__":
    main()
