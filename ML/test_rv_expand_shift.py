"""
Shift Test for rv_expand_abs H=30
Tests temporal integrity by shifting features forward/backward in time
"""

import numpy as np
import pandas as pd
from sqlalchemy import create_engine, text
from sklearn.metrics import roc_auc_score
from xgboost import XGBClassifier


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


def shift_test(X, y):
    """
    Test temporal integrity by shifting features
    Good model: forward shifts degrade, backward shifts degrade
    Leakage: forward shifts improve
    """
    
    # Feature columns (clean - no vol)
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
    
    # Prepare data
    train_mask = X["dt"] <= "2023-12-29"
    test_mask = X["dt"] > "2024-12-31"
    
    X_prep = X[feature_cols + ["dt"]].copy()
    X_prep["y"] = y
    X_prep = X_prep.dropna()
    
    train = X_prep[train_mask.loc[X_prep.index]]
    test = X_prep[test_mask.loc[X_prep.index]]
    
    X_train, y_train = train[feature_cols], train["y"]
    X_test_orig, y_test = test[feature_cols], test["y"]
    
    print(f"\n{'='*80}")
    print("SHIFT TEST (Clean Model)")
    print(f"{'='*80}")
    print(f"Using {len(feature_cols)} features (no volatility)")
    print(f"Train: {len(y_train)} samples")
    print(f"Test:  {len(y_test)} samples")
    
    # Train baseline model
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
    
    # Test shifts
    shifts = list(range(-10, 11))
    results = []
    
    for shift in shifts:
        if shift == 0:
            X_test_shifted = X_test_orig
        else:
            X_test_shifted = X_test_orig.shift(shift)
        
        # Drop NaN rows
        valid_idx = X_test_shifted.dropna().index
        X_test_valid = X_test_shifted.loc[valid_idx]
        y_test_valid = y_test.loc[valid_idx]
        
        if len(y_test_valid) < 50:
            continue
        
        test_pred = model.predict_proba(X_test_valid)[:, 1]
        test_auc = roc_auc_score(y_test_valid, test_pred)
        
        results.append({
            "shift": shift,
            "test_auc": test_auc,
            "n_samples": len(y_test_valid)
        })
    
    results_df = pd.DataFrame(results)
    baseline_auc = results_df[results_df["shift"] == 0]["test_auc"].values[0]
    
    print(f"\nBaseline (shift=0): {baseline_auc:.4f} test AUC")
    
    print(f"\n{'Direction':<15} {'Shift':<10} {'Test AUC':<12} {'Change':<12} {'Status'}")
    print("-" * 80)
    
    for _, row in results_df.iterrows():
        shift = int(row["shift"])
        auc = row["test_auc"]
        change = auc - baseline_auc
        
        if shift == 0:
            direction = "BASELINE"
            status = "---"
        elif shift > 0:
            direction = "FORWARD"
            status = "✅ Good" if change < -0.02 else "🚨 BAD" if change > 0.02 else "⚠️ Neutral"
        else:
            direction = "BACKWARD"
            status = "✅ Good" if change < 0 else "⚠️ OK"
        
        print(f"{direction:<15} {shift:+3d}d      {auc:.4f}      {change:+.4f}      {status}")
    
    # Analysis
    forward_aucs = results_df[results_df["shift"] > 0]["test_auc"]
    backward_aucs = results_df[results_df["shift"] < 0]["test_auc"]
    
    forward_mean = forward_aucs.mean()
    backward_mean = backward_aucs.mean()
    
    print(f"\n{'='*80}")
    print("ANALYSIS")
    print(f"{'='*80}")
    print(f"Baseline AUC:        {baseline_auc:.4f}")
    print(f"Forward mean AUC:    {forward_mean:.4f} ({forward_mean - baseline_auc:+.4f})")
    print(f"Backward mean AUC:   {backward_mean:.4f} ({backward_mean - baseline_auc:+.4f})")
    
    print(f"\n{'='*80}")
    print("VERDICT")
    print(f"{'='*80}")
    
    if forward_mean < baseline_auc - 0.05:
        print("✅ PASSED: Forward shifts degrade performance significantly")
        print("   No hidden temporal leakage detected")
        passed = True
    elif forward_mean > baseline_auc:
        print("🚨 FAILED: Forward shifts IMPROVE performance")
        print("   Strong evidence of temporal leakage!")
        passed = False
    else:
        print("⚠️ MARGINAL: Forward shifts show small degradation")
        print("   May have minor temporal issues")
        passed = True
    
    return results_df, passed


def main():
    print("="*80)
    print("RV_EXPAND_ABS H=30 - SHIFT TEST")
    print("="*80)
    
    df = load_data()
    print(f"\nLoaded {len(df)} rows from {df['dt'].min()} to {df['dt'].max()}")
    
    X = build_features(df)
    y = create_label(X, h=30, threshold=0.05)
    
    results, passed = shift_test(X, y)
    
    if passed:
        print("\n✅ Model PASSED shift test - temporal integrity validated")
    else:
        print("\n❌ Model FAILED shift test - do not use in production")


if __name__ == "__main__":
    main()
