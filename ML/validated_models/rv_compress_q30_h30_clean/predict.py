"""
Predict rv_compress_q30 H=30 for Recent Days
============================================
Shows percentile thresholds and predictions for last 3 trading days.
"""

import os
import sys
from pathlib import Path
import numpy as np
import pandas as pd
from sqlalchemy import create_engine, text
from xgboost import XGBClassifier

# Add project root to path (go up 3 levels: predict.py -> rv_compress_q30_h30_clean -> validated_models -> ML -> project_root)
project_root = Path(__file__).parent.parent.parent.parent
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
    
    # CCI approximation
    tp = (X["high"] + X["low"] + X["close"]) / 3
    X["cci"] = (tp - tp.rolling(20).mean()) / (0.015 * tp.rolling(20).std())
    
    # Williams %R
    hh = X["high"].rolling(14).max()
    ll = X["low"].rolling(14).min()
    X["willr"] = -100 * (hh - X["close"]) / (hh - ll)
    
    # MFI approximation
    X["mfi"] = 50 + 50 * X["roc_20"] / 100
    
    return X

def create_label(df, h=30):
    """rv_compress_q30: Will volatility compress in next 30 days?"""
    rv_now = df["rv_20"].astype(float)
    rv_future = rv_now.shift(-h)
    rv_change = rv_future - rv_now
    
    # Use training data to calculate threshold
    train_mask = df["dt"] <= "2023-12-29"
    q30 = rv_change[train_mask].quantile(0.30)
    
    label = (rv_change <= q30).astype(int)
    return label, q30

def main():
    print("="*80)
    print("RV_COMPRESS_Q30 H=30 CLEAN MODEL - RECENT PREDICTIONS")
    print("="*80)
    
    df = load_data()
    print(f"\nLoaded {len(df)} rows from {df['dt'].min()} to {df['dt'].max()}")
    
    # Build features
    X = build_features(df)
    y, q30_threshold = create_label(df, h=30)
    X["y"] = y
    
    print(f"Label threshold (q30): {q30_threshold:.6f}")
    
    # Feature columns (NO volatility features)
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
    
    # Drop NaN
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
    
    print(f"\nData splits: train={len(y_train)} val={len(y_val)} test={len(y_test)}")
    
    # Train model
    print("\nTraining model...")
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
    
    # Get predictions for ALL data
    X_all = X[feature_cols]
    all_probs = model.predict_proba(X_all)[:, 1]
    
    # Add probabilities to the dataframe
    X["prob"] = all_probs
    
    # Calculate percentile thresholds on TEST set
    test_probs = model.predict_proba(X_test)[:, 1]
    p75 = np.percentile(test_probs, 75)
    p90 = np.percentile(test_probs, 90)
    p95 = np.percentile(test_probs, 95)
    
    print(f"\n{'='*80}")
    print("PERCENTILE THRESHOLDS (from test set)")
    print(f"{'='*80}")
    print(f"p75 threshold: {p75:.4f}")
    print(f"p90 threshold: {p90:.4f}")
    print(f"p95 threshold: {p95:.4f}")
    
    # Calculate hit rates by percentile
    test_df = X.loc[test_mask].copy()
    test_df["y_true"] = y_test.values
    
    top75 = test_df[test_df["prob"] >= p75]
    top90 = test_df[test_df["prob"] >= p90]
    top95 = test_df[test_df["prob"] >= p95]
    
    base_rate = y_test.mean()
    
    print(f"\nBase rate (test): {base_rate:.1%}")
    print(f"\nHit rates by percentile:")
    print(f"  ≥ p75: {top75['y_true'].mean():.1%} ({len(top75)} samples, {top75['y_true'].mean()/base_rate:.2f}x lift)")
    print(f"  ≥ p90: {top90['y_true'].mean():.1%} ({len(top90)} samples, {top90['y_true'].mean()/base_rate:.2f}x lift)")
    print(f"  ≥ p95: {top95['y_true'].mean():.1%} ({len(top95)} samples, {top95['y_true'].mean()/base_rate:.2f}x lift)")
    
    # Get last 3 trading days
    print(f"\n{'='*80}")
    print("LAST 3 TRADING DAYS - PREDICTIONS")
    print(f"{'='*80}")
    
    last_3 = X.iloc[-3:].copy()
    
    results = []
    for idx in last_3.index:
        row = X.loc[idx]
        prob = row["prob"]
        
        # Determine classification
        if prob >= p95:
            classification = "TOP 5%"
            expected_hit = top95['y_true'].mean() if len(top95) > 0 else np.nan
        elif prob >= p90:
            classification = "TOP 10%"
            expected_hit = top90['y_true'].mean() if len(top90) > 0 else np.nan
        elif prob >= p75:
            classification = "TOP 25%"
            expected_hit = top75['y_true'].mean() if len(top75) > 0 else np.nan
        else:
            classification = "BELOW TOP 25%"
            expected_hit = base_rate
        
        results.append({
            "date": row["dt"].strftime("%Y-%m-%d"),
            "prob": prob,
            "classification": classification,
            "expected_hit": expected_hit,
            "close": row["close"],
            "rv_20": row["rv_20"],
            "rsi": row["rsi_14"],
            "adx": row["adx_14"]
        })
    
    for r in results:
        print(f"\n{r['date']} ({r['date'][:10]})")
        print(f"  Probability: {r['prob']:.4f} ({r['prob']:.1%})")
        print(f"  Classification: {r['classification']}")
        print(f"  Expected hit rate: {r['expected_hit']:.1%}")
        print(f"  Close: ${r['close']:.2f}")
        print(f"  RV(20): {r['rv_20']:.4f}")
        print(f"  RSI: {r['rsi']:.1f}")
        print(f"  ADX: {r['adx']:.1f}")
        
        # Trading signal
        if r['prob'] >= p95:
            print(f"  🔥 SIGNAL: STRONG - High confidence vol compression expected")
        elif r['prob'] >= p90:
            print(f"  ✅ SIGNAL: GOOD - Vol compression likely")
        elif r['prob'] >= p75:
            print(f"  ⚠️  SIGNAL: MODERATE - Above average vol compression probability")
        else:
            print(f"  ❌ SIGNAL: WEAK - Below threshold")
    
    print(f"\n{'='*80}")
    print("INTERPRETATION")
    print(f"{'='*80}")
    print("\nVol compression (rv_compress_q30) means:")
    print("  - Realized volatility will DECREASE over next 30 days")
    print("  - Current vol is HIGH relative to future vol")
    print("  - Good for: Selling options (premium decay)")
    print("  - Bad for: Buying options (lose vega)")
    print("\nStrategy recommendations:")
    print("  - TOP 5%: Sell straddles/strangles (high conviction)")
    print("  - TOP 10%: Sell credit spreads (good probability)")
    print("  - TOP 25%: Consider iron condors (moderate conviction)")
    print("  - BELOW: Avoid selling premium (vol may expand)")

if __name__ == "__main__":
    main()
