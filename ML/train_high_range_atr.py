"""
Train and validate high_range_atr models for H=4 and H=5
Predicts wide trading ranges (range/ATR >= 2.0)
"""

import os
import sys
from pathlib import Path
import numpy as np
import pandas as pd
from sqlalchemy import create_engine, text
from sklearn.metrics import roc_auc_score, brier_score_loss
from sklearn.model_selection import ParameterGrid
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


def create_label(df, h=4, threshold=2.0):
    """
    high_range_atr: Will the next H days have a wide range?
    Label = 1 if (max_high - min_low) / ATR >= threshold
    """
    high = df["high"].astype(float)
    low = df["low"].astype(float)
    atr = df["atr_14"].astype(float)
    
    # Forward-looking max/min
    f_max_high = high.shift(-1).rolling(h).max().shift(-(h-1))
    f_min_low = low.shift(-1).rolling(h).min().shift(-(h-1))
    
    # Range in ATR units
    range_atr = (f_max_high - f_min_low) / atr.replace(0, np.nan)
    
    label = (range_atr >= threshold).astype(int)
    return label


def train_and_evaluate(X, y, horizon):
    """Train with hyperparameter search and evaluate"""
    
    # Split data
    train_mask = X["dt"] <= "2023-12-29"
    val_mask = (X["dt"] > "2023-12-29") & (X["dt"] <= "2024-12-31")
    test_mask = X["dt"] > "2024-12-31"
    
    # Feature columns (use ALL features including vol)
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
    
    print(f"\nH={horizon} Data splits:")
    print(f"  Train: {len(y_train)} samples, base rate: {y_train.mean():.1%}")
    print(f"  Val:   {len(y_val)} samples, base rate: {y_val.mean():.1%}")
    print(f"  Test:  {len(y_test)} samples, base rate: {y_test.mean():.1%}")
    
    if y_train.mean() < 0.1 or y_train.mean() > 0.9:
        print(f"  ⚠️  Base rate outside 10-90% range")
        return None
    
    # Hyperparameter grid
    param_grid = {
        "max_depth": [2, 3, 4],
        "learning_rate": [0.03, 0.05, 0.08],
        "n_estimators": [200, 400, 600],
        "subsample": [0.7, 0.85],
        "colsample_bytree": [0.7, 0.85],
        "reg_lambda": [1.0, 3.0],
        "min_child_weight": [3, 5, 10]
    }
    
    best_val_auc = 0
    best_params = None
    best_model = None
    
    print(f"\nTraining models (testing {len(list(ParameterGrid(param_grid)))} combinations)...")
    
    for i, params in enumerate(ParameterGrid(param_grid)):
        if i % 50 == 0:
            print(f"  Progress: {i}/{len(list(ParameterGrid(param_grid)))}")
        
        model = XGBClassifier(
            objective="binary:logistic",
            eval_metric="logloss",
            random_state=42,
            n_jobs=-1,
            **params
        )
        
        model.fit(X_train, y_train, verbose=False)
        
        val_pred = model.predict_proba(X_val)[:, 1]
        val_auc = roc_auc_score(y_val, val_pred)
        
        if val_auc > best_val_auc:
            best_val_auc = val_auc
            best_params = params
            best_model = model
    
    print(f"\nBest params (val AUC: {best_val_auc:.4f}):")
    for k, v in best_params.items():
        print(f"  {k}: {v}")
    
    # Final evaluation
    val_pred = best_model.predict_proba(X_val)[:, 1]
    test_pred = best_model.predict_proba(X_test)[:, 1]
    
    val_auc = roc_auc_score(y_val, val_pred)
    val_brier = brier_score_loss(y_val, val_pred)
    test_auc = roc_auc_score(y_test, test_pred)
    test_brier = brier_score_loss(y_test, test_pred)
    
    print(f"\n{'='*60}")
    print(f"RESULTS: high_range_atr H={horizon}")
    print(f"{'='*60}")
    print(f"Val  AUC: {val_auc:.4f} | Brier: {val_brier:.4f}")
    print(f"Test AUC: {test_auc:.4f} | Brier: {test_brier:.4f}")
    
    # Feature importance
    feature_imp = pd.DataFrame({
        "feature": feature_cols,
        "importance": best_model.feature_importances_
    }).sort_values("importance", ascending=False)
    
    print(f"\nTop 10 Features:")
    for i, row in feature_imp.head(10).iterrows():
        print(f"  {row['feature']:25s} {row['importance']:.4f}")
    
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
    
    print(f"\nLift Analysis:")
    print(f"  Base rate: {base_rate:.1%}")
    print(f"  ≥ p75 (>{p75:.2f}): {top75_rate:.1%} ({top75_rate/base_rate:.2f}x lift)")
    print(f"  ≥ p90 (>{p90:.2f}): {top90_rate:.1%} ({top90_rate/base_rate:.2f}x lift)")
    print(f"  ≥ p95 (>{p95:.2f}): {top95_rate:.1%} ({top95_rate/base_rate:.2f}x lift)")
    
    return {
        "horizon": horizon,
        "best_params": best_params,
        "best_model": best_model,
        "val_auc": val_auc,
        "val_brier": val_brier,
        "test_auc": test_auc,
        "test_brier": test_brier,
        "feature_importance": feature_imp,
        "base_rate": base_rate,
        "p75": p75,
        "p90": p90,
        "p95": p95,
        "top75_rate": top75_rate,
        "top90_rate": top90_rate,
        "top95_rate": top95_rate,
        "n_features": len(feature_cols)
    }


def main():
    print("="*60)
    print("HIGH_RANGE_ATR MODEL TRAINING & VALIDATION")
    print("="*60)
    print("\nTarget: Wide trading range (range/ATR >= 2.0)")
    print("Testing horizons: H=4, H=5")
    
    df = load_data()
    print(f"\nLoaded {len(df)} rows from {df['dt'].min()} to {df['dt'].max()}")
    
    X = build_features(df)
    
    results = {}
    
    # Train H=4
    print(f"\n{'='*60}")
    print("TRAINING: H=4")
    print(f"{'='*60}")
    y4 = create_label(X, h=4, threshold=2.0)
    results[4] = train_and_evaluate(X, y4, horizon=4)
    
    # Train H=5
    print(f"\n{'='*60}")
    print("TRAINING: H=5")
    print(f"{'='*60}")
    y5 = create_label(X, h=5, threshold=2.0)
    results[5] = train_and_evaluate(X, y5, horizon=5)
    
    # Compare
    print(f"\n{'='*60}")
    print("COMPARISON")
    print(f"{'='*60}")
    print(f"\n{'Horizon':<10} {'Val AUC':<10} {'Test AUC':<10} {'Base Rate':<12} {'Recommendation'}")
    print("-" * 60)
    
    for h in [4, 5]:
        if results[h]:
            r = results[h]
            rec = "✅ GOOD" if r["test_auc"] >= 0.75 else "⚠️  MODERATE" if r["test_auc"] >= 0.70 else "❌ WEAK"
            print(f"H={h:<8} {r['val_auc']:.4f}     {r['test_auc']:.4f}     {r['base_rate']:.1%}         {rec}")
    
    # Recommend best
    best_h = max([4, 5], key=lambda h: results[h]["test_auc"] if results[h] else 0)
    print(f"\n{'='*60}")
    print(f"RECOMMENDATION: Use H={best_h}")
    print(f"{'='*60}")
    
    if results[best_h]:
        r = results[best_h]
        print(f"  Test AUC: {r['test_auc']:.4f}")
        print(f"  Base rate: {r['base_rate']:.1%}")
        print(f"  Top 10% hit rate: {r['top90_rate']:.1%} ({r['top90_rate']/r['base_rate']:.2f}x lift)")
        print(f"\n  Ready for validation tests!")


if __name__ == "__main__":
    main()
