#!/usr/bin/env python3
"""
Test rv_compress_q30 H=30 WITHOUT volatility features to verify true predictive power.

Excludes: rv_5d, rv_21d, atr, atr_pct, bb_width, bb_width_pct, 
          realized_volatility_20, parkinson_volatility_20, rv20_logret

Uses only: price, momentum, trend, volume, breadth features
"""
import sys
import os
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

import pandas as pd
import numpy as np
from xgboost import XGBClassifier
from sklearn.metrics import roc_auc_score, brier_score_loss
from src.config import settings

# ============================================================================
# Configuration
# ============================================================================
class Cfg:
    def __init__(self):
        self.ticker = "AD.AS"
        self.train_end = "2023-12-31"
        self.val_end = "2024-12-31"
        self.seed = 42

cfg = Cfg()

# ============================================================================
# Data Loading
# ============================================================================
def load_data():
    """Load technical indicators from database"""
    import sqlalchemy as sa
    engine = sa.create_engine(settings.database_url)
    
    query = f"""
    SELECT trade_date as dt, ticker, close, volume,
           sma_20, sma_50, ema_12, ema_26,
           macd, macd_signal, macd_histogram as macd_hist,
           rsi_14 as rsi, adx_14 as adx,
           stochastic_k as stoch_k, stochastic_d as stoch_d,
           roc_20 as cci, minus_di_14 as willr,
           volume_ratio as mfi,
           obv, obv_sma_20,
           pct_from_high_20d, pct_from_low_20d,
           pct_from_high_52w, pct_from_low_52w
    FROM fact_technical_indicators
    WHERE ticker = '{cfg.ticker}'
    ORDER BY trade_date
    """
    
    df = pd.read_sql(query, engine)
    df['obv_norm'] = df['obv'] / df['obv_sma_20']
    engine.dispose()
    return df

# ============================================================================
# Feature Engineering (NO VOLATILITY FEATURES)
# ============================================================================
def build_features(df):
    """Build features WITHOUT volatility measures"""
    df = df.copy()
    
    # Returns
    df['ret_1d'] = df['close'].pct_change(1)
    df['ret_5d'] = df['close'].pct_change(5)
    df['ret_21d'] = df['close'].pct_change(21)
    
    # Price vs moving averages
    df['px_vs_sma20'] = df['close'] / df['sma_20'] - 1
    df['px_vs_sma50'] = df['close'] / df['sma_50'] - 1
    
    # Volume
    df['vol_ma_ratio'] = df['volume'] / df['volume'].rolling(21).mean()
    
    return df

# ============================================================================
# Label Building
# ============================================================================
def build_rv_compress_q30(df, h):
    """
    rv_compress_q30: future RV falls into bottom 30% quantile
    """
    rv_now = df['ret_1d'].rolling(21).std()
    rv_future = df['ret_1d'].shift(-h).rolling(21).std()
    
    return rv_now, rv_future

# ============================================================================
# Main Test
# ============================================================================
def main():
    print(f"{'='*80}")
    print(f"CLEAN TEST: rv_compress_q30 H=30 WITHOUT Volatility Features")
    print(f"{'='*80}")
    print(f"Excluded features: rv_5d, rv_21d, atr, atr_pct, bb_width, bb_width_pct,")
    print(f"                   realized_volatility_20, parkinson_volatility_20, rv20_logret")
    print(f"\nUsing only: price, momentum, trend, volume, breadth features")
    print(f"{'='*80}\n")
    
    df = load_data()
    df = build_features(df)
    
    h = 30
    
    # Feature set WITHOUT volatility
    feature_cols = [
        # Returns (momentum)
        'ret_1d', 'ret_5d', 'ret_21d',
        
        # Price position (trend)
        'px_vs_sma20', 'px_vs_sma50',
        
        # Momentum indicators
        'rsi', 'stoch_k', 'stoch_d',
        'macd', 'macd_signal', 'macd_hist',
        
        # Trend strength
        'adx', 'cci', 'willr',
        
        # Volume
        'mfi', 'obv_norm', 'vol_ma_ratio',
        
        # Breadth (price extremes)
        'pct_from_high_20d', 'pct_from_low_20d',
        'pct_from_high_52w', 'pct_from_low_52w'
    ]
    
    print(f"Using {len(feature_cols)} features (NO volatility measures)")
    
    # Build label - quantile on TRAINING data only
    dt = pd.to_datetime(df['dt'])
    train_end = pd.to_datetime(cfg.train_end)
    val_end = pd.to_datetime(cfg.val_end)
    
    m_train = dt <= train_end
    m_val = (dt > train_end) & (dt <= val_end)
    m_test = dt > val_end
    
    rv_now, rv_future = build_rv_compress_q30(df, h)
    rv_diff = rv_future - rv_now
    
    # Calculate quantile ONLY on training data
    q30_threshold = rv_diff[m_train].quantile(0.30)
    print(f"q30 threshold (from training): {q30_threshold:.6f}")
    
    y = (rv_diff <= q30_threshold).astype(int)
    
    # Prepare dataset
    tmp = df[['dt'] + feature_cols].copy()
    tmp['y'] = y
    tmp = tmp.dropna()
    
    # Apply masks
    m_train_clean = m_train.loc[tmp.index]
    m_val_clean = m_val.loc[tmp.index]
    m_test_clean = m_test.loc[tmp.index]
    
    X_train = tmp.loc[m_train_clean, feature_cols]
    y_train = tmp.loc[m_train_clean, 'y'].astype(int)
    
    X_val = tmp.loc[m_val_clean, feature_cols]
    y_val = tmp.loc[m_val_clean, 'y'].astype(int)
    
    X_test = tmp.loc[m_test_clean, feature_cols]
    y_test = tmp.loc[m_test_clean, 'y'].astype(int)
    
    print(f"\nData splits:")
    print(f"Train: {len(y_train)} samples | Base rate: {y_train.mean():.1%}")
    print(f"Val:   {len(y_val)} samples | Base rate: {y_val.mean():.1%}")
    print(f"Test:  {len(y_test)} samples | Base rate: {y_test.mean():.1%}")
    
    # Train model with best params from supersweep
    print(f"\nTraining XGBoost model...")
    model = XGBClassifier(
        max_depth=5, 
        learning_rate=0.05, 
        n_estimators=400,
        subsample=1.0, 
        colsample_bytree=0.85, 
        reg_lambda=1.0,
        min_child_weight=5, 
        random_state=cfg.seed, 
        verbosity=0
    )
    model.fit(X_train, y_train)
    
    # Predictions
    p_val = model.predict_proba(X_val)[:, 1]
    p_test = model.predict_proba(X_test)[:, 1]
    
    # Metrics
    val_auc = roc_auc_score(y_val, p_val)
    val_brier = brier_score_loss(y_val, p_val)
    test_auc = roc_auc_score(y_test, p_test)
    test_brier = brier_score_loss(y_test, p_test)
    
    print(f"\n{'='*80}")
    print(f"RESULTS WITHOUT VOLATILITY FEATURES")
    print(f"{'='*80}")
    print(f"Val  AUC: {val_auc:.4f} | Brier: {val_brier:.4f}")
    print(f"Test AUC: {test_auc:.4f} | Brier: {test_brier:.4f}")
    
    print(f"\n{'='*80}")
    print(f"COMPARISON TO ORIGINAL")
    print(f"{'='*80}")
    print(f"Original (with vol features): Test AUC = 0.9361")
    print(f"Clean (no vol features):      Test AUC = {test_auc:.4f}")
    print(f"\nDifference: {0.9361 - test_auc:.4f} AUC points")
    print(f"Percentage of performance from leakage: {(0.9361 - test_auc) / 0.9361 * 100:.1f}%")
    
    # Feature importance
    importance = pd.DataFrame({
        'feature': feature_cols,
        'importance': model.feature_importances_
    }).sort_values('importance', ascending=False)
    
    print(f"\n{'='*80}")
    print(f"TOP 10 FEATURES (No Volatility)")
    print(f"{'='*80}")
    print(importance.head(10).to_string(index=False))
    
    # Lift analysis
    print(f"\n{'='*80}")
    print(f"LIFT ANALYSIS (Clean Model)")
    print(f"{'='*80}")
    
    base_rate = y_test.mean()
    percentiles = [0.05, 0.10, 0.25, 0.75, 0.90, 0.95]
    thresholds = np.percentile(p_test, [pct * 100 for pct in percentiles])
    
    results = []
    for pct, thr in zip(percentiles, thresholds):
        if pct <= 0.5:
            mask = p_test <= thr
            segment = f"≤ p{int(pct*100):02d}"
        else:
            mask = p_test >= thr
            segment = f"≥ p{int(pct*100):02d}"
        
        n = mask.sum()
        if n > 0:
            hit_rate = float(y_test[mask].mean())
            lift = hit_rate / base_rate if base_rate > 0 else 0
        else:
            hit_rate = 0
            lift = 0
        
        results.append({
            'segment': segment,
            'threshold': f"{thr:.4f}",
            'n': int(n),
            'hit_rate': f"{hit_rate:.1%}",
            'lift': f"{lift:.2f}x"
        })
    
    df_lift = pd.DataFrame(results)
    print(f"\nBase Rate: {base_rate:.1%}")
    print(df_lift.to_string(index=False))
    
    print(f"\n{'='*80}")
    print(f"FINAL ASSESSMENT")
    print(f"{'='*80}")
    
    if test_auc >= 0.65:
        print(f"✅ LEGITIMATE SIGNAL: {test_auc:.4f} AUC without volatility features")
        print(f"   The model has real predictive power beyond just current vol.")
        print(f"   However, {(0.9361 - test_auc) / 0.9361 * 100:.1f}% of original performance was from leakage.")
    elif test_auc >= 0.55:
        print(f"⚠️  WEAK SIGNAL: {test_auc:.4f} AUC without volatility features")
        print(f"   Some predictive power exists, but most came from leakage.")
        print(f"   Consider using with extreme caution or not at all.")
    else:
        print(f"❌ NO SIGNAL: {test_auc:.4f} AUC without volatility features")
        print(f"   Nearly all performance was from leakage (current vol features).")
        print(f"   This model should NOT be used for trading.")
    
    print(f"\n✅ TRUE PREDICTIVE POWER: {test_auc:.4f} AUC")
    print(f"🚨 LEAKAGE CONTRIBUTION: {0.9361 - test_auc:.4f} AUC")

if __name__ == "__main__":
    main()
