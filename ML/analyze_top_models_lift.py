#!/usr/bin/env python3
"""
Analyze lift for top models from supersweep:
- low_range_atr H=3
- rv_expand_abs H=21  
- rv_compress_q30 H=30
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
           rsi_14 as rsi, atr_14 as atr, adx_14 as adx,
           bollinger_width as bb_width,
           (close - bollinger_lower) / NULLIF(bollinger_upper - bollinger_lower, 0) as bb_position,
           stochastic_k as stoch_k, stochastic_d as stoch_d,
           roc_20 as cci, minus_di_14 as willr,
           volume_ratio as mfi,
           obv, obv_sma_20,
           realized_volatility_20 as rv_20,
           high_20d as high, low_20d as low
    FROM fact_technical_indicators
    WHERE ticker = '{cfg.ticker}'
    ORDER BY trade_date
    """
    
    df = pd.read_sql(query, engine)
    
    # Calculate OBV normalization
    df['obv_norm'] = df['obv'] / df['obv_sma_20']
    
    engine.dispose()
    return df

# ============================================================================
# Feature Engineering
# ============================================================================
def build_features(df):
    """Build core features"""
    df = df.copy()
    
    # Returns
    df['ret_1d'] = df['close'].pct_change(1)
    df['ret_5d'] = df['close'].pct_change(5)
    df['ret_21d'] = df['close'].pct_change(21)
    
    # Volatility
    df['rv_5d'] = df['ret_1d'].rolling(5).std()
    df['rv_21d'] = df['ret_1d'].rolling(21).std()
    
    # Volume
    df['vol_ma_ratio'] = df['volume'] / df['volume'].rolling(21).mean()
    
    return df

# ============================================================================
# Label Building Functions
# ============================================================================
def build_low_range_atr(df, h, thr=1.0):
    """
    low_range_atr: (high - low) within next h days < thr * ATR
    """
    # Need to get actual high/low from close
    fwd_high = df['close'].shift(-1).rolling(h).max()
    fwd_low = df['close'].shift(-1).rolling(h).min()
    fwd_range = fwd_high - fwd_low
    
    threshold = thr * df['atr']
    return (fwd_range < threshold).astype(int)

def build_rv_expand_abs(df, h, d=0.05):
    """
    rv_expand_abs: realized vol increases by absolute amount d
    """
    rv_now = df['ret_1d'].rolling(21).std()
    rv_future = df['ret_1d'].shift(-h).rolling(21).std()
    
    return ((rv_future - rv_now) > d).astype(int)

def build_rv_compress_q30(df, h):
    """
    rv_compress_q30: future RV falls into bottom 30% quantile
    """
    rv_future = df['ret_1d'].shift(-h).rolling(21).std()
    q30 = rv_future.quantile(0.30)
    
    return (rv_future <= q30).astype(int)

# ============================================================================
# Model Training & Evaluation
# ============================================================================
def train_and_evaluate(df, h, label_name, label_builder, label_params):
    """Train model and compute lift analysis"""
    
    print(f"\n{'='*80}")
    print(f"Model: {label_name} | Horizon: {h} | Params: {label_params}")
    print(f"{'='*80}")
    
    # Build label
    y = label_builder(df, h, **label_params)
    
    # Core features
    feature_cols = [
        'ret_1d', 'ret_5d', 'ret_21d',
        'rv_5d', 'rv_21d',
        'rsi', 'atr', 'adx',
        'bb_width', 'bb_position',
        'macd', 'macd_signal', 'macd_hist',
        'obv_norm',
        'stoch_k', 'stoch_d',
        'cci', 'willr',
        'mfi',
        'vol_ma_ratio'
    ]
    
    # Prepare dataset
    tmp = df[['dt'] + feature_cols].copy()
    tmp['y'] = y
    tmp = tmp.dropna()
    
    if len(tmp) < 100:
        print(f"❌ Insufficient data: {len(tmp)} rows")
        return
    
    # Split data
    dt = pd.to_datetime(tmp['dt'])
    train_end = pd.to_datetime(cfg.train_end)
    val_end = pd.to_datetime(cfg.val_end)
    
    m_train = dt <= train_end
    m_val = (dt > train_end) & (dt <= val_end)
    m_test = dt > val_end
    
    X_train = tmp.loc[m_train, feature_cols]
    y_train = tmp.loc[m_train, 'y'].astype(int)
    
    X_val = tmp.loc[m_val, feature_cols]
    y_val = tmp.loc[m_val, 'y'].astype(int)
    
    X_test = tmp.loc[m_test, feature_cols]
    y_test = tmp.loc[m_test, 'y'].astype(int)
    
    print(f"Train: {len(y_train)} | Val: {len(y_val)} | Test: {len(y_test)}")
    print(f"Base rates - Train: {y_train.mean():.1%} | Val: {y_val.mean():.1%} | Test: {y_test.mean():.1%}")
    
    # Check for valid base rates
    if y_train.mean() < 0.05 or y_train.mean() > 0.95:
        print(f"❌ Invalid base rate: {y_train.mean():.1%} - skipping model")
        return
    
    if len(np.unique(y_train)) < 2 or len(np.unique(y_val)) < 2 or len(np.unique(y_test)) < 2:
        print(f"❌ Insufficient class diversity - skipping model")
        return
    
    # Get best params from supersweep results
    results_df = pd.read_csv('ML/supersweep_staged_results.csv')
    model_row = results_df[
        (results_df['horizon'] == h) & 
        (results_df['label'] == label_name)
    ].sort_values('val_auc', ascending=False).iloc[0]
    
    best_params = eval(model_row['best_params'])
    print(f"Best params from supersweep: {best_params}")
    
    # Train model
    model = XGBClassifier(**best_params, random_state=cfg.seed, verbosity=0)
    model.fit(X_train, y_train)
    
    # Predictions
    p_val = model.predict_proba(X_val)[:, 1]
    p_test = model.predict_proba(X_test)[:, 1]
    
    # Metrics
    val_auc = roc_auc_score(y_val, p_val)
    val_brier = brier_score_loss(y_val, p_val)
    test_auc = roc_auc_score(y_test, p_test)
    test_brier = brier_score_loss(y_test, p_test)
    
    print(f"\n📊 METRICS")
    print(f"Val  - AUC: {val_auc:.4f} | Brier: {val_brier:.4f}")
    print(f"Test - AUC: {test_auc:.4f} | Brier: {test_brier:.4f}")
    
    # Lift Analysis
    lift_quantiles(p_test, y_test, test_auc)

def lift_quantiles(p, y, auc):
    """
    Compute lift at extreme percentiles: p05, p10, p25, p75, p90, p95
    """
    base_rate = float(y.mean())
    
    percentiles = [0.05, 0.10, 0.25, 0.75, 0.90, 0.95]
    thresholds = np.percentile(p, [pct * 100 for pct in percentiles])
    
    results = []
    for pct, thr in zip(percentiles, thresholds):
        if pct <= 0.5:
            # Bottom percentiles
            mask = p <= thr
            segment = f"≤ p{int(pct*100):02d}"
        else:
            # Top percentiles
            mask = p >= thr
            segment = f"≥ p{int(pct*100):02d}"
        
        n = mask.sum()
        if n > 0:
            hit_rate = float(y[mask].mean())
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
    
    print(f"\n🎯 LIFT ANALYSIS (Base Rate: {base_rate:.1%})")
    print(df_lift.to_string(index=False))
    
    # Highlight best segments
    print(f"\n⭐ TOP CONVICTION SIGNALS:")
    print(f"   Bottom 5%:  {results[0]['hit_rate']} hit rate ({results[0]['lift']} lift)")
    print(f"   Bottom 10%: {results[1]['hit_rate']} hit rate ({results[1]['lift']} lift)")
    print(f"   Top 10%:    {results[-2]['hit_rate']} hit rate ({results[-2]['lift']} lift)")
    print(f"   Top 5%:     {results[-1]['hit_rate']} hit rate ({results[-1]['lift']} lift)")

# ============================================================================
# Main
# ============================================================================
def main():
    print(f"Loading data for {cfg.ticker}...")
    df0 = load_data()
    print(f"Loaded {len(df0)} rows")
    
    df = build_features(df0)
    
    # Model 1: low_range_atr H=3
    train_and_evaluate(
        df, h=3, 
        label_name='low_range_atr',
        label_builder=build_low_range_atr,
        label_params={'thr': 1.0}
    )
    
    # Model 2: rv_expand_abs H=21
    train_and_evaluate(
        df, h=21,
        label_name='rv_expand_abs', 
        label_builder=build_rv_expand_abs,
        label_params={'d': 0.05}
    )
    
    # Model 3: rv_compress_q30 H=30
    train_and_evaluate(
        df, h=30,
        label_name='rv_compress_q30',
        label_builder=build_rv_compress_q30,
        label_params={}
    )
    
    print(f"\n{'='*80}")
    print("✓ Analysis complete!")
    print(f"{'='*80}")

if __name__ == "__main__":
    main()
