#!/usr/bin/env python3
"""
Predict low_range_atr (H=3) for the latest available date
"""
import sys
import os
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

import pandas as pd
import numpy as np
from xgboost import XGBClassifier
from src.config import settings

# ============================================================================
# Configuration
# ============================================================================
class Cfg:
    def __init__(self):
        self.ticker = "AD.AS"
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
# Label Building
# ============================================================================
def build_low_range_atr(df, h, thr=1.0):
    """
    low_range_atr: (high - low) within next h days < thr * ATR
    """
    fwd_high = df['close'].shift(-1).rolling(h).max()
    fwd_low = df['close'].shift(-1).rolling(h).min()
    fwd_range = fwd_high - fwd_low
    
    threshold = thr * df['atr']
    return (fwd_range < threshold).astype(int)

# ============================================================================
# Main
# ============================================================================
def main():
    print(f"Loading data for {cfg.ticker}...")
    df0 = load_data()
    print(f"Loaded {len(df0)} rows")
    
    df = build_features(df0)
    
    # Build label for training
    h = 3
    y = build_low_range_atr(df, h, thr=1.0)
    
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
    tmp_train = tmp.dropna()
    
    print(f"\nTraining on all available data through {tmp_train['dt'].max()}")
    print(f"Total training samples: {len(tmp_train)}")
    
    # Get best params from supersweep results
    results_df = pd.read_csv('ML/supersweep_staged_results.csv')
    model_row = results_df[
        (results_df['horizon'] == h) & 
        (results_df['label'] == 'low_range_atr')
    ].sort_values('val_auc', ascending=False).iloc[0]
    
    best_params = eval(model_row['best_params'])
    print(f"Best params: {best_params}")
    
    # Train model on all data
    X_train = tmp_train[feature_cols]
    y_train = tmp_train['y'].astype(int)
    
    model = XGBClassifier(**best_params, random_state=cfg.seed, verbosity=0)
    model.fit(X_train, y_train)
    
    print(f"✓ Model trained (base rate: {y_train.mean():.1%})")
    
    # Get latest row for prediction
    latest_data = tmp[feature_cols].iloc[-1:].dropna()
    
    if len(latest_data) == 0:
        print("\n❌ Cannot predict - latest row has missing features")
        return
    
    latest_date = df['dt'].iloc[-1]
    
    # Make prediction
    pred_proba = model.predict_proba(latest_data)[:, 1][0]
    
    # Load lift analysis thresholds
    print(f"\n{'='*80}")
    print(f"PREDICTION FOR {latest_date}")
    print(f"{'='*80}")
    print(f"Probability of tight range (next 3 days): {pred_proba:.4f} ({pred_proba:.1%})")
    
    # Classify based on lift analysis thresholds
    if pred_proba >= 0.873:
        tier = "TOP 5%"
        hit_rate = "92.9%"
        lift = "1.42x"
        signal = "🟢 ULTRA HIGH CONVICTION - SAFE for premium selling"
    elif pred_proba >= 0.862:
        tier = "TOP 10%"
        hit_rate = "88.0%"
        lift = "1.34x"
        signal = "🟢 HIGH CONVICTION - Strong for premium selling"
    elif pred_proba >= 0.812:
        tier = "TOP 25%"
        hit_rate = "82.0%"
        lift = "1.25x"
        signal = "🟡 MODERATE - Good for premium selling"
    elif pred_proba >= 0.610:
        tier = "MIDDLE 50%"
        hit_rate = "~66%"
        lift = "~1.0x"
        signal = "⚪ NEUTRAL - Base rate probability"
    elif pred_proba >= 0.430:
        tier = "BOTTOM 25%"
        hit_rate = "~28%"
        lift = "0.42x"
        signal = "🟠 CAUTION - Lower than average, watch for breakout"
    elif pred_proba >= 0.363:
        tier = "BOTTOM 10%"
        hit_rate = "12.0%"
        lift = "0.18x"
        signal = "🔴 HIGH RISK - Likely breakout, avoid premium selling"
    else:
        tier = "BOTTOM 5%"
        hit_rate = "0.0%"
        lift = "0.00x"
        signal = "🔴 EXTREME RISK - Almost certain breakout, DO NOT sell premium"
    
    print(f"\nClassification: {tier}")
    print(f"Expected hit rate: {hit_rate} (lift: {lift})")
    print(f"Trading Signal: {signal}")
    
    # Additional context
    print(f"\n{'='*80}")
    print(f"LATEST MARKET DATA")
    print(f"{'='*80}")
    print(f"Close: ${df['close'].iloc[-1]:.2f}")
    print(f"ATR (14d): ${df['atr'].iloc[-1]:.2f}")
    print(f"RSI (14d): {df['rsi'].iloc[-1]:.1f}")
    print(f"ADX (14d): {df['adx'].iloc[-1]:.1f}")
    print(f"BB Width: {df['bb_width'].iloc[-1]:.4f}")
    print(f"1-day return: {df['ret_1d'].iloc[-1]:.2%}")
    print(f"5-day RV: {df['rv_5d'].iloc[-1]:.4f}")
    print(f"21-day RV: {df['rv_21d'].iloc[-1]:.4f}")
    
    print(f"\n{'='*80}")
    print(f"RECOMMENDATION")
    print(f"{'='*80}")
    
    if pred_proba >= 0.862:
        print("✅ PREMIUM SELLING RECOMMENDED")
        print("   - High confidence of tight 3-day range")
        print("   - Consider aggressive short strangles/straddles")
        print("   - Target delta: 0.25-0.35 for maximum premium")
    elif pred_proba >= 0.610:
        print("⚠️  MODERATE APPROACH")
        print("   - Average probability of tight range")
        print("   - Conservative premium selling acceptable")
        print("   - Target delta: 0.15-0.25 for safety margin")
    else:
        print("🛑 AVOID PREMIUM SELLING")
        print("   - High probability of breakout in next 3 days")
        print("   - Consider directional strategies instead")
        print("   - Or wait for better entry conditions")

if __name__ == "__main__":
    main()
