#!/usr/bin/env python3
"""
Leakage Detection for rv_compress_q30 H=30 model

Tests:
1. Walk-forward validation (rolling windows)
2. Feature importance analysis (check for future-leaking features)
3. Label construction verification (ensure no forward-looking data in features)
4. Time-series split validation
5. Permutation test (shuffle labels to check if AUC drops)
"""
import sys
import os
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

import pandas as pd
import numpy as np
from xgboost import XGBClassifier
from sklearn.metrics import roc_auc_score, brier_score_loss
from sklearn.model_selection import TimeSeriesSplit
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
def build_rv_compress_q30(df, h):
    """
    rv_compress_q30: future RV falls into bottom 30% quantile
    
    CRITICAL: This must be calculated ONLY on training data to avoid leakage
    """
    rv_now = df['ret_1d'].rolling(21).std()
    rv_future = df['ret_1d'].shift(-h).rolling(21).std()
    
    return rv_now, rv_future

# ============================================================================
# Test 1: Walk-Forward Validation
# ============================================================================
def test_walk_forward():
    """
    Walk-forward validation with rolling windows.
    If there's leakage, performance should degrade significantly.
    """
    print(f"\n{'='*80}")
    print("TEST 1: WALK-FORWARD VALIDATION")
    print(f"{'='*80}")
    
    df = load_data()
    df = build_features(df)
    
    h = 30
    
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
    
    # Walk-forward splits
    train_size = 500
    val_size = 150
    test_size = 100
    step_size = 50  # Move forward by 50 days each time
    
    results = []
    
    for i in range(0, len(df) - train_size - val_size - test_size, step_size):
        train_end = i + train_size
        val_end = train_end + val_size
        test_end = val_end + test_size
        
        if test_end > len(df):
            break
        
        # Split data
        df_train = df.iloc[i:train_end].copy()
        df_val = df.iloc[train_end:val_end].copy()
        df_test = df.iloc[val_end:test_end].copy()
        
        # Build label - CRITICAL: quantile calculated ONLY on training data
        rv_now_train, rv_future_train = build_rv_compress_q30(df_train, h)
        rv_diff_train = rv_future_train - rv_now_train
        q30_threshold = rv_diff_train.quantile(0.30)
        
        # Apply threshold to all splits
        rv_now_val, rv_future_val = build_rv_compress_q30(df_val, h)
        rv_now_test, rv_future_test = build_rv_compress_q30(df_test, h)
        
        y_train = ((rv_future_train - rv_now_train) <= q30_threshold).astype(int)
        y_val = ((rv_future_val - rv_now_val) <= q30_threshold).astype(int)
        y_test = ((rv_future_test - rv_now_test) <= q30_threshold).astype(int)
        
        # Prepare features
        X_train = df_train[feature_cols].copy()
        X_val = df_val[feature_cols].copy()
        X_test = df_test[feature_cols].copy()
        
        # Combine with labels and drop NaN
        X_train['y'] = y_train
        X_val['y'] = y_val
        X_test['y'] = y_test
        
        X_train = X_train.dropna()
        X_val = X_val.dropna()
        X_test = X_test.dropna()
        
        if len(X_train) < 100 or len(X_val) < 30 or len(X_test) < 30:
            continue
        
        y_train = X_train['y'].astype(int)
        y_val = X_val['y'].astype(int)
        y_test = X_test['y'].astype(int)
        
        X_train = X_train[feature_cols]
        X_val = X_val[feature_cols]
        X_test = X_test[feature_cols]
        
        # Check class balance
        if len(np.unique(y_train)) < 2 or len(np.unique(y_test)) < 2:
            continue
        
        # Train model
        model = XGBClassifier(
            max_depth=5, learning_rate=0.05, n_estimators=400,
            subsample=1.0, colsample_bytree=0.85, reg_lambda=1.0,
            min_child_weight=5, random_state=cfg.seed, verbosity=0
        )
        
        try:
            model.fit(X_train, y_train)
            
            p_val = model.predict_proba(X_val)[:, 1]
            p_test = model.predict_proba(X_test)[:, 1]
            
            val_auc = roc_auc_score(y_val, p_val) if len(np.unique(y_val)) > 1 else np.nan
            test_auc = roc_auc_score(y_test, p_test) if len(np.unique(y_test)) > 1 else np.nan
            
            results.append({
                'fold': len(results) + 1,
                'train_dates': f"{df_train['dt'].iloc[0]} to {df_train['dt'].iloc[-1]}",
                'test_dates': f"{df_test['dt'].iloc[0]} to {df_test['dt'].iloc[-1]}",
                'train_size': len(X_train),
                'test_size': len(X_test),
                'train_base_rate': y_train.mean(),
                'test_base_rate': y_test.mean(),
                'val_auc': val_auc,
                'test_auc': test_auc,
            })
            
            print(f"Fold {len(results)}: Test {df_test['dt'].iloc[0].date()} - AUC: {test_auc:.4f} | Base rate: {y_test.mean():.1%}")
        
        except Exception as e:
            print(f"Fold {len(results) + 1}: Error - {e}")
            continue
    
    if results:
        df_results = pd.DataFrame(results)
        print(f"\n📊 WALK-FORWARD SUMMARY ({len(results)} folds)")
        print(f"Mean Test AUC: {df_results['test_auc'].mean():.4f} ± {df_results['test_auc'].std():.4f}")
        print(f"Min Test AUC: {df_results['test_auc'].min():.4f}")
        print(f"Max Test AUC: {df_results['test_auc'].max():.4f}")
        print(f"\nIf leakage exists, AUC should be much lower than 0.936!")
        
        return df_results
    else:
        print("❌ No valid folds generated")
        return None

# ============================================================================
# Test 2: Feature Importance Analysis
# ============================================================================
def test_feature_importance():
    """
    Check which features are most important.
    If future-looking features dominate, there's leakage.
    """
    print(f"\n{'='*80}")
    print("TEST 2: FEATURE IMPORTANCE ANALYSIS")
    print(f"{'='*80}")
    
    df = load_data()
    df = build_features(df)
    
    h = 30
    
    # Build label on full data for quantile
    rv_now, rv_future = build_rv_compress_q30(df, h)
    rv_diff = rv_future - rv_now
    q30 = rv_diff.quantile(0.30)
    y = ((rv_future - rv_now) <= q30).astype(int)
    
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
    
    tmp = df[feature_cols].copy()
    tmp['y'] = y
    tmp = tmp.dropna()
    
    X = tmp[feature_cols]
    y = tmp['y'].astype(int)
    
    # Split
    split_idx = int(len(X) * 0.7)
    X_train, y_train = X.iloc[:split_idx], y.iloc[:split_idx]
    X_test, y_test = X.iloc[split_idx:], y.iloc[split_idx:]
    
    # Train model
    model = XGBClassifier(
        max_depth=5, learning_rate=0.05, n_estimators=400,
        subsample=1.0, colsample_bytree=0.85, reg_lambda=1.0,
        min_child_weight=5, random_state=cfg.seed, verbosity=0
    )
    model.fit(X_train, y_train)
    
    # Get feature importance
    importance = pd.DataFrame({
        'feature': feature_cols,
        'importance': model.feature_importances_
    }).sort_values('importance', ascending=False)
    
    print(f"\n📊 TOP 10 MOST IMPORTANT FEATURES:")
    print(importance.head(10).to_string(index=False))
    
    print(f"\n⚠️  SUSPICIOUS FEATURES TO WATCH:")
    print(f"   - rv_21d: Current 21-day realized vol (should be low importance if no leakage)")
    print(f"   - rv_5d: Current 5-day realized vol (should be low importance if no leakage)")
    print(f"   - ret_21d: 21-day return (overlaps with H=30 horizon)")
    
    return importance

# ============================================================================
# Test 3: Label Timing Check
# ============================================================================
def test_label_timing():
    """
    Verify label is constructed correctly without lookahead bias.
    """
    print(f"\n{'='*80}")
    print("TEST 3: LABEL TIMING VERIFICATION")
    print(f"{'='*80}")
    
    df = load_data()
    df = build_features(df)
    
    h = 30
    
    # Show first few rows to verify timing
    rv_now, rv_future = build_rv_compress_q30(df, h)
    
    sample = pd.DataFrame({
        'dt': df['dt'],
        'close': df['close'],
        'ret_1d': df['ret_1d'],
        'rv_now': rv_now,
        'rv_future': rv_future,
        'rv_diff': rv_future - rv_now
    }).iloc[50:60]
    
    print(f"\n📊 SAMPLE ROWS (showing timing):")
    print(sample.to_string(index=False))
    
    print(f"\n✅ VERIFICATION:")
    print(f"   - rv_now: Rolling 21-day std of PAST returns (includes today)")
    print(f"   - rv_future: Rolling 21-day std of returns {h} days AHEAD")
    print(f"   - Label = 1 if rv_future - rv_now <= q30 (compression)")
    print(f"\n   ⚠️  POTENTIAL ISSUE:")
    print(f"   - rv_future uses shift(-{h}), which means it looks at returns")
    print(f"     from day {h} to day {h}+21")
    print(f"   - But rv_now INCLUDES today's return in the rolling window!")
    print(f"   - This creates a {h}-day gap but rv_now still sees recent vol")

# ============================================================================
# Test 4: Permutation Test
# ============================================================================
def test_permutation():
    """
    Shuffle labels and retrain. If AUC stays high, there's severe leakage.
    """
    print(f"\n{'='*80}")
    print("TEST 4: PERMUTATION TEST (Shuffle Labels)")
    print(f"{'='*80}")
    
    df = load_data()
    df = build_features(df)
    
    h = 30
    
    # Build label
    rv_now, rv_future = build_rv_compress_q30(df, h)
    rv_diff = rv_future - rv_now
    q30 = rv_diff.quantile(0.30)
    y = ((rv_future - rv_now) <= q30).astype(int)
    
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
    
    tmp = df[feature_cols].copy()
    tmp['y'] = y
    tmp = tmp.dropna()
    
    X = tmp[feature_cols]
    y_real = tmp['y'].astype(int)
    
    # Split
    split_idx = int(len(X) * 0.7)
    X_train, X_test = X.iloc[:split_idx], X.iloc[split_idx:]
    y_train_real, y_test_real = y_real.iloc[:split_idx], y_real.iloc[split_idx:]
    
    # Train on REAL labels
    model_real = XGBClassifier(
        max_depth=5, learning_rate=0.05, n_estimators=400,
        subsample=1.0, colsample_bytree=0.85, reg_lambda=1.0,
        min_child_weight=5, random_state=cfg.seed, verbosity=0
    )
    model_real.fit(X_train, y_train_real)
    p_test_real = model_real.predict_proba(X_test)[:, 1]
    auc_real = roc_auc_score(y_test_real, p_test_real)
    
    # Train on SHUFFLED labels (5 trials)
    auc_shuffled = []
    for trial in range(5):
        y_train_shuffled = y_train_real.sample(frac=1, random_state=cfg.seed + trial).values
        
        model_shuffled = XGBClassifier(
            max_depth=5, learning_rate=0.05, n_estimators=400,
            subsample=1.0, colsample_bytree=0.85, reg_lambda=1.0,
            min_child_weight=5, random_state=cfg.seed, verbosity=0
        )
        model_shuffled.fit(X_train, y_train_shuffled)
        p_test_shuffled = model_shuffled.predict_proba(X_test)[:, 1]
        auc_shuffled.append(roc_auc_score(y_test_real, p_test_shuffled))
        print(f"Trial {trial + 1}: Shuffled AUC = {auc_shuffled[-1]:.4f}")
    
    print(f"\n📊 PERMUTATION TEST RESULTS:")
    print(f"Real labels AUC: {auc_real:.4f}")
    print(f"Shuffled labels AUC: {np.mean(auc_shuffled):.4f} ± {np.std(auc_shuffled):.4f}")
    print(f"\n✅ INTERPRETATION:")
    print(f"   - If shuffled AUC ≈ 0.5: Good! Model learned real patterns")
    print(f"   - If shuffled AUC > 0.6: LEAKAGE! Model uses future information")

# ============================================================================
# Main
# ============================================================================
def main():
    print(f"{'='*80}")
    print(f"LEAKAGE DETECTION: rv_compress_q30 H=30")
    print(f"{'='*80}")
    print(f"Model claims 93.6% test AUC - testing for data leakage...")
    
    # Test 1: Walk-forward validation
    wf_results = test_walk_forward()
    
    # Test 2: Feature importance
    importance = test_feature_importance()
    
    # Test 3: Label timing
    test_label_timing()
    
    # Test 4: Permutation test
    test_permutation()
    
    print(f"\n{'='*80}")
    print(f"FINAL ASSESSMENT")
    print(f"{'='*80}")
    print(f"Review the tests above:")
    print(f"1. Walk-forward AUC should be << 0.936 if no leakage")
    print(f"2. Top features should NOT be rv_5d, rv_21d (current volatility)")
    print(f"3. Label timing should show proper {30}-day forward gap")
    print(f"4. Shuffled labels should produce AUC ≈ 0.5")
    print(f"\nIf any test fails, there's likely leakage in the model!")

if __name__ == "__main__":
    main()
