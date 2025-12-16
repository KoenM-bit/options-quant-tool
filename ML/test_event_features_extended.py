"""
Extended event features test - more labels, more feature combinations
Tests event features with proven supersweep labels plus additional variations
"""
import os
import sys
import math
from pathlib import Path
import numpy as np
import pandas as pd
from sqlalchemy import create_engine, text

from sklearn.metrics import roc_auc_score, brier_score_loss
from sklearn.calibration import CalibratedClassifierCV
from xgboost import XGBClassifier

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))


def load_data_with_events(ticker="AD.AS", start="2020-01-01", end="2025-12-31"):
    """Load data from ml_features_calendar_v2 view"""
    db_url = "postgresql://airflow:airflow@192.168.1.201:5433/ahold_options"
    eng = create_engine(db_url, pool_pre_ping=True)
    
    query = """
    SELECT 
        dt,
        ticker,
        open, high, low, close, volume,
        -- Technical features
        sma_20, sma_50, sma_200,
        ema_12, ema_26,
        macd, macd_signal, macd_histogram,
        rsi_14,
        stochastic_k, stochastic_d,
        roc_20,
        atr_14,
        bollinger_upper, bollinger_middle, bollinger_lower,
        bollinger_width,
        realized_volatility_20,
        parkinson_volatility_20,
        high_20d, low_20d,
        high_52w, low_52w,
        pct_from_high_20d, pct_from_low_20d,
        pct_from_high_52w, pct_from_low_52w,
        volume_sma_20, volume_ratio,
        adx_14, plus_di_14, minus_di_14,
        obv, obv_sma_20,
        -- Event features
        days_to_earnings,
        days_since_earnings,
        is_earnings_week,
        days_to_exdiv,
        is_exdiv_week,
        days_to_opex,
        is_opex_week
    FROM ml_features_calendar_v2
    WHERE ticker = :t 
      AND dt BETWEEN :s AND :e
    ORDER BY dt
    """
    
    df = pd.read_sql(text(query), eng, params={"t": ticker, "s": start, "e": end})
    df["dt"] = pd.to_datetime(df["dt"])
    return df


def build_derived_features(df: pd.DataFrame) -> pd.DataFrame:
    """Add derived features"""
    X = df.copy()
    
    # Price/momentum features
    X["logret_1d"] = np.log(X["close"] / X["close"].shift(1))
    X["logret_5d"] = X["logret_1d"].rolling(5).sum()
    X["logret_10d"] = X["logret_1d"].rolling(10).sum()
    X["gap_1d"] = (X["open"] - X["close"].shift(1)) / X["close"].shift(1)
    
    X["px_vs_sma20"] = X["close"] / X["sma_20"] - 1
    X["px_vs_sma50"] = X["close"] / X["sma_50"] - 1
    X["px_vs_sma200"] = X["close"] / X["sma_200"] - 1
    
    X["atr_pct"] = X["atr_14"] / X["close"]
    X["rv20_logret"] = X["logret_1d"].rolling(20).std() * math.sqrt(252)
    
    X["bb_width_pct"] = X["bollinger_width"] / X["close"]
    X["di_diff"] = X["plus_di_14"] - X["minus_di_14"]
    
    X["macd_z"] = (X["macd"] - X["macd"].rolling(60).mean()) / (X["macd"].rolling(60).std().replace(0, np.nan))
    
    return X


def make_all_labels(df: pd.DataFrame, horizon: int):
    """Create comprehensive label set"""
    close = df["close"]
    high = df["high"]
    low = df["low"]
    atr = df["atr_14"]
    
    # Forward looking values
    fwd_max_high = high.shift(-1).rolling(horizon).max().shift(-(horizon-1))
    fwd_min_low = low.shift(-1).rolling(horizon).min().shift(-(horizon-1))
    fwd_max_close = close.shift(-1).rolling(horizon).max().shift(-(horizon-1))
    fwd_min_close = close.shift(-1).rolling(horizon).min().shift(-(horizon-1))
    fwd_ret = close.shift(-horizon) / close - 1.0
    
    # Up/down days
    up_days = (df["logret_1d"] > 0).astype(float).shift(-1).rolling(horizon).sum().shift(-(horizon-1))
    total_days = horizon
    
    labels = {}
    
    # ============= Range labels (our best performers) =============
    range_atr = (fwd_max_high - fwd_min_low) / atr.replace(0, np.nan)
    labels[f"low_range_1.0_H{horizon}"] = (range_atr <= 1.0).astype(int)
    labels[f"low_range_1.25_H{horizon}"] = (range_atr <= 1.25).astype(int)
    labels[f"high_range_2.0_H{horizon}"] = (range_atr >= 2.0).astype(int)
    labels[f"high_range_1.75_H{horizon}"] = (range_atr >= 1.75).astype(int)
    
    # ============= Volatility expansion/compression =============
    rv = df["realized_volatility_20"].fillna(df["rv20_logret"])
    rv_change = rv.shift(-horizon) - rv
    
    # Absolute thresholds
    labels[f"rv_expand_abs_0.05_H{horizon}"] = (rv_change >= 0.05).astype(int)
    labels[f"rv_expand_abs_0.03_H{horizon}"] = (rv_change >= 0.03).astype(int)
    labels[f"rv_compress_abs_-0.05_H{horizon}"] = (rv_change <= -0.05).astype(int)
    labels[f"rv_compress_abs_-0.03_H{horizon}"] = (rv_change <= -0.03).astype(int)
    
    # ============= ATR-based labels =============
    # Touch labels (high/low touch threshold)
    labels[f"touch_up_0.75atr_H{horizon}"] = (fwd_max_high >= (close + 0.75 * atr)).astype(int)
    labels[f"touch_up_1.0atr_H{horizon}"] = (fwd_max_high >= (close + 1.0 * atr)).astype(int)
    labels[f"touch_up_1.25atr_H{horizon}"] = (fwd_max_high >= (close + 1.25 * atr)).astype(int)
    labels[f"touch_down_0.75atr_H{horizon}"] = (fwd_min_low <= (close - 0.75 * atr)).astype(int)
    labels[f"touch_down_1.0atr_H{horizon}"] = (fwd_min_low <= (close - 1.0 * atr)).astype(int)
    labels[f"touch_down_1.25atr_H{horizon}"] = (fwd_min_low <= (close - 1.25 * atr)).astype(int)
    
    # Close break labels (close must break threshold)
    labels[f"close_break_up_1.0atr_H{horizon}"] = (fwd_max_close >= (close + 1.0 * atr)).astype(int)
    labels[f"close_break_down_1.0atr_H{horizon}"] = (fwd_min_close <= (close - 1.0 * atr)).astype(int)
    
    # Drawdown/drawup labels
    labels[f"dd_1.0atr_H{horizon}"] = (fwd_min_low <= (close - 1.0 * atr)).astype(int)
    labels[f"ud_1.0atr_H{horizon}"] = (fwd_max_high >= (close + 1.0 * atr)).astype(int)
    
    # ============= Trend persistence =============
    labels[f"trend_up_65pct_H{horizon}"] = (up_days >= 0.65 * total_days).astype(int)
    labels[f"trend_down_65pct_H{horizon}"] = (up_days <= 0.35 * total_days).astype(int)
    
    return labels


def train_and_evaluate(X_train, y_train, X_val, y_val, X_test, y_test, seed=42):
    """Train XGBoost model and evaluate"""
    params = {
        "max_depth": 3,
        "learning_rate": 0.05,
        "n_estimators": 400,
        "subsample": 0.85,
        "colsample_bytree": 0.85,
        "reg_lambda": 3.0,
        "min_child_weight": 5,
    }
    
    base = XGBClassifier(
        objective="binary:logistic",
        eval_metric="logloss",
        random_state=seed,
        n_jobs=max(1, os.cpu_count() or 1),
        tree_method="hist",
        **params
    )
    base.fit(X_train, y_train)
    
    # Calibrate
    calib = CalibratedClassifierCV(base, method="sigmoid", cv="prefit")
    calib.fit(X_val, y_val)
    
    # Predictions
    p_val = calib.predict_proba(X_val)[:, 1]
    p_test = calib.predict_proba(X_test)[:, 1]
    
    # Feature importance
    importance = pd.DataFrame({
        'feature': X_train.columns,
        'importance': base.feature_importances_
    }).sort_values('importance', ascending=False)
    
    return {
        "val_auc": roc_auc_score(y_val, p_val) if len(np.unique(y_val)) > 1 else np.nan,
        "val_brier": brier_score_loss(y_val, p_val),
        "test_auc": roc_auc_score(y_test, p_test) if len(np.unique(y_test)) > 1 else np.nan,
        "test_brier": brier_score_loss(y_test, p_test),
        "val_base_rate": float(y_val.mean()),
        "test_base_rate": float(y_test.mean()),
        "importance": importance
    }


def main():
    print("="*80)
    print("EXTENDED EVENT FEATURES TEST")
    print("="*80)
    
    # Load data
    print("\n1. Loading data with event features...")
    df = load_data_with_events()
    df = build_derived_features(df)
    print(f"   Loaded {len(df)} rows")
    
    # Split data
    train_end = pd.to_datetime("2023-12-29")
    val_end = pd.to_datetime("2024-12-31")
    
    m_train = df["dt"] <= train_end
    m_val = (df["dt"] > train_end) & (df["dt"] <= val_end)
    m_test = df["dt"] > val_end
    
    print(f"   Train: {m_train.sum()} | Val: {m_val.sum()} | Test: {m_test.sum()}")
    
    # Define feature sets
    event_features = [
        "days_to_earnings",
        "days_since_earnings", 
        "is_earnings_week",
        "days_to_exdiv",
        "is_exdiv_week",
        "days_to_opex",
        "is_opex_week"
    ]
    
    core_technical = [
        "px_vs_sma20", "px_vs_sma50", "px_vs_sma200",
        "macd", "macd_signal", "rsi_14",
        "stochastic_k", "stochastic_d",
        "adx_14", "di_diff",
        "atr_14", "atr_pct",
        "bb_width_pct",
        "volume_ratio",
        "pct_from_high_20d", "pct_from_low_20d",
        "logret_1d", "logret_5d", "logret_10d"
    ]
    
    # Focus on best horizons from initial test
    horizons = [3, 7, 14]
    
    results = []
    
    for h in horizons:
        print(f"\n{'='*80}")
        print(f"HORIZON {h}")
        print(f"{'='*80}")
        
        # Create labels
        labels = make_all_labels(df, h)
        
        print(f"Testing {len(labels)} labels...")
        
        for label_name, y in labels.items():
            # Prepare data
            data = df.copy()
            data["y"] = y
            data = data.dropna()
            
            if len(data) < 800:
                continue
            
            tr = m_train.loc[data.index]
            va = m_val.loc[data.index]
            te = m_test.loc[data.index]
            
            y_tr = data.loc[tr, "y"].astype(int)
            y_va = data.loc[va, "y"].astype(int)
            y_te = data.loc[te, "y"].astype(int)
            
            # Check base rates
            train_br = y_tr.mean()
            if not (0.10 <= train_br <= 0.90):
                continue
            
            # Tech only
            tech_cols = [c for c in core_technical if c in data.columns]
            X_tr_tech = data.loc[tr, tech_cols]
            X_va_tech = data.loc[va, tech_cols]
            X_te_tech = data.loc[te, tech_cols]
            
            res_tech = train_and_evaluate(X_tr_tech, y_tr, X_va_tech, y_va, X_te_tech, y_te)
            
            # Combined (tech + events)
            event_cols = [c for c in event_features if c in data.columns and data[c].notna().sum() > 100]
            combined_cols = tech_cols + event_cols
            X_tr_comb = data.loc[tr, combined_cols]
            X_va_comb = data.loc[va, combined_cols]
            X_te_comb = data.loc[te, combined_cols]
            
            res_comb = train_and_evaluate(X_tr_comb, y_tr, X_va_comb, y_va, X_te_comb, y_te)
            
            # Calculate improvement
            auc_gain_val = res_comb['val_auc'] - res_tech['val_auc']
            auc_gain_test = res_comb['test_auc'] - res_tech['test_auc']
            
            # Get top event features
            event_importance = res_comb['importance'][res_comb['importance']['feature'].isin(event_cols)]
            top_event = event_importance.head(1)['feature'].values[0] if len(event_importance) > 0 else None
            top_event_imp = event_importance.head(1)['importance'].values[0] if len(event_importance) > 0 else 0.0
            
            # Store results
            results.append({
                "horizon": h,
                "label": label_name,
                "tech_only_val": res_tech['val_auc'],
                "tech_only_test": res_tech['test_auc'],
                "combined_val": res_comb['val_auc'],
                "combined_test": res_comb['test_auc'],
                "gain_val": auc_gain_val,
                "gain_test": auc_gain_test,
                "base_rate": train_br,
                "top_event_feature": top_event,
                "top_event_importance": top_event_imp
            })
            
            # Print progress for significant gains
            if auc_gain_test > 0.03:
                print(f"   ✅ {label_name}: Test AUC {res_tech['test_auc']:.3f} → {res_comb['test_auc']:.3f} (+{auc_gain_test:+.3f}) [{top_event}]")
            elif auc_gain_test < -0.03:
                print(f"   ⚠️  {label_name}: Test AUC {res_tech['test_auc']:.3f} → {res_comb['test_auc']:.3f} ({auc_gain_test:+.3f})")
    
    # Summary
    print("\n" + "="*80)
    print("SUMMARY")
    print("="*80)
    
    if results:
        res_df = pd.DataFrame(results)
        
        # Overall stats
        print(f"\n1. Overall Statistics:")
        print(f"   Labels tested: {len(res_df)}")
        print(f"   Avg tech-only test AUC: {res_df['tech_only_test'].mean():.4f}")
        print(f"   Avg combined test AUC: {res_df['combined_test'].mean():.4f}")
        print(f"   Avg test gain: {res_df['gain_test'].mean():+.4f} ({res_df['gain_test'].mean()/res_df['tech_only_test'].mean()*100:+.1f}%)")
        
        # Best improvements
        print(f"\n2. Top 10 Improvements (by test AUC gain):")
        top_gains = res_df.nlargest(10, 'gain_test')
        for idx, row in top_gains.iterrows():
            print(f"   {row['label']:<40} {row['tech_only_test']:.3f} → {row['combined_test']:.3f} (+{row['gain_test']:.3f}) [{row['top_event_feature']}]")
        
        # Worst cases
        print(f"\n3. Worst 5 Cases (negative impact):")
        worst = res_df.nsmallest(5, 'gain_test')
        for idx, row in worst.iterrows():
            print(f"   {row['label']:<40} {row['tech_only_test']:.3f} → {row['combined_test']:.3f} ({row['gain_test']:.3f})")
        
        # By label type
        print(f"\n4. Results by Label Type:")
        res_df['label_type'] = res_df['label'].str.extract(r'([a-z_]+)_')[0]
        by_type = res_df.groupby('label_type').agg({
            'gain_test': ['mean', 'count'],
            'combined_test': 'mean',
            'tech_only_test': 'mean'
        }).round(4)
        print(by_type)
        
        # By horizon
        print(f"\n5. Results by Horizon:")
        by_h = res_df.groupby('horizon').agg({
            'gain_test': 'mean',
            'combined_test': 'mean',
            'tech_only_test': 'mean'
        }).round(4)
        print(by_h)
        
        # Event feature effectiveness
        print(f"\n6. Most Effective Event Features:")
        event_counts = res_df[res_df['gain_test'] > 0.02]['top_event_feature'].value_counts()
        print(event_counts.head(5))
        
        # Save results
        res_df.to_csv("ML/event_features_extended_results.csv", index=False)
        print(f"\n✅ Results saved to ML/event_features_extended_results.csv")
    
    print("\nDONE")


if __name__ == "__main__":
    main()
