"""
Test predictive power of event features (earnings, dividends, OPEX)
Simple focused test using the same framework as supersweep
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

from src.config import settings


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


def make_labels(df: pd.DataFrame, horizon: int):
    """Create simple labels to test"""
    close = df["close"]
    high = df["high"]
    low = df["low"]
    atr = df["atr_14"]
    
    # Forward looking values
    fwd_max_high = high.shift(-1).rolling(horizon).max().shift(-(horizon-1))
    fwd_min_low = low.shift(-1).rolling(horizon).min().shift(-(horizon-1))
    fwd_ret = close.shift(-horizon) / close - 1.0
    
    labels = {}
    
    # Range labels (our best performers)
    range_atr = (fwd_max_high - fwd_min_low) / atr.replace(0, np.nan)
    labels[f"low_range_H{horizon}"] = (range_atr <= 1.0).astype(int)
    labels[f"high_range_H{horizon}"] = (range_atr >= 2.0).astype(int)
    
    # Volatility expansion/compression
    rv = df["realized_volatility_20"].fillna(df["rv20_logret"])
    rv_change = rv.shift(-horizon) - rv
    labels[f"rv_expand_H{horizon}"] = (rv_change >= 0.05).astype(int)
    labels[f"rv_compress_H{horizon}"] = (rv_change <= -0.05).astype(int)
    
    # ATR touch labels
    labels[f"touch_up_1atr_H{horizon}"] = (fwd_max_high >= (close + 1.0 * atr)).astype(int)
    labels[f"touch_down_1atr_H{horizon}"] = (fwd_min_low <= (close - 1.0 * atr)).astype(int)
    
    return labels


def train_and_evaluate(X_train, y_train, X_val, y_val, X_test, y_test, seed=42):
    """Train XGBoost model and evaluate"""
    # Simple params
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
        "importance": importance,
        "model": calib
    }


def main():
    print("="*80)
    print("EVENT FEATURES PREDICTIVE POWER TEST")
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
    
    # Test horizons that worked well before
    horizons = [3, 7, 14, 30]
    
    results = []
    
    for h in horizons:
        print(f"\n{'='*80}")
        print(f"HORIZON {h}")
        print(f"{'='*80}")
        
        # Create labels
        labels = make_labels(df, h)
        
        for label_name, y in labels.items():
            print(f"\n--- Label: {label_name} ---")
            
            # Prepare data
            data = df.copy()
            data["y"] = y
            data = data.dropna()
            
            if len(data) < 800:
                print(f"   ⚠️  Insufficient data ({len(data)} rows)")
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
                print(f"   ⚠️  Extreme base rate: {train_br:.2%}")
                continue
            
            print(f"   Base rates - Train: {train_br:.2%}, Val: {y_va.mean():.2%}, Test: {y_te.mean():.2%}")
            
            # Test 1: Technical features only (baseline)
            tech_cols = [c for c in core_technical if c in data.columns]
            X_tr_tech = data.loc[tr, tech_cols]
            X_va_tech = data.loc[va, tech_cols]
            X_te_tech = data.loc[te, tech_cols]
            
            print(f"\n   Test 1: Technical features only ({len(tech_cols)} features)")
            res_tech = train_and_evaluate(X_tr_tech, y_tr, X_va_tech, y_va, X_te_tech, y_te)
            print(f"   Val AUC: {res_tech['val_auc']:.4f} | Test AUC: {res_tech['test_auc']:.4f}")
            
            # Test 2: Event features only
            event_cols = [c for c in event_features if c in data.columns and data[c].notna().sum() > 100]
            if len(event_cols) >= 3:
                X_tr_event = data.loc[tr, event_cols]
                X_va_event = data.loc[va, event_cols]
                X_te_event = data.loc[te, event_cols]
                
                print(f"\n   Test 2: Event features only ({len(event_cols)} features)")
                res_event = train_and_evaluate(X_tr_event, y_tr, X_va_event, y_va, X_te_event, y_te)
                print(f"   Val AUC: {res_event['val_auc']:.4f} | Test AUC: {res_event['test_auc']:.4f}")
                print(f"   Top 3 event features:")
                for _, row in res_event['importance'].head(3).iterrows():
                    print(f"      {row['feature']}: {row['importance']:.4f}")
            else:
                print(f"\n   Test 2: SKIPPED (only {len(event_cols)} event features available)")
                res_event = None
            
            # Test 3: Combined (technical + event)
            combined_cols = tech_cols + event_cols
            X_tr_comb = data.loc[tr, combined_cols]
            X_va_comb = data.loc[va, combined_cols]
            X_te_comb = data.loc[te, combined_cols]
            
            print(f"\n   Test 3: Technical + Event ({len(combined_cols)} features)")
            res_comb = train_and_evaluate(X_tr_comb, y_tr, X_va_comb, y_va, X_te_comb, y_te)
            print(f"   Val AUC: {res_comb['val_auc']:.4f} | Test AUC: {res_comb['test_auc']:.4f}")
            
            # Calculate improvement
            auc_gain_val = res_comb['val_auc'] - res_tech['val_auc']
            auc_gain_test = res_comb['test_auc'] - res_tech['test_auc']
            
            print(f"\n   📊 GAIN from adding events:")
            print(f"      Val: {auc_gain_val:+.4f} ({auc_gain_val/res_tech['val_auc']*100:+.1f}%)")
            print(f"      Test: {auc_gain_test:+.4f} ({auc_gain_test/res_tech['test_auc']*100:+.1f}%)")
            
            # Show top event features in combined model
            event_importance = res_comb['importance'][res_comb['importance']['feature'].isin(event_cols)]
            if len(event_importance) > 0:
                print(f"\n   Top event features in combined model:")
                for _, row in event_importance.head(5).iterrows():
                    print(f"      {row['feature']}: {row['importance']:.4f}")
            
            # Store results
            results.append({
                "horizon": h,
                "label": label_name,
                "tech_only_val": res_tech['val_auc'],
                "tech_only_test": res_tech['test_auc'],
                "event_only_val": res_event['val_auc'] if res_event else np.nan,
                "event_only_test": res_event['test_auc'] if res_event else np.nan,
                "combined_val": res_comb['val_auc'],
                "combined_test": res_comb['test_auc'],
                "gain_val": auc_gain_val,
                "gain_test": auc_gain_test,
                "base_rate": train_br
            })
    
    # Summary
    print("\n" + "="*80)
    print("SUMMARY")
    print("="*80)
    
    if results:
        res_df = pd.DataFrame(results)
        
        print("\n1. Average AUC by approach:")
        print(f"   Technical only: {res_df['tech_only_test'].mean():.4f}")
        print(f"   Event only: {res_df['event_only_test'].mean():.4f}")
        print(f"   Combined: {res_df['combined_test'].mean():.4f}")
        
        print(f"\n2. Average gain from adding events:")
        print(f"   Val: {res_df['gain_val'].mean():+.4f} ({res_df['gain_val'].mean()/res_df['tech_only_val'].mean()*100:+.1f}%)")
        print(f"   Test: {res_df['gain_test'].mean():+.4f} ({res_df['gain_test'].mean()/res_df['tech_only_test'].mean()*100:+.1f}%)")
        
        print(f"\n3. Best improvements:")
        top_gains = res_df.nlargest(3, 'gain_test')
        for _, row in top_gains.iterrows():
            print(f"   {row['label']}: {row['gain_test']:+.4f} (Test AUC: {row['tech_only_test']:.4f} → {row['combined_test']:.4f})")
        
        print(f"\n4. Cases where events hurt performance:")
        negative = res_df[res_df['gain_test'] < -0.01]
        if len(negative) > 0:
            print(f"   {len(negative)}/{len(res_df)} cases with negative impact")
            for _, row in negative.iterrows():
                print(f"   {row['label']}: {row['gain_test']:.4f}")
        else:
            print(f"   ✅ No significant negative impact!")
        
        # Save results
        res_df.to_csv("ML/event_features_test_results.csv", index=False)
        print(f"\n✅ Results saved to ML/event_features_test_results.csv")
    
    print("\nDONE")


if __name__ == "__main__":
    main()
