"""
Event features test with FULL LIFT ANALYSIS
Includes percentile thresholds, baseline rates, bucket analysis, and practical signal strength
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
    """Create key labels"""
    close = df["close"]
    high = df["high"]
    low = df["low"]
    atr = df["atr_14"]
    
    fwd_max_high = high.shift(-1).rolling(horizon).max().shift(-(horizon-1))
    fwd_min_low = low.shift(-1).rolling(horizon).min().shift(-(horizon-1))
    
    labels = {}
    
    # Range labels (proven winners)
    range_atr = (fwd_max_high - fwd_min_low) / atr.replace(0, np.nan)
    labels[f"low_range_H{horizon}"] = (range_atr <= 1.0).astype(int)
    labels[f"high_range_H{horizon}"] = (range_atr >= 2.0).astype(int)
    
    # Volatility (proven winners)
    rv = df["realized_volatility_20"].fillna(df["rv20_logret"])
    rv_change = rv.shift(-horizon) - rv
    labels[f"rv_expand_H{horizon}"] = (rv_change >= 0.05).astype(int)
    labels[f"rv_compress_H{horizon}"] = (rv_change <= -0.05).astype(int)
    
    # Close break (strong from extended test)
    fwd_max_close = close.shift(-1).rolling(horizon).max().shift(-(horizon-1))
    fwd_min_close = close.shift(-1).rolling(horizon).min().shift(-(horizon-1))
    labels[f"close_break_up_H{horizon}"] = (fwd_max_close >= (close + 1.0 * atr)).astype(int)
    labels[f"close_break_down_H{horizon}"] = (fwd_min_close <= (close - 1.0 * atr)).astype(int)
    
    return labels


def analyze_lift(y_true, y_pred, baseline_rate, model_name="Model"):
    """
    Full lift analysis with percentile buckets
    Returns dictionary with all metrics
    """
    # Percentile thresholds
    p50 = np.percentile(y_pred, 50)
    p75 = np.percentile(y_pred, 75)
    p90 = np.percentile(y_pred, 90)
    p95 = np.percentile(y_pred, 95)
    
    # Bucket analysis
    buckets = {
        'p0-50': (y_pred < p50),
        'p50-75': (y_pred >= p50) & (y_pred < p75),
        'p75-90': (y_pred >= p75) & (y_pred < p90),
        'p90-95': (y_pred >= p90) & (y_pred < p95),
        'p95-100': (y_pred >= p95)
    }
    
    bucket_stats = {}
    for bucket_name, mask in buckets.items():
        if mask.sum() > 0:
            rate = y_true[mask].mean()
            lift = rate / baseline_rate if baseline_rate > 0 else 0
            bucket_stats[bucket_name] = {
                'count': int(mask.sum()),
                'rate': float(rate),
                'lift': float(lift)
            }
        else:
            bucket_stats[bucket_name] = {'count': 0, 'rate': 0.0, 'lift': 0.0}
    
    return {
        'baseline': float(baseline_rate),
        'p50_threshold': float(p50),
        'p75_threshold': float(p75),
        'p90_threshold': float(p90),
        'p95_threshold': float(p95),
        'p75_rate': bucket_stats.get('p75-90', {}).get('rate', 0.0),
        'p90_rate': bucket_stats.get('p90-95', {}).get('rate', 0.0),
        'p95_rate': bucket_stats.get('p95-100', {}).get('rate', 0.0),
        'p75_lift': bucket_stats.get('p75-90', {}).get('lift', 0.0),
        'p90_lift': bucket_stats.get('p90-95', {}).get('lift', 0.0),
        'p95_lift': bucket_stats.get('p95-100', {}).get('lift', 0.0),
        'bucket_stats': bucket_stats
    }


def train_and_evaluate_full(X_train, y_train, X_val, y_val, X_test, y_test, seed=42):
    """Train XGBoost model and evaluate with full lift analysis"""
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
    
    # Lift analysis
    val_lift = analyze_lift(y_val.values, p_val, y_val.mean())
    test_lift = analyze_lift(y_test.values, p_test, y_test.mean())
    
    return {
        "val_auc": roc_auc_score(y_val, p_val) if len(np.unique(y_val)) > 1 else np.nan,
        "val_brier": brier_score_loss(y_val, p_val),
        "test_auc": roc_auc_score(y_test, p_test) if len(np.unique(y_test)) > 1 else np.nan,
        "test_brier": brier_score_loss(y_test, p_test),
        "val_base_rate": float(y_val.mean()),
        "test_base_rate": float(y_test.mean()),
        "val_lift": val_lift,
        "test_lift": test_lift,
        "importance": importance
    }


def main():
    print("="*80)
    print("EVENT FEATURES TEST WITH FULL LIFT ANALYSIS")
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
    
    # Test key horizons and labels
    test_cases = [
        (3, "low_range_H3"),
        (3, "high_range_H3"),
        (7, "high_range_H7"),
        (7, "rv_expand_H7"),
        (7, "rv_compress_H7"),
        (7, "close_break_up_H7"),
        (7, "close_break_down_H7"),
        (14, "rv_expand_H14"),
        (14, "rv_compress_H14"),
    ]
    
    results = []
    
    for h, label_key in test_cases:
        print(f"\n{'='*80}")
        print(f"TESTING: {label_key}")
        print(f"{'='*80}")
        
        # Create label
        labels = make_labels(df, h)
        y = labels.get(label_key)
        
        if y is None:
            print(f"   ⚠️  Label {label_key} not found")
            continue
        
        # Prepare data
        data = df.copy()
        data["y"] = y
        data = data.dropna()
        
        if len(data) < 800:
            print(f"   ⚠️  Insufficient data")
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
        
        print(f"\n📊 Base Rates:")
        print(f"   Train: {y_tr.mean():.2%}")
        print(f"   Val:   {y_va.mean():.2%}")
        print(f"   Test:  {y_te.mean():.2%}")
        
        # Tech only
        tech_cols = [c for c in core_technical if c in data.columns]
        X_tr_tech = data.loc[tr, tech_cols]
        X_va_tech = data.loc[va, tech_cols]
        X_te_tech = data.loc[te, tech_cols]
        
        print(f"\n🔧 Technical Features Only ({len(tech_cols)} features):")
        res_tech = train_and_evaluate_full(X_tr_tech, y_tr, X_va_tech, y_va, X_te_tech, y_te)
        
        print(f"   Test AUC: {res_tech['test_auc']:.4f}")
        print(f"   Test Brier: {res_tech['test_brier']:.4f}")
        print(f"\n   Lift Analysis (Test):")
        print(f"   Baseline: {res_tech['test_lift']['baseline']:.2%}")
        print(f"   p75: {res_tech['test_lift']['p75_rate']:.2%} (lift: {res_tech['test_lift']['p75_lift']:.2f}x)")
        print(f"   p90: {res_tech['test_lift']['p90_rate']:.2%} (lift: {res_tech['test_lift']['p90_lift']:.2f}x)")
        print(f"   p95: {res_tech['test_lift']['p95_rate']:.2%} (lift: {res_tech['test_lift']['p95_lift']:.2f}x)")
        
        print(f"\n   Bucket Breakdown (Test):")
        for bucket_name, stats in res_tech['test_lift']['bucket_stats'].items():
            print(f"   {bucket_name:>10}: n={stats['count']:3d}  rate={stats['rate']:6.2%}  lift={stats['lift']:5.2f}x")
        
        # Combined (tech + events)
        event_cols = [c for c in event_features if c in data.columns and data[c].notna().sum() > 100]
        combined_cols = tech_cols + event_cols
        X_tr_comb = data.loc[tr, combined_cols]
        X_va_comb = data.loc[va, combined_cols]
        X_te_comb = data.loc[te, combined_cols]
        
        print(f"\n🎯 Technical + Event Features ({len(combined_cols)} features):")
        res_comb = train_and_evaluate_full(X_tr_comb, y_tr, X_va_comb, y_va, X_te_comb, y_te)
        
        print(f"   Test AUC: {res_comb['test_auc']:.4f}")
        print(f"   Test Brier: {res_comb['test_brier']:.4f}")
        print(f"\n   Lift Analysis (Test):")
        print(f"   Baseline: {res_comb['test_lift']['baseline']:.2%}")
        print(f"   p75: {res_comb['test_lift']['p75_rate']:.2%} (lift: {res_comb['test_lift']['p75_lift']:.2f}x)")
        print(f"   p90: {res_comb['test_lift']['p90_rate']:.2%} (lift: {res_comb['test_lift']['p90_lift']:.2f}x)")
        print(f"   p95: {res_comb['test_lift']['p95_rate']:.2%} (lift: {res_comb['test_lift']['p95_lift']:.2f}x)")
        
        print(f"\n   Bucket Breakdown (Test):")
        for bucket_name, stats in res_comb['test_lift']['bucket_stats'].items():
            print(f"   {bucket_name:>10}: n={stats['count']:3d}  rate={stats['rate']:6.2%}  lift={stats['lift']:5.2f}x")
        
        # Calculate improvement
        auc_gain = res_comb['test_auc'] - res_tech['test_auc']
        lift_p95_gain = res_comb['test_lift']['p95_lift'] - res_tech['test_lift']['p95_lift']
        
        print(f"\n📈 IMPROVEMENT with Events:")
        print(f"   AUC gain: {auc_gain:+.4f} ({auc_gain/res_tech['test_auc']*100:+.1f}%)")
        print(f"   p95 lift gain: {lift_p95_gain:+.2f}x")
        print(f"   p95 rate: {res_tech['test_lift']['p95_rate']:.2%} → {res_comb['test_lift']['p95_rate']:.2%}")
        
        # Top event features
        event_importance = res_comb['importance'][res_comb['importance']['feature'].isin(event_cols)]
        print(f"\n   Top Event Features:")
        for idx, (_, row) in enumerate(event_importance.head(3).iterrows(), 1):
            print(f"   {idx}. {row['feature']}: {row['importance']:.4f}")
        
        # Store results
        results.append({
            "label": label_key,
            "horizon": h,
            "tech_test_auc": res_tech['test_auc'],
            "comb_test_auc": res_comb['test_auc'],
            "auc_gain": auc_gain,
            "baseline": res_tech['test_lift']['baseline'],
            "tech_p75_rate": res_tech['test_lift']['p75_rate'],
            "tech_p90_rate": res_tech['test_lift']['p90_rate'],
            "tech_p95_rate": res_tech['test_lift']['p95_rate'],
            "tech_p95_lift": res_tech['test_lift']['p95_lift'],
            "comb_p75_rate": res_comb['test_lift']['p75_rate'],
            "comb_p90_rate": res_comb['test_lift']['p90_rate'],
            "comb_p95_rate": res_comb['test_lift']['p95_rate'],
            "comb_p95_lift": res_comb['test_lift']['p95_lift'],
            "lift_p95_gain": lift_p95_gain,
        })
    
    # Summary
    print("\n" + "="*80)
    print("SUMMARY")
    print("="*80)
    
    if results:
        res_df = pd.DataFrame(results)
        
        print(f"\n1. Overall Performance:")
        print(f"   Avg tech AUC: {res_df['tech_test_auc'].mean():.4f}")
        print(f"   Avg combined AUC: {res_df['comb_test_auc'].mean():.4f}")
        print(f"   Avg AUC gain: {res_df['auc_gain'].mean():+.4f}")
        
        print(f"\n2. Lift Improvements:")
        print(f"   Avg tech p95 lift: {res_df['tech_p95_lift'].mean():.2f}x")
        print(f"   Avg combined p95 lift: {res_df['comb_p95_lift'].mean():.2f}x")
        print(f"   Avg p95 lift gain: {res_df['lift_p95_gain'].mean():+.2f}x")
        
        print(f"\n3. Best Cases (by AUC gain):")
        top3 = res_df.nlargest(3, 'auc_gain')
        for _, row in top3.iterrows():
            print(f"   {row['label']:<25} AUC: {row['tech_test_auc']:.3f}→{row['comb_test_auc']:.3f} (+{row['auc_gain']:.3f})  p95: {row['tech_p95_rate']:.1%}→{row['comb_p95_rate']:.1%} (lift: {row['tech_p95_lift']:.2f}x→{row['comb_p95_lift']:.2f}x)")
        
        print(f"\n4. Best Cases (by p95 lift gain):")
        top3_lift = res_df.nlargest(3, 'lift_p95_gain')
        for _, row in top3_lift.iterrows():
            print(f"   {row['label']:<25} p95 lift: {row['tech_p95_lift']:.2f}x→{row['comb_p95_lift']:.2f}x (+{row['lift_p95_gain']:.2f}x)  rate: {row['tech_p95_rate']:.1%}→{row['comb_p95_rate']:.1%}")
        
        # Save results
        res_df.to_csv("ML/event_features_lift_analysis.csv", index=False)
        print(f"\n✅ Results saved to ML/event_features_lift_analysis.csv")
    
    print("\nDONE")


if __name__ == "__main__":
    main()
