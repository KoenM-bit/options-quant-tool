"""
Generate predictions for latest trading days using event features
Shows both tech-only and tech+events predictions for comparison
"""
import os
import sys
import math
from pathlib import Path
from datetime import datetime
import numpy as np
import pandas as pd
from sqlalchemy import create_engine, text

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
    """Create labels for training"""
    close = df["close"]
    high = df["high"]
    low = df["low"]
    atr = df["atr_14"]
    
    fwd_max_high = high.shift(-1).rolling(horizon).max().shift(-(horizon-1))
    fwd_min_low = low.shift(-1).rolling(horizon).min().shift(-(horizon-1))
    
    labels = {}
    
    # Range labels
    range_atr = (fwd_max_high - fwd_min_low) / atr.replace(0, np.nan)
    labels[f"low_range_H{horizon}"] = (range_atr <= 1.0).astype(int)
    labels[f"high_range_H{horizon}"] = (range_atr >= 2.0).astype(int)
    
    # Volatility
    rv = df["realized_volatility_20"].fillna(df["rv20_logret"])
    rv_change = rv.shift(-horizon) - rv
    labels[f"rv_expand_H{horizon}"] = (rv_change >= 0.05).astype(int)
    labels[f"rv_compress_H{horizon}"] = (rv_change <= -0.05).astype(int)
    
    # Close breaks
    fwd_max_close = close.shift(-1).rolling(horizon).max().shift(-(horizon-1))
    fwd_min_close = close.shift(-1).rolling(horizon).min().shift(-(horizon-1))
    labels[f"close_break_up_H{horizon}"] = (fwd_max_close >= (close + 1.0 * atr)).astype(int)
    labels[f"close_break_down_H{horizon}"] = (fwd_min_close <= (close - 1.0 * atr)).astype(int)
    
    return labels


def train_model(X_train, y_train, X_val, y_val, seed=42):
    """Train and calibrate model"""
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
    base.fit(X_train, y_train, verbose=False)
    
    # Calibrate
    calib = CalibratedClassifierCV(base, method="sigmoid", cv="prefit")
    calib.fit(X_val, y_val)
    
    return calib


def main():
    print("="*80)
    print("LATEST TRADING DAY PREDICTIONS - Event Features Analysis")
    print("="*80)
    
    # Load data
    print("\n1. Loading data...")
    df = load_data_with_events()
    df = build_derived_features(df)
    print(f"   Loaded {len(df)} rows")
    
    # Get latest date
    latest_date = df["dt"].max()
    latest_3_dates = df["dt"].nlargest(3).sort_values()
    
    print(f"\n   Latest date: {latest_date.strftime('%Y-%m-%d')}")
    print(f"   Last 3 dates: {', '.join([d.strftime('%Y-%m-%d') for d in latest_3_dates])}")
    
    # Split data
    train_end = pd.to_datetime("2023-12-29")
    val_end = pd.to_datetime("2024-12-31")
    
    m_train = df["dt"] <= train_end
    m_val = (df["dt"] > train_end) & (df["dt"] <= val_end)
    m_recent = df["dt"] > val_end
    
    print(f"\n   Train: {m_train.sum()} | Val: {m_val.sum()} | Recent: {m_recent.sum()}")
    
    # Feature sets
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
    
    # Models to predict (best performers from lift analysis)
    models_config = [
        (3, "low_range_H3", "Tight 3-day range (≤1.0 ATR)"),
        (3, "high_range_H3", "Wide 3-day range (≥2.0 ATR)"),
        (7, "rv_compress_H7", "Vol compression next 7 days"),
        (7, "rv_expand_H7", "Vol expansion next 7 days"),
        (7, "close_break_up_H7", "Upside breakout (1 ATR)"),
        (7, "close_break_down_H7", "Downside breakout (1 ATR)"),
        (14, "rv_compress_H14", "Vol compression next 14 days"),
        (14, "rv_expand_H14", "Vol expansion next 14 days"),
    ]
    
    all_predictions = []
    
    for h, label_key, description in models_config:
        print(f"\n{'='*80}")
        print(f"{label_key.upper()}: {description}")
        print(f"{'='*80}")
        
        # Create labels
        labels = make_labels(df, h)
        y = labels.get(label_key)
        
        if y is None:
            continue
        
        # Prepare data
        data = df.copy()
        data["y"] = y
        data = data.dropna()
        
        tr = m_train.loc[data.index]
        va = m_val.loc[data.index]
        rec = m_recent.loc[data.index]
        
        y_tr = data.loc[tr, "y"].astype(int)
        y_va = data.loc[va, "y"].astype(int)
        
        baseline = y_tr.mean()
        
        # Train tech-only model
        tech_cols = [c for c in core_technical if c in data.columns]
        X_tr_tech = data.loc[tr, tech_cols]
        X_va_tech = data.loc[va, tech_cols]
        X_rec_tech = data.loc[rec, tech_cols]
        
        model_tech = train_model(X_tr_tech, y_tr, X_va_tech, y_va)
        p_tech = model_tech.predict_proba(X_rec_tech)[:, 1]
        
        # Train combined model
        event_cols = [c for c in event_features if c in data.columns and data[c].notna().sum() > 100]
        combined_cols = tech_cols + event_cols
        X_tr_comb = data.loc[tr, combined_cols]
        X_va_comb = data.loc[va, combined_cols]
        X_rec_comb = data.loc[rec, combined_cols]
        
        model_comb = train_model(X_tr_comb, y_tr, X_va_comb, y_va)
        p_comb = model_comb.predict_proba(X_rec_comb)[:, 1]
        
        # Get percentile thresholds
        p75_tech = np.percentile(p_tech, 75)
        p90_tech = np.percentile(p_tech, 90)
        p95_tech = np.percentile(p_tech, 95)
        
        p75_comb = np.percentile(p_comb, 75)
        p90_comb = np.percentile(p_comb, 90)
        p95_comb = np.percentile(p_comb, 95)
        
        print(f"\n📊 Baseline rate: {baseline:.2%}")
        
        print(f"\n🔧 Technical Only Thresholds:")
        print(f"   p75: {p75_tech:.4f} ({p75_tech*100:.1f}%)")
        print(f"   p90: {p90_tech:.4f} ({p90_tech*100:.1f}%)")
        print(f"   p95: {p95_tech:.4f} ({p95_tech*100:.1f}%)")
        
        print(f"\n🎯 Tech+Events Thresholds:")
        print(f"   p75: {p75_comb:.4f} ({p75_comb*100:.1f}%)")
        print(f"   p90: {p90_comb:.4f} ({p90_comb*100:.1f}%)")
        print(f"   p95: {p95_comb:.4f} ({p95_comb*100:.1f}%)")
        
        # Get recent dates (last 3 only)
        recent_dates = data.loc[rec, "dt"].tail(3)
        recent_indices = recent_dates.index
        
        print(f"\n📅 Latest Predictions (last 3 days):")
        print(f"\n{'Date':<12} {'Close':<8} {'Tech':<12} {'Tech+Evt':<12} {'Delta':<12} {'Signal':<15} {'Event Context'}")
        print("-" * 110)
        
        for i, idx in enumerate(recent_indices):
            date = data.loc[idx, "dt"]
            close = data.loc[idx, "close"]
            pred_tech = p_tech[-(i+1)]
            pred_comb = p_comb[-(i+1)]
            delta = pred_comb - pred_tech
            
            # Determine signal strength (combined model)
            if pred_comb >= p95_comb:
                signal = "🔥 STRONG"
            elif pred_comb >= p90_comb:
                signal = "✅ MODERATE"
            elif pred_comb >= p75_comb:
                signal = "⚠️  WEAK"
            else:
                signal = "❌ NO SIGNAL"
            
            # Get event context
            events = []
            if pd.notna(data.loc[idx, "days_to_earnings"]) and data.loc[idx, "days_to_earnings"] <= 7:
                events.append(f"E-{int(data.loc[idx, 'days_to_earnings'])}d")
            if pd.notna(data.loc[idx, "days_since_earnings"]) and data.loc[idx, "days_since_earnings"] <= 5:
                events.append(f"E+{int(data.loc[idx, 'days_since_earnings'])}d")
            if data.loc[idx, "is_opex_week"] == 1:
                events.append(f"OPEX-{int(data.loc[idx, 'days_to_opex'])}d")
            if data.loc[idx, "is_exdiv_week"] == 1:
                events.append(f"ExDiv-{int(data.loc[idx, 'days_to_exdiv'])}d")
            event_str = ", ".join(events) if events else "-"
            
            print(f"{date.strftime('%Y-%m-%d'):<12} {close:<8.2f} {pred_tech:>6.2%} ({pred_tech/baseline:>4.2f}x)  {pred_comb:>6.2%} ({pred_comb/baseline:>4.2f}x)  {delta:>+6.2%}      {signal:<15} {event_str}")
            
            # Store for summary
            all_predictions.append({
                "date": date,
                "label": label_key,
                "description": description,
                "close": close,
                "baseline": baseline,
                "tech_pred": pred_tech,
                "comb_pred": pred_comb,
                "delta": delta,
                "signal": signal,
                "events": event_str
            })
    
    # Overall summary
    print(f"\n{'='*80}")
    print("SUMMARY - Latest Trading Day")
    print(f"{'='*80}")
    
    latest_pred = [p for p in all_predictions if p["date"] == latest_date]
    
    if latest_pred:
        print(f"\nDate: {latest_date.strftime('%Y-%m-%d')}")
        print(f"Close: ${latest_pred[0]['close']:.2f}")
        
        print(f"\n🔥 STRONG Signals (p95+):")
        strong = [p for p in latest_pred if "STRONG" in p["signal"]]
        if strong:
            for p in strong:
                print(f"   {p['label']:<20} {p['comb_pred']:>6.2%} (vs baseline {p['baseline']:>5.2%}) - {p['description']}")
                print(f"      Events: {p['events']}")
        else:
            print("   None")
        
        print(f"\n✅ MODERATE Signals (p90-p95):")
        moderate = [p for p in latest_pred if "MODERATE" in p["signal"]]
        if moderate:
            for p in moderate:
                print(f"   {p['label']:<20} {p['comb_pred']:>6.2%} (vs baseline {p['baseline']:>5.2%}) - {p['description']}")
                print(f"      Events: {p['events']}")
        else:
            print("   None")
        
        print(f"\n📈 Biggest Event Feature Impact (delta):")
        sorted_pred = sorted(latest_pred, key=lambda x: abs(x['delta']), reverse=True)
        for p in sorted_pred[:3]:
            print(f"   {p['label']:<20} {p['delta']:>+6.2%}  (Tech: {p['tech_pred']:>5.2%} → Combined: {p['comb_pred']:>5.2%})")
        
        # Trading recommendations
        print(f"\n💡 Trading Implications:")
        
        strong_compress = [p for p in strong if "compress" in p["label"]]
        strong_expand = [p for p in strong if "expand" in p["label"]]
        strong_low_range = [p for p in strong if "low_range" in p["label"]]
        strong_high_range = [p for p in strong if "high_range" in p["label"]]
        
        if strong_compress:
            print(f"   ✅ SELL PREMIUM: Strong vol compression signal(s)")
        if strong_expand:
            print(f"   ⚠️  CAUTION: Strong vol expansion signal(s) - avoid selling premium")
        if strong_low_range:
            print(f"   ✅ IRON CONDOR: Strong tight range signal(s)")
        if strong_high_range:
            print(f"   ⚠️  WIDE RANGE: Avoid tight spreads")
        
        if not (strong_compress or strong_expand or strong_low_range or strong_high_range):
            print(f"   ℹ️  No strong directional signals - neutral positioning")
    
    # Save predictions
    pred_df = pd.DataFrame(all_predictions)
    pred_df.to_csv("ML/latest_predictions_with_events.csv", index=False)
    print(f"\n✅ Predictions saved to ML/latest_predictions_with_events.csv")
    
    print("\nDONE")


if __name__ == "__main__":
    main()
