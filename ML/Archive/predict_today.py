"""
Generate regime predictions for the latest available trading day.
This script trains models on all available data and predicts the current regime.
"""
import os
import sys
from pathlib import Path
from datetime import datetime

import numpy as np
import pandas as pd
from sqlalchemy import create_engine

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from src.config import settings

from sklearn.calibration import CalibratedClassifierCV
from xgboost import XGBClassifier


def load_data(ticker="AD.AS"):
    """Load technical indicators from database."""
    eng = create_engine(settings.database_url)
    
    query = f"""
    SELECT *
    FROM fact_technical_indicators
    WHERE ticker = '{ticker}'
    ORDER BY trade_date
    """
    
    df = pd.read_sql(query, eng)
    df = df.rename(columns={"trade_date": "dt"})
    return df


def make_range_target(df, horizon, quantile=0.30):
    """Create range target: low forward efficiency ratio."""
    fwd = df["close"].shift(-horizon)
    high_h = df["close"].rolling(horizon).max().shift(-horizon)
    low_h = df["close"].rolling(horizon).min().shift(-horizon)
    
    eff_ratio = ((fwd - df["close"]).abs()) / ((high_h - low_h).replace(0, np.nan))
    threshold = eff_ratio.quantile(quantile)
    
    return (eff_ratio <= threshold).astype(int)


def make_down_target(df, horizon, quantile=0.30):
    """Create down target: low forward return."""
    fwd_ret = (df["close"].shift(-horizon) / df["close"] - 1.0) * 100
    threshold = fwd_ret.quantile(quantile)
    
    return (fwd_ret <= threshold).astype(int)


def train_model(X_train, y_train, seed=42):
    """Train calibrated XGBoost model."""
    base_model = XGBClassifier(
        n_estimators=100,
        max_depth=3,
        learning_rate=0.05,
        subsample=0.8,
        colsample_bytree=0.8,
        random_state=seed,
        eval_metric="logloss",
        verbosity=0,
    )
    
    calib = CalibratedClassifierCV(
        base_model,
        method="sigmoid",
        cv=5,
        n_jobs=4,
    )
    
    calib.fit(X_train, y_train)
    return calib


def predict_latest(ticker="AD.AS", horizons=[4, 5, 7, 21], seed=42):
    """Generate predictions for the latest available day."""
    
    print(f"Loading data for {ticker}...")
    df = load_data(ticker)
    
    if len(df) < 100:
        raise ValueError(f"Not enough data: only {len(df)} rows")
    
    latest_date = df["dt"].max()
    print(f"Latest date in data: {latest_date}")
    print(f"Total rows: {len(df)}")
    
    # Feature columns (drop non-features)
    drop_cols = {"dt", "ticker", "indicator_id", "created_at", "updated_at", "calculated_at"}
    feature_cols = [c for c in df.columns if c not in drop_cols]
    
    X_all = df[feature_cols].apply(pd.to_numeric, errors="coerce").astype(float)
    
    # Get latest row for prediction
    X_latest = X_all.iloc[[-1]]
    latest_dt = df["dt"].iloc[-1]
    
    print(f"\nGenerating predictions for {latest_dt}...")
    print(f"Features: {len(feature_cols)}")
    print("-" * 80)
    
    results = []
    
    for h in horizons:
        # Create targets
        y_range = make_range_target(df, h, quantile=0.30)
        y_down = make_down_target(df, h, quantile=0.30)
        
        for target_name, y in [("range", y_range), ("down", y_down)]:
            # Prepare training data (exclude NaN targets and latest row which has no target yet)
            data = pd.concat([df[["dt"]], X_all, y.rename("y")], axis=1).dropna(subset=["y"])
            
            if len(data) < 500:
                print(f"Skipping H={h} {target_name}: insufficient data ({len(data)} rows)")
                continue
            
            # Use all available historical data for training
            X_train = data[feature_cols].iloc[:-1]  # Exclude the row we're predicting
            y_train = data["y"].iloc[:-1].astype(int)
            
            # Train model
            model = train_model(X_train, y_train, seed=seed)
            
            # Predict on latest
            prob = model.predict_proba(X_latest)[0, 1]
            
            results.append({
                "date": latest_dt,
                "horizon": h,
                "target": target_name,
                "probability": prob,
            })
            
            print(f"H={h:2d} {target_name:5s}: {prob:.4f}")
    
    return pd.DataFrame(results), latest_dt


def classify_tier(predictions):
    """Classify current regime into trading tiers based on predictions."""
    
    # Extract key predictions
    h4_range = predictions[(predictions["horizon"] == 4) & (predictions["target"] == "range")]["probability"].values[0]
    h5_range = predictions[(predictions["horizon"] == 5) & (predictions["target"] == "range")]["probability"].values[0]
    h7_range = predictions[(predictions["horizon"] == 7) & (predictions["target"] == "range")]["probability"].values[0]
    h21_down = predictions[(predictions["horizon"] == 21) & (predictions["target"] == "down")]["probability"].values[0]
    
    print("\n" + "=" * 80)
    print("REGIME CLASSIFICATION")
    print("=" * 80)
    print(f"\nKey Signals:")
    print(f"  H=4  RANGE: {h4_range:.4f} (threshold: p90=0.46, p75=0.35)")
    print(f"  H=5  RANGE: {h5_range:.4f} (threshold: p90=0.37, p75=0.34)")
    print(f"  H=7  RANGE: {h7_range:.4f} (threshold: p90=0.52, p75=0.37)")
    print(f"  H=21 DOWN:  {h21_down:.4f} (threshold: p25=0.22, p10=0.14)")
    
    # Tier classification
    tier = None
    action = None
    delta_range = None
    dte_range = None
    position_size = None
    
    if h4_range >= 0.46 and h21_down <= 0.14:
        tier = 1
        action = "MAXIMUM PREMIUM SELLING"
        delta_range = "0.30-0.35"
        dte_range = "35-42"
        position_size = "100%"
        conviction = "ULTRA HIGH"
        
    elif h4_range >= 0.46 and h21_down <= 0.22:
        tier = 2
        action = "AGGRESSIVE PREMIUM SELLING"
        delta_range = "0.25-0.30"
        dte_range = "28-35"
        position_size = "75-100%"
        conviction = "HIGH"
        
    elif h4_range >= 0.35 and h21_down <= 0.31:
        tier = 3
        action = "CONSERVATIVE PREMIUM SELLING"
        delta_range = "0.18-0.25"
        dte_range = "21-28"
        position_size = "50-75%"
        conviction = "MODERATE"
        
    elif h21_down >= 0.33:
        tier = 5
        action = "HEDGE MODE - DEFENSIVE"
        delta_range = "N/A (close shorts)"
        dte_range = "N/A"
        position_size = "0% (buy puts)"
        conviction = "HIGH RISK"
        
    else:
        tier = 4
        action = "SKIP / AVOID"
        delta_range = "N/A"
        dte_range = "N/A"
        position_size = "0%"
        conviction = "LOW"
    
    print(f"\n{'=' * 80}")
    print(f"TIER {tier}: {action}")
    print(f"{'=' * 80}")
    print(f"Conviction: {conviction}")
    print(f"Recommended Delta: {delta_range}")
    print(f"Recommended DTE: {dte_range}")
    print(f"Position Size: {position_size}")
    
    # Additional insights
    print(f"\n{'=' * 80}")
    print("SIGNAL ANALYSIS")
    print(f"{'=' * 80}")
    
    # H4 RANGE analysis
    if h4_range >= 0.52:
        print(f"✅ H=4 RANGE in TOP 5% ({h4_range:.4f} >= 0.52): 58% ranging probability")
    elif h4_range >= 0.46:
        print(f"✅ H=4 RANGE in TOP 10% ({h4_range:.4f} >= 0.46): 58% ranging probability")
    elif h4_range >= 0.35:
        print(f"⚠️  H=4 RANGE in TOP 25% ({h4_range:.4f} >= 0.35): 50% ranging probability")
    elif h4_range >= 0.28:
        print(f"⚠️  H=4 RANGE above P25 ({h4_range:.4f} >= 0.28): 30% ranging probability")
    else:
        print(f"❌ H=4 RANGE in BOTTOM 25% ({h4_range:.4f} < 0.28): 17% ranging probability")
    
    # H21 DOWN analysis
    if h21_down <= 0.10:
        print(f"✅ H=21 DOWN in BOTTOM 5% ({h21_down:.4f} <= 0.10): 8% downside risk - MAXIMUM SAFETY")
    elif h21_down <= 0.14:
        print(f"✅ H=21 DOWN in BOTTOM 10% ({h21_down:.4f} <= 0.14): 9% downside risk - ULTRA SAFE")
    elif h21_down <= 0.22:
        print(f"✅ H=21 DOWN in BOTTOM 25% ({h21_down:.4f} <= 0.22): 5% downside risk - VERY SAFE")
    elif h21_down <= 0.31:
        print(f"⚠️  H=21 DOWN below median ({h21_down:.4f} <= 0.31): 5-30% downside risk")
    elif h21_down <= 0.33:
        print(f"⚠️  H=21 DOWN in TOP 25% ({h21_down:.4f} <= 0.33): 30% downside risk")
    else:
        print(f"❌ H=21 DOWN in TOP 10% ({h21_down:.4f} >= 0.33): 30% downside risk - AVOID SELLING")
    
    # Multi-timeframe consensus
    range_consensus = sum([h4_range >= 0.35, h5_range >= 0.34, h7_range >= 0.37])
    print(f"\nMulti-timeframe RANGE consensus: {range_consensus}/3 models agree")
    
    if range_consensus == 3:
        print("  ✅ Strong consensus: All short-term models predict ranging")
    elif range_consensus == 2:
        print("  ⚠️  Moderate consensus: 2/3 models predict ranging")
    else:
        print("  ❌ Weak consensus: Only 1 or fewer models predict ranging")
    
    print(f"\n{'=' * 80}")
    
    return {
        "tier": tier,
        "action": action,
        "delta_range": delta_range,
        "dte_range": dte_range,
        "position_size": position_size,
        "conviction": conviction,
        "h4_range": h4_range,
        "h5_range": h5_range,
        "h7_range": h7_range,
        "h21_down": h21_down,
        "range_consensus": range_consensus,
    }


def main():
    ticker = os.getenv("TICKER", "AD.AS")
    
    print("=" * 80)
    print("REGIME PREDICTION FOR TODAY")
    print("=" * 80)
    print(f"Ticker: {ticker}")
    print(f"Current time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print("=" * 80)
    
    # Generate predictions
    predictions, latest_date = predict_latest(ticker=ticker, horizons=[4, 5, 7, 21])
    
    # Save predictions
    os.makedirs("ML", exist_ok=True)
    predictions.to_csv("ML/predictions_today.csv", index=False)
    print(f"\nPredictions saved to ML/predictions_today.csv")
    
    # Classify regime
    classification = classify_tier(predictions)
    
    # Save classification
    classification_df = pd.DataFrame([classification])
    classification_df["date"] = latest_date
    classification_df.to_csv("ML/regime_classification_today.csv", index=False)
    print(f"\nClassification saved to ML/regime_classification_today.csv")
    
    print("\n" + "=" * 80)
    print("DONE")
    print("=" * 80)


if __name__ == "__main__":
    main()
