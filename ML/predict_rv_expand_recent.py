"""
Get recent predictions for rv_expand_abs H=30
"""

import numpy as np
import pandas as pd
from sqlalchemy import create_engine, text
from xgboost import XGBClassifier
from sklearn.calibration import CalibratedClassifierCV


def load_data():
    db_url = "postgresql://airflow:airflow@192.168.1.201:5433/ahold_options"
    eng = create_engine(db_url, pool_pre_ping=True)
    
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
    
    # Momentum
    X["di_diff"] = X["plus_di_14"] - X["minus_di_14"]
    X["macd_z"] = (X["macd"] - X["macd"].rolling(60).mean()) / (X["macd"].rolling(60).std().replace(0, np.nan))
    
    # Volume
    X["obv_norm"] = X["obv"] / X["obv"].rolling(50).mean()
    
    # Gap
    X["gap_1d"] = (X["open"] - X["close"].shift(1)) / X["close"].shift(1)
    
    return X


def create_label(df, h=30, threshold=0.05):
    rv = df["realized_volatility_20"]
    if rv.isna().all():
        rv = df["rv20_logret"]
    rv = rv.astype(float)
    
    rv_future = rv.shift(-h)
    rv_change = rv_future - rv
    
    label = (rv_change >= threshold).astype(int)
    return label


def main():
    print("="*80)
    print("RV_EXPAND_ABS H=30 - RECENT PREDICTIONS")
    print("="*80)
    
    df = load_data()
    X = build_features(df)
    y = create_label(X, h=30, threshold=0.05)
    
    # Feature columns (clean)
    feature_cols = [
        "ret_1d", "ret_5d", "ret_21d",
        "logret_1d", "logret_5d",
        "px_vs_sma20", "px_vs_sma50", "px_vs_sma200",
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
    train_mask = X["dt"] <= "2023-12-29"
    val_mask = (X["dt"] > "2023-12-29") & (X["dt"] <= "2024-12-31")
    test_mask = X["dt"] > "2024-12-31"
    
    X_prep = X[feature_cols + ["dt"]].copy()
    X_prep["y"] = y
    X_prep = X_prep.dropna()
    
    train = X_prep[train_mask.loc[X_prep.index]]
    val = X_prep[val_mask.loc[X_prep.index]]
    test = X_prep[test_mask.loc[X_prep.index]]
    
    X_train, y_train = train[feature_cols], train["y"]
    X_val, y_val = val[feature_cols], val["y"]
    X_test, y_test = test[feature_cols], test["y"]
    
    # Train model
    print("\nTraining model...")
    model = XGBClassifier(
        max_depth=5,
        learning_rate=0.05,
        n_estimators=400,
        subsample=1.0,
        colsample_bytree=0.85,
        reg_lambda=1.0,
        min_child_weight=5,
        objective="binary:logistic",
        eval_metric="logloss",
        random_state=42,
        n_jobs=-1
    )
    
    model.fit(X_train, y_train)
    
    # Calibrate
    calib_model = CalibratedClassifierCV(model, method="sigmoid", cv="prefit")
    calib_model.fit(X_val, y_val)
    
    # Get test predictions
    test_pred = calib_model.predict_proba(X_test)[:, 1]
    
    # Calculate percentiles
    p75 = np.percentile(test_pred, 75)
    p90 = np.percentile(test_pred, 90)
    p95 = np.percentile(test_pred, 95)
    
    # Hit rates
    test_df = test.copy()
    test_df["pred"] = test_pred
    
    base_rate = y_test.mean()
    top75_rate = test_df[test_df["pred"] >= p75]["y"].mean()
    top90_rate = test_df[test_df["pred"] >= p90]["y"].mean()
    top95_rate = test_df[test_df["pred"] >= p95]["y"].mean()
    
    print(f"\n{'='*80}")
    print("PERCENTILE THRESHOLDS (Test Set)")
    print(f"{'='*80}")
    print(f"Base rate: {base_rate:.1%}")
    print(f"  p75 ≥ {p75:.4f} ({p75*100:.1f}%): Hit rate {top75_rate:.1%} ({top75_rate/base_rate:.2f}x lift)")
    print(f"  p90 ≥ {p90:.4f} ({p90*100:.1f}%): Hit rate {top90_rate:.1%} ({top90_rate/base_rate:.2f}x lift)")
    print(f"  p95 ≥ {p95:.4f} ({p95*100:.1f}%): Hit rate {top95_rate:.1%} ({top95_rate/base_rate:.2f}x lift)")
    
    # Recent predictions
    recent = test_df.tail(3).copy()
    recent["pred_pct"] = recent["pred"] * 100
    
    print(f"\n{'='*80}")
    print("RECENT PREDICTIONS (Last 3 Trading Days)")
    print(f"{'='*80}")
    
    for _, row in recent.iterrows():
        date = row["dt"].strftime("%b %d")
        pred = row["pred"]
        pred_pct = row["pred_pct"]
        
        if pred >= p95:
            signal = "🔥 VERY STRONG"
        elif pred >= p90:
            signal = "🔴 STRONG"
        elif pred >= p75:
            signal = "🟡 MODERATE"
        else:
            signal = "❌ WEAK"
        
        print(f"{date}: {pred_pct:5.1f}% {signal}")
    
    print(f"\n{'='*80}")
    print("INTERPRETATION")
    print(f"{'='*80}")
    latest = recent.iloc[-1]
    latest_pred = latest["pred_pct"]
    
    if latest_pred >= p95 * 100:
        print(f"🔥 VERY HIGH probability ({latest_pred:.1f}%) of volatility EXPANSION")
        print("   → Consider buying options (long straddles/strangles)")
        print("   → Avoid selling premium")
        print("   → Increase position vega exposure")
    elif latest_pred >= p90 * 100:
        print(f"🔴 HIGH probability ({latest_pred:.1f}%) of volatility EXPANSION")
        print("   → Good setup for long volatility strategies")
        print("   → Consider delta-hedged long gamma positions")
    elif latest_pred >= p75 * 100:
        print(f"🟡 MODERATE probability ({latest_pred:.1f}%) of volatility EXPANSION")
        print("   → Monitor for confirmation")
        print("   → Small long vol positions acceptable")
    else:
        print(f"❌ LOW probability ({latest_pred:.1f}%) of volatility EXPANSION")
        print("   → Vol may stay stable or compress")
        print("   → Premium selling strategies may work")


if __name__ == "__main__":
    main()
