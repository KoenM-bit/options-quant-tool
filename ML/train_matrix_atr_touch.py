import os
import sys
import math
from datetime import datetime
from pathlib import Path
import numpy as np
import pandas as pd
from sqlalchemy import create_engine, text

from sklearn.metrics import roc_auc_score, brier_score_loss, log_loss
from sklearn.calibration import CalibratedClassifierCV
from xgboost import XGBClassifier

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from src.config import settings

# --- config ---
class Cfg:
    """Config that reads from environment at instantiation time."""
    
    def __init__(self):
        self.ticker = os.getenv("TICKER", "AD.AS")
        self.start = os.getenv("START", "2020-01-01")
        self.end = os.getenv("END", "2025-12-31")
        self.train_end = os.getenv("TRAIN_END", "2023-12-29")
        self.val_end = os.getenv("VAL_END", "2024-12-31")
        self.horizons = [4, 5, 7, 10, 21]
        self.targets = ["touch_up", "touch_down"]
        self.seed = int(os.getenv("SEED", "42"))
        self.n_jobs = int(os.getenv("N_JOBS", "8"))
        self.out_csv = os.getenv("OUT_CSV", "ML/matrix_atr_touch_results.csv")

def lift_table(p, y):
    df = pd.DataFrame({"p": p, "y": y}).dropna()
    p05 = df["p"].quantile(0.05)
    p10 = df["p"].quantile(0.10)
    p25 = df["p"].quantile(0.25)
    p75 = df["p"].quantile(0.75)
    p90 = df["p"].quantile(0.90)
    p95 = df["p"].quantile(0.95)

    base = df["y"].mean()
    def seg(mask, name):
        d = df[mask]
        if len(d) == 0:
            return {"segment": name, "n": 0, "avg_p": np.nan, "hit_rate": np.nan, "lift": np.nan}
        hr = d["y"].mean()
        return {"segment": name, "n": len(d), "avg_p": float(d["p"].mean()), "hit_rate": float(hr), "lift": float(hr / base) if base > 0 else np.nan}

    tab = pd.DataFrame([
        seg(df["p"] >= p95, "top_5%"),
        seg(df["p"] >= p90, "top_10%"),
        seg(df["p"] >= p75, "top_25%"),
        seg((df["p"] > p25) & (df["p"] < p75), "mid_50%"),
        seg(df["p"] <= p25, "bottom_25%"),
        seg(df["p"] <= p10, "bottom_10%"),
        seg(df["p"] <= p05, "bottom_5%"),
        seg(np.ones(len(df), dtype=bool), "all"),
    ])
    return {"p05":p05,"p10":p10,"p25":p25,"p75":p75,"p90":p90,"p95":p95}, tab

def main():
    cfg = Cfg()
    eng = create_engine(settings.database_url, pool_pre_ping=True)

    # 1) features (36ish) from fact layer
    qX = """
      SELECT
        trade_date::date AS dt,
        ticker,
        close::float,
        volume::float,
        sma_20::float, sma_50::float, sma_200::float,
        ema_12::float, ema_26::float,
        macd::float, macd_signal::float, macd_histogram::float,
        rsi_14::float,
        stochastic_k::float, stochastic_d::float,
        roc_20::float,
        atr_14::float,
        bollinger_upper::float, bollinger_middle::float, bollinger_lower::float,
        bollinger_width::float,
        realized_volatility_20::float,
        parkinson_volatility_20::float,
        high_20d::float, low_20d::float,
        high_52w::float, low_52w::float,
        pct_from_high_20d::float, pct_from_low_20d::float,
        pct_from_high_52w::float, pct_from_low_52w::float,
        volume_sma_20::float, volume_ratio::float,
        adx_14::float, plus_di_14::float, minus_di_14::float,
        obv::float, obv_sma_20::float
      FROM fact_technical_indicators
      WHERE ticker=:t
        AND trade_date BETWEEN :s AND :e
      ORDER BY trade_date;
    """
    X = pd.read_sql(text(qX), eng, params={"t": cfg.ticker, "s": cfg.start, "e": cfg.end})
    X["dt"] = pd.to_datetime(X["dt"])

    # 2) labels from ATR view
    qY = """
      SELECT *
      FROM gold_labels_atr_ad
      WHERE ticker=:t
        AND dt BETWEEN :s AND :e
      ORDER BY dt;
    """
    Y = pd.read_sql(text(qY), eng, params={"t": cfg.ticker, "s": cfg.start, "e": cfg.end})
    Y["dt"] = pd.to_datetime(Y["dt"])

    # Merge - drop overlapping columns from Y (keep only labels)
    label_cols = [c for c in Y.columns if c.startswith("y_")]
    Y_labels = Y[["ticker", "dt"] + label_cols]
    
    df = X.merge(Y_labels, on=["ticker","dt"], how="inner")

    # split masks
    train_end = pd.to_datetime(cfg.train_end)
    val_end   = pd.to_datetime(cfg.val_end)

    m_train = df["dt"] <= train_end
    m_val   = (df["dt"] > train_end) & (df["dt"] <= val_end)
    m_test  = df["dt"] > val_end

    results = []

    for h in cfg.horizons:
        for tgt in cfg.targets:
            ycol = f"y_{tgt}_atr_{h}"
            if ycol not in df.columns:
                continue

            data = df[["dt"] + [c for c in df.columns if c not in ["ticker"]] ].copy()
            data = data.dropna(subset=[ycol])  # label exists
            # drop rows with missing features
            feat_cols = [c for c in X.columns if c not in ["dt","ticker"]]
            data = data.dropna(subset=feat_cols)

            y = data[ycol].astype(int)
            Xmat = data[feat_cols]

            X_train, y_train = Xmat[m_train.loc[data.index]], y[m_train.loc[data.index]]
            X_val,   y_val   = Xmat[m_val.loc[data.index]],   y[m_val.loc[data.index]]
            X_test,  y_test  = Xmat[m_test.loc[data.index]],  y[m_test.loc[data.index]]

            if len(X_train) < 400 or len(X_val) < 120 or len(X_test) < 120:
                continue

            model = XGBClassifier(
                objective="binary:logistic",
                eval_metric="logloss",
                random_state=cfg.seed,
                n_jobs=cfg.n_jobs,
                tree_method="hist",
                max_depth=3,
                learning_rate=0.05,
                n_estimators=400,
                subsample=1.0,
                colsample_bytree=1.0,
                reg_lambda=3.0,
                min_child_weight=1,
            )
            model.fit(X_train, y_train)

            calib = CalibratedClassifierCV(model, method="sigmoid", cv="prefit")
            calib.fit(X_val, y_val)

            p_val  = calib.predict_proba(X_val)[:,1]
            p_test = calib.predict_proba(X_test)[:,1]

            out = {
                "timestamp": datetime.now().isoformat(timespec="seconds"),
                "ticker": cfg.ticker,
                "horizon": h,
                "target": tgt,
                "n_features": Xmat.shape[1],
                "n_train": len(X_train),
                "n_val": len(X_val),
                "n_test": len(X_test),
                "val_auc": roc_auc_score(y_val, p_val),
                "val_brier": brier_score_loss(y_val, p_val),
                "val_logloss": log_loss(y_val, p_val, eps=1e-15),
                "val_base_rate": float(y_val.mean()),
                "test_auc": roc_auc_score(y_test, p_test),
                "test_brier": brier_score_loss(y_test, p_test),
                "test_logloss": log_loss(y_test, p_test, eps=1e-15),
                "test_base_rate": float(y_test.mean()),
                "p_test_min": float(p_test.min()),
                "p_test_max": float(p_test.max()),
            }
            results.append(out)

            # print lift for best-ish candidates quickly
            pct, tab = lift_table(p_test, y_test.values)
            print("\n" + "="*80)
            print(f"[LIFT TEST] target={tgt} horizon={h}  base_rate={out['test_base_rate']:.3f}  test_auc={out['test_auc']:.3f}")
            print(f"Percentiles: p10={pct['p10']:.4f} p25={pct['p25']:.4f} p75={pct['p75']:.4f} p90={pct['p90']:.4f}")
            print(tab.to_string(index=False))
            print("="*80)

    res = pd.DataFrame(results).sort_values(["test_auc","val_auc"], ascending=False)
    os.makedirs("ML", exist_ok=True)
    res.to_csv(cfg.out_csv, index=False)

    print("\n=== TOP 10 by TEST AUC ===")
    print(res.head(10)[["horizon","target","test_auc","val_auc","test_brier","test_base_rate","p_test_min","p_test_max"]].to_string(index=False))
    print(f"\nSaved: {cfg.out_csv}")

if __name__ == "__main__":
    main()