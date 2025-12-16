import os
import sys
from dataclasses import dataclass
from datetime import datetime
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


# -----------------------------
# Config
# -----------------------------
@dataclass
class Cfg:
    ticker: str = os.getenv("TICKER", "AD.AS")

    # Choose your best setup
    horizon: int = int(os.getenv("HORIZON", "5"))
    label_q: float = float(os.getenv("LABEL_Q", "0.30"))
    featureset: str = os.getenv("FEATURESET", "base")  # keep for metadata

    # Split boundaries
    train_end: str = os.getenv("TRAIN_END", "2023-12-29")
    val_end: str = os.getenv("VAL_END", "2024-12-31")

    pred_table: str = os.getenv("PRED_TABLE", "gold_regime_predictions")
    model_version: str = os.getenv("MODEL_VERSION", f"rangeH{os.getenv('HORIZON','5')}_xgb_sigmoid_{datetime.now().date()}")

    seed: int = 42


def make_engine():
    """Get database engine using project settings."""
    return create_engine(settings.database_url, pool_pre_ping=True)


# -----------------------------
# Load base
# -----------------------------
def load_base(engine, cfg: Cfg) -> pd.DataFrame:
    q = f"""
      SELECT dt, open, high, low, px, volume
      FROM silver_prices_ad
      WHERE ticker = :ticker
      ORDER BY dt;
    """
    df = pd.read_sql(text(q), engine, params={"ticker": cfg.ticker})
    df["dt"] = pd.to_datetime(df["dt"])
    return df.sort_values("dt").reset_index(drop=True)


# -----------------------------
# Label (forward ER bottom q)
# -----------------------------
def forward_er(price: pd.Series, h: int) -> pd.Series:
    dp_abs = price.diff().abs()
    denom = dp_abs.shift(-1).rolling(h).sum()
    num = (price.shift(-h) - price).abs()
    return num / denom.replace(0, np.nan)


def make_label(df: pd.DataFrame, cfg: Cfg):
    px = df["px"].astype(float)
    fer = forward_er(px, cfg.horizon)

    train_mask = df["dt"] <= pd.to_datetime(cfg.train_end)
    thr = float(np.nanquantile(fer[train_mask].dropna().values, cfg.label_q))

    y = (fer <= thr).astype(float)
    y[fer.isna()] = np.nan
    return y, fer, thr


# -----------------------------
# Features (BASE set: 12 cols)
# -----------------------------
def rolling_er_past(price: pd.Series, n: int) -> pd.Series:
    dp = price.diff().abs()
    denom = dp.rolling(n).sum()
    num = (price - price.shift(n)).abs()
    return num / denom.replace(0, np.nan)

def realized_vol(logret: pd.Series, n: int) -> pd.Series:
    return logret.rolling(n).std()

def range_width(price: pd.Series, n: int) -> pd.Series:
    hi = price.rolling(n).max()
    lo = price.rolling(n).min()
    return (hi - lo) / price.replace(0, np.nan)

def make_base_features(df: pd.DataFrame) -> pd.DataFrame:
    px = df["px"].astype(float)
    logret = np.log(px / px.shift(1))

    X = pd.DataFrame({
        "ret_1d": logret,
        "ret_5d": logret.rolling(5).sum(),
        "ret_10d": logret.rolling(10).sum(),
        "rv_10d": realized_vol(logret, 10),
        "rv_20d": realized_vol(logret, 20),
        "er_past_10d": rolling_er_past(px, 10),
        "vol_ratio_20": df["volume"].astype(float) / df["volume"].astype(float).rolling(20).mean(),
        "range_width_20": range_width(px, 20),
        "gap_1d": (df["open"].astype(float) - px.shift(1)) / px.shift(1),
        "price_vs_sma20": px / px.rolling(20).mean() - 1,
        "price_vs_sma50": px / px.rolling(50).mean() - 1,
        "price_vs_sma200": px / px.rolling(200).mean() - 1,
    })

    # Force numeric (for SHAP later if needed)
    return X.astype(float)


# -----------------------------
# Split
# -----------------------------
def split_masks(dt: pd.Series, cfg: Cfg):
    train = dt <= pd.to_datetime(cfg.train_end)
    val = (dt > pd.to_datetime(cfg.train_end)) & (dt <= pd.to_datetime(cfg.val_end))
    test = dt > pd.to_datetime(cfg.val_end)
    return train, val, test


# -----------------------------
# Reports
# -----------------------------
def bucket_report(y_true: np.ndarray, p: np.ndarray, name: str):
    p25 = float(np.quantile(p, 0.25))
    p75 = float(np.quantile(p, 0.75))

    def seg(mask, label):
        if mask.sum() == 0:
            return {"segment": label, "n": 0, "actual_rate": np.nan, "avg_p": np.nan}
        return {"segment": label, "n": int(mask.sum()), "actual_rate": float(y_true[mask].mean()), "avg_p": float(p[mask].mean())}

    rows = [
        seg(p >= p75, "top_25%"),
        seg((p > p25) & (p < p75), "mid_50%"),
        seg(p <= p25, "bottom_25%"),
        seg(np.ones_like(p, dtype=bool), "all"),
    ]
    print(f"\n[{name}] thresholds: p25={p25:.4f}, p75={p75:.4f}")
    print(pd.DataFrame(rows).to_string(index=False))
    return p25, p75


def calibration_buckets(y_true: np.ndarray, p: np.ndarray, bins=10, name="TEST"):
    edges = np.linspace(0, 1, bins + 1)
    idx = np.clip(np.digitize(p, edges) - 1, 0, bins - 1)
    out = []
    for b in range(bins):
        m = idx == b
        if m.sum() == 0:
            continue
        out.append({
            "bin": f"{edges[b]:.1f}-{edges[b+1]:.1f}",
            "n": int(m.sum()),
            "p_mean": float(p[m].mean()),
            "y_mean": float(y_true[m].mean()),
        })
    print(f"\n[Calibration buckets] {name}")
    print(pd.DataFrame(out).to_string(index=False))


# -----------------------------
# Write predictions
# -----------------------------
def upsert_predictions(engine, cfg: Cfg, pred_df: pd.DataFrame):
    sql = f"""
    INSERT INTO {cfg.pred_table} (ticker, dt, horizon, label_q, model_version, featureset, p_range, split)
    VALUES (:ticker, :dt, :horizon, :label_q, :model_version, :featureset, :p_range, :split)
    ON CONFLICT (ticker, dt, horizon, label_q, model_version)
    DO UPDATE SET
      p_range = EXCLUDED.p_range,
      split = EXCLUDED.split,
      featureset = EXCLUDED.featureset;
    """
    with engine.begin() as conn:
        conn.execute(text(sql), pred_df.to_dict(orient="records"))


def main():
    cfg = Cfg()
    eng = make_engine()

    df = load_base(eng, cfg)
    y, fer, thr = make_label(df, cfg)
    X = make_base_features(df)

    data = pd.concat([df[["dt"]], X, y.rename("y"), fer.rename("fer_fwd")], axis=1).dropna()
    dt = data["dt"]
    y_all = data["y"].astype(int)
    X_all = data.drop(columns=["dt", "y", "fer_fwd"]).astype(float)

    train_m, val_m, test_m = split_masks(dt, cfg)

    X_train, y_train = X_all[train_m], y_all[train_m]
    X_val, y_val = X_all[val_m], y_all[val_m]
    X_test, y_test = X_all[test_m], y_all[test_m]

    print("Rows:", len(data), "| train/val/test:", len(X_train), len(X_val), len(X_test))
    print(f"Label threshold (train q{cfg.label_q}): {thr:.6f}")
    print("Base rates:", {
        "train": float(y_train.mean()),
        "val": float(y_val.mean()),
        "test": float(y_test.mean()),
    })

    # Train
    base_model = XGBClassifier(
        objective="binary:logistic",
        eval_metric="logloss",
        random_state=cfg.seed,
        n_jobs=max(1, os.cpu_count() or 1),
        tree_method="hist",
        max_depth=3,
        learning_rate=0.05,
        n_estimators=400,
        subsample=1.0,
        colsample_bytree=1.0,
        reg_lambda=3.0,
        min_child_weight=1,
        base_score=0.5,  # Fix for SHAP compatibility
    )
    base_model.fit(X_train, y_train)

    calib = CalibratedClassifierCV(base_model, method="sigmoid", cv="prefit")
    calib.fit(X_val, y_val)

    # Metrics
    p_val = calib.predict_proba(X_val)[:, 1]
    p_test = calib.predict_proba(X_test)[:, 1]

    print("\n[Metrics]")
    print("Val AUC:", roc_auc_score(y_val, p_val))
    print("Val Brier:", brier_score_loss(y_val, p_val))
    print("Test AUC:", roc_auc_score(y_test, p_test))
    print("Test Brier:", brier_score_loss(y_test, p_test))

    # Reports
    bucket_report(y_val.values, p_val, "VAL")
    bucket_report(y_test.values, p_test, "TEST")
    calibration_buckets(y_test.values, p_test, bins=10, name="TEST")

    # Predictions for all available dt
    p_all = calib.predict_proba(X_all)[:, 1]
    split = np.where(dt <= pd.to_datetime(cfg.train_end), "train",
             np.where(dt <= pd.to_datetime(cfg.val_end), "val", "test"))

    out = pd.DataFrame({
        "ticker": cfg.ticker,
        "dt": dt.dt.date.astype(str),
        "horizon": cfg.horizon,
        "label_q": cfg.label_q,
        "model_version": cfg.model_version,
        "featureset": cfg.featureset,
        "p_range": np.clip(p_all, 0, 1).round(6),
        "split": split
    })

    upsert_predictions(eng, cfg, out)
    print(f"\n✅ Wrote {len(out)} rows into {cfg.pred_table} (model_version={cfg.model_version})")

    # Show last few
    with eng.connect() as conn:
        r = conn.execute(text(f"""
          SELECT dt, p_range, split
          FROM {cfg.pred_table}
          WHERE ticker=:t AND horizon=:h AND label_q=:q AND model_version=:mv
          ORDER BY dt DESC
          LIMIT 10;
        """), {"t": cfg.ticker, "h": cfg.horizon, "q": cfg.label_q, "mv": cfg.model_version}).fetchall()
    print("\nLatest predictions:")
    for row in r:
        print(row)


if __name__ == "__main__":
    main()