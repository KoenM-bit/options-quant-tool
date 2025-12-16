import os
import sys
import json
import math
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd

from sqlalchemy import create_engine, text

from sklearn.linear_model import LogisticRegression
from sklearn.metrics import brier_score_loss, roc_auc_score
from sklearn.calibration import CalibratedClassifierCV
from sklearn.model_selection import ParameterGrid

from xgboost import XGBClassifier

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from src.config import settings


FEATURE_COLS = [
    "ret_1d", "ret_5d", "ret_10d",
    "rv_10d", "rv_20d",
    "er_past_10d",
    "price_vs_sma20", "price_vs_sma50", "price_vs_sma200",
    "vol_ratio_20",
    "range_width_20",
    "gap_1d",
]

LABEL_COL = "y_range_next_10d"
SPLIT_COL = "split"


@dataclass
class Config:
    source_view: str = "ml_dataset_range10_ad_clean"
    pred_table: str = "gold_predictions_range10_ad"

    model_version: str = ""
    random_seed: int = 42

    # Decision thresholds (for lift diagnostics)
    high_p: float = 0.70
    low_p: float = 0.30


def get_engine():
    """Get database engine using project settings."""
    return create_engine(settings.database_url, pool_pre_ping=True)


def load_dataset(engine, cfg: Config) -> pd.DataFrame:
    q = f"""
    SELECT
      ticker, dt, split,
      {", ".join(FEATURE_COLS)},
      {LABEL_COL}::int AS {LABEL_COL}
    FROM {cfg.source_view}
    ORDER BY dt;
    """
    df = pd.read_sql(q, engine)
    return df


def summarize_split(df: pd.DataFrame, split: str):
    d = df[df[SPLIT_COL] == split]
    y = d[LABEL_COL].values
    return {
        "n": int(len(d)),
        "pct_range": float(np.mean(y)) if len(y) else float("nan"),
        "start": str(d["dt"].min()) if len(d) else None,
        "end": str(d["dt"].max()) if len(d) else None,
    }


def bucket_calibration(y_true: np.ndarray, p: np.ndarray, n_bins: int = 10) -> pd.DataFrame:
    # bins: [0,0.1), ..., [0.9,1.0]
    bins = np.linspace(0.0, 1.0, n_bins + 1)
    idx = np.digitize(p, bins, right=False) - 1
    idx = np.clip(idx, 0, n_bins - 1)

    rows = []
    for b in range(n_bins):
        m = idx == b
        if not np.any(m):
            rows.append({"bin": f"{bins[b]:.1f}-{bins[b+1]:.1f}", "n": 0, "p_mean": np.nan, "y_mean": np.nan})
        else:
            rows.append({
                "bin": f"{bins[b]:.1f}-{bins[b+1]:.1f}",
                "n": int(np.sum(m)),
                "p_mean": float(np.mean(p[m])),
                "y_mean": float(np.mean(y_true[m])),
            })
    return pd.DataFrame(rows)


def lift_table(y_true: np.ndarray, p: np.ndarray, high_p: float, low_p: float) -> pd.DataFrame:
    rows = []
    for name, mask in [
        (f"p>={high_p:.2f}", p >= high_p),
        (f"p<={low_p:.2f}", p <= low_p),
        ("all", np.ones_like(p, dtype=bool)),
    ]:
        if mask.sum() == 0:
            rows.append({"segment": name, "n": 0, "range_rate": np.nan})
        else:
            rows.append({"segment": name, "n": int(mask.sum()), "range_rate": float(y_true[mask].mean())})
    return pd.DataFrame(rows)


def train_logistic_baseline(X_train, y_train, X_val, y_val, seed: int):
    clf = LogisticRegression(
        max_iter=2000,
        solver="lbfgs",
        random_state=seed,
    )
    clf.fit(X_train, y_train)

    p_val = clf.predict_proba(X_val)[:, 1]
    return clf, p_val


def train_xgb_with_light_tuning(X_train, y_train, X_val, y_val, seed: int):
    # Keep tuning small on purpose (dataset ~1k rows)
    grid = ParameterGrid({
        "max_depth": [2, 3, 4],
        "learning_rate": [0.03, 0.07, 0.12],
        "n_estimators": [200, 400, 700],
        "subsample": [0.8, 1.0],
        "colsample_bytree": [0.8, 1.0],
        "min_child_weight": [1, 3],
        "reg_lambda": [1.0, 3.0],
    })

    best = None
    best_brier = float("inf")

    for params in grid:
        clf = XGBClassifier(
            objective="binary:logistic",
            eval_metric="logloss",
            random_state=seed,
            n_jobs=max(1, os.cpu_count() or 1),
            tree_method="hist",
            **params
        )
        clf.fit(X_train, y_train)

        p_val = clf.predict_proba(X_val)[:, 1]
        brier = brier_score_loss(y_val, p_val)

        if brier < best_brier:
            best_brier = brier
            best = (clf, params, p_val)

    assert best is not None
    return best[0], best[1], best[2], best_brier


def calibrate_model(base_model, X_val, y_val, method: str = "isotonic"):
    # Calibrate on validation set only (simple and effective for probabilities)
    calib = CalibratedClassifierCV(base_model, method=method, cv="prefit")
    calib.fit(X_val, y_val)
    return calib


def ensure_pred_table(engine, cfg: Config):
    ddl = f"""
    CREATE TABLE IF NOT EXISTS {cfg.pred_table} (
      ticker text NOT NULL,
      dt date NOT NULL,
      model_version text NOT NULL,
      p_range_next_10d numeric(10,6) NOT NULL,
      split text NULL,
      created_at timestamp NOT NULL DEFAULT now(),
      PRIMARY KEY (ticker, dt, model_version)
    );
    """
    with engine.begin() as conn:
        conn.execute(text(ddl))


def upsert_predictions(engine, cfg: Config, pred_df: pd.DataFrame):
    # pred_df columns: ticker, dt, model_version, p_range_next_10d, split
    sql = f"""
    INSERT INTO {cfg.pred_table} (ticker, dt, model_version, p_range_next_10d, split)
    VALUES (:ticker, :dt, :model_version, :p_range_next_10d, :split)
    ON CONFLICT (ticker, dt, model_version)
    DO UPDATE SET
      p_range_next_10d = EXCLUDED.p_range_next_10d,
      split = EXCLUDED.split;
    """
    records = pred_df.to_dict(orient="records")
    with engine.begin() as conn:
        conn.execute(text(sql), records)


def main():
    cfg = Config(
        model_version=os.getenv("MODEL_VERSION", f"range10_xgb_v1_{datetime.now().date()}"),
    )

    engine = get_engine()

    df = load_dataset(engine, cfg)
    print("Loaded rows:", len(df))
    print("Split summary:", json.dumps({
        "train": summarize_split(df, "train"),
        "val": summarize_split(df, "val"),
        "test": summarize_split(df, "test"),
    }, indent=2))

    # Split
    train_df = df[df[SPLIT_COL] == "train"].copy()
    val_df   = df[df[SPLIT_COL] == "val"].copy()
    test_df  = df[df[SPLIT_COL] == "test"].copy()

    # Ensure no NaNs in features
    for name, d in [("train", train_df), ("val", val_df), ("test", test_df)]:
        na = d[FEATURE_COLS].isna().any(axis=1).sum()
        if na > 0:
            print(f"[WARN] {name} has {na} rows with NaN features; dropping them.")
            d.dropna(subset=FEATURE_COLS + [LABEL_COL], inplace=True)

    X_train = train_df[FEATURE_COLS].values
    y_train = train_df[LABEL_COL].values
    X_val   = val_df[FEATURE_COLS].values
    y_val   = val_df[LABEL_COL].values
    X_test  = test_df[FEATURE_COLS].values
    y_test  = test_df[LABEL_COL].values

    # Baseline
    base_lr, p_val_lr = train_logistic_baseline(X_train, y_train, X_val, y_val, cfg.random_seed)
    print("\n[Baseline] Logistic Regression")
    print("  Val Brier:", brier_score_loss(y_val, p_val_lr))
    try:
        print("  Val AUC:", roc_auc_score(y_val, p_val_lr))
    except Exception:
        pass

    # XGBoost + light tuning
    print("\n[Model] XGBoost tuning (small grid)...")
    xgb_model, xgb_params, p_val_xgb, best_brier = train_xgb_with_light_tuning(
        X_train, y_train, X_val, y_val, cfg.random_seed
    )
    print("  Best params:", xgb_params)
    print("  Val Brier (uncalibrated):", best_brier)
    try:
        print("  Val AUC (uncalibrated):", roc_auc_score(y_val, p_val_xgb))
    except Exception:
        pass

    # Calibrate probabilities on validation set
    calib = calibrate_model(xgb_model, X_val, y_val, method="sigmoid")
    p_val_cal = calib.predict_proba(X_val)[:, 1]
    p_test_cal = calib.predict_proba(X_test)[:, 1]

    print("\n[Calibrated] XGBoost + isotonic")
    print("  Val Brier (calibrated):", brier_score_loss(y_val, p_val_cal))
    try:
        print("  Val AUC (calibrated):", roc_auc_score(y_val, p_val_cal))
    except Exception:
        pass
    print("  Test Brier (calibrated):", brier_score_loss(y_test, p_test_cal))
    try:
        print("  Test AUC (calibrated):", roc_auc_score(y_test, p_test_cal))
    except Exception:
        pass

    # Diagnostics tables
    print("\n[Calibration buckets] TEST")
    print(bucket_calibration(y_test, p_test_cal, n_bins=10).to_string(index=False))

    print("\n[Lift table] TEST")
    print(lift_table(y_test, p_test_cal, cfg.high_p, cfg.low_p).to_string(index=False))

    # Write predictions (val + test + optionally train)
    ensure_pred_table(engine, cfg)

    pred_out = pd.concat([
        train_df[["ticker", "dt", "split"]].assign(p_range_next_10d=calib.predict_proba(X_train)[:, 1]),
        val_df[["ticker", "dt", "split"]].assign(p_range_next_10d=p_val_cal),
        test_df[["ticker", "dt", "split"]].assign(p_range_next_10d=p_test_cal),
    ], ignore_index=True)

    pred_out["model_version"] = cfg.model_version
    pred_out["p_range_next_10d"] = pred_out["p_range_next_10d"].clip(0, 1).round(6)

    upsert_predictions(engine, cfg, pred_out[["ticker", "dt", "model_version", "p_range_next_10d", "split"]])

    print(f"\n✅ Wrote predictions to {cfg.pred_table} with model_version={cfg.model_version}")
    print("Tip: query a few latest rows:")
    print(f"  SELECT * FROM {cfg.pred_table} ORDER BY dt DESC LIMIT 10;")

if __name__ == "__main__":
    main()