import os
import sys
import math
from datetime import datetime
from typing import Dict, List, Tuple
from pathlib import Path

import numpy as np
import pandas as pd
from sqlalchemy import create_engine, text

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from src.config import settings

from sklearn.metrics import roc_auc_score, brier_score_loss, accuracy_score, log_loss
from sklearn.calibration import CalibratedClassifierCV
from xgboost import XGBClassifier


# -----------------------------
# Lift Analysis
# -----------------------------
def lift_quantiles(p, y):
    """Compute hit rates by prediction probability quantiles - using extreme percentiles for high conviction."""
    df = pd.DataFrame({"p": p, "y": y}).dropna()
    p05 = df["p"].quantile(0.05)
    p10 = df["p"].quantile(0.10)
    p25 = df["p"].quantile(0.25)
    p75 = df["p"].quantile(0.75)
    p90 = df["p"].quantile(0.90)
    p95 = df["p"].quantile(0.95)
    
    out = []
    for name, mask in [
        ("top_5%", df["p"] >= p95),
        ("top_10%", df["p"] >= p90),
        ("top_25%", df["p"] >= p75),
        ("mid_50%", (df["p"] > p25) & (df["p"] < p75)),
        ("bottom_25%", df["p"] <= p25),
        ("bottom_10%", df["p"] <= p10),
        ("bottom_5%", df["p"] <= p05),
        ("all", np.ones(len(df), dtype=bool)),
    ]:
        d = df[mask]
        if len(d) > 0:
            lift = float(d["y"].mean()) / float(df["y"].mean()) if df["y"].mean() > 0 else np.nan
        else:
            lift = np.nan
        out.append({
            "segment": name,
            "n": len(d),
            "avg_p": float(d["p"].mean()) if len(d) else np.nan,
            "hit_rate": float(d["y"].mean()) if len(d) else np.nan,
            "lift": lift,
        })
    
    return {
        "p05": p05, "p10": p10, "p25": p25,
        "p75": p75, "p90": p90, "p95": p95,
        "table": pd.DataFrame(out)
    }


# -----------------------------
# Config
# -----------------------------
class Cfg:
    """Config that reads from environment at instantiation time."""
    
    def __init__(self):
        self.ticker = os.getenv("TICKER", "AD.AS")
        # Data
        self.fact_table = os.getenv("FACT_TABLE", "fact_technical_indicators")
        self.start = os.getenv("START", "2020-01-01")
        self.end = os.getenv("END", "2025-12-31")
        # Splits
        self.train_end = os.getenv("TRAIN_END", "2023-12-29")
        self.val_end = os.getenv("VAL_END", "2024-12-31")
        # Horizons (comma-separated)
        self.horizons = None
        # Label configs
        self.range_q = float(os.getenv("RANGE_Q", "0.30"))     # bottom-q forward efficiency ratio
        self.up_q = float(os.getenv("UP_Q", "0.70"))           # top-q forward return
        self.down_q = float(os.getenv("DOWN_Q", "0.30"))       # bottom-q forward return
        # Model
        self.seed = int(os.getenv("SEED", "42"))
        self.n_jobs = int(os.getenv("N_JOBS", "8"))
        # Output
        self.out_csv = os.getenv("OUT_CSV", "ML/matrix_regime_results.csv")
        self.topn = int(os.getenv("TOPN", "10"))
        # Horizons (comma-separated)
        hs = os.getenv("HORIZONS", "3,4,5,7,10,14,21")
        self.horizons = [int(x.strip()) for x in hs.split(",") if x.strip()]
def make_engine():
    """Get database engine using project settings."""
    return create_engine(settings.database_url, pool_pre_ping=True)
def load_fact(engine, cfg: Cfg) -> pd.DataFrame:
    q = f"""
      SELECT
        trade_date::date AS dt,
        ticker,
        close::float AS close,
        volume::float AS volume,

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

        volume_sma_20::float,
        volume_ratio::float,

        adx_14::float, plus_di_14::float, minus_di_14::float,
        obv::float, obv_sma_20::float
      FROM {cfg.fact_table}
      WHERE ticker = :t
        AND trade_date BETWEEN :s AND :e
      ORDER BY trade_date;
    """
    df = pd.read_sql(text(q), engine, params={"t": cfg.ticker, "s": cfg.start, "e": cfg.end})
    df["dt"] = pd.to_datetime(df["dt"])
    return df.sort_values("dt").reset_index(drop=True)


# -----------------------------
# Labels (from close only)
# -----------------------------
def forward_return(close: pd.Series, h: int) -> pd.Series:
    return (close.shift(-h) / close - 1.0)

def forward_efficiency_ratio(close: pd.Series, h: int) -> pd.Series:
    # forward ER proxy: |P(t+h)-P(t)| / sum(|dP| over next h days)
    dp = close.diff().abs()
    denom = dp.shift(-1).rolling(h).sum()
    num = (close.shift(-h) - close).abs()
    return num / denom.replace(0, np.nan)


def make_targets(df: pd.DataFrame, cfg: Cfg, h: int) -> Dict[str, pd.Series]:
    """
    Returns dict of label series for a given horizon h.
    Labels computed with thresholds based on TRAIN only.
    """
    close = df["close"].astype(float)
    fr = forward_return(close, h)
    fer = forward_efficiency_ratio(close, h)

    train_mask = df["dt"] <= pd.to_datetime(cfg.train_end)

    # thresholds computed on train
    thr_range = float(np.nanquantile(fer[train_mask].dropna().values, cfg.range_q))
    thr_up = float(np.nanquantile(fr[train_mask].dropna().values, cfg.up_q))
    thr_down = float(np.nanquantile(fr[train_mask].dropna().values, cfg.down_q))

    y_range = (fer <= thr_range).astype(float)   # "range-like" = low movement efficiency
    y_up = (fr >= thr_up).astype(float)          # "up impulse"
    y_down = (fr <= thr_down).astype(float)      # "down impulse"

    y_range[fer.isna()] = np.nan
    y_up[fr.isna()] = np.nan
    y_down[fr.isna()] = np.nan

    return {
        "range": y_range.astype("float"),
        "up": y_up.astype("float"),
        "down": y_down.astype("float"),
    }


# -----------------------------
# Splits
# -----------------------------
def split_masks(dt: pd.Series, cfg: Cfg):
    train = dt <= pd.to_datetime(cfg.train_end)
    val = (dt > pd.to_datetime(cfg.train_end)) & (dt <= pd.to_datetime(cfg.val_end))
    test = dt > pd.to_datetime(cfg.val_end)
    return train, val, test


# -----------------------------
# Model
# -----------------------------
def fit_calibrated_xgb(X_train, y_train, X_val, y_val, cfg: Cfg):
    base = XGBClassifier(
        objective="binary:logistic",
        eval_metric="logloss",
        random_state=cfg.seed,
        n_jobs=cfg.n_jobs,
        tree_method="hist",
        max_depth=3,
        learning_rate=0.05,
        n_estimators=500,
        subsample=1.0,
        colsample_bytree=1.0,
        reg_lambda=3.0,
        min_child_weight=1,
    )
    base.fit(X_train, y_train)

    calib = CalibratedClassifierCV(base, method="sigmoid", cv="prefit")
    calib.fit(X_val, y_val)
    return base, calib


def eval_binary(calib, X, y) -> Dict[str, float]:
    p = calib.predict_proba(X)[:, 1]
    out = {
        "auc": roc_auc_score(y, p) if len(np.unique(y)) > 1 else np.nan,
        "brier": brier_score_loss(y, p),
        "logloss": log_loss(y, p, labels=[0, 1]),
        "p_min": float(np.min(p)),
        "p_max": float(np.max(p)),
        "base_rate": float(np.mean(y)),
    }
    return out


# -----------------------------
# 3D Screen helpers
# -----------------------------
def print_3d_screen(df_res: pd.DataFrame, metric: str):
    """
    Prints pivot tables: horizon x target with chosen metric.
    """
    piv = df_res.pivot_table(index="horizon", columns="target", values=metric, aggfunc="max")
    print(f"\n=== 3D SCREEN: {metric} (higher better for AUC, lower better for Brier/LogLoss) ===")
    print(piv.round(4).to_string())


def main():
    cfg = Cfg()
    eng = make_engine()

    df = load_fact(eng, cfg)
    if len(df) < 500:
        raise RuntimeError(f"Too few rows: {len(df)}. Check date range / table.")

    # Feature matrix: drop non-features
    drop_cols = {"dt", "ticker"}
    feature_cols = [c for c in df.columns if c not in drop_cols]

    # Remove any obviously non-feature identifiers if present
    feature_cols = [c for c in feature_cols if c not in ["indicator_id", "created_at", "updated_at", "calculated_at"]]

    X_all = df[feature_cols].copy()

    # enforce numeric
    X_all = X_all.apply(pd.to_numeric, errors="coerce").astype(float)

    train_m, val_m, test_m = split_masks(df["dt"], cfg)

    results = []

    for h in cfg.horizons:
        targets = make_targets(df, cfg, h)

        for target_name, y in targets.items():
            data = pd.concat([df[["dt"]], X_all, y.rename("y")], axis=1).dropna(subset=["y"])
            data = data.dropna()  # strict: drop rows where any feature null

            if len(data) < 600:
                continue

            dt = data["dt"]
            X = data.drop(columns=["dt", "y"])
            yv = data["y"].astype(int)

            tr = dt <= pd.to_datetime(cfg.train_end)
            va = (dt > pd.to_datetime(cfg.train_end)) & (dt <= pd.to_datetime(cfg.val_end))
            te = dt > pd.to_datetime(cfg.val_end)

            if tr.sum() < 300 or va.sum() < 80 or te.sum() < 80:
                continue

            X_train, y_train = X[tr], yv[tr]
            X_val, y_val = X[va], yv[va]
            X_test, y_test = X[te], yv[te]

            base_model, calib = fit_calibrated_xgb(X_train, y_train, X_val, y_val, cfg)

            val_metrics = eval_binary(calib, X_val, y_val)
            test_metrics = eval_binary(calib, X_test, y_test)

            # Lift analysis for best models
            if (target_name == "range" and h in [4, 5, 7]) or (target_name == "down" and h == 21):
                p_test = calib.predict_proba(X_test)[:, 1]
                lift_result = lift_quantiles(p_test, y_test.values)
                print(f"\n{'='*80}")
                print(f"[LIFT TEST] target={target_name} horizon={h}")
                print(f"Percentiles: p05={lift_result['p05']:.4f} p10={lift_result['p10']:.4f} p25={lift_result['p25']:.4f}")
                print(f"             p75={lift_result['p75']:.4f} p90={lift_result['p90']:.4f} p95={lift_result['p95']:.4f}")
                print(f"{'='*80}")
                print(lift_result['table'].to_string(index=False))
                print(f"{'='*80}\n")

            results.append({
                "timestamp": datetime.now().isoformat(timespec="seconds"),
                "ticker": cfg.ticker,
                "horizon": h,
                "target": target_name,
                "n_features": X.shape[1],
                "n_train": int(tr.sum()),
                "n_val": int(va.sum()),
                "n_test": int(te.sum()),
                "val_auc": val_metrics["auc"],
                "val_brier": val_metrics["brier"],
                "val_logloss": val_metrics["logloss"],
                "val_base_rate": val_metrics["base_rate"],
                "test_auc": test_metrics["auc"],
                "test_brier": test_metrics["brier"],
                "test_logloss": test_metrics["logloss"],
                "test_base_rate": test_metrics["base_rate"],
                "p_test_min": test_metrics["p_min"],
                "p_test_max": test_metrics["p_max"],
            })

            print("DONE", results[-1])

    out = pd.DataFrame(results)
    os.makedirs("ML", exist_ok=True)
    out.to_csv(cfg.out_csv, index=False)

    print("\nSaved:", cfg.out_csv)

    # “3D screen” views
    # AUC: higher better, Brier/logloss: lower better
    print_3d_screen(out, "test_auc")
    print_3d_screen(out, "test_brier")

    # Leaderboards per target
    for t in sorted(out["target"].unique()):
        sub = out[out["target"] == t].sort_values(["test_auc", "val_auc"], ascending=[False, False]).head(cfg.topn)
        print(f"\n=== TOP {cfg.topn} (target={t}) by test_auc ===")
        print(sub[["horizon", "val_auc", "test_auc", "val_brier", "test_brier", "n_features"]].to_string(index=False))


if __name__ == "__main__":
    main()