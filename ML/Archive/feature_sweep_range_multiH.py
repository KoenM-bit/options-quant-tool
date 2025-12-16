import os
import sys
import itertools
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd
from sqlalchemy import create_engine

from sklearn.metrics import roc_auc_score, brier_score_loss
from sklearn.calibration import CalibratedClassifierCV
from sklearn.inspection import permutation_importance

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
    ticker: str = "AD.AS"

    # Horizons to sweep
    horizons: tuple = (3, 4, 5, 10)
    label_quantile: float = 0.30  # bottom q = range

    # Calendar split boundaries
    train_end: str = "2023-12-29"
    val_end: str = "2024-12-31"

    out_csv: str = "ML/experiments_range_multiH.csv"
    seed: int = 42


def make_engine():
    """Get database engine using project settings."""
    return create_engine(settings.database_url, pool_pre_ping=True)


# -----------------------------
# Load base series (no labels here)
# -----------------------------
def load_base(cfg: Cfg) -> pd.DataFrame:
    eng = make_engine()
    q_px = f"""
      SELECT dt, open, high, low, px, volume
      FROM silver_prices_ad
      WHERE ticker = '{cfg.ticker}'
      ORDER BY dt;
    """
    df = pd.read_sql(q_px, eng)
    df["dt"] = pd.to_datetime(df["dt"])
    return df.sort_values("dt").reset_index(drop=True)


# -----------------------------
# Leakage-safe label creation for horizon H
# -----------------------------
def forward_er(price: pd.Series, h: int) -> pd.Series:
    # fER_t = |P_{t+h}-P_t| / sum_{i=1..h} |P_{t+i}-P_{t+i-1}|
    dp_abs = price.diff().abs()
    denom = dp_abs.shift(-1).rolling(h).sum()  # sums t+1..t+h
    num = (price.shift(-h) - price).abs()
    return num / denom.replace(0, np.nan)


def make_label_from_forward_er(df: pd.DataFrame, h: int, q: float, train_end: str):
    px = df["px"].astype(float)
    fer = forward_er(px, h)

    train_mask = df["dt"] <= pd.to_datetime(train_end)
    fer_train = fer[train_mask].dropna()
    if len(fer_train) < 200:
        raise RuntimeError(f"Not enough train rows to estimate threshold for h={h}")

    thr = float(np.nanquantile(fer_train.values, q))
    y = (fer <= thr).astype(float)
    y[fer.isna()] = np.nan  # last h rows -> NaN label
    return y, fer, thr


# -----------------------------
# Feature engineering (all leakage-safe: uses only past)
# -----------------------------
def rolling_er(price: pd.Series, n: int) -> pd.Series:
    dp = price.diff().abs()
    denom = dp.rolling(n).sum()
    num = (price - price.shift(n)).abs()
    return num / denom.replace(0, np.nan)

def realized_vol(logret: pd.Series, n: int) -> pd.Series:
    return logret.rolling(n).std()

def zscore(series: pd.Series, n: int) -> pd.Series:
    mu = series.rolling(n).mean()
    sd = series.rolling(n).std()
    return (series - mu) / sd.replace(0, np.nan)

def range_width(price: pd.Series, n: int) -> pd.Series:
    hi = price.rolling(n).max()
    lo = price.rolling(n).min()
    return (hi - lo) / price.replace(0, np.nan)

def choppiness(high: pd.Series, low: pd.Series, close: pd.Series, n: int = 14) -> pd.Series:
    prev_close = close.shift(1)
    tr = pd.concat([
        (high - low),
        (high - prev_close).abs(),
        (low - prev_close).abs()
    ], axis=1).max(axis=1)

    tr_sum = tr.rolling(n).sum()
    denom = (high.rolling(n).max() - low.rolling(n).min()).replace(0, np.nan)
    chop = 100.0 * np.log10(tr_sum / denom) / np.log10(n)
    return chop

def directional_persistence(logret: pd.Series, n: int) -> pd.Series:
    sign_1 = np.sign(logret)
    sign_n = np.sign(logret.rolling(n).sum())
    return (sign_1 == sign_n).rolling(n).mean()

def autocorr_1(logret: pd.Series, n: int) -> pd.Series:
    x = logret
    y = logret.shift(1)
    return x.rolling(n).corr(y)


def make_feature_families(df: pd.DataFrame) -> dict:
    px = df["px"].astype(float)
    logret = np.log(px / px.shift(1))

    families = {}

    base = pd.DataFrame({
        "ret_1d": logret,
        "ret_5d": logret.rolling(5).sum(),
        "ret_10d": logret.rolling(10).sum(),
        "rv_10d": realized_vol(logret, 10),
        "rv_20d": realized_vol(logret, 20),
        "er_past_10d": rolling_er(px, 10),

        "vol_ratio_20": df["volume"].astype(float) / df["volume"].astype(float).rolling(20).mean(),
        "range_width_20": range_width(px, 20),
        "gap_1d": (df["open"].astype(float) - px.shift(1)) / px.shift(1),

        "price_vs_sma20": px / px.rolling(20).mean() - 1,
        "price_vs_sma50": px / px.rolling(50).mean() - 1,
        "price_vs_sma200": px / px.rolling(200).mean() - 1,
    })
    families["base"] = base

    families["range_physics"] = pd.DataFrame({
        "zscore_20": zscore(px, 20),
        "chop_14": choppiness(df["high"].astype(float), df["low"].astype(float), px, 14),
        "autocorr_ret_20": autocorr_1(logret, 20),
        "persistence_10": directional_persistence(logret, 10),
    })

    return families


# -----------------------------
# Splits & model
# -----------------------------
def split_sets(dt: pd.Series, X: pd.DataFrame, y: pd.Series, cfg: Cfg):
    train_mask = dt <= pd.to_datetime(cfg.train_end)
    val_mask = (dt > pd.to_datetime(cfg.train_end)) & (dt <= pd.to_datetime(cfg.val_end))
    test_mask = dt > pd.to_datetime(cfg.val_end)

    return (
        X[train_mask], y[train_mask],
        X[val_mask], y[val_mask],
        X[test_mask], y[test_mask]
    )


def train_calibrated_xgb(X_train, y_train, X_val, y_val, cfg: Cfg):
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
    return base_model, calib


def eval_model(calib, X_val, y_val, X_test, y_test):
    p_val = calib.predict_proba(X_val)[:, 1]
    p_test = calib.predict_proba(X_test)[:, 1]

    return {
        "val_auc": roc_auc_score(y_val, p_val) if len(np.unique(y_val)) > 1 else np.nan,
        "val_brier": brier_score_loss(y_val, p_val),
        "test_auc": roc_auc_score(y_test, p_test) if len(np.unique(y_test)) > 1 else np.nan,
        "test_brier": brier_score_loss(y_test, p_test),
        "p_test_min": float(np.min(p_test)),
        "p_test_max": float(np.max(p_test)),
    }


# -----------------------------
# Permutation importance + SHAP
# -----------------------------
def permutation_importance_proba(model, X: pd.DataFrame, y: pd.Series, n_repeats=30, seed=42) -> pd.DataFrame:
    def brier_scorer(estimator, Xn, yn):
        p = estimator.predict_proba(Xn)[:, 1]
        return -brier_score_loss(yn, p)

    r = permutation_importance(
        model,
        X,
        y,
        scoring=brier_scorer,
        n_repeats=n_repeats,
        random_state=seed,
        n_jobs=max(1, os.cpu_count() or 1),
    )
    return (pd.DataFrame({
        "feature": X.columns,
        "importance_mean": r.importances_mean,
        "importance_std": r.importances_std,
    }).sort_values("importance_mean", ascending=False))


def try_shap_importance(xgb_model, X: pd.DataFrame) -> pd.DataFrame:
    """Try SHAP importance, fallback to XGBoost native feature importance if SHAP fails."""
    try:
        import shap
        explainer = shap.TreeExplainer(xgb_model)
        shap_vals = explainer.shap_values(X)
        if isinstance(shap_vals, list):
            shap_vals = shap_vals[1]
        mean_abs = np.abs(shap_vals).mean(axis=0)
        return (pd.DataFrame({"feature": X.columns, "mean_abs_shap": mean_abs})
                .sort_values("mean_abs_shap", ascending=False))
    except ImportError:
        print("[INFO] shap not installed; using XGBoost native feature importance instead")
    except Exception as e:
        print(f"[WARN] SHAP analysis failed: {e}")
        print("[INFO] Using XGBoost native feature importance instead")
    
    # Fallback to XGBoost native feature importance (gain-based)
    if hasattr(xgb_model, 'feature_importances_'):
        importance = xgb_model.feature_importances_
        return (pd.DataFrame({
            "feature": X.columns,
            "xgb_importance_gain": importance
        }).sort_values("xgb_importance_gain", ascending=False))
    
    return pd.DataFrame()


# -----------------------------
# Main sweep
# -----------------------------
def run_sweep(cfg: Cfg):
    df = load_base(cfg)
    families = make_feature_families(df)

    optional = [k for k in families.keys() if k != "base"]
    results = []

    for h in cfg.horizons:
        y, fer, thr = make_label_from_forward_er(df, h, cfg.label_quantile, cfg.train_end)

        base_frame = df[["dt"]].copy()
        base_frame["y"] = y
        base_frame["fer_fwd"] = fer  # useful diagnostics

        print(f"\n--- Horizon H={h} | label q={cfg.label_quantile} | threshold={thr:.6f} ---")

        for r in range(0, len(optional) + 1):
            for combo in itertools.combinations(optional, r):
                fams = ("base",) + combo

                X = pd.concat([families[k] for k in fams], axis=1)
                data = pd.concat([base_frame, X], axis=1).dropna()

                X2 = data.drop(columns=["dt", "y", "fer_fwd"])
                y2 = data["y"].astype(int)
                dt = data["dt"]

                X_train, y_train, X_val, y_val, X_test, y_test = split_sets(dt, X2, y2, cfg)
                if len(X_train) < 300 or len(X_val) < 50 or len(X_test) < 50:
                    continue

                base_model, calib = train_calibrated_xgb(X_train, y_train, X_val, y_val, cfg)
                m = eval_model(calib, X_val, y_val, X_test, y_test)

                results.append({
                    "timestamp": datetime.now().isoformat(timespec="seconds"),
                    "horizon": h,
                    "label_q": cfg.label_quantile,
                    "label_thr": thr,
                    "featureset": "+".join(fams),
                    "n_features": X2.shape[1],
                    "n_train": int(len(X_train)),
                    "n_val": int(len(X_val)),
                    "n_test": int(len(X_test)),
                    "train_base_rate": float(y_train.mean()),
                    "val_base_rate": float(y_val.mean()),
                    "test_base_rate": float(y_test.mean()),
                    **m
                })
                print("DONE", results[-1])

    out = pd.DataFrame(results).sort_values(["val_brier", "val_auc"], ascending=[True, False])
    os.makedirs(os.path.dirname(cfg.out_csv), exist_ok=True)
    out.to_csv(cfg.out_csv, index=False)

    print("\nSaved:", cfg.out_csv)
    print("\nTop 15 by val_brier (then val_auc):")
    print(out.head(15).to_string(index=False))

    # -----------------------------
    # Analyze best run
    # -----------------------------
    best = out.iloc[0].to_dict()
    best_h = int(best["horizon"])
    best_featureset = best["featureset"]
    best_fams = best_featureset.split("+")

    print("\n=== Best run (by val_brier) ===")
    print(best)

    # rebuild label + data for best horizon/featureset
    y, fer, thr = make_label_from_forward_er(df, best_h, cfg.label_quantile, cfg.train_end)
    base_frame = df[["dt"]].copy()
    base_frame["y"] = y
    base_frame["fer_fwd"] = fer

    X_best = pd.concat([families[k] for k in best_fams], axis=1)
    data_best = pd.concat([base_frame, X_best], axis=1).dropna()

    Xb = data_best.drop(columns=["dt", "y", "fer_fwd"])
    yb = data_best["y"].astype(int)
    dtb = data_best["dt"]

    X_train, y_train, X_val, y_val, X_test, y_test = split_sets(dtb, Xb, yb, cfg)

    base_model, calib = train_calibrated_xgb(X_train, y_train, X_val, y_val, cfg)

    # Permutation importance
    pi_val = permutation_importance_proba(calib, X_val, y_val, n_repeats=40, seed=cfg.seed)
    pi_test = permutation_importance_proba(calib, X_test, y_test, n_repeats=40, seed=cfg.seed)

    pi_val_path = "ML/best_perm_importance_val.csv"
    pi_test_path = "ML/best_perm_importance_test.csv"
    pi_val.to_csv(pi_val_path, index=False)
    pi_test.to_csv(pi_test_path, index=False)

    print("\n[Permutation importance] VAL (top 20)")
    print(pi_val.head(20).to_string(index=False))
    print("\n[Permutation importance] TEST (top 20)")
    print(pi_test.head(20).to_string(index=False))
    print(f"\nSaved: {pi_val_path} and {pi_test_path}")

    # SHAP (optional) - fallback to XGBoost native importance
    feature_imp = try_shap_importance(base_model, X_train)
    if len(feature_imp):
        imp_path = "ML/best_feature_importance.csv"
        feature_imp.to_csv(imp_path, index=False)
        print("\n[Feature importance] (top 20)")
        print(feature_imp.head(20).to_string(index=False))
        print(f"\nSaved: {imp_path}")

    return out


if __name__ == "__main__":
    cfg = Cfg()
    run_sweep(cfg)