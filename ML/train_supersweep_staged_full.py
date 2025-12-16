import os
import sys
import math
import itertools
from pathlib import Path
from datetime import datetime
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


# =============================================================================
# Config
# =============================================================================
class Cfg:
    def __init__(self):
        self.ticker = os.getenv("TICKER", "AD.AS")
        self.start = os.getenv("START", "2020-01-01")
        self.end = os.getenv("END", "2025-12-31")
        self.train_end = os.getenv("TRAIN_END", "2023-12-29")
        self.val_end = os.getenv("VAL_END", "2024-12-31")
        self.horizons = tuple(int(x) for x in os.getenv("HORIZONS", "3,4,5,7,10,14,21,30").split(","))

        # Feature-set breadth
        self.max_featureset_combos = int(os.getenv("MAX_FEATURESET_COMBOS", "60"))

        # Label breadth
        self.max_labels_per_h = int(os.getenv("MAX_LABELS_PER_H", "400"))

        # Filters
        self.min_train = int(os.getenv("MIN_TRAIN", "500"))
        self.min_val = int(os.getenv("MIN_VAL", "150"))
        self.min_test = int(os.getenv("MIN_TEST", "150"))
        self.base_rate_min = float(os.getenv("BASE_RATE_MIN", "0.10"))
        self.base_rate_max = float(os.getenv("BASE_RATE_MAX", "0.90"))

        # flat probability filter
        self.min_pred_spread = float(os.getenv("MIN_PRED_SPREAD", "0.08"))

        # staged controls
        self.stage1_params = int(os.getenv("STAGE1_PARAMS", "5"))
        self.stage2_params = int(os.getenv("STAGE2_PARAMS", "60"))
        self.topk_per_family = int(os.getenv("TOPK_PER_FAMILY", "4"))
        self.stage1_featureset = os.getenv("STAGE1_FEATURESET", "core+vol")

        # param sampling
        self.seed = int(os.getenv("SEED", "42"))

        # output
        self.out_csv = os.getenv("OUT_CSV", "ML/supersweep_staged_results.csv")


# =============================================================================
# Data load
# =============================================================================
def load_data(cfg: Cfg) -> pd.DataFrame:
    eng = create_engine(settings.database_url, pool_pre_ping=True)

    q_px = """
      SELECT trade_date::date AS dt, ticker,
             open::float, high::float, low::float, close::float, volume::float
      FROM bronze_ohlcv
      WHERE ticker=:t AND trade_date BETWEEN :s AND :e
      ORDER BY trade_date;
    """
    px = pd.read_sql(text(q_px), eng, params={"t": cfg.ticker, "s": cfg.start, "e": cfg.end})
    px["dt"] = pd.to_datetime(px["dt"])

    q_f = """
      SELECT trade_date::date AS dt, ticker,
        close::float AS f_close,
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
      WHERE ticker=:t AND trade_date BETWEEN :s AND :e
      ORDER BY trade_date;
    """
    f = pd.read_sql(text(q_f), eng, params={"t": cfg.ticker, "s": cfg.start, "e": cfg.end})
    f["dt"] = pd.to_datetime(f["dt"])

    df = px.merge(f, on=["ticker", "dt"], how="inner").sort_values("dt").reset_index(drop=True)
    df = df.dropna(subset=["close", "high", "low", "atr_14"])
    return df


# =============================================================================
# Features
# =============================================================================
def build_features(df: pd.DataFrame) -> pd.DataFrame:
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


def make_feature_sets(X: pd.DataFrame, cfg: Cfg, rng: np.random.Generator):
    core = [
        "px_vs_sma20","px_vs_sma50","px_vs_sma200",
        "ema_12","ema_26",
        "macd","macd_signal","macd_histogram","macd_z",
        "rsi_14","stochastic_k","stochastic_d","roc_20",
        "adx_14","plus_di_14","minus_di_14","di_diff",
        "atr_14","atr_pct",
        "bollinger_width","bb_width_pct",
        "realized_volatility_20","parkinson_volatility_20","rv20_logret",
        "volume_ratio","obv","obv_sma_20",
        "pct_from_high_20d","pct_from_low_20d","pct_from_high_52w","pct_from_low_52w",
        "logret_1d","logret_5d","logret_10d","gap_1d"
    ]
    core = [c for c in core if c in X.columns]

    families = {
        "core": core,
        "trend": [c for c in core if c in ["px_vs_sma20","px_vs_sma50","px_vs_sma200","adx_14","di_diff","roc_20","logret_10d"]],
        "momentum": [c for c in core if c in ["rsi_14","stochastic_k","stochastic_d","macd","macd_signal","macd_histogram","macd_z","logret_1d","logret_5d","logret_10d","gap_1d"]],
        "vol": [c for c in core if c in ["atr_14","atr_pct","bollinger_width","bb_width_pct","realized_volatility_20","parkinson_volatility_20","rv20_logret"]],
        "range": [c for c in core if c in ["bb_width_pct","bollinger_width","atr_pct","rv20_logret","realized_volatility_20"]],
        "volume": [c for c in core if c in ["volume_ratio","obv","obv_sma_20"]],
        "breadth": [c for c in core if c.startswith("pct_from_")],
    }

    keys = ["trend","momentum","vol","range","volume","breadth"]

    combos = []
    combos.append(("core",))
    for k in keys:
        combos.append(("core", k))
    for a,b in itertools.combinations(keys, 2):
        combos.append(("core", a, b))
    for a,b,c in itertools.combinations(keys, 3):
        combos.append(("core", a, b, c))

    combos = list(dict.fromkeys(combos))
    if len(combos) > cfg.max_featureset_combos:
        idx = list(rng.choice(len(combos), size=cfg.max_featureset_combos, replace=False))
        combos = [combos[i] for i in idx]

    out = {}
    for combo in combos:
        cols = []
        for fam in combo:
            cols += families[fam]
        cols = list(dict.fromkeys(cols))
        out["+".join(combo)] = cols

    return out


# =============================================================================
# Forward helpers (cached per horizon)
# =============================================================================
def fwd_max(s: pd.Series, h: int) -> pd.Series:
    return s.shift(-1).rolling(h).max().shift(-(h-1))

def fwd_min(s: pd.Series, h: int) -> pd.Series:
    return s.shift(-1).rolling(h).min().shift(-(h-1))

def fwd_ret(close: pd.Series, h: int) -> pd.Series:
    return close.shift(-h) / close - 1.0

def pct_up_days(logret_1d: pd.Series, h: int) -> pd.Series:
    up = (logret_1d > 0).astype(float)
    return up.shift(-1).rolling(h).mean().shift(-(h-1))

def pct_down_days(logret_1d: pd.Series, h: int) -> pd.Series:
    dn = (logret_1d < 0).astype(float)
    return dn.shift(-1).rolling(h).mean().shift(-(h-1))

def precompute_forward_cache(df: pd.DataFrame, h: int) -> dict:
    close = df["close"].astype(float)
    high  = df["high"].astype(float)
    low   = df["low"].astype(float)
    return {
        "f_max_high": fwd_max(high, h),
        "f_min_low": fwd_min(low, h),
        "f_max_close": fwd_max(close, h),
        "f_min_close": fwd_min(close, h),
        "f_ret": fwd_ret(close, h),
        "pct_up": pct_up_days(df["logret_1d"], h),
        "pct_down": pct_down_days(df["logret_1d"], h),
    }


# =============================================================================
# Label zoo
# =============================================================================
def multipliers_for_h(h: int):
    if h <= 5:
        return [0.75, 1.0, 1.25]
    if h <= 10:
        return [1.0, 1.25, 1.5]
    if h <= 14:
        return [1.25, 1.5, 1.75]
    return [1.5, 1.75, 2.0, 2.25]

def make_label_specs(h: int):
    specs = []
    specs.append(("ret_down_q30", {"q": 0.30}))
    specs.append(("ret_up_q70", {"q": 0.70}))

    for m in multipliers_for_h(h):
        specs += [
            (f"touch_up_atr_{m}", {"m": m}),
            (f"touch_down_atr_{m}", {"m": m}),
            (f"close_break_up_atr_{m}", {"m": m}),
            (f"close_break_down_atr_{m}", {"m": m}),
            (f"dd_atr_{m}", {"m": m}),
            (f"ud_atr_{m}", {"m": m}),
        ]

    specs.append(("rv_expand_q70", {"q": 0.70}))
    specs.append(("rv_compress_q30", {"q": 0.30}))
    specs.append(("rv_expand_abs", {"d": 0.05}))
    specs.append(("rv_compress_abs", {"d": -0.05}))

    specs.append(("low_range_atr", {"thr": 1.0}))
    specs.append(("high_range_atr", {"thr": 2.0}))

    specs.append(("trend_up_persist", {"thr": 0.65}))
    specs.append(("trend_down_persist", {"thr": 0.65}))
    return specs

def build_label_series_cached(df: pd.DataFrame, h: int, name: str, params: dict,
                              train_mask: pd.Series, cache: dict) -> pd.Series:
    close = df["close"].astype(float)
    atr = df["atr_14"].astype(float)

    f_max_high  = cache["f_max_high"]
    f_min_low   = cache["f_min_low"]
    f_max_close = cache["f_max_close"]
    f_min_close = cache["f_min_close"]
    f_r         = cache["f_ret"]

    rv = df["realized_volatility_20"]
    if rv.isna().all():
        rv = df["rv20_logret"]
    rv = rv.astype(float)

    if name == "ret_down_q30":
        thr = f_r[train_mask].quantile(params["q"])
        return (f_r <= thr).astype(int)
    if name == "ret_up_q70":
        thr = f_r[train_mask].quantile(params["q"])
        return (f_r >= thr).astype(int)

    if name.startswith("touch_up_atr_"):
        m = float(params["m"])
        return (f_max_high >= (close + m * atr)).astype(int)
    if name.startswith("touch_down_atr_"):
        m = float(params["m"])
        return (f_min_low <= (close - m * atr)).astype(int)

    if name.startswith("close_break_up_atr_"):
        m = float(params["m"])
        return (f_max_close >= (close + m * atr)).astype(int)
    if name.startswith("close_break_down_atr_"):
        m = float(params["m"])
        return (f_min_close <= (close - m * atr)).astype(int)

    if name.startswith("dd_atr_"):
        m = float(params["m"])
        return (f_min_low <= (close - m * atr)).astype(int)
    if name.startswith("ud_atr_"):
        m = float(params["m"])
        return (f_max_high >= (close + m * atr)).astype(int)

    if name == "rv_expand_q70":
        d = rv.shift(-h) - rv
        thr = d[train_mask].quantile(params["q"])
        return (d >= thr).astype(int)
    if name == "rv_compress_q30":
        d = rv.shift(-h) - rv
        thr = d[train_mask].quantile(params["q"])
        return (d <= thr).astype(int)

    if name == "rv_expand_abs":
        d = rv.shift(-h) - rv
        return (d >= params["d"]).astype(int)
    if name == "rv_compress_abs":
        d = rv.shift(-h) - rv
        return (d <= params["d"]).astype(int)

    if name == "low_range_atr":
        rng_ = (cache["f_max_high"] - cache["f_min_low"]) / atr.replace(0, np.nan)
        return (rng_ <= params["thr"]).astype(int)
    if name == "high_range_atr":
        rng_ = (cache["f_max_high"] - cache["f_min_low"]) / atr.replace(0, np.nan)
        return (rng_ >= params["thr"]).astype(int)

    if name == "trend_up_persist":
        return (cache["pct_up"] >= params["thr"]).astype(int)
    if name == "trend_down_persist":
        return (cache["pct_down"] >= params["thr"]).astype(int)

    raise ValueError(f"Unknown label: {name}")


def label_family(label: str) -> str:
    if label.startswith("ret_"):
        return "ret_quantile"
    if label.startswith("touch_"):
        return "atr_touch"
    if label.startswith("close_break_"):
        return "atr_close_break"
    if label.startswith("dd_") or label.startswith("ud_"):
        return "atr_tail"
    if label.startswith("rv_"):
        return "vol_regime"
    if label.endswith("range_atr"):
        return "range_atr"
    if label.startswith("trend_"):
        return "trend_persist"
    return "other"


# =============================================================================
# Model training
# =============================================================================
def sample_params(rng: np.random.Generator, n: int):
    out = []
    for _ in range(n):
        out.append({
            "max_depth": int(rng.integers(2, 6)),
            "learning_rate": float(rng.choice([0.01, 0.03, 0.05, 0.08, 0.10])),
            "n_estimators": int(rng.choice([200, 400, 600, 800])),
            "subsample": float(rng.choice([0.7, 0.85, 1.0])),
            "colsample_bytree": float(rng.choice([0.7, 0.85, 1.0])),
            "reg_lambda": float(rng.choice([1.0, 3.0, 6.0])),
            "min_child_weight": int(rng.choice([1, 5, 10])),
        })
    return out

def train_eval(X_train, y_train, X_val, y_val, X_test, y_test, params, seed: int):
    base = XGBClassifier(
        objective="binary:logistic",
        eval_metric="logloss",
        random_state=seed,
        n_jobs=max(1, os.cpu_count() or 1),
        tree_method="hist",
        **params
    )
    base.fit(X_train, y_train)

    calib = CalibratedClassifierCV(base, method="sigmoid", cv="prefit")
    calib.fit(X_val, y_val)

    p_val = calib.predict_proba(X_val)[:, 1]
    p_test = calib.predict_proba(X_test)[:, 1]

    return {
        "val_auc": roc_auc_score(y_val, p_val) if len(np.unique(y_val)) > 1 else np.nan,
        "val_brier": brier_score_loss(y_val, p_val),
        "val_logloss": log_loss(y_val, p_val, eps=1e-15),
        "val_base_rate": float(y_val.mean()),
        "test_auc": roc_auc_score(y_test, p_test) if len(np.unique(y_test)) > 1 else np.nan,
        "test_brier": brier_score_loss(y_test, p_test),
        "test_logloss": log_loss(y_test, p_test, eps=1e-15),
        "test_base_rate": float(y_test.mean()),
        "p_test_min": float(np.min(p_test)),
        "p_test_max": float(np.max(p_test)),
        "p_spread": float(np.max(p_test) - np.min(p_test)),
    }

def select_score(met: dict):
    # choose on VAL only (no test leakage)
    return (met["val_auc"], -met["val_brier"])


# =============================================================================
# Main staged sweep
# =============================================================================
def main():
    cfg = Cfg()
    rng = np.random.default_rng(cfg.seed)

    df0 = load_data(cfg)
    df = build_features(df0)

    dt = df["dt"]
    train_end = pd.to_datetime(cfg.train_end)
    val_end = pd.to_datetime(cfg.val_end)

    m_train = dt <= train_end
    m_val = (dt > train_end) & (dt <= val_end)
    m_test = dt > val_end

    print(f"Loaded rows: {len(df)}  splits: train={m_train.sum()} val={m_val.sum()} test={m_test.sum()}")

    feat_sets = make_feature_sets(df, cfg, rng)
    print(f"Feature sets: {len(feat_sets)} (capped by MAX_FEATURESET_COMBOS={cfg.max_featureset_combos})")

    params_stage1 = sample_params(rng, cfg.stage1_params)
    params_stage2 = sample_params(rng, cfg.stage2_params)

    results = []
    total_models = 0

    # choose stage1 featureset safely
    if cfg.stage1_featureset not in feat_sets:
        cfg.stage1_featureset = list(feat_sets.keys())[0]
    fs1_cols = [c for c in feat_sets[cfg.stage1_featureset] if c in df.columns]

    for h in cfg.horizons:
        specs = make_label_specs(h)
        if len(specs) > cfg.max_labels_per_h:
            specs = list(rng.choice(specs, size=cfg.max_labels_per_h, replace=False))
        print(f"\n=== Horizon {h} | labels to try: {len(specs)} ===")

        cache = precompute_forward_cache(df, h)

        # ---------------- Stage 1 ----------------
        stage1_rows = []
        for lname, lparams in specs:
            y = build_label_series_cached(df, h, lname, lparams, m_train, cache)

            # Don't copy entire df - only select needed columns
            tmp = df[["dt"] + fs1_cols].copy()
            tmp["y"] = y
            tmp = tmp.dropna()
            if len(tmp) < (cfg.min_train + cfg.min_val + cfg.min_test):
                continue

            tr = m_train.loc[tmp.index]
            va = m_val.loc[tmp.index]
            te = m_test.loc[tmp.index]

            y_tr = tmp.loc[tr, "y"].astype(int)
            y_va = tmp.loc[va, "y"].astype(int)
            y_te = tmp.loc[te, "y"].astype(int)

            if len(y_tr) < cfg.min_train or len(y_va) < cfg.min_val or len(y_te) < cfg.min_test:
                continue

            br = float(y_tr.mean())
            if not (cfg.base_rate_min <= br <= cfg.base_rate_max):
                continue

            if len(np.unique(y_tr)) < 2 or len(np.unique(y_va)) < 2 or len(np.unique(y_te)) < 2:
                continue

            Xtr = tmp.loc[tr, fs1_cols]
            Xva = tmp.loc[va, fs1_cols]
            Xte = tmp.loc[te, fs1_cols]

            best = None
            for p in params_stage1:
                met = train_eval(Xtr, y_tr, Xva, y_va, Xte, y_te, p, cfg.seed)
                total_models += 1
                if met["p_spread"] < cfg.min_pred_spread:
                    continue
                sc = select_score(met)
                if best is None or sc > best["score"]:
                    best = {"score": sc, "params": p, "met": met}

            if best is None:
                continue

            stage1_rows.append({
                "horizon": h,
                "label": lname,
                "label_params": str(lparams),
                "label_family": label_family(lname),
                "featureset": cfg.stage1_featureset,
                **best["met"],
                "best_params": str(best["params"]),
                "n_features": len(fs1_cols)
            })

        stage1 = pd.DataFrame(stage1_rows)
        if stage1.empty:
            print("Stage1: no viable labels.")
            continue

        shortlist = (stage1.sort_values(["val_auc", "val_brier"], ascending=[False, True])
                          .groupby("label_family", as_index=False)
                          .head(cfg.topk_per_family))
        print(f"Stage1 shortlist: {len(shortlist)} labels ({cfg.topk_per_family}/family max)")

        # ---------------- Stage 2 ----------------
        for _, srow in shortlist.iterrows():
            lname = srow["label"]
            lparams = eval(srow["label_params"]) if isinstance(srow["label_params"], str) else srow["label_params"]
            y = build_label_series_cached(df, h, lname, lparams, m_train, cache)

            for fs_name, cols in feat_sets.items():
                cols = [c for c in cols if c in df.columns]
                if len(cols) < 10:
                    continue

                # Don't copy entire df - only select needed columns + y
                tmp = df[["dt"] + cols].copy()
                tmp["y"] = y
                tmp = tmp.dropna()
                if len(tmp) < (cfg.min_train + cfg.min_val + cfg.min_test):
                    continue

                tr = m_train.loc[tmp.index]
                va = m_val.loc[tmp.index]
                te = m_test.loc[tmp.index]

                y_tr = tmp.loc[tr, "y"].astype(int)
                y_va = tmp.loc[va, "y"].astype(int)
                y_te = tmp.loc[te, "y"].astype(int)

                if len(y_tr) < cfg.min_train or len(y_va) < cfg.min_val or len(y_te) < cfg.min_test:
                    continue
                if len(np.unique(y_tr)) < 2 or len(np.unique(y_va)) < 2 or len(np.unique(y_te)) < 2:
                    continue

                Xtr = tmp.loc[tr, cols]
                Xva = tmp.loc[va, cols]
                Xte = tmp.loc[te, cols]

                best = None
                for p in params_stage2:
                    met = train_eval(Xtr, y_tr, Xva, y_va, Xte, y_te, p, cfg.seed)
                    total_models += 1
                    if met["p_spread"] < cfg.min_pred_spread:
                        continue
                    sc = select_score(met)
                    if best is None or sc > best["score"]:
                        best = {"score": sc, "params": p, "met": met}

                if best is None:
                    continue

                results.append({
                    "timestamp": datetime.now().isoformat(timespec="seconds"),
                    "ticker": cfg.ticker,
                    "horizon": h,
                    "label": lname,
                    "label_params": str(lparams),
                    "label_family": label_family(lname),
                    "featureset": fs_name,
                    "n_features": len(cols),
                    "n_train": int(len(y_tr)),
                    "n_val": int(len(y_va)),
                    "n_test": int(len(y_te)),
                    "train_base_rate": float(y_tr.mean()),
                    "val_base_rate": float(y_va.mean()),
                    "test_base_rate": float(y_te.mean()),
                    **best["met"],
                    "best_params": str(best["params"]),
                })

                if len(results) % 25 == 0:
                    print(f"progress: results={len(results)} models_trained={total_models}")
        
        # Save results after each horizon completes (incremental save)
        if results:
            res_partial = pd.DataFrame(results)
            os.makedirs(os.path.dirname(cfg.out_csv), exist_ok=True)
            
            # Check if file exists to determine write mode
            if os.path.exists(cfg.out_csv):
                # Append without header
                res_partial.to_csv(cfg.out_csv, mode='a', header=False, index=False)
                print(f"✓ Appended {len(res_partial)} rows to {cfg.out_csv} (horizon {h} complete)")
            else:
                # First write with header
                res_partial.to_csv(cfg.out_csv, mode='w', header=True, index=False)
                print(f"✓ Created {cfg.out_csv} with {len(res_partial)} rows (horizon {h} complete)")
            
            # Clear results list to free memory
            results = []
        
        # Clear cache for this horizon to free memory
        del cache
        import gc
        gc.collect()
        print(f"✓ Memory cleared after horizon {h}")

    # Final summary (results list should be empty now due to incremental saves)
    if os.path.exists(cfg.out_csv):
        res = pd.read_csv(cfg.out_csv)
        print(f"\n{'='*80}")
        print(f"Final results saved: {cfg.out_csv}")
        print(f"{'='*80}")
        print(f"Total trained candidate models: {total_models}")
        print(f"Total kept rows: {len(res)}")
    else:
        print("\nNo results generated.")
        return

    if len(res):
        top = res.sort_values(["test_auc","val_auc"], ascending=False).head(25)
        print("\n=== TOP 25 by test_auc (reporting only; selected on VAL) ===")
        print(top[["horizon","label_family","label","featureset","val_auc","test_auc","val_brier","test_brier","p_spread","test_base_rate"]].to_string(index=False))

        best_fam = (res.sort_values(["val_auc","val_brier"], ascending=[False, True])
                      .groupby(["horizon","label_family"], as_index=False).head(1))
        screen_auc = best_fam.pivot_table(index="horizon", columns="label_family", values="test_auc", aggfunc="first")
        screen_brier = best_fam.pivot_table(index="horizon", columns="label_family", values="test_brier", aggfunc="first")

        print("\n=== 3D SCREEN: best-per-(horizon,label_family) test_auc ===")
        print(screen_auc.round(4).to_string())
        print("\n=== 3D SCREEN: best-per-(horizon,label_family) test_brier ===")
        print(screen_brier.round(4).to_string())

    print("\nDONE")


if __name__ == "__main__":
    main()