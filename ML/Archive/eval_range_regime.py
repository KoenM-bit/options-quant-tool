import os
import sys
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd
from sqlalchemy import create_engine, text

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from src.config import settings


@dataclass
class Cfg:
    ticker: str = os.getenv("TICKER", "AD.AS")
    horizon: int = int(os.getenv("HORIZON", "5"))
    label_q: float = float(os.getenv("LABEL_Q", "0.30"))

    model_version: str = os.getenv("MODEL_VERSION")  # REQUIRED
    pred_table: str = os.getenv("PRED_TABLE", "gold_regime_predictions")

    train_end: str = os.getenv("TRAIN_END", "2023-12-29")
    val_end: str = os.getenv("VAL_END", "2024-12-31")


def make_engine():
    """Get database engine using project settings."""
    return create_engine(settings.database_url, pool_pre_ping=True)


def validate_config(cfg: Cfg):
    """Validate required configuration."""
    if not cfg.model_version:
        raise RuntimeError("Set MODEL_VERSION env var (the one you wrote).")


def forward_er(price: pd.Series, h: int) -> pd.Series:
    dp_abs = price.diff().abs()
    denom = dp_abs.shift(-1).rolling(h).sum()
    num = (price.shift(-h) - price).abs()
    return num / denom.replace(0, np.nan)


def bucket_report(df: pd.DataFrame, split: str):
    d = df[df["split"] == split].copy()
    p = d["p"].values
    y = d["y"].values

    p25 = float(np.quantile(p, 0.25))
    p75 = float(np.quantile(p, 0.75))

    def seg(mask, label):
        if mask.sum() == 0:
            return {"segment": label, "n": 0, "actual_rate": np.nan, "avg_p": np.nan}
        return {"segment": label, "n": int(mask.sum()), "actual_rate": float(y[mask].mean()), "avg_p": float(p[mask].mean())}

    rows = [
        seg(p >= p75, "top_25%"),
        seg((p > p25) & (p < p75), "mid_50%"),
        seg(p <= p25, "bottom_25%"),
        seg(np.ones_like(p, dtype=bool), "all"),
    ]
    print(f"\n[{split}] thresholds: p25={p25:.4f}, p75={p75:.4f}")
    print(pd.DataFrame(rows).to_string(index=False))
    return p25, p75


def main():
    cfg = Cfg()
    validate_config(cfg)
    eng = make_engine()

    # Pull predictions
    q_pred = f"""
      SELECT dt::date AS dt, p_range::float AS p,
             CASE
               WHEN dt::date <= :train_end THEN 'train'
               WHEN dt::date <= :val_end THEN 'val'
               ELSE 'test'
             END AS split
      FROM {cfg.pred_table}
      WHERE ticker=:t AND horizon=:h AND label_q=:q AND model_version=:mv
      ORDER BY dt;
    """
    pred = pd.read_sql(text(q_pred), eng, params={
        "t": cfg.ticker, "h": cfg.horizon, "q": cfg.label_q, "mv": cfg.model_version,
        "train_end": cfg.train_end, "val_end": cfg.val_end
    })
    pred["dt"] = pd.to_datetime(pred["dt"])

    # Pull prices to rebuild labels consistently (same def)
    q_px = """
      SELECT dt::date AS dt, px::float AS px
      FROM silver_prices_ad
      WHERE ticker=:t
      ORDER BY dt;
    """
    px = pd.read_sql(text(q_px), eng, params={"t": cfg.ticker})
    px["dt"] = pd.to_datetime(px["dt"])

    df = px.merge(pred, on="dt", how="inner").sort_values("dt").reset_index(drop=True)

    fer = forward_er(df["px"], cfg.horizon)
    train_mask = df["dt"] <= pd.to_datetime(cfg.train_end)
    thr = float(np.nanquantile(fer[train_mask].dropna().values, cfg.label_q))
    y = (fer <= thr).astype(float)
    y[fer.isna()] = np.nan

    df["y"] = y
    df = df.dropna(subset=["y", "p"]).copy()
    df["y"] = df["y"].astype(int)

    print("Rows:", len(df), "splits:", df["split"].value_counts().to_dict())
    print(f"Label threshold (train q{cfg.label_q}): {thr:.6f}")

    # Reports
    p25_val, p75_val = bucket_report(df, "val")
    p25_test, p75_test = bucket_report(df, "test")

    print("\nPolicy suggestion (use percentiles):")
    print(f"- Range regime  : p >= p75  (VAL p75={p75_val:.4f}, TEST p75={p75_test:.4f})")
    print(f"- Trend regime  : p <= p25  (VAL p25={p25_val:.4f}, TEST p25={p25_test:.4f})")
    print("- Middle        : otherwise (trade conservatively)")

    # Show latest signals
    latest = df.tail(10)[["dt", "p", "split", "y"]]
    print("\nLatest rows:")
    print(latest.to_string(index=False))


if __name__ == "__main__":
    main()