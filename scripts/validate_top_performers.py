#!/usr/bin/env python3
"""
Top performer removal validation for consolidation breakout model (v2).

Upgrades vs v1:
- Supports removal by COUNT (e.g., 5,10,15) or PERCENT (e.g., 10,20,30)
- Saves per-ticker ranking for inspection
- Prints a clear run signature so you never mix US/NL outputs
- Single, consistent GO/NO-GO assessment per run

Usage examples:

# Remove by counts
python scripts/validate_top_performers_v2.py \
  --bundle ML/production/v20251223_212702 \
  --events data/ml_datasets/accum_distrib_events.parquet \
  --market US \
  --threshold 0.75 \
  --cost_bps 10 \
  --remove_counts 5,10,15 \
  --output data/backtests/validation/top_perf_us_t075.csv

# Remove by % of tickers (recommended)
python scripts/validate_top_performers_v2.py \
  --bundle ML/production/v20251223_212702 \
  --events data/ml_datasets/accum_distrib_events.parquet \
  --market NL \
  --threshold 0.65 \
  --cost_bps 10 \
  --remove_percents 10,20,30 \
  --output data/backtests/validation/top_perf_nl_t065.csv
"""

import sys
import os
from pathlib import Path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

import argparse
import json
import logging
import numpy as np
import pandas as pd
import joblib
from sqlalchemy import create_engine, text

from src.config import settings

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)


def load_events(events_path: str, market: str = None) -> pd.DataFrame:
    ev = pd.read_parquet(events_path)
    ev["t_end"] = pd.to_datetime(ev["t_end"])
    ev["t_start"] = pd.to_datetime(ev["t_start"])
    if "label_valid" in ev.columns:
        ev = ev[ev["label_valid"] == True].copy()
    if market:
        ev = ev[ev["market"] == market].copy()
    return ev


def get_prices(engine, tickers, start_ts, end_ts, market=None):
    q = """
    SELECT ticker, market, timestamp, open, high, low, close, volume
    FROM bronze_ohlcv_intraday
    WHERE timestamp >= :start_ts AND timestamp <= :end_ts
      AND ticker = ANY(:tickers)
    """
    params = {"start_ts": start_ts, "end_ts": end_ts, "tickers": list(tickers)}
    if market:
        q += " AND market = :market"
        params["market"] = market
    q += " ORDER BY market, ticker, timestamp"
    return pd.read_sql(text(q), engine, params=params)


def run_backtest(events, prices, model, feature_names, threshold, cost_bps, exclude_tickers=None, H=40, ATR_K=1.5):
    if exclude_tickers is None:
        exclude_tickers = set()

    prices["timestamp"] = pd.to_datetime(prices["timestamp"])
    prices = prices.sort_values(["ticker", "timestamp"])
    by_ticker = {t: g.set_index("timestamp") for t, g in prices.groupby("ticker")}

    trades = []
    cost_fraction = cost_bps / 10000.0

    for _, ev in events.iterrows():
        tkr = ev["ticker"]
        if tkr in exclude_tickers:
            continue
        if tkr not in by_ticker:
            continue

        t_end = pd.to_datetime(ev["t_end"])
        entry_ts = t_end + pd.Timedelta(hours=1)

        g = by_ticker[tkr]
        if entry_ts not in g.index:
            continue

        entry_px = float(g.loc[entry_ts, "open"])
        if not np.isfinite(entry_px) or entry_px <= 0:
            continue

        try:
            X = ev[feature_names].to_frame().T.astype(float).fillna(0.0)
        except KeyError:
            continue

        p_up = float(model.predict_proba(X)[:, 1][0])

        if p_up >= threshold:
            side = "LONG"
        elif p_up <= (1 - threshold):
            side = "SHORT"
        else:
            continue

        path = g.loc[entry_ts:].iloc[:H+1]
        if len(path) < 2:
            continue

        atr_pct = float(ev.get("atr_pct_last", np.nan))
        if not np.isfinite(atr_pct) or atr_pct <= 0:
            continue

        atr_end = atr_pct * entry_px

        if side == "LONG":
            stop = entry_px - ATR_K * atr_end
            target = entry_px + ATR_K * atr_end
        else:
            stop = entry_px + ATR_K * atr_end
            target = entry_px - ATR_K * atr_end

        exit_ts = None
        exit_px = None
        exit_reason = None

        for ts, row in path.iterrows():
            if ts == entry_ts:
                continue

            hi, lo = float(row["high"]), float(row["low"])

            if side == "LONG":
                hit_target = hi >= target
                hit_stop = lo <= stop
            else:
                hit_target = lo <= target
                hit_stop = hi >= stop

            if hit_target and hit_stop:
                exit_ts, exit_px, exit_reason = ts, stop, "BOTH_WORST"
                break
            elif hit_target:
                exit_ts, exit_px, exit_reason = ts, target, "TARGET"
                break
            elif hit_stop:
                exit_ts, exit_px, exit_reason = ts, stop, "STOP"
                break

        if exit_ts is None:
            exit_ts = path.index[-1]
            exit_px = float(path.iloc[-1]["close"])
            exit_reason = "TIME"

        if side == "LONG":
            gross_ret = (exit_px - entry_px) / entry_px
        else:
            gross_ret = (entry_px - exit_px) / entry_px

        net_ret = gross_ret - 2 * cost_fraction

        trades.append({
            "ticker": tkr,
            "entry_ts": entry_ts,
            "exit_ts": exit_ts,
            "side": side,
            "net_ret": net_ret,
            "exit_reason": exit_reason,
        })

    if len(trades) == 0:
        return {"n_trades": 0}, pd.DataFrame()

    trades_df = pd.DataFrame(trades)
    win_rate = (trades_df["net_ret"] > 0).mean()
    avg_ret = trades_df["net_ret"].mean()
    med_ret = trades_df["net_ret"].median()
    std_ret = trades_df["net_ret"].std()
    n_tickers = trades_df["ticker"].nunique()

    eq = (1 + trades_df.sort_values("entry_ts")["net_ret"]).cumprod()
    peak = eq.cummax()
    dd = (eq / peak) - 1
    max_dd = float(dd.min())
    total_ret = float(eq.iloc[-1] - 1)

    sharpe = 0
    if std_ret > 0 and len(trades_df) > 1:
        sharpe = (avg_ret / std_ret) * np.sqrt(252 * 6.5)

    metrics = {
        "n_trades": int(len(trades_df)),
        "win_rate": float(win_rate),
        "avg_ret": float(avg_ret),
        "median_ret": float(med_ret),
        "std_ret": float(std_ret),
        "total_ret": float(total_ret),
        "max_dd": float(max_dd),
        "sharpe": float(sharpe),
        "n_tickers": int(n_tickers),
    }

    return metrics, trades_df


def main():
    ap = argparse.ArgumentParser(description="Top performer removal validation (v2)")
    ap.add_argument("--bundle", required=True)
    ap.add_argument("--events", required=True)
    ap.add_argument("--market", default=None)
    ap.add_argument("--threshold", type=float, default=0.65)
    ap.add_argument("--cost_bps", type=float, default=10.0)
    ap.add_argument("--remove_counts", default=None, help="Comma-separated counts e.g. 5,10,15")
    ap.add_argument("--remove_percents", default=None, help="Comma-separated percents e.g. 10,20,30")
    ap.add_argument("--output", required=True)
    args = ap.parse_args()

    print("\n" + "="*90)
    print("TOP PERFORMER REMOVAL VALIDATION (v2)")
    print(f"RUNNING FILE: {os.path.abspath(__file__)}")
    print(f"market={args.market} threshold={args.threshold} cost_bps={args.cost_bps}")
    print(f"remove_counts={args.remove_counts} remove_percents={args.remove_percents}")
    print("="*90 + "\n")

    # Load bundle
    bundle_dir = Path(args.bundle)
    features_data = json.loads((bundle_dir / "features.json").read_text())
    feature_names = features_data["feature_names"]

    model_file = bundle_dir / "model_calibrated.pkl"
    if not model_file.exists():
        model_file = bundle_dir / "model.pkl"
    model_bundle = joblib.load(model_file)
    model = model_bundle["model"] if isinstance(model_bundle, dict) and "model" in model_bundle else model_bundle

    # Load events
    events = load_events(args.events, args.market)
    if len(events) == 0:
        raise RuntimeError("No events after filtering.")

    # Price data range
    engine = create_engine(settings.database_url)
    tickers = events["ticker"].unique()
    price_start = events["t_end"].min() - pd.Timedelta(days=2)
    price_end = events["t_end"].max() + pd.Timedelta(hours=50)
    prices = get_prices(engine, tickers, price_start, price_end, args.market)

    # Baseline
    baseline_metrics, baseline_trades = run_backtest(events, prices, model, feature_names, args.threshold, args.cost_bps)

    # Per-ticker totals
    ticker_stats = baseline_trades.groupby("ticker")["net_ret"].sum().sort_values(ascending=False).reset_index()
    ticker_stats.columns = ["ticker", "total_ret"]
    n_universe = len(ticker_stats)

    # Decide removal sets
    removals = []
    if args.remove_counts:
        removals += [("count", int(x)) for x in args.remove_counts.split(",")]
    if args.remove_percents:
        removals += [("percent", float(x)) for x in args.remove_percents.split(",")]

    results = [{"scenario": "baseline", "removed": 0, "removed_type": "none", **baseline_metrics}]

    for rtype, rval in removals:
        if rtype == "count":
            n_remove = int(rval)
        else:
            n_remove = max(1, int(round(n_universe * (rval / 100.0))))

        top_set = set(ticker_stats.head(n_remove)["ticker"])
        metrics, _ = run_backtest(events, prices, model, feature_names, args.threshold, args.cost_bps, exclude_tickers=top_set)

        results.append({
            "scenario": f"remove_top_{n_remove}",
            "removed": n_remove,
            "removed_type": rtype,
            **metrics
        })

    # Save results + ticker ranking
    out = Path(args.output)
    out.parent.mkdir(parents=True, exist_ok=True)

    results_df = pd.DataFrame(results)
    results_df.to_csv(out, index=False)

    ticker_rank_file = out.with_suffix(".ticker_rank.csv")
    ticker_stats.to_csv(ticker_rank_file, index=False)

    print("\nSaved results:", out)
    print("Saved ticker ranking:", ticker_rank_file)

    # Print summary
    print("\n" + "="*70)
    print("TOP PERFORMER REMOVAL ANALYSIS")
    print("="*70)
    print(results_df[["scenario", "removed_type", "removed", "n_trades", "win_rate", "avg_ret", "sharpe", "max_dd"]])

    # GO/NO-GO
    print("\nGO/NO-GO ASSESSMENT")
    print("-"*70)

    base = results_df.iloc[0]
    for _, row in results_df.iterrows():
        if row["scenario"] == "baseline":
            continue

        sharpe_drop_pct = (base["sharpe"] - row["sharpe"]) / base["sharpe"] if base["sharpe"] != 0 else 1.0

        checks = {
            "Avg return > 0": row["avg_ret"] > 0,
            "Win rate > 60%": row["win_rate"] > 0.60,
            "Sharpe > 1.0": row["sharpe"] > 1.0,
            "Max DD < -20%": row["max_dd"] > -0.20,
            "Sharpe drop < 50%": sharpe_drop_pct < 0.50,
        }

        all_pass = all(checks.values())
        status = "✅ PASS" if all_pass else "❌ FAIL"
        print(f"\n{row['scenario']}: {status}")
        for k, v in checks.items():
            print(f"  • {k}: {'✓' if v else '✗'}")

    print("\nDone.\n")


if __name__ == "__main__":
    main()