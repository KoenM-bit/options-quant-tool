#!/usr/bin/env python3
"""
Slippage curve validation for NL Top 25 Liquidity universe.

Tests whether Top 25 Liquidity NL tolerates more slippage than full universe.

Previous result: Full NL universe tolerates only ~3 bps
Expected: Top 25 should tolerate ≥5-6 bps (like US market)

Usage:
    python scripts/validate_slippage_nl_top25.py \
        --bundle ML/production/v20251223_212702 \
        --events data/ml_datasets/accum_distrib_events.parquet \
        --threshold 0.65 \
        --cost_bps 10 \
        --slippage_levels 0,1,2,3,4,5,6,8,10,12,15 \
        --output data/backtests/validation/slippage_nl_top25.csv
"""

import sys
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


def compute_liquidity_metrics(engine, tickers, market="NL"):
    """Compute average daily volume for each ticker."""
    q = """
    SELECT 
        ticker,
        AVG(volume) as avg_volume
    FROM bronze_ohlcv_intraday
    WHERE ticker = ANY(:tickers)
      AND market = :market
      AND volume > 0
    GROUP BY ticker
    """
    params = {"tickers": list(tickers), "market": market}
    df = pd.read_sql(text(q), engine, params=params)
    return df


def load_events(events_path: str, market: str = "NL") -> pd.DataFrame:
    """Load and prepare events dataset."""
    ev = pd.read_parquet(events_path)
    ev["t_end"] = pd.to_datetime(ev["t_end"])
    ev["t_start"] = pd.to_datetime(ev["t_start"])

    if "label_valid" in ev.columns:
        ev = ev[ev["label_valid"] == True].copy()

    ev = ev[ev["market"] == market].copy()
    return ev


def get_prices(engine, tickers, start_ts, end_ts, market="NL"):
    """Fetch OHLCV data."""
    q = """
    SELECT ticker, market, timestamp, open, high, low, close, volume
    FROM bronze_ohlcv_intraday
    WHERE timestamp >= :start_ts AND timestamp <= :end_ts
      AND ticker = ANY(:tickers)
      AND market = :market
    ORDER BY ticker, timestamp
    """
    params = {"start_ts": start_ts, "end_ts": end_ts, "tickers": list(tickers), "market": market}
    return pd.read_sql(text(q), engine, params=params)


def backtest_with_slippage(
    events: pd.DataFrame,
    prices: pd.DataFrame,
    model,
    feature_names: list,
    threshold: float,
    cost_bps: float,
    slippage_bps: float,
    H: int = 40,
    ATR_K: float = 1.5
) -> dict:
    """
    Run backtest with directional slippage.
    
    LONG: Pay slippage (entry_px * (1 + slippage))
    SHORT: Receive less (entry_px * (1 - slippage))
    """
    prices["timestamp"] = pd.to_datetime(prices["timestamp"])
    prices = prices.sort_values(["ticker", "timestamp"])
    by_ticker = {t: g.set_index("timestamp") for t, g in prices.groupby("ticker")}

    trades = []
    cost_fraction = cost_bps / 10000.0
    slip_fraction = slippage_bps / 10000.0

    for _, ev in events.iterrows():
        tkr = ev["ticker"]
        if tkr not in by_ticker:
            continue

        t_end = pd.to_datetime(ev["t_end"])
        entry_ts = t_end + pd.Timedelta(hours=1)

        g = by_ticker[tkr]
        if entry_ts not in g.index:
            continue

        entry_px_raw = float(g.loc[entry_ts, "open"])
        if not np.isfinite(entry_px_raw) or entry_px_raw <= 0:
            continue

        # Features
        try:
            X = ev[feature_names].to_frame().T
            X = X.astype(float).fillna(0.0)
        except KeyError:
            continue

        # Probability
        p_up = float(model.predict_proba(X)[:, 1][0])

        # Decision
        if p_up >= threshold:
            side = "LONG"
        elif p_up <= (1 - threshold):
            side = "SHORT"
        else:
            continue

        # Apply directional slippage
        if side == "LONG":
            entry_px = entry_px_raw * (1 + slip_fraction)  # Pay more
        else:
            entry_px = entry_px_raw * (1 - slip_fraction)  # Receive less

        # Path window
        path = g.loc[entry_ts:].iloc[:H+1]
        if len(path) < 2:
            continue

        # ATR
        atr_pct = float(ev.get("atr_pct_last", np.nan))
        if not np.isfinite(atr_pct) or atr_pct <= 0:
            continue
        atr_end = atr_pct * entry_px

        # Stops/targets
        if side == "LONG":
            stop = entry_px - ATR_K * atr_end
            target = entry_px + ATR_K * atr_end
        else:
            stop = entry_px + ATR_K * atr_end
            target = entry_px - ATR_K * atr_end

        # Exit logic
        exit_ts = None
        exit_px = None
        exit_reason = None

        for ts, row in path.iterrows():
            if ts == entry_ts:
                continue

            hi = float(row["high"])
            lo = float(row["low"])

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

        # Returns
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
            "p_up": p_up,
            "net_ret": net_ret,
            "exit_reason": exit_reason
        })

    if len(trades) == 0:
        return {
            "n_trades": 0, "win_rate": 0, "avg_ret": 0, "median_ret": 0,
            "std_ret": 0, "total_ret": 0, "max_dd": 0, "sharpe": 0
        }

    trades_df = pd.DataFrame(trades)
    trades_df = trades_df.sort_values("entry_ts")

    win_rate = (trades_df["net_ret"] > 0).mean()
    avg_ret = trades_df["net_ret"].mean()
    med_ret = trades_df["net_ret"].median()
    std_ret = trades_df["net_ret"].std()

    # Equity curve
    eq = (1 + trades_df["net_ret"]).cumprod()
    peak = eq.cummax()
    dd = (eq / peak) - 1
    max_dd = float(dd.min())
    total_ret = float(eq.iloc[-1] - 1)

    # Sharpe
    if std_ret > 0 and len(trades_df) > 1:
        sharpe = (avg_ret / std_ret) * np.sqrt(252 * 6.5)
    else:
        sharpe = 0

    return {
        "n_trades": int(len(trades_df)),
        "win_rate": float(win_rate),
        "avg_ret": float(avg_ret),
        "median_ret": float(med_ret),
        "std_ret": float(std_ret),
        "total_ret": float(total_ret),
        "max_dd": float(max_dd),
        "sharpe": float(sharpe),
    }


def main():
    ap = argparse.ArgumentParser(description="NL Top 25 Liquidity slippage curve validation")
    ap.add_argument("--bundle", required=True, help="Production bundle directory")
    ap.add_argument("--events", required=True, help="Events parquet file")
    ap.add_argument("--threshold", type=float, default=0.65, help="Decision threshold")
    ap.add_argument("--cost_bps", type=float, default=10.0, help="Transaction costs (bps)")
    ap.add_argument("--slippage_levels", default="0,1,2,3,4,5,6,8,10,12,15", help="Comma-separated slippage levels (bps)")
    ap.add_argument("--output", required=True, help="Output CSV path")
    args = ap.parse_args()

    slippage_levels = [float(x) for x in args.slippage_levels.split(",")]

    logger.info("="*70)
    logger.info("NL TOP 25 LIQUIDITY SLIPPAGE CURVE VALIDATION")
    logger.info("="*70)
    logger.info(f"Threshold: {args.threshold}, Cost: {args.cost_bps} bps")
    logger.info(f"Slippage levels: {slippage_levels}")
    logger.info("")

    # Load bundle
    bundle_dir = Path(args.bundle)
    features_data = json.loads((bundle_dir / "features.json").read_text())
    feature_names = features_data["feature_names"]

    model_file = bundle_dir / "model_calibrated.pkl"
    if not model_file.exists():
        model_file = bundle_dir / "model.pkl"

    model_bundle = joblib.load(model_file)
    model = model_bundle["model"] if isinstance(model_bundle, dict) and "model" in model_bundle else model_bundle

    logger.info(f"Loaded model with {len(feature_names)} features")

    # Load events
    events = load_events(args.events)
    logger.info(f"Loaded {len(events):,} NL events")

    if len(events) == 0:
        logger.error("No NL events found")
        return

    # Get liquidity metrics and filter to Top 25
    engine = create_engine(settings.database_url)
    tickers = events["ticker"].unique()
    
    logger.info("\nComputing liquidity metrics...")
    liq_metrics = compute_liquidity_metrics(engine, tickers)
    top_25_tickers = set(liq_metrics.nlargest(25, "avg_volume")["ticker"])
    
    logger.info(f"Top 25 liquidity tickers: {', '.join(sorted(top_25_tickers))}")

    # Filter events and prices to Top 25
    events = events[events["ticker"].isin(top_25_tickers)].copy()
    logger.info(f"Filtered to {len(events):,} events in Top 25 universe")

    # Get prices
    price_start = events["t_end"].min() - pd.Timedelta(days=2)
    price_end = events["t_end"].max() + pd.Timedelta(hours=50)

    logger.info(f"Fetching prices for Top 25 tickers...")
    prices = get_prices(engine, top_25_tickers, price_start, price_end)
    logger.info(f"Loaded {len(prices):,} price bars")

    # Run slippage curve
    results = []

    for slip_bps in slippage_levels:
        logger.info(f"\nTesting slippage: {slip_bps} bps")
        
        metrics = backtest_with_slippage(
            events, prices, model, feature_names,
            args.threshold, args.cost_bps, slip_bps
        )

        results.append({"slippage_bps": slip_bps, **metrics})

        logger.info(
            f"  {metrics['n_trades']} trades, {metrics['win_rate']:.1%} WR, "
            f"{metrics['avg_ret']:.3%} avg, Sharpe {metrics['sharpe']:.2f}, DD {metrics['max_dd']:.1%}"
        )

    # Save results
    results_df = pd.DataFrame(results)
    out = Path(args.output)
    out.parent.mkdir(parents=True, exist_ok=True)
    results_df.to_csv(out, index=False)
    logger.info(f"\nSaved results: {out}")

    # Analysis
    print("\n" + "="*90)
    print("NL TOP 25 LIQUIDITY SLIPPAGE CURVE ANALYSIS")
    print("="*90)
    print(f"\n{'Slippage (bps)':<15} {'Trades':<8} {'WR':<8} {'Avg Ret':<12} {'Sharpe':<10} {'Max DD':<10}")
    print("-"*90)

    for _, row in results_df.iterrows():
        print(f"{row['slippage_bps']:<15.0f} {row['n_trades']:<8} {row['win_rate']:.1%}    "
              f"{row['avg_ret']:.3%}       {row['sharpe']:.2f}       {row['max_dd']:.1%}")

    print("="*90)

    # Find breakeven thresholds
    print("\nBREAKEVEN THRESHOLDS")
    print("-"*90)

    # Positive returns
    pos_ret = results_df[results_df["avg_ret"] > 0]
    if len(pos_ret) > 0:
        max_slip_pos = pos_ret["slippage_bps"].max()
        print(f"✓ Positive returns up to: {max_slip_pos:.0f} bps")
    else:
        print("✗ No positive returns at any slippage level")

    # Sharpe > 1.0
    good_sharpe = results_df[results_df["sharpe"] > 1.0]
    if len(good_sharpe) > 0:
        max_slip_sharpe = good_sharpe["slippage_bps"].max()
        print(f"✓ Sharpe > 1.0 up to: {max_slip_sharpe:.0f} bps")
    else:
        print("✗ No Sharpe > 1.0 at any slippage level")

    # Max DD > -15%
    acceptable_dd = results_df[results_df["max_dd"] > -0.15]
    if len(acceptable_dd) > 0:
        max_slip_dd = acceptable_dd["slippage_bps"].max()
        print(f"✓ Max DD < -15% up to: {max_slip_dd:.0f} bps")
    else:
        print("✗ Excessive drawdown at all levels")

    # ALL criteria
    all_pass = results_df[
        (results_df["avg_ret"] > 0) & 
        (results_df["sharpe"] > 1.0) & 
        (results_df["max_dd"] > -0.15)
    ]
    
    if len(all_pass) > 0:
        max_slip_all = all_pass["slippage_bps"].max()
        print(f"\n🎯 ALL CRITERIA MET up to: {max_slip_all:.0f} bps")
        
        if max_slip_all >= 5:
            print(f"\n✅ VERDICT: NL Top 25 tolerates ≥5 bps slippage (ROBUST)")
            print("   Execution risk is manageable with limit orders")
        elif max_slip_all >= 3:
            print(f"\n⚠️  VERDICT: NL Top 25 tolerates {max_slip_all:.0f} bps (MARGINAL)")
            print("   Requires tight execution control")
        else:
            print(f"\n❌ VERDICT: NL Top 25 tolerates <3 bps (FRAGILE)")
            print("   Execution risk remains high")
    else:
        print("\n❌ VERDICT: No slippage level meets all criteria")

    # Comparison with baseline
    baseline = results_df[results_df["slippage_bps"] == 0].iloc[0]
    print(f"\nBaseline (0 bps): Sharpe {baseline['sharpe']:.2f}, Avg {baseline['avg_ret']:.3%}, DD {baseline['max_dd']:.1%}")
    print("Previous full NL universe: ~3 bps tolerance")
    print(f"Top 25 Liquidity: {max_slip_all:.0f} bps tolerance ({max_slip_all/3:.1f}x improvement)" if len(all_pass) > 0 else "Top 25 Liquidity: No improvement")
    
    print("="*90 + "\n")


if __name__ == "__main__":
    main()
