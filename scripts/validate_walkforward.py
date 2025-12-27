#!/usr/bin/env python3
"""
Walk-forward validation for consolidation breakout model.

Tests temporal stability by rolling train/test windows through time.
This is THE critical test - reveals if edge is regime-dependent.

Usage:
    python scripts/validate_walkforward.py \\
        --bundle ML/production/v20251223_212702 \\
        --events data/ml_datasets/accum_distrib_events.parquet \\
        --market US \\
        --train_months 12 \\
        --test_months 3 \\
        --threshold 0.65 \\
        --cost_bps 10 \\
        --output data/backtests/validation/walkforward_us.csv
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
from datetime import datetime, timedelta

from src.config import settings

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)


def load_events(events_path: str, market: str = None) -> pd.DataFrame:
    """Load and prepare events dataset."""
    ev = pd.read_parquet(events_path)
    ev["t_end"] = pd.to_datetime(ev["t_end"])
    ev["t_start"] = pd.to_datetime(ev["t_start"])
    
    # Filter by label validity and market
    if "label_valid" in ev.columns:
        ev = ev[ev["label_valid"] == True].copy()
    
    if market:
        ev = ev[ev["market"] == market].copy()
    
    return ev


def get_prices(engine, tickers, start_ts, end_ts, market=None):
    """Fetch OHLCV data for backtest period."""
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
    
    df = pd.read_sql(text(q), engine, params=params)
    return df


def backtest_window(
    events: pd.DataFrame,
    prices: pd.DataFrame,
    model,
    feature_names: list,
    threshold: float,
    cost_bps: float,
    H: int = 40,
    ATR_K: float = 1.5
) -> dict:
    """
    Run backtest on a single time window.
    Returns summary metrics dict.
    """
    # Index prices by ticker
    prices["timestamp"] = pd.to_datetime(prices["timestamp"])
    prices = prices.sort_values(["ticker", "timestamp"])
    by_ticker = {t: g.set_index("timestamp") for t, g in prices.groupby("ticker")}
    
    trades = []
    cost_fraction = cost_bps / 10000.0
    
    for _, ev in events.iterrows():
        tkr = ev["ticker"]
        if tkr not in by_ticker:
            continue
        
        t_end = pd.to_datetime(ev["t_end"])
        entry_ts = t_end + pd.Timedelta(hours=1)
        
        g = by_ticker[tkr]
        if entry_ts not in g.index:
            continue
        
        entry_open = float(g.loc[entry_ts, "open"])
        if not np.isfinite(entry_open) or entry_open <= 0:
            continue
        
        # Build features
        try:
            X = ev[feature_names].to_frame().T
            X = X.astype(float).fillna(0.0)
        except KeyError:
            continue
        
        # Predict
        p_up = float(model.predict_proba(X)[:, 1][0])
        
        # Threshold logic
        if p_up >= threshold:
            side = "LONG"
        elif p_up <= (1 - threshold):
            side = "SHORT"
        else:
            continue
        
        # Get path
        path = g.loc[entry_ts:].iloc[:H+1]
        if len(path) < 2:
            continue
        
        # ATR
        atr_pct = float(ev.get("atr_pct_last", np.nan))
        if not np.isfinite(atr_pct) or atr_pct <= 0:
            continue
        atr_end = atr_pct * entry_open
        
        # Stops/targets
        if side == "LONG":
            stop = entry_open - ATR_K * atr_end
            target = entry_open + ATR_K * atr_end
        else:
            stop = entry_open + ATR_K * atr_end
            target = entry_open - ATR_K * atr_end
        
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
            gross_ret = (exit_px - entry_open) / entry_open
        else:
            gross_ret = (entry_open - exit_px) / entry_open
        
        net_ret = gross_ret - 2 * cost_fraction
        
        trades.append({
            "entry_ts": entry_ts,
            "exit_ts": exit_ts,
            "side": side,
            "net_ret": net_ret,
        })
    
    # Compute metrics
    if len(trades) == 0:
        return {
            "n_trades": 0,
            "win_rate": 0,
            "avg_ret": 0,
            "median_ret": 0,
            "std_ret": 0,
            "total_ret": 0,
            "max_dd": 0,
            "sharpe": 0,
        }
    
    trades_df = pd.DataFrame(trades)
    win_rate = (trades_df["net_ret"] > 0).mean()
    avg_ret = trades_df["net_ret"].mean()
    med_ret = trades_df["net_ret"].median()
    std_ret = trades_df["net_ret"].std()
    
    # Equity curve
    eq = (1 + trades_df.sort_values("entry_ts")["net_ret"]).cumprod()
    peak = eq.cummax()
    dd = (eq / peak) - 1
    max_dd = float(dd.min())
    total_ret = float(eq.iloc[-1] - 1)
    
    # Sharpe (annualized)
    if std_ret > 0 and len(trades) > 1:
        sharpe = (avg_ret / std_ret) * np.sqrt(252 * 6.5)  # hourly → daily → annual
    else:
        sharpe = 0
    
    return {
        "n_trades": len(trades),
        "win_rate": float(win_rate),
        "avg_ret": float(avg_ret),
        "median_ret": float(med_ret),
        "std_ret": float(std_ret),
        "total_ret": float(total_ret),
        "max_dd": float(max_dd),
        "sharpe": float(sharpe),
    }


def main():
    ap = argparse.ArgumentParser(description="Walk-forward validation")
    ap.add_argument("--bundle", required=True, help="Production bundle directory")
    ap.add_argument("--events", required=True, help="Events parquet file")
    ap.add_argument("--market", default=None, help="Market filter (US or NL)")
    ap.add_argument("--train_months", type=int, default=12, help="Training window months")
    ap.add_argument("--test_months", type=int, default=3, help="Test window months")
    ap.add_argument("--threshold", type=float, default=0.65, help="Decision threshold")
    ap.add_argument("--cost_bps", type=float, default=10.0, help="Transaction costs (bps)")
    ap.add_argument("--output", required=True, help="Output CSV path")
    args = ap.parse_args()
    
    logger.info(f"Walk-forward validation: market={args.market}, train={args.train_months}m, test={args.test_months}m")
    
    # Load bundle
    bundle_dir = Path(args.bundle)
    features_data = json.loads((bundle_dir / "features.json").read_text())
    feature_names = features_data["feature_names"]
    
    # Load model (use calibrated if available)
    model_file = bundle_dir / "model_calibrated.pkl"
    if not model_file.exists():
        model_file = bundle_dir / "model.pkl"
    
    model_bundle = joblib.load(model_file)
    if isinstance(model_bundle, dict) and "model" in model_bundle:
        model = model_bundle["model"]
    else:
        model = model_bundle
    
    logger.info(f"Loaded model with {len(feature_names)} features")
    
    # Load events
    events = load_events(args.events, args.market)
    logger.info(f"Loaded {len(events):,} events")
    
    # Create walk-forward windows
    min_date = events["t_end"].min()
    max_date = events["t_end"].max()
    
    logger.info(f"Date range: {min_date.date()} to {max_date.date()}")
    
    # Generate windows
    windows = []
    current_start = min_date
    
    while True:
        train_end = current_start + pd.DateOffset(months=args.train_months)
        test_start = train_end
        test_end = test_start + pd.DateOffset(months=args.test_months)
        
        if test_end > max_date:
            break
        
        windows.append({
            "train_start": current_start,
            "train_end": train_end,
            "test_start": test_start,
            "test_end": test_end,
        })
        
        # Roll forward by test_months
        current_start = test_start
    
    logger.info(f"Generated {len(windows)} walk-forward windows")
    
    # Run backtest on each window
    engine = create_engine(settings.database_url)
    results = []
    
    for i, win in enumerate(windows):
        logger.info(f"\nWindow {i+1}/{len(windows)}: Test {win['test_start'].date()} to {win['test_end'].date()}")
        
        # Filter events for test period
        test_events = events[
            (events["t_end"] >= win["test_start"]) &
            (events["t_end"] < win["test_end"])
        ].copy()
        
        if len(test_events) == 0:
            logger.warning(f"  No events in test window, skipping")
            continue
        
        logger.info(f"  Test events: {len(test_events):,}")
        
        # Get prices
        tickers = test_events["ticker"].unique()
        price_start = win["test_start"] - pd.Timedelta(days=2)
        price_end = win["test_end"] + pd.Timedelta(hours=50)
        
        prices = get_prices(engine, tickers, price_start, price_end, args.market)
        logger.info(f"  Loaded {len(prices):,} price bars")
        
        # Run backtest
        metrics = backtest_window(
            test_events, prices, model, feature_names,
            args.threshold, args.cost_bps
        )
        
        # Store results
        result = {
            "window": i + 1,
            "test_start": win["test_start"],
            "test_end": win["test_end"],
            **metrics
        }
        results.append(result)
        
        logger.info(f"  Results: {metrics['n_trades']} trades, "
                   f"{metrics['win_rate']:.1%} WR, "
                   f"{metrics['avg_ret']:.3%} avg, "
                   f"Sharpe {metrics['sharpe']:.2f}")
    
    # Save results
    results_df = pd.DataFrame(results)
    out = Path(args.output)
    out.parent.mkdir(parents=True, exist_ok=True)
    results_df.to_csv(out, index=False)
    logger.info(f"\nSaved results: {out}")
    
    # Summary statistics
    print("\n" + "="*70)
    print("WALK-FORWARD SUMMARY")
    print("="*70)
    
    print(f"\nTotal windows: {len(results)}")
    print(f"Total trades: {results_df['n_trades'].sum():,}")
    print(f"Avg trades per window: {results_df['n_trades'].mean():.1f}")
    
    print(f"\nWin Rate:")
    print(f"  Mean: {results_df['win_rate'].mean():.1%}")
    print(f"  Median: {results_df['win_rate'].median():.1%}")
    print(f"  Std: {results_df['win_rate'].std():.1%}")
    print(f"  Min: {results_df['win_rate'].min():.1%}")
    print(f"  Max: {results_df['win_rate'].max():.1%}")
    
    print(f"\nAvg Return per Trade:")
    print(f"  Mean: {results_df['avg_ret'].mean():.3%}")
    print(f"  Median: {results_df['avg_ret'].median():.3%}")
    print(f"  Std: {results_df['avg_ret'].std():.3%}")
    print(f"  Min: {results_df['avg_ret'].min():.3%}")
    print(f"  Max: {results_df['avg_ret'].max():.3%}")
    
    print(f"\nSharpe Ratio:")
    print(f"  Mean: {results_df['sharpe'].mean():.2f}")
    print(f"  Median: {results_df['sharpe'].median():.2f}")
    print(f"  Min: {results_df['sharpe'].min():.2f}")
    print(f"  Max: {results_df['sharpe'].max():.2f}")
    
    print(f"\nMax Drawdown:")
    print(f"  Mean: {results_df['max_dd'].mean():.1%}")
    print(f"  Worst: {results_df['max_dd'].min():.1%}")
    
    # Windows with positive returns
    positive_windows = (results_df['total_ret'] > 0).sum()
    pct_positive = positive_windows / len(results) * 100
    
    print(f"\nPositive Windows: {positive_windows}/{len(results)} ({pct_positive:.1f}%)")
    
    # Pass/Fail criteria
    print("\n" + "="*70)
    print("GO/NO-GO ASSESSMENT")
    print("="*70)
    
    pass_checks = []
    
    # Check 1: 70%+ windows positive
    check1 = pct_positive >= 70
    pass_checks.append(check1)
    print(f"✓ 70%+ windows positive: {pct_positive:.1f}% {'PASS' if check1 else 'FAIL'}")
    
    # Check 2: Mean win rate > 60%
    check2 = results_df['win_rate'].mean() > 0.60
    pass_checks.append(check2)
    print(f"✓ Mean win rate > 60%: {results_df['win_rate'].mean():.1%} {'PASS' if check2 else 'FAIL'}")
    
    # Check 3: Mean Sharpe > 1.0
    check3 = results_df['sharpe'].mean() > 1.0
    pass_checks.append(check3)
    print(f"✓ Mean Sharpe > 1.0: {results_df['sharpe'].mean():.2f} {'PASS' if check3 else 'FAIL'}")
    
    # Check 4: No catastrophic drawdown (worst DD < -30%)
    check4 = results_df['max_dd'].min() > -0.30
    pass_checks.append(check4)
    print(f"✓ Worst DD < -30%: {results_df['max_dd'].min():.1%} {'PASS' if check4 else 'FAIL'}")
    
    # Check 5: Avg return positive
    check5 = results_df['avg_ret'].mean() > 0
    pass_checks.append(check5)
    print(f"✓ Mean avg return > 0: {results_df['avg_ret'].mean():.3%} {'PASS' if check5 else 'FAIL'}")
    
    print("-"*70)
    if all(pass_checks):
        print("🎯 VERDICT: GO - Edge is temporally stable")
    else:
        print("⚠️  VERDICT: NO-GO - Edge is not stable over time")
    print("="*70 + "\n")


if __name__ == "__main__":
    main()
