#!/usr/bin/env python3
"""
Entry price realism validation for consolidation breakout model.

Tests impact of different entry price assumptions:
1. Baseline: t_end + 1 bar open (current assumption)
2. VWAP: Simulated as (open + close) / 2
3. Close: Worst case - entered at next bar close
4. Slippage: Open + 5 bps adverse slippage

This validates that execution is realistic and achievable.

Usage:
    python scripts/validate_entry_realism.py \\
        --bundle ML/production/v20251223_212702 \\
        --events data/ml_datasets/accum_distrib_events.parquet \\
        --market US \\
        --threshold 0.75 \\
        --cost_bps 10 \\
        --output data/backtests/validation/entry_realism_us_t075.csv
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


def load_events(events_path: str, market: str = None) -> pd.DataFrame:
    """Load and prepare events dataset."""
    ev = pd.read_parquet(events_path)
    ev["t_end"] = pd.to_datetime(ev["t_end"])
    ev["t_start"] = pd.to_datetime(ev["t_start"])
    
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


def backtest_entry_scenario(
    events: pd.DataFrame,
    prices: pd.DataFrame,
    model,
    feature_names: list,
    threshold: float,
    cost_bps: float,
    entry_type: str,
    H: int = 40,
    ATR_K: float = 1.5
) -> dict:
    """
    Run backtest with specific entry price assumption.
    
    Args:
        entry_type: 'open', 'vwap', 'close', 'slippage'
    """
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
        
        entry_bar = g.loc[entry_ts]
        
        # Calculate entry price based on scenario
        if entry_type == "open":
            entry_px = float(entry_bar["open"])
        elif entry_type == "vwap":
            # Simulate VWAP as midpoint between open and close
            entry_px = (float(entry_bar["open"]) + float(entry_bar["close"])) / 2
        elif entry_type == "close":
            # Worst case - entered at close
            entry_px = float(entry_bar["close"])
        elif entry_type == "slippage":
            # Open + 5 bps adverse slippage
            open_px = float(entry_bar["open"])
            entry_px = open_px * 1.0005  # Always worse (higher for long, higher for short)
        else:
            raise ValueError(f"Unknown entry_type: {entry_type}")
        
        if not np.isfinite(entry_px) or entry_px <= 0:
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
        atr_end = atr_pct * entry_px
        
        # Stops/targets (based on actual entry price)
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
            "entry_ts": entry_ts,
            "exit_ts": exit_ts,
            "side": side,
            "entry_px": entry_px,
            "net_ret": net_ret,
        })
    
    # Compute metrics
    if len(trades) == 0:
        return {
            "entry_type": entry_type,
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
        sharpe = (avg_ret / std_ret) * np.sqrt(252 * 6.5)
    else:
        sharpe = 0
    
    return {
        "entry_type": entry_type,
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
    ap = argparse.ArgumentParser(description="Entry price realism validation")
    ap.add_argument("--bundle", required=True, help="Production bundle directory")
    ap.add_argument("--events", required=True, help="Events parquet file")
    ap.add_argument("--market", default=None, help="Market filter (US or NL)")
    ap.add_argument("--threshold", type=float, default=0.65, help="Decision threshold")
    ap.add_argument("--cost_bps", type=float, default=10.0, help="Transaction costs (bps)")
    ap.add_argument("--output", required=True, help="Output CSV path")
    args = ap.parse_args()
    
    logger.info(f"Entry realism validation: market={args.market}, threshold={args.threshold}")
    
    # Load bundle
    bundle_dir = Path(args.bundle)
    features_data = json.loads((bundle_dir / "features.json").read_text())
    feature_names = features_data["feature_names"]
    
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
    
    # Get prices
    engine = create_engine(settings.database_url)
    tickers = events["ticker"].unique()
    price_start = events["t_end"].min() - pd.Timedelta(days=2)
    price_end = events["t_end"].max() + pd.Timedelta(hours=50)
    
    logger.info(f"Fetching prices for {len(tickers)} tickers...")
    prices = get_prices(engine, tickers, price_start, price_end, args.market)
    logger.info(f"Loaded {len(prices):,} price bars")
    
    # Run all entry scenarios
    entry_types = ["open", "vwap", "close", "slippage"]
    results = []
    
    for entry_type in entry_types:
        logger.info(f"\n{'='*70}")
        logger.info(f"Testing entry type: {entry_type.upper()}")
        logger.info(f"{'='*70}")
        
        metrics = backtest_entry_scenario(
            events, prices, model, feature_names,
            args.threshold, args.cost_bps, entry_type
        )
        
        results.append(metrics)
        
        logger.info(f"Results: {metrics['n_trades']} trades, "
                   f"{metrics['win_rate']:.1%} WR, "
                   f"{metrics['avg_ret']:.3%} avg, "
                   f"Sharpe {metrics['sharpe']:.2f}, "
                   f"DD {metrics['max_dd']:.1%}")
    
    # Save results
    results_df = pd.DataFrame(results)
    out = Path(args.output)
    out.parent.mkdir(parents=True, exist_ok=True)
    results_df.to_csv(out, index=False)
    logger.info(f"\nSaved results: {out}")
    
    # Summary comparison
    print("\n" + "="*70)
    print("ENTRY PRICE REALISM COMPARISON")
    print("="*70)
    
    baseline = results_df[results_df["entry_type"] == "open"].iloc[0]
    
    print(f"\n{'Entry Type':<12} {'Trades':<8} {'Win Rate':<10} {'Avg Ret':<10} {'Sharpe':<8} {'Max DD':<10}")
    print("-"*70)
    
    for _, row in results_df.iterrows():
        etype = row["entry_type"].upper()
        
        # Calculate deltas vs baseline
        wr_delta = row["win_rate"] - baseline["win_rate"]
        ret_delta = row["avg_ret"] - baseline["avg_ret"]
        sharpe_delta = row["sharpe"] - baseline["sharpe"]
        
        # Format with deltas
        wr_str = f"{row['win_rate']:.1%}"
        if etype != "OPEN":
            wr_str += f" ({wr_delta:+.1%})"
        
        ret_str = f"{row['avg_ret']:.3%}"
        if etype != "OPEN":
            ret_str += f" ({ret_delta:+.3%})"
        
        sharpe_str = f"{row['sharpe']:.2f}"
        if etype != "OPEN":
            sharpe_str += f" ({sharpe_delta:+.2f})"
        
        print(f"{etype:<12} {row['n_trades']:<8} {wr_str:<18} {ret_str:<18} "
              f"{sharpe_str:<14} {row['max_dd']:.1%}")
    
    print("="*70)
    
    # Pass/Fail assessment
    print("\nGO/NO-GO ASSESSMENT")
    print("-"*70)
    
    baseline_sharpe = baseline["sharpe"]
    baseline_ret = baseline["avg_ret"]
    
    checks = []
    
    for _, row in results_df.iterrows():
        if row["entry_type"] == "open":
            continue
        
        etype = row["entry_type"].upper()
        
        # Check 1: Sharpe still > 1.0
        check1 = row["sharpe"] > 1.0
        
        # Check 2: Avg return still positive
        check2 = row["avg_ret"] > 0
        
        # Check 3: Sharpe degradation < 50%
        sharpe_drop = (baseline_sharpe - row["sharpe"]) / baseline_sharpe
        check3 = sharpe_drop < 0.50
        
        # Check 4: DD acceptable (< -15%)
        check4 = row["max_dd"] > -0.15
        
        all_pass = all([check1, check2, check3, check4])
        checks.append(all_pass)
        
        status = "✅ PASS" if all_pass else "❌ FAIL"
        print(f"\n{etype} scenario: {status}")
        print(f"  • Sharpe > 1.0: {row['sharpe']:.2f} {'✓' if check1 else '✗'}")
        print(f"  • Avg return > 0: {row['avg_ret']:.3%} {'✓' if check2 else '✗'}")
        print(f"  • Sharpe drop < 50%: {sharpe_drop:.1%} {'✓' if check3 else '✗'}")
        print(f"  • Max DD < -15%: {row['max_dd']:.1%} {'✓' if check4 else '✗'}")
    
    print("-"*70)
    
    if all(checks):
        print("🎯 VERDICT: GO - Edge survives realistic entry prices")
    else:
        print("⚠️  VERDICT: NO-GO - Edge vulnerable to execution quality")
    
    print("="*70 + "\n")


if __name__ == "__main__":
    main()
