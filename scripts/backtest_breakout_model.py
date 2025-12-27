#!/usr/bin/env python3
"""
Event-based backtest for consolidation breakout model.

Features:
- No look-ahead bias: enters at t_end + 1 bar open
- Hybrid exit: barrier (stop/target) OR time-based (H bars)
- Round-trip transaction costs (slippage + commission as bps)
- Symmetrical threshold logic for LONG/SHORT/NO-TRADE
- Validation split support for fair evaluation

Exit modes:
- fixed: Exit at t_end + 1 + H bars close
- barrier: Exit when price hits stop OR target (±ATR_K * atr14_end)
- hybrid: First barrier hit, else time stop at H bars

Usage:
    python scripts/backtest_breakout_model.py \\
        --bundle ML/production/v20251223_212702 \\
        --events data/ml_datasets/accum_distrib_events.parquet \\
        --market US \\
        --threshold 0.50 \\
        --exit hybrid \\
        --cost_bps 5 \\
        --output data/backtests/bt_us_t050.csv
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
from typing import Dict, Tuple, List

from src.config import settings

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)


def load_bundle(bundle_dir: str) -> Tuple[List[str], Dict]:
    """Load production bundle with feature schema and calibrated model."""
    bundle_dir = Path(bundle_dir)
    
    if not bundle_dir.exists():
        raise FileNotFoundError(f"Bundle directory not found: {bundle_dir}")
    
    # Load feature schema (ORDERED)
    features_file = bundle_dir / "features.json"
    if not features_file.exists():
        raise FileNotFoundError(f"features.json not found in bundle: {bundle_dir}")
    
    features_data = json.loads(features_file.read_text())
    # Handle both list and dict formats
    if isinstance(features_data, list):
        feature_names = [f["name"] for f in features_data]
    elif isinstance(features_data, dict) and "feature_names" in features_data:
        feature_names = features_data["feature_names"]
    else:
        raise ValueError(f"Unexpected features.json format: {type(features_data)}")
    
    # Load calibrated model (preferred) or fall back to uncalibrated
    calib_file = bundle_dir / "model_calibrated.pkl"
    model_file = bundle_dir / "model.pkl"
    
    if calib_file.exists():
        logger.info(f"Loading calibrated model from {calib_file}")
        model_bundle = joblib.load(calib_file)
        calibrated = True
    elif model_file.exists():
        logger.info(f"Loading uncalibrated model from {model_file}")
        model_bundle = joblib.load(model_file)
        calibrated = False
    else:
        raise FileNotFoundError(f"No model file found in bundle: {bundle_dir}")
    
    logger.info(f"Loaded bundle: {len(feature_names)} features, calibrated={calibrated}")
    return feature_names, model_bundle


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
    
    logger.info(f"Fetching prices for {len(tickers)} tickers from {start_ts} to {end_ts}")
    df = pd.read_sql(text(q), engine, params=params)
    logger.info(f"Loaded {len(df):,} price bars")
    
    return df


def apply_calibrator(model_bundle, p_uncal):
    """Apply probability calibration if available."""
    # Support both dict bundles {'model': ..., 'calibrator': ...}
    # and pipeline objects
    if isinstance(model_bundle, dict):
        if "calibrator" in model_bundle:
            # Reshape for sklearn calibrator (expects 2D)
            p_uncal_2d = p_uncal.reshape(-1, 1) if p_uncal.ndim == 1 else p_uncal
            return model_bundle["calibrator"].predict(p_uncal_2d)
        elif "model" in model_bundle:
            # Uncalibrated dict bundle
            return p_uncal
    
    # Fallback: assume uncalibrated pipeline
    return p_uncal


def backtest(
    events: pd.DataFrame,
    prices: pd.DataFrame,
    feature_names: List[str],
    model_bundle: Dict,
    threshold: float,
    H: int,
    ATR_K: float,
    cost_bps: float,
    exit_mode: str
) -> pd.DataFrame:
    """
    Run event-based backtest with realistic entry/exit.
    
    Args:
        events: Event dataset with features
        prices: OHLCV price data
        feature_names: ORDERED list of features (from features.json)
        model_bundle: Trained model + calibrator
        threshold: Decision threshold (0.50-0.90)
        H: Time horizon for exits (bars)
        ATR_K: ATR multiplier for stops/targets
        cost_bps: Round-trip transaction costs (basis points)
        exit_mode: 'fixed', 'barrier', or 'hybrid'
    
    Returns:
        DataFrame with trade log (one row per trade)
    """
    # Index prices by ticker for fast slicing
    prices["timestamp"] = pd.to_datetime(prices["timestamp"])
    prices = prices.sort_values(["ticker", "timestamp"])
    by_ticker = {t: g.set_index("timestamp") for t, g in prices.groupby("ticker")}
    
    trades = []
    cost_fraction = cost_bps / 10000.0  # bps -> fraction
    
    # Extract model from bundle
    if isinstance(model_bundle, dict) and "model" in model_bundle:
        model = model_bundle["model"]
    else:
        model = model_bundle
    
    logger.info(f"Running backtest: threshold={threshold}, exit={exit_mode}, H={H}, ATR_K={ATR_K}, cost={cost_bps}bps")
    
    # Debug counters
    skip_reasons = {
        "no_ticker_data": 0,
        "missing_entry_bar": 0,
        "invalid_entry_price": 0,
        "missing_feature": 0,
        "no_trade_region": 0,
        "insufficient_path": 0,
        "invalid_atr": 0,
    }
    
    for idx, ev in events.iterrows():
        if idx > 0 and idx % 1000 == 0:
            logger.info(f"Processing event {idx:,}/{len(events):,} ({100*idx/len(events):.1f}%)")
        
        tkr = ev["ticker"]
        if tkr not in by_ticker:
            skip_reasons["no_ticker_data"] += 1
            continue
        
        # Entry at next bar OPEN after consolidation ends (no look-ahead)
        t_end = pd.to_datetime(ev["t_end"])
        entry_ts = t_end + pd.Timedelta(hours=1)
        
        g = by_ticker[tkr]
        if entry_ts not in g.index:
            # Missing bar - skip this event
            skip_reasons["missing_entry_bar"] += 1
            continue
        
        entry_open = float(g.loc[entry_ts, "open"])
        if not np.isfinite(entry_open) or entry_open <= 0:
            skip_reasons["invalid_entry_price"] += 1
            continue
        
        # Build feature vector (MUST match order in features.json)
        try:
            X = ev[feature_names].to_frame().T
            X = X.astype(float).fillna(0.0)
        except KeyError as e:
            logger.warning(f"Missing feature in event: {e}")
            skip_reasons["missing_feature"] += 1
            continue
        
        # Predict probability (with calibration if available)
        p_uncal = model.predict_proba(X)[:, 1]
        p_up = float(apply_calibrator(model_bundle, p_uncal)[0])
        
        # Symmetrical threshold logic:
        # - Long if P(UP) >= threshold
        # - Short if P(UP) <= (1 - threshold)
        # - Else no-trade
        if p_up >= threshold:
            side = "LONG"
        elif p_up <= (1 - threshold):
            side = "SHORT"
        else:
            skip_reasons["no_trade_region"] += 1
            continue  # No-trade region
        
        # Get price path for exit logic (up to H bars after entry)
        path = g.loc[entry_ts:].iloc[:H+1]  # Include entry bar
        if len(path) < 2:
            skip_reasons["insufficient_path"] += 1
            continue  # Need at least 1 bar after entry
        
        # Get ATR for stops/targets (convert from percentage to price units)
        atr_pct = float(ev.get("atr_pct_last", np.nan))
        if not np.isfinite(atr_pct) or atr_pct <= 0:
            skip_reasons["invalid_atr"] += 1
            continue
        
        # Convert ATR from percentage to price units
        atr_end = atr_pct * entry_open
        
        # Calculate stop and target levels
        if side == "LONG":
            stop = entry_open - ATR_K * atr_end
            target = entry_open + ATR_K * atr_end
        else:  # SHORT
            stop = entry_open + ATR_K * atr_end
            target = entry_open - ATR_K * atr_end
        
        # Exit logic
        exit_ts = None
        exit_px = None
        exit_reason = None
        
        if exit_mode in ("barrier", "hybrid"):
            # Check each bar for barrier hits
            for ts, row in path.iterrows():
                if ts == entry_ts:
                    continue  # Skip entry bar (enter at open, can't exit same bar)
                
                hi = float(row["high"])
                lo = float(row["low"])
                
                if side == "LONG":
                    hit_target = hi >= target
                    hit_stop = lo <= stop
                else:  # SHORT
                    hit_target = lo <= target
                    hit_stop = hi >= stop
                
                # Conservative: if both hit same bar, assume worst case (stop)
                if hit_target and hit_stop:
                    exit_ts, exit_px, exit_reason = ts, stop, "BOTH_WORST"
                    break
                elif hit_target:
                    exit_ts, exit_px, exit_reason = ts, target, "TARGET"
                    break
                elif hit_stop:
                    exit_ts, exit_px, exit_reason = ts, stop, "STOP"
                    break
        
        # If no barrier hit (or fixed exit mode), use time exit
        if exit_ts is None:
            exit_ts = path.index[-1]
            exit_px = float(path.iloc[-1]["close"])
            exit_reason = "TIME"
        
        # Compute returns (gross and net of costs)
        if side == "LONG":
            gross_ret = (exit_px - entry_open) / entry_open
        else:  # SHORT
            gross_ret = (entry_open - exit_px) / entry_open
        
        # Round-trip costs (enter + exit)
        net_ret = gross_ret - 2 * cost_fraction
        
        trades.append({
            "ticker": tkr,
            "market": ev.get("market", None),
            "t_start": pd.to_datetime(ev["t_start"]),
            "t_end": t_end,
            "entry_ts": entry_ts,
            "exit_ts": exit_ts,
            "side": side,
            "p_up": p_up,
            "threshold": threshold,
            "entry": entry_open,
            "exit": exit_px,
            "stop": stop,
            "target": target,
            "exit_reason": exit_reason,
            "gross_ret": gross_ret,
            "net_ret": net_ret,
            "label": ev.get("label_generic", None),
        })
    
    logger.info(f"Generated {len(trades):,} trades from {len(events):,} events")
    logger.info(f"Skip reasons: {skip_reasons}")
    return pd.DataFrame(trades)


def summarize(trades: pd.DataFrame) -> Dict:
    """Compute backtest summary statistics."""
    if len(trades) == 0:
        return {
            "n_trades": 0,
            "error": "No trades generated"
        }
    
    # Overall metrics
    win_rate = (trades["net_ret"] > 0).mean()
    avg_ret = trades["net_ret"].mean()
    med_ret = trades["net_ret"].median()
    std_ret = trades["net_ret"].std()
    
    # Equity curve and drawdown
    trades_sorted = trades.sort_values("entry_ts")
    eq = (1 + trades_sorted["net_ret"]).cumprod()
    peak = eq.cummax()
    dd = (eq / peak) - 1
    max_dd = float(dd.min())
    
    # Sharpe ratio (assuming ~250 trading days, ~6 bars per day = 1500 bars/year)
    # Annualization factor: sqrt(trades per year)
    if len(trades) > 1:
        date_range = (trades["entry_ts"].max() - trades["entry_ts"].min()).days
        if date_range > 0:
            trades_per_year = len(trades) * 365.25 / date_range
            sharpe = (avg_ret / std_ret) * np.sqrt(trades_per_year) if std_ret > 0 else 0
        else:
            sharpe = 0
    else:
        sharpe = 0
    
    # Breakdown by side
    long_trades = trades[trades["side"] == "LONG"]
    short_trades = trades[trades["side"] == "SHORT"]
    
    # Breakdown by exit reason
    exit_counts = trades["exit_reason"].value_counts().to_dict()
    
    # Time-based metrics
    first_trade = trades["entry_ts"].min()
    last_trade = trades["entry_ts"].max()
    days = (last_trade - first_trade).days
    trades_per_month = len(trades) / (days / 30.0) if days > 0 else 0
    
    summary = {
        "n_trades": int(len(trades)),
        "win_rate": float(win_rate),
        "avg_net_ret": float(avg_ret),
        "median_net_ret": float(med_ret),
        "std_ret": float(std_ret),
        "max_drawdown": float(max_dd),
        "sharpe_ratio": float(sharpe),
        "total_return": float(eq.iloc[-1] - 1),
        "trades_per_month": float(trades_per_month),
        "first_trade": str(first_trade),
        "last_trade": str(last_trade),
        "days": int(days),
        "long": {
            "n": int(len(long_trades)),
            "win_rate": float((long_trades["net_ret"] > 0).mean()) if len(long_trades) > 0 else 0,
            "avg_ret": float(long_trades["net_ret"].mean()) if len(long_trades) > 0 else 0,
        },
        "short": {
            "n": int(len(short_trades)),
            "win_rate": float((short_trades["net_ret"] > 0).mean()) if len(short_trades) > 0 else 0,
            "avg_ret": float(short_trades["net_ret"].mean()) if len(short_trades) > 0 else 0,
        },
        "exit_reasons": exit_counts,
    }
    
    return summary


def main():
    ap = argparse.ArgumentParser(description="Event-based backtest for breakout model")
    ap.add_argument("--bundle", required=True, help="Path to production bundle directory")
    ap.add_argument("--events", required=True, help="Path to events parquet file")
    ap.add_argument("--market", default=None, help="Filter by market (US or NL)")
    ap.add_argument("--threshold", type=float, default=0.65, help="Decision threshold (0.50-0.90)")
    ap.add_argument("--exit", choices=["fixed", "barrier", "hybrid"], default="hybrid",
                    help="Exit mode: fixed (time), barrier (stop/target), or hybrid")
    ap.add_argument("--H", type=int, default=40, help="Time horizon for exits (bars)")
    ap.add_argument("--ATR_K", type=float, default=1.5, help="ATR multiplier for stops/targets")
    ap.add_argument("--cost_bps", type=float, default=5.0, help="Round-trip transaction costs (basis points)")
    ap.add_argument("--output", required=True, help="Output path for trade log CSV")
    ap.add_argument("--split", default=None, choices=["train", "val", "test"],
                    help="Filter events by data split (train/val/test)")
    args = ap.parse_args()
    
    # Load production bundle
    logger.info(f"Loading bundle: {args.bundle}")
    feature_names, model_bundle = load_bundle(args.bundle)
    
    # Load events
    logger.info(f"Loading events: {args.events}")
    ev = pd.read_parquet(args.events)
    ev["t_end"] = pd.to_datetime(ev["t_end"])
    ev["t_start"] = pd.to_datetime(ev["t_start"])
    
    logger.info(f"Loaded {len(ev):,} total events")
    
    # Filter by label validity (use only valid events)
    if "label_valid" in ev.columns:
        ev = ev[ev["label_valid"] == True].copy()
        logger.info(f"Filtered to {len(ev):,} label-valid events")
    
    # Filter by market
    if args.market:
        ev = ev[ev["market"] == args.market].copy()
        logger.info(f"Filtered to {len(ev):,} events in {args.market} market")
    
    # Add split column if missing (for backward compatibility)
    if "split" not in ev.columns:
        ev["year_month"] = pd.to_datetime(ev["t_end"]).dt.to_period("M")
        ev["split"] = "train"
        ev.loc[ev["year_month"] >= "2025-01", "split"] = "val"
        ev.loc[ev["year_month"] >= "2025-09", "split"] = "test"
        logger.info(f"Added split column: {ev['split'].value_counts().to_dict()}")
    
    # Filter by data split
    if args.split:
        ev = ev[ev["split"] == args.split].copy()
        logger.info(f"Filtered to {len(ev):,} events in {args.split} split")
    
    if len(ev) == 0:
        logger.error("No events after filtering - exiting")
        return
    
    # Determine time span for price data (need H bars after last event)
    start_ts = ev["t_end"].min() - pd.Timedelta(days=1)  # Buffer before first event
    end_ts = ev["t_end"].max() + pd.Timedelta(hours=args.H + 10)  # Buffer after last event
    
    # Fetch price data
    engine = create_engine(settings.database_url)
    tickers = ev["ticker"].unique()
    prices = get_prices(engine, tickers, start_ts, end_ts, market=args.market)
    
    if len(prices) == 0:
        logger.error("No price data found - exiting")
        return
    
    # Run backtest
    trades = backtest(
        ev, prices, feature_names, model_bundle,
        args.threshold, args.H, args.ATR_K, args.cost_bps, args.exit
    )
    
    # Save trade log
    out = Path(args.output)
    out.parent.mkdir(parents=True, exist_ok=True)
    trades.to_csv(out, index=False)
    logger.info(f"Saved trades: {out} ({len(trades):,} rows)")
    
    # Compute and save summary
    summ = summarize(trades)
    summ_file = out.with_suffix(".summary.json")
    summ_file.write_text(json.dumps(summ, indent=2))
    logger.info(f"Saved summary: {summ_file}")
    
    # Print summary
    print("\n" + "="*60)
    print("BACKTEST SUMMARY")
    print("="*60)
    print(json.dumps(summ, indent=2))
    print("="*60)
    
    # Go/No-Go assessment
    if "win_rate" in summ:
        print("\nGO/NO-GO ASSESSMENT:")
        print("-" * 60)
        
        go_criteria = []
        
        # Win rate > 60%
        wr_pass = summ["win_rate"] > 0.60
        go_criteria.append(wr_pass)
        print(f"✓ Win rate > 60%: {summ['win_rate']:.1%} {'PASS' if wr_pass else 'FAIL'}")
        
        # Max DD < 15%
        dd_pass = summ["max_drawdown"] > -0.15
        go_criteria.append(dd_pass)
        print(f"✓ Max DD < 15%: {summ['max_drawdown']:.1%} {'PASS' if dd_pass else 'FAIL'}")
        
        # Sharpe > 1.0
        sharpe_pass = summ["sharpe_ratio"] > 1.0
        go_criteria.append(sharpe_pass)
        print(f"✓ Sharpe > 1.0: {summ['sharpe_ratio']:.2f} {'PASS' if sharpe_pass else 'FAIL'}")
        
        # Both sides profitable
        both_pass = summ["long"]["avg_ret"] > 0 and summ["short"]["avg_ret"] > 0
        go_criteria.append(both_pass)
        print(f"✓ Both sides profitable: Long {summ['long']['avg_ret']:.2%}, Short {summ['short']['avg_ret']:.2%} {'PASS' if both_pass else 'FAIL'}")
        
        print("-" * 60)
        if all(go_criteria):
            print("🎯 VERDICT: GO - Proceed to paper trading")
        else:
            print("⚠️  VERDICT: NO-GO - Model needs improvement")
        print("="*60 + "\n")
    else:
        print(f"⚠️  ERROR: {summ.get('error', 'Unknown error')}")
        print("="*60 + "\n")


if __name__ == "__main__":
    main()
