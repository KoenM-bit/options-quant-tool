#!/usr/bin/env python3
"""
Top performer removal validation for NL Top 25 Liquidity universe.

Tests whether concentration risk disappears after filtering to liquid tickers.

Previous result: Full NL universe had extreme concentration (top 5 = 76% of returns)
Expected: Top 25 should show more diversified edge

Usage:
    python scripts/validate_top_performers_nl_top25.py \
        --bundle ML/production/v20251223_212702 \
        --events data/ml_datasets/accum_distrib_events.parquet \
        --threshold 0.65 \
        --cost_bps 10 \
        --remove_counts 3,5,8 \
        --output data/backtests/validation/top_perf_nl_top25.csv
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


def compounded_return(r: pd.Series) -> float:
    """Compounded return: Π(1+ri) - 1"""
    if len(r) == 0:
        return 0.0
    return float((1.0 + r).prod() - 1.0)


def backtest_with_exclusions(
    events: pd.DataFrame,
    prices: pd.DataFrame,
    model,
    feature_names: list,
    threshold: float,
    cost_bps: float,
    exclude_tickers: set = None,
    H: int = 40,
    ATR_K: float = 1.5
) -> tuple:
    """
    Run backtest with optional ticker exclusions.
    Returns: (metrics_dict, trades_df)
    """
    if exclude_tickers is None:
        exclude_tickers = set()

    # Filter out excluded tickers
    events = events[~events["ticker"].isin(exclude_tickers)].copy()
    prices = prices[~prices["ticker"].isin(exclude_tickers)].copy()

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

        entry_px = float(g.loc[entry_ts, "open"])
        if not np.isfinite(entry_px) or entry_px <= 0:
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
            "n_trades": 0, "n_tickers": 0, "win_rate": 0, "avg_ret": 0,
            "median_ret": 0, "std_ret": 0, "total_ret": 0, "max_dd": 0, "sharpe": 0
        }, pd.DataFrame()

    trades_df = pd.DataFrame(trades)
    trades_df = trades_df.sort_values("entry_ts")

    win_rate = (trades_df["net_ret"] > 0).mean()
    avg_ret = trades_df["net_ret"].mean()
    med_ret = trades_df["net_ret"].median()
    std_ret = trades_df["net_ret"].std()
    n_tickers = trades_df["ticker"].nunique()

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
        "n_tickers": int(n_tickers),
        "win_rate": float(win_rate),
        "avg_ret": float(avg_ret),
        "median_ret": float(med_ret),
        "std_ret": float(std_ret),
        "total_ret": float(total_ret),
        "max_dd": float(max_dd),
        "sharpe": float(sharpe),
    }, trades_df


def main():
    ap = argparse.ArgumentParser(description="NL Top 25 Liquidity top performer removal")
    ap.add_argument("--bundle", required=True, help="Production bundle directory")
    ap.add_argument("--events", required=True, help="Events parquet file")
    ap.add_argument("--threshold", type=float, default=0.65, help="Decision threshold")
    ap.add_argument("--cost_bps", type=float, default=10.0, help="Transaction costs (bps)")
    ap.add_argument("--remove_counts", default="3,5,8", help="Comma-separated removal counts")
    ap.add_argument("--output", required=True, help="Output CSV path")
    args = ap.parse_args()

    remove_counts = [int(x) for x in args.remove_counts.split(",")]

    logger.info("="*70)
    logger.info("NL TOP 25 LIQUIDITY TOP PERFORMER REMOVAL")
    logger.info("="*70)
    logger.info(f"Threshold: {args.threshold}, Cost: {args.cost_bps} bps")
    logger.info(f"Removal counts: {remove_counts}")
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

    # Baseline
    logger.info("\n" + "="*70)
    logger.info("BASELINE (Top 25 Liquidity)")
    logger.info("="*70)

    baseline_metrics, baseline_trades = backtest_with_exclusions(
        events, prices, model, feature_names,
        args.threshold, args.cost_bps, exclude_tickers=None
    )

    logger.info(
        f"Baseline: {baseline_metrics['n_trades']} trades, {baseline_metrics['n_tickers']} tickers, "
        f"{baseline_metrics['win_rate']:.1%} WR, {baseline_metrics['avg_ret']:.3%} avg, "
        f"Sharpe {baseline_metrics['sharpe']:.2f}, DD {baseline_metrics['max_dd']:.1%}"
    )

    # Rank tickers by compounded return
    ticker_stats = baseline_trades.groupby("ticker").agg(
        total_ret=("net_ret", compounded_return),
        n_trades=("net_ret", "count"),
        avg_ret=("net_ret", "mean"),
    ).reset_index()

    ticker_stats = ticker_stats.sort_values("total_ret", ascending=False)

    logger.info("\nTop 10 performers (ranked by compounded return):")
    for rank, (_, row) in enumerate(ticker_stats.head(10).iterrows(), start=1):
        logger.info(
            f"  {rank:2d}. {row['ticker']:10s} "
            f"{row['total_ret']:7.2%} total "
            f"({int(row['n_trades']):3d} trades, avg {row['avg_ret']:.3%})"
        )

    # Calculate concentration
    top_5_pct = ticker_stats.head(5)["total_ret"].sum() / ticker_stats["total_ret"].sum() * 100
    logger.info(f"\nTop 5 concentration: {top_5_pct:.1f}% of total returns")

    # Run removal tests
    results = [{"scenario": "baseline", "removed": 0, **baseline_metrics}]

    for n_remove in remove_counts:
        logger.info("\n" + "="*70)
        logger.info(f"REMOVING TOP {n_remove} PERFORMERS")
        logger.info("="*70)

        top_n = set(ticker_stats.head(n_remove)["ticker"])
        logger.info(f"Excluding: {', '.join(sorted(top_n))}")

        metrics, trades = backtest_with_exclusions(
            events, prices, model, feature_names,
            args.threshold, args.cost_bps, exclude_tickers=top_n
        )

        results.append({"scenario": f"remove_top_{n_remove}", "removed": n_remove, **metrics})

        logger.info(
            f"Results: {metrics['n_trades']} trades, {metrics['n_tickers']} tickers, "
            f"{metrics['win_rate']:.1%} WR, {metrics['avg_ret']:.3%} avg, "
            f"Sharpe {metrics['sharpe']:.2f}, DD {metrics['max_dd']:.1%}"
        )

    # Save results
    results_df = pd.DataFrame(results)
    out = Path(args.output)
    out.parent.mkdir(parents=True, exist_ok=True)
    results_df.to_csv(out, index=False)
    
    # Save ticker ranking
    ticker_out = out.parent / f"{out.stem}.ticker_rank.csv"
    ticker_stats.to_csv(ticker_out, index=False)
    
    logger.info(f"\nSaved results: {out}")
    logger.info(f"Saved ticker ranking: {ticker_out}")

    # Summary table
    print("\n" + "="*90)
    print("NL TOP 25 LIQUIDITY TOP PERFORMER REMOVAL ANALYSIS")
    print("="*90)
    print(f"\n{'Scenario':<20} {'Trades':<8} {'Tickers':<8} {'WR':<8} {'Avg Ret':<10} {'Sharpe':<10} {'Max DD':<10}")
    print("-"*90)

    baseline_sharpe = float(baseline_metrics["sharpe"])
    baseline_avg_ret = float(baseline_metrics["avg_ret"])

    for _, row in results_df.iterrows():
        scenario = row["scenario"]

        if scenario == "baseline":
            print(f"{scenario:<20} {row['n_trades']:<8} {row['n_tickers']:<8} {row['win_rate']:.1%}    "
                  f"{row['avg_ret']:.3%}     {row['sharpe']:.2f}       {row['max_dd']:.1%}")
        else:
            wr_delta = row["win_rate"] - baseline_metrics["win_rate"]
            ret_delta = row["avg_ret"] - baseline_avg_ret
            sharpe_delta = row["sharpe"] - baseline_sharpe

            print(f"{scenario:<20} {row['n_trades']:<8} {row['n_tickers']:<8} {row['win_rate']:.1%} ({wr_delta:+.1%}) "
                  f"{row['avg_ret']:.3%} ({ret_delta:+.3%}) "
                  f"{row['sharpe']:.2f} ({sharpe_delta:+.2f}) "
                  f"{row['max_dd']:.1%}")

    print("="*90)

    # GO/NO-GO Assessment
    print("\nGO/NO-GO ASSESSMENT")
    print("-"*90)

    checks = []
    for _, row in results_df.iterrows():
        if row["scenario"] == "baseline":
            continue

        scenario = row["scenario"]

        # Checks
        check1 = row["avg_ret"] > 0
        check2 = row["win_rate"] > 0.60
        check3 = row["sharpe"] > 1.0
        check4 = row["max_dd"] > -0.20

        # Sharpe drop
        sharpe_drop = (baseline_sharpe - row["sharpe"]) / max(baseline_sharpe, 1e-9)
        check5 = True
        if baseline_sharpe >= 2.0:
            check5 = sharpe_drop < 0.50

        all_pass = all([check1, check2, check3, check4, check5])
        checks.append(all_pass)

        status = "✅ PASS" if all_pass else "❌ FAIL"
        print(f"\n{scenario}: {status}")
        print(f"  • Avg return > 0: {row['avg_ret']:.3%} {'✓' if check1 else '✗'}")
        print(f"  • Win rate > 60%: {row['win_rate']:.1%} {'✓' if check2 else '✗'}")
        print(f"  • Sharpe > 1.0: {row['sharpe']:.2f} {'✓' if check3 else '✗'}")
        print(f"  • Max DD < -20%: {row['max_dd']:.1%} {'✓' if check4 else '✗'}")
        if baseline_sharpe >= 2.0:
            print(f"  • Sharpe drop < 50%: {sharpe_drop:.1%} {'✓' if check5 else '✗'}")

    print("-"*90)

    if all(checks):
        print("🎯 VERDICT: GO - Edge is diversified in Top 25 universe")
        print(f"   Top 5 concentration: {top_5_pct:.1f}% (vs 76% in full universe)")
    elif any(checks):
        print("⚠️  VERDICT: MARGINAL - Some concentration remains")
        print(f"   Top 5 concentration: {top_5_pct:.1f}%")
    else:
        print("❌ VERDICT: NO-GO - Edge still driven by few tickers")
        print(f"   Top 5 concentration: {top_5_pct:.1f}%")

    print("\nCOMPARISON WITH FULL NL UNIVERSE:")
    print(f"  Full universe: Top 5 = 76% concentration, fails all removal tests")
    print(f"  Top 25 Liquidity: Top 5 = {top_5_pct:.1f}% concentration")
    print("="*90 + "\n")


if __name__ == "__main__":
    main()
