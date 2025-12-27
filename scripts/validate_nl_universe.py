#!/usr/bin/env python3
"""
NL Universe Robustness Validation

Tests whether NL market performance improves by filtering universe:
1) AEX only (large caps)
2) Top 25 by liquidity (avg daily volume)
3) Exclude worst 10 spreads (high spread = wide bid-ask)

Purpose: Determine if concentration risk can be mitigated via universe selection.

Usage:
    python scripts/validate_nl_universe.py \
        --bundle ML/production/v20251223_212702 \
        --events data/ml_datasets/accum_distrib_events.parquet \
        --threshold 0.65 \
        --cost_bps 10 \
        --output data/backtests/validation/nl_universe_robustness.csv
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


# AEX components (25 largest Dutch companies as of late 2024/early 2025)
AEX_TICKERS = {
    "ASML.AS", "SHELL.AS", "INGA.AS", "HEIA.AS", "PHIA.AS",
    "ADYEN.AS", "UNA.AS", "AKZA.AS", "KPN.AS", "ABN.AS",
    "RAND.AS", "DSM.AS", "WKL.AS", "IMCD.AS", "ASM.AS",
    "BESI.AS", "NN.AS", "JDE.AS", "UMG.AS", "ASRNL.AS",
    "ABN.AS", "AGN.AS", "AD.AS", "LIGHT.AS", "GLPG.AS"
}


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


def compute_liquidity_metrics(engine, tickers, market="NL"):
    """
    Compute average daily volume and spread proxy for each ticker.
    
    Spread proxy = (high - low) / close
    Higher values = wider intraday range (proxy for poor liquidity / wide spreads)
    """
    q = """
    SELECT 
        ticker,
        AVG(volume) as avg_volume,
        AVG((high - low) / NULLIF(close, 0)) as avg_spread_proxy
    FROM bronze_ohlcv_intraday
    WHERE ticker = ANY(:tickers)
      AND market = :market
      AND volume > 0
      AND close > 0
    GROUP BY ticker
    """
    params = {"tickers": list(tickers), "market": market}
    df = pd.read_sql(text(q), engine, params=params)
    return df


def compounded_return(r: pd.Series) -> float:
    """Compounded return: Π(1+ri) - 1"""
    if len(r) == 0:
        return 0.0
    return float((1.0 + r).prod() - 1.0)


def backtest_universe(
    events: pd.DataFrame,
    prices: pd.DataFrame,
    model,
    feature_names: list,
    threshold: float,
    cost_bps: float,
    ticker_filter: set = None,
    H: int = 40,
    ATR_K: float = 1.5
) -> dict:
    """
    Run backtest on filtered universe.
    
    Args:
        ticker_filter: If provided, only trade these tickers
    """
    if ticker_filter is not None:
        events = events[events["ticker"].isin(ticker_filter)].copy()
        prices = prices[prices["ticker"].isin(ticker_filter)].copy()

    if len(events) == 0:
        return {
            "n_trades": 0, "n_tickers": 0, "win_rate": 0, "avg_ret": 0,
            "median_ret": 0, "std_ret": 0, "total_ret": 0, "max_dd": 0, "sharpe": 0
        }

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
        }

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
    }


def main():
    ap = argparse.ArgumentParser(description="NL universe robustness validation")
    ap.add_argument("--bundle", required=True, help="Production bundle directory")
    ap.add_argument("--events", required=True, help="Events parquet file")
    ap.add_argument("--threshold", type=float, default=0.65, help="Decision threshold")
    ap.add_argument("--cost_bps", type=float, default=10.0, help="Transaction costs (bps)")
    ap.add_argument("--output", required=True, help="Output CSV path")
    args = ap.parse_args()

    logger.info("="*70)
    logger.info("NL UNIVERSE ROBUSTNESS VALIDATION")
    logger.info("="*70)
    logger.info(f"Threshold: {args.threshold}, Cost: {args.cost_bps} bps")
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

    # Get prices
    engine = create_engine(settings.database_url)
    tickers = events["ticker"].unique()
    price_start = events["t_end"].min() - pd.Timedelta(days=2)
    price_end = events["t_end"].max() + pd.Timedelta(hours=50)

    logger.info(f"Fetching prices for {len(tickers)} NL tickers...")
    prices = get_prices(engine, tickers, price_start, price_end)
    logger.info(f"Loaded {len(prices):,} price bars")

    # Compute liquidity metrics
    logger.info("\nComputing liquidity metrics...")
    liq_metrics = compute_liquidity_metrics(engine, tickers)
    logger.info(f"Got metrics for {len(liq_metrics)} tickers")

    # Identify universe filters
    logger.info("\n" + "="*70)
    logger.info("UNIVERSE DEFINITIONS")
    logger.info("="*70)

    # 1) AEX only
    aex_tickers = set(tickers) & AEX_TICKERS
    logger.info(f"\n1) AEX ONLY: {len(aex_tickers)} tickers")
    logger.info(f"   Tickers: {', '.join(sorted(aex_tickers))}")

    # 2) Top 25 by liquidity (avg volume)
    top_25_liq = set(liq_metrics.nlargest(25, "avg_volume")["ticker"])
    logger.info(f"\n2) TOP 25 LIQUIDITY: {len(top_25_liq)} tickers")
    logger.info(f"   Tickers: {', '.join(sorted(top_25_liq))}")

    # 3) Exclude worst 10 spreads
    worst_10_spreads = set(liq_metrics.nlargest(10, "avg_spread_proxy")["ticker"])
    exclude_spreads = set(tickers) - worst_10_spreads
    logger.info(f"\n3) EXCLUDE WORST 10 SPREADS: {len(exclude_spreads)} tickers")
    logger.info(f"   Excluded: {', '.join(sorted(worst_10_spreads))}")
    logger.info(f"   Remaining: {', '.join(sorted(exclude_spreads))}")

    # Run backtests
    results = []

    # Baseline (all tickers)
    logger.info("\n" + "="*70)
    logger.info("SCENARIO 1: BASELINE (All 41 NL tickers)")
    logger.info("="*70)
    baseline = backtest_universe(events, prices, model, feature_names, args.threshold, args.cost_bps)
    results.append({"scenario": "baseline", "n_tickers": baseline["n_tickers"], **baseline})
    logger.info(
        f"Results: {baseline['n_trades']} trades, {baseline['n_tickers']} tickers, "
        f"{baseline['win_rate']:.1%} WR, {baseline['avg_ret']:.3%} avg, "
        f"Sharpe {baseline['sharpe']:.2f}, DD {baseline['max_dd']:.1%}"
    )

    # AEX only
    logger.info("\n" + "="*70)
    logger.info("SCENARIO 2: AEX ONLY (Large caps)")
    logger.info("="*70)
    aex_result = backtest_universe(events, prices, model, feature_names, args.threshold, args.cost_bps, aex_tickers)
    results.append({"scenario": "aex_only", "n_tickers": len(aex_tickers), **aex_result})
    logger.info(
        f"Results: {aex_result['n_trades']} trades, {aex_result['n_tickers']} tickers, "
        f"{aex_result['win_rate']:.1%} WR, {aex_result['avg_ret']:.3%} avg, "
        f"Sharpe {aex_result['sharpe']:.2f}, DD {aex_result['max_dd']:.1%}"
    )

    # Top 25 liquidity
    logger.info("\n" + "="*70)
    logger.info("SCENARIO 3: TOP 25 LIQUIDITY")
    logger.info("="*70)
    liq_result = backtest_universe(events, prices, model, feature_names, args.threshold, args.cost_bps, top_25_liq)
    results.append({"scenario": "top_25_liquidity", "n_tickers": len(top_25_liq), **liq_result})
    logger.info(
        f"Results: {liq_result['n_trades']} trades, {liq_result['n_tickers']} tickers, "
        f"{liq_result['win_rate']:.1%} WR, {liq_result['avg_ret']:.3%} avg, "
        f"Sharpe {liq_result['sharpe']:.2f}, DD {liq_result['max_dd']:.1%}"
    )

    # Exclude worst 10 spreads
    logger.info("\n" + "="*70)
    logger.info("SCENARIO 4: EXCLUDE WORST 10 SPREADS")
    logger.info("="*70)
    spread_result = backtest_universe(events, prices, model, feature_names, args.threshold, args.cost_bps, exclude_spreads)
    results.append({"scenario": "exclude_worst_spreads", "n_tickers": len(exclude_spreads), **spread_result})
    logger.info(
        f"Results: {spread_result['n_trades']} trades, {spread_result['n_tickers']} tickers, "
        f"{spread_result['win_rate']:.1%} WR, {spread_result['avg_ret']:.3%} avg, "
        f"Sharpe {spread_result['sharpe']:.2f}, DD {spread_result['max_dd']:.1%}"
    )

    # Save results
    results_df = pd.DataFrame(results)
    out = Path(args.output)
    out.parent.mkdir(parents=True, exist_ok=True)
    results_df.to_csv(out, index=False)
    logger.info(f"\nSaved results: {out}")

    # Summary table
    print("\n" + "="*90)
    print("NL UNIVERSE ROBUSTNESS ANALYSIS")
    print("="*90)
    print(f"\n{'Scenario':<25} {'Tickers':<8} {'Trades':<8} {'WR':<8} {'Avg Ret':<10} {'Sharpe':<10} {'Max DD':<10}")
    print("-"*90)

    for _, row in results_df.iterrows():
        print(f"{row['scenario']:<25} {row['n_tickers']:<8} {row['n_trades']:<8} "
              f"{row['win_rate']:.1%}    {row['avg_ret']:.3%}     "
              f"{row['sharpe']:.2f}       {row['max_dd']:.1%}")

    print("="*90)

    # Assessment
    print("\nGO/NO-GO ASSESSMENT")
    print("-"*90)

    baseline_sharpe = baseline["sharpe"]
    baseline_avg = baseline["avg_ret"]

    for _, row in results_df.iterrows():
        if row["scenario"] == "baseline":
            continue

        scenario = row["scenario"]

        # Checks
        check1 = row["avg_ret"] > 0
        check2 = row["win_rate"] > 0.60
        check3 = row["sharpe"] > 1.0
        check4 = row["max_dd"] > -0.20

        # Improvement checks
        sharpe_improved = row["sharpe"] > baseline_sharpe * 1.1  # 10% improvement
        avg_ret_improved = row["avg_ret"] > baseline_avg * 1.1

        all_pass = all([check1, check2, check3, check4])
        improved = sharpe_improved or avg_ret_improved

        if all_pass and improved:
            status = "✅ PASS + IMPROVED"
        elif all_pass:
            status = "✅ PASS (no improvement)"
        else:
            status = "❌ FAIL"

        print(f"\n{scenario}: {status}")
        print(f"  • Avg return > 0: {row['avg_ret']:.3%} {'✓' if check1 else '✗'}")
        print(f"  • Win rate > 60%: {row['win_rate']:.1%} {'✓' if check2 else '✗'}")
        print(f"  • Sharpe > 1.0: {row['sharpe']:.2f} {'✓' if check3 else '✗'}")
        print(f"  • Max DD < -20%: {row['max_dd']:.1%} {'✓' if check4 else '✗'}")
        print(f"  • Sharpe vs baseline: {row['sharpe']:.2f} vs {baseline_sharpe:.2f} ({(row['sharpe']/baseline_sharpe - 1)*100:+.1f}%)")

    print("-"*90)

    # Final verdict
    best_scenario = results_df.iloc[1:].loc[results_df.iloc[1:]["sharpe"].idxmax()]
    
    print(f"\n🎯 BEST SCENARIO: {best_scenario['scenario'].upper()}")
    print(f"   Sharpe: {best_scenario['sharpe']:.2f} (vs baseline {baseline_sharpe:.2f})")
    print(f"   Avg Return: {best_scenario['avg_ret']:.3%} (vs baseline {baseline_avg:.3%})")
    print(f"   Tickers: {best_scenario['n_tickers']} (vs baseline {baseline['n_tickers']})")

    if best_scenario["sharpe"] > 3.0 and best_scenario["sharpe"] > baseline_sharpe * 1.2:
        print("\n✅ VERDICT: NL becomes production-safe with universe filtering")
        print(f"   Recommended: Trade {best_scenario['scenario']} only")
    elif best_scenario["sharpe"] > 2.0:
        print("\n⚠️  VERDICT: NL improved but still marginal")
        print(f"   Consider: Trade {best_scenario['scenario']} with reduced position size")
    else:
        print("\n❌ VERDICT: NL remains too risky even with filtering")
        print("   Recommendation: Skip NL market entirely, focus on US")

    print("="*90 + "\n")


if __name__ == "__main__":
    main()
