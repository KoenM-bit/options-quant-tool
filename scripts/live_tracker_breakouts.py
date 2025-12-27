#!/usr/bin/env python3
"""
Config-driven LIVE signal generator for breakout model (Option A).

Reads a universe JSON config containing:
- enabled markets
- thresholds
- ticker universe
- model bundle path
- feature schema

Then:
1) Loads calibrated model bundle
2) Loads events (parquet)
3) Filters to completed events (t_end <= asof) and correct market
4) Ensures entry bar exists at t_end + 1 hour
5) Predicts P(UP) for each event
6) Applies decision thresholds -> LONG/SHORT/NO_TRADE
7) Writes signals CSV + summary JSON

Usage:
    python scripts/live_tracker_breakouts.py \
      --config config/live_universe.json \
      --events data/ml_datasets/accum_distrib_events.parquet \
      --asof "2025-12-24 15:30:00" \
      --output data/signals/breakout_signals_2025-12-24.csv

Notes:
- No lookahead: uses only event features (computed inside event window),
  and requires next bar to exist for tradeability.
- This script emits entry signals only (no exit management).
"""

import argparse
import json
import logging
from pathlib import Path
from typing import Dict, Any, List, Tuple

import numpy as np
import pandas as pd
import joblib
from sqlalchemy import create_engine, text

# Project imports
import sys
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))
from src.config import settings

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)


# ---------------------------
# Helpers
# ---------------------------

def load_json(path: str) -> Dict[str, Any]:
    p = Path(path)
    if not p.exists():
        raise FileNotFoundError(f"JSON file not found: {p}")
    return json.loads(p.read_text())


def load_bundle(bundle_dir: str) -> Tuple[List[str], Dict[str, Any], bool]:
    """
    Load production bundle:
    - features.json (ordered)
    - model_calibrated.pkl (preferred) or model.pkl
    Returns:
      feature_names, model_bundle, calibrated_flag
    """
    bundle_dir = Path(bundle_dir)
    if not bundle_dir.exists():
        raise FileNotFoundError(f"Bundle dir not found: {bundle_dir}")

    features_path = bundle_dir / "features.json"
    if not features_path.exists():
        raise FileNotFoundError(f"features.json not found in {bundle_dir}")

    features_data = json.loads(features_path.read_text())

    # support list or dict style
    if isinstance(features_data, dict) and "feature_names" in features_data:
        feature_names = features_data["feature_names"]
    elif isinstance(features_data, list):
        feature_names = [f["name"] for f in features_data]
    else:
        raise ValueError(f"Unexpected features.json format: {type(features_data)}")

    model_cal = bundle_dir / "model_calibrated.pkl"
    model_raw = bundle_dir / "model.pkl"

    if model_cal.exists():
        model_bundle = joblib.load(model_cal)
        return feature_names, model_bundle, True
    elif model_raw.exists():
        model_bundle = joblib.load(model_raw)
        return feature_names, model_bundle, False
    else:
        raise FileNotFoundError(f"No model.pkl or model_calibrated.pkl found in {bundle_dir}")


def get_model_and_calibrator(model_bundle: Any):
    """
    Bundle formats:
      - dict: {'model': ..., 'calibrator': ...}
      - model object directly (uncalibrated)
    """
    if isinstance(model_bundle, dict) and "model" in model_bundle:
        model = model_bundle["model"]
        calibrator = model_bundle.get("calibrator", None)
        return model, calibrator

    # Otherwise model bundle is model object
    return model_bundle, None


def apply_calibration(calibrator, p_uncal: np.ndarray) -> np.ndarray:
    if calibrator is None:
        return p_uncal

    # calibrators often expect 2D input
    p2 = p_uncal.reshape(-1, 1)
    return calibrator.predict(p2)


def validate_config(config: Dict[str, Any], bundle_features: List[str]) -> None:
    """
    Fail-fast if config and bundle mismatch in dangerous ways.
    """
    if "markets" not in config:
        raise ValueError("Config missing 'markets'")

    # Validate model bundle path exists
    bundle = config.get("model_bundle")
    if not bundle:
        raise ValueError("Config missing top-level 'model_bundle'")

    # Validate features match bundle (if provided)
    cfg_feats = config.get("features", {}).get("feature_names", None)
    if cfg_feats is not None:
        if list(cfg_feats) != list(bundle_features):
            raise ValueError(
                "FEATURE MISMATCH: config.features.feature_names does not match bundle features.json.\n"
                f"Config:  {cfg_feats}\n"
                f"Bundle:  {bundle_features}\n"
                "Fix your universe.json features list to EXACTLY match the bundle ordering."
            )


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


def resolve_universe(market_cfg: Dict[str, Any]) -> List[str]:
    """
    Allows:
      "tickers": "ALL"
      "tickers": ["AAPL", ...]
    """
    tickers = market_cfg.get("tickers", [])
    if isinstance(tickers, str) and tickers.upper() == "ALL":
        return []  # meaning no explicit restriction
    if isinstance(tickers, list):
        return tickers
    raise ValueError(f"Unexpected tickers format: {type(tickers)}")


def get_decision_rule(market_cfg: Dict[str, Any]) -> Tuple[float, float]:
    """
    Decision rule:
      - explicit: decision_rule.long_if_p_up_gte, short_if_p_up_lte
      - fallback: symmetric threshold logic
    """
    if "decision_rule" in market_cfg:
        dr = market_cfg["decision_rule"]
        return float(dr["long_if_p_up_gte"]), float(dr["short_if_p_up_lte"])

    th = float(market_cfg["threshold"])
    return th, 1.0 - th


# ---------------------------
# Core signal generation
# ---------------------------

def generate_signals(
    config: Dict[str, Any],
    events: pd.DataFrame,
    asof_ts: pd.Timestamp,
    lookback_hours: int,
    feature_names: List[str],
    model_bundle: Any,
    engine=None,
) -> pd.DataFrame:

    model, calibrator = get_model_and_calibrator(model_bundle)

    # standardize
    events = events.copy()
    events["t_end"] = pd.to_datetime(events["t_end"])
    events["t_start"] = pd.to_datetime(events["t_start"])

    # only completed by asof
    events = events[events["t_end"] <= asof_ts].copy()
    
    # CRITICAL: Only look at RECENT events (within lookback window)
    lookback_start = asof_ts - pd.Timedelta(hours=lookback_hours)
    events = events[events["t_end"] >= lookback_start].copy()
    
    logger.info(f"Lookback window: {lookback_start} to {asof_ts} ({lookback_hours} hours)")
    logger.info(f"Events after lookback filter: {len(events):,}")

    # NOTE: For live trading, we do NOT filter by label_valid because:
    # 1. label_valid requires forward-looking data to validate the label
    # 2. In live trading, we're predicting the future - we don't have the validation data yet
    # 3. The model was trained on label_valid=True events, but for inference we apply to all events
    # if "label_valid" in events.columns:
    #     before_valid = len(events)
    #     events = events[events["label_valid"] == True].copy()
    #     logger.info(f"Events after label_valid filter: {len(events):,} (filtered out {before_valid - len(events)} invalid)")

    logger.info(f"Total events to process: {len(events):,} across {events['market'].nunique() if len(events) > 0 else 0} markets")
    
    signals = []
    total_candidates = 0
    total_tradeable = 0

    # per market loop
    for mkt, mkt_cfg in config["markets"].items():
        if not mkt_cfg.get("enabled", False):
            continue

        th_long, th_short = get_decision_rule(mkt_cfg)

        # filter events to market
        ev_m = events[events["market"] == mkt].copy()
        
        logger.info(f"[{mkt}] Events in lookback window: {len(ev_m):,}")
        
        if len(ev_m) == 0:
            logger.info(f"[{mkt}] No completed events in lookback window")
            continue

        # universe filter
        explicit_universe = resolve_universe(mkt_cfg)
        if explicit_universe:
            before_filter = len(ev_m)
            ev_m = ev_m[ev_m["ticker"].isin(explicit_universe)].copy()
            logger.info(f"[{mkt}] After Top 25 filter: {len(ev_m):,} events (filtered out {before_filter - len(ev_m)})")

        if len(ev_m) == 0:
            logger.info(f"[{mkt}] No events after universe filtering")
            continue

        logger.info(f"[{mkt}] Completed events: {len(ev_m):,}")

        total_candidates += len(ev_m)

        # Need prices to confirm entry bar exists
        if engine is None:
            engine = create_engine(settings.database_url)

        tickers = ev_m["ticker"].unique()
        start_ts = ev_m["t_end"].min() - pd.Timedelta(days=2)
        end_ts = ev_m["t_end"].max() + pd.Timedelta(hours=50)

        prices = get_prices(engine, tickers, start_ts, end_ts, market=mkt)
        prices["timestamp"] = pd.to_datetime(prices["timestamp"])
        prices = prices.sort_values(["ticker", "timestamp"])
        by_ticker = {t: g.set_index("timestamp") for t, g in prices.groupby("ticker")}

        # iterate each event
        for _, ev in ev_m.iterrows():
            tkr = ev["ticker"]
            if tkr not in by_ticker:
                continue

            t_end = pd.to_datetime(ev["t_end"])
            entry_ts = t_end + pd.Timedelta(hours=1)

            g = by_ticker[tkr]
            if entry_ts not in g.index:
                continue  # not tradeable -> no next bar

            entry_open = float(g.loc[entry_ts, "open"])
            if not np.isfinite(entry_open) or entry_open <= 0:
                continue

            # Build X exactly in feature order
            try:
                X = ev[feature_names].to_frame().T.astype(float).fillna(0.0)
            except KeyError:
                continue

            # Predict
            p_uncal = model.predict_proba(X)[:, 1]
            p_up = float(apply_calibration(calibrator, p_uncal)[0])

            # Decision
            if p_up >= th_long:
                direction = "LONG"
                confidence = "HIGH" if p_up >= 0.85 else ("MEDIUM" if p_up >= 0.75 else "LOW")
            elif p_up <= th_short:
                direction = "SHORT"
                confidence = "HIGH" if p_up <= 0.15 else ("MEDIUM" if p_up <= 0.25 else "LOW")
            else:
                direction = "NO_TRADE"
                confidence = "NONE"

            total_tradeable += 1

            # Only emit trades (or optionally include NO_TRADE rows)
            if direction == "NO_TRADE":
                continue

            row = {
                "ticker": tkr,
                "market": mkt,
                "t_start": pd.to_datetime(ev["t_start"]),
                "t_end": t_end,
                "entry_ts": entry_ts,
                "entry_open": entry_open,
                "p_up": p_up,
                "direction": direction,
                "confidence": confidence,
                "threshold_long": th_long,
                "threshold_short": th_short,
            }

            # include features for debugging/monitoring
            for f in feature_names:
                row[f] = float(ev.get(f, np.nan))

            signals.append(row)

    if len(signals) == 0:
        logger.warning("No signals generated (no recent completed events or all filtered out)")
        return pd.DataFrame()

    out = pd.DataFrame(signals).sort_values(["market", "entry_ts", "p_up"], ascending=[True, True, False])

    logger.info(f"Total candidates (completed events): {total_candidates:,}")
    logger.info(f"Total tradeable (has entry bar):     {total_tradeable:,}")
    logger.info(f"Signals emitted (LONG/SHORT):        {len(out):,}")
    return out


# ---------------------------
# Main
# ---------------------------

def main():
    ap = argparse.ArgumentParser(description="Live breakout signal tracker (config-driven)")
    ap.add_argument("--config", required=True, help="Universe JSON config file")
    ap.add_argument("--events", required=True, help="Events parquet file")
    ap.add_argument("--asof", required=True, help="As-of timestamp (e.g. '2025-12-24 15:30:00')")
    ap.add_argument("--lookback_hours", type=int, default=2, help="Only consider events that ended within this many hours before asof (default: 2 for live trading)")
    ap.add_argument("--output", required=True, help="Output signals CSV file")
    ap.add_argument("--summary_out", default=None, help="Optional JSON summary output")
    args = ap.parse_args()

    config = load_json(args.config)
    asof_ts = pd.to_datetime(args.asof)

    # Load bundle once
    bundle_dir = config["model_bundle"]
    bundle_features, model_bundle, calibrated = load_bundle(bundle_dir)

    # Validate config against bundle
    validate_config(config, bundle_features)

    # Load events
    events = pd.read_parquet(args.events)

    # Generate signals
    engine = create_engine(settings.database_url)
    signals = generate_signals(
        config=config,
        events=events,
        asof_ts=asof_ts,
        lookback_hours=args.lookback_hours,
        feature_names=bundle_features,
        model_bundle=model_bundle,
        engine=engine
    )

    # Save
    out_path = Path(args.output)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    signals.to_csv(out_path, index=False)
    logger.info(f"Saved signals: {out_path} ({len(signals):,} rows)")

    # Optional run summary
    if args.summary_out:
        summary = {
            "asof": str(asof_ts),
            "lookback_hours": args.lookback_hours,
            "lookback_start": str(asof_ts - pd.Timedelta(hours=args.lookback_hours)),
            "n_signals": int(len(signals)),
            "markets": signals["market"].value_counts().to_dict() if len(signals) else {},
            "calibrated": bool(calibrated),
            "bundle": str(bundle_dir),
            "output": str(out_path),
        }
        s_path = Path(args.summary_out)
        s_path.parent.mkdir(parents=True, exist_ok=True)
        s_path.write_text(json.dumps(summary, indent=2))
        logger.info(f"Saved run summary: {s_path}")

    print("\nDONE ✅")


if __name__ == "__main__":
    main()