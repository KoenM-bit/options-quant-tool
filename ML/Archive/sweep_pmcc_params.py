import os
import sys
import itertools
from datetime import datetime
from pathlib import Path
import pandas as pd

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

# Import your working backtest module
import ML.backtest_pmcc_stockproxy as bt


def run_once(env_overrides: dict):
    """
    Run 1 backtest with certain env overrides.
    Returns dict with metrics + params.
    """
    # backup env
    old = {}
    for k, v in env_overrides.items():
        old[k] = os.environ.get(k)
        os.environ[k] = str(v)

    try:
        cfg = bt.Cfg()
        eng = bt.make_engine()

        px = bt.load_prices(eng, cfg)
        pr = bt.load_predictions(eng, cfg)
        df = px.merge(pr, on="dt", how="inner").dropna()

        trades_df, equity_df = bt.backtest_pmcc(df, cfg)
        summ = bt.summarize(equity_df)

        out = {
            "ts": datetime.now().isoformat(timespec="seconds"),
            **env_overrides,
            **{k: float(v) if hasattr(v, "__float__") else v for k, v in summ.items()},
            "n_days": int(len(equity_df)),
        }
        return out

    finally:
        # restore env
        for k, v in old.items():
            if v is None:
                os.environ.pop(k, None)
            else:
                os.environ[k] = v


def main():
    # ---------
    # REQUIRED base env
    # ---------
    base_env = {
        "TICKER": os.getenv("TICKER", "AD.AS"),
        "MODEL_VERSION": os.environ["MODEL_VERSION"],  # must be set
        "HORIZON": os.getenv("HORIZON", "5"),
        "LABEL_Q": os.getenv("LABEL_Q", "0.30"),
        "BT_START": os.getenv("BT_START", "2021-01-01"),
        "BT_END": os.getenv("BT_END", "2025-12-31"),
        "INIT_CASH": os.getenv("INIT_CASH", "10000"),
    }

    # ---------
    # Sweep grid (edit freely)
    # ---------
    grid = {
        "DELTA_RANGE": [0.18, 0.22, 0.25, 0.28, 0.32],
        "DELTA_MIDDLE": [0.15, 0.18, 0.22],
        # Trend behavior: either skip calls or write very low delta
        "SKIP_TREND": ["1", "0"],
        "DELTA_TREND": [0.08, 0.10, 0.12],
        "SHORT_DAYS": [14, 21, 28],
        "REB_EVERY": [14, 21],
        # Pricing sensitivity:
        "DIV_YIELD": [0.00, 0.03],
        "RV_WINDOW": [10, 20],
    }

    # Build combinations
    keys = list(grid.keys())
    combos = list(itertools.product(*[grid[k] for k in keys]))
    print(f"Running sweep with {len(combos)} runs...")

    results = []
    for i, vals in enumerate(combos, 1):
        env = dict(base_env)
        env.update({k: v for k, v in zip(keys, vals)})

        r = run_once(env)
        results.append(r)

        if i % 10 == 0:
            print(f"  done {i}/{len(combos)}")

    out = pd.DataFrame(results)

    # Rank: primary sharpe, secondary max_dd, tertiary end_eq
    out["rank_key"] = out["sharpe_like"] - 0.5 * out["max_drawdown"].abs()
    out = out.sort_values(["rank_key", "sharpe_like", "end_eq"], ascending=[False, False, False])

    os.makedirs("ML", exist_ok=True)
    out_csv = f"ML/sweep_pmcc_policy_{datetime.now().date()}.csv"
    out.to_csv(out_csv, index=False)

    print("\nSaved:", out_csv)
    print("\nTop 15 configs:")
    cols = (
        ["rank_key", "sharpe_like", "cagr", "max_drawdown", "end_eq"]
        + ["DELTA_RANGE", "DELTA_MIDDLE", "SKIP_TREND", "DELTA_TREND", "SHORT_DAYS", "REB_EVERY", "DIV_YIELD", "RV_WINDOW"]
    )
    print(out[cols].head(15).to_string(index=False))

    # Also print baseline row if present (all deltas equal)
    baseline = out[
        (out["DELTA_RANGE"] == 0.18)
        & (out["DELTA_MIDDLE"] == 0.18)
        & (out["DELTA_TREND"] == 0.18)
        & (out["SKIP_TREND"] == "0")
    ]
    if len(baseline):
        print("\nBaseline-ish example:")
        print(baseline[cols].head(1).to_string(index=False))


if __name__ == "__main__":
    if "MODEL_VERSION" not in os.environ:
        raise RuntimeError("Set MODEL_VERSION first, e.g. MODEL_VERSION=rangeH5_xgb_sigmoid_2025-12-15")
    main()
