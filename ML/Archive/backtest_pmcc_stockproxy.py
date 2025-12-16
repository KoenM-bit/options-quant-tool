import os
import sys
import math
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd
from sqlalchemy import create_engine, text

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from src.config import settings

# -------------------------
# Black-Scholes helpers
# -------------------------
def norm_cdf(x: float) -> float:
    return 0.5 * (1.0 + math.erf(x / math.sqrt(2.0)))

def bs_call_price(S, K, T, r, q, sigma):
    if T <= 0 or sigma <= 0 or S <= 0 or K <= 0:
        return max(0.0, S - K)
    d1 = (math.log(S / K) + (r - q + 0.5 * sigma * sigma) * T) / (sigma * math.sqrt(T))
    d2 = d1 - sigma * math.sqrt(T)
    return S * math.exp(-q * T) * norm_cdf(d1) - K * math.exp(-r * T) * norm_cdf(d2)

def bs_call_delta(S, K, T, r, q, sigma):
    if T <= 0 or sigma <= 0 or S <= 0 or K <= 0:
        return 1.0 if S > K else 0.0
    d1 = (math.log(S / K) + (r - q + 0.5 * sigma * sigma) * T) / (sigma * math.sqrt(T))
    return math.exp(-q * T) * norm_cdf(d1)

def find_strike_for_delta_call(S, target_delta, T, r, q, sigma, K_min=0.1, K_max_mult=3.0):
    """
    Find strike K such that call delta approx target_delta.
    Monotone: higher K -> lower delta.
    Uses bisection on [low, high].
    """
    low = max(K_min, S * 0.2)
    high = S * K_max_mult

    # ensure bracket: delta(low) > target, delta(high) < target
    for _ in range(20):
        if bs_call_delta(S, low, T, r, q, sigma) < target_delta:
            low *= 0.8
        elif bs_call_delta(S, high, T, r, q, sigma) > target_delta:
            high *= 1.2
        else:
            break

    for _ in range(80):
        mid = 0.5 * (low + high)
        d = bs_call_delta(S, mid, T, r, q, sigma)
        if d > target_delta:
            low = mid
        else:
            high = mid
    return 0.5 * (low + high)


# -------------------------
# Config
# -------------------------
class Cfg:
    """Config that reads from environment at instantiation time."""
    
    def __init__(self):
        self.ticker = os.getenv("TICKER", "AD.AS")
        
        # Use your regime predictions
        self.pred_table = os.getenv("PRED_TABLE", "gold_regime_predictions")
        self.model_version = os.getenv("MODEL_VERSION", "rangeH5_xgb_sigmoid_2025-12-15")
        self.horizon = int(os.getenv("HORIZON", "5"))
        self.label_q = float(os.getenv("LABEL_Q", "0.30"))
        
        # Backtest period
        self.start = os.getenv("BT_START", "2021-01-01")
        self.end = os.getenv("BT_END", "2025-12-31")
        
        # Option proxy assumptions
        self.r = float(os.getenv("RISK_FREE", "0.02"))       # 2% risk-free
        self.q_div = float(os.getenv("DIV_YIELD", "0.03"))   # 3% dividend yield proxy
        self.iv_floor = float(os.getenv("IV_FLOOR", "0.10")) # don't let sigma go too low
        self.iv_cap = float(os.getenv("IV_CAP", "0.80"))
        
        self.rv_window = int(os.getenv("RV_WINDOW", "20"))     # realized vol window (days)
        self.trading_days = 252
        
        # PMCC rules
        self.leap_days = int(os.getenv("LEAP_DAYS", "365"))     # buy LEAP ~1y
        self.leap_delta = float(os.getenv("LEAP_DELTA", "0.80"))
        
        self.short_days = int(os.getenv("SHORT_DAYS", "21"))    # ~1 month
        self.rebalance_every = int(os.getenv("REB_EVERY", "21"))# sell new short every N trading days
        
        # Delta targets per regime
        self.delta_range = float(os.getenv("DELTA_RANGE", "0.25"))
        self.delta_middle = float(os.getenv("DELTA_MIDDLE", "0.18"))
        self.delta_trend = float(os.getenv("DELTA_TREND", "0.10"))
        
        # If TREND regime: write calls or skip?
        self.skip_trend = os.getenv("SKIP_TREND", "1") == "1"
        
        # Initial capital and position sizing
        self.init_cash = float(os.getenv("INIT_CASH", "10000"))  # Starting capital
        self.contract_multiplier = 100  # 1 option contract = 100 shares
        
        # Costs (per contract equivalent)
        self.fee_per_trade = float(os.getenv("FEE", "0.10")) # small fee proxy
        
        self.out_trades = "ML/pmcc_backtest_trades.csv"
        self.out_equity = "ML/pmcc_backtest_equity.csv"


def make_engine():
    """Get database engine using project settings."""
    return create_engine(settings.database_url, pool_pre_ping=True)


# -------------------------
# Data load
# -------------------------
def load_prices(engine, cfg: Cfg) -> pd.DataFrame:
    q = """
      SELECT dt::date AS dt, open::float AS open, high::float AS high, low::float AS low, px::float AS close, volume::float AS volume
      FROM silver_prices_ad
      WHERE ticker=:t AND dt::date BETWEEN :s AND :e
      ORDER BY dt;
    """
    df = pd.read_sql(text(q), engine, params={"t": cfg.ticker, "s": cfg.start, "e": cfg.end})
    df["dt"] = pd.to_datetime(df["dt"])
    return df

def load_predictions(engine, cfg: Cfg) -> pd.DataFrame:
    q = f"""
      SELECT dt::date AS dt, p_range::float AS p
      FROM {cfg.pred_table}
      WHERE ticker=:t AND horizon=:h AND label_q=:q AND model_version=:mv
        AND dt::date BETWEEN :s AND :e
      ORDER BY dt;
    """
    df = pd.read_sql(text(q), engine, params={
        "t": cfg.ticker, "h": cfg.horizon, "q": cfg.label_q, "mv": cfg.model_version,
        "s": cfg.start, "e": cfg.end
    })
    df["dt"] = pd.to_datetime(df["dt"])
    return df


# -------------------------
# Backtest
# -------------------------
def compute_iv_proxy(df: pd.DataFrame, cfg: Cfg) -> pd.Series:
    # realized vol of log returns (annualized)
    lr = np.log(df["close"] / df["close"].shift(1))
    rv = lr.rolling(cfg.rv_window).std() * math.sqrt(cfg.trading_days)
    rv = rv.clip(cfg.iv_floor, cfg.iv_cap)
    return rv

def classify_regime(p, p25, p75):
    if p >= p75:
        return "RANGE"
    if p <= p25:
        return "TREND"
    return "MIDDLE"

def backtest_pmcc(df: pd.DataFrame, cfg: Cfg):
    df = df.copy()
    df["iv"] = compute_iv_proxy(df, cfg)

    # rolling p25/p75 on last 252 obs (like your SQL)
    df["p25"] = df["p"].rolling(252, min_periods=120).quantile(0.25)
    df["p75"] = df["p"].rolling(252, min_periods=120).quantile(0.75)

    # state - START WITH INITIAL CASH
    cash = cfg.init_cash
    trades = []
    equity = []

    leap = None   # dict: K, entry_price, entry_dt, contracts
    short = None  # dict: K, entry_price, entry_dt, expiry_idx, contracts

    # Position sizing: 1 contract = 100 shares exposure
    contracts = 1  # Fixed: 1 LEAP contract

    # helper to price current option positions (per share, multiply by multiplier * contracts for notional)
    def price_call(S, K, days, sigma):
        T = max(days, 0) / cfg.trading_days
        return bs_call_price(S, K, T, cfg.r, cfg.q_div, sigma)

    def choose_short_delta(regime):
        if regime == "RANGE":
            return cfg.delta_range
        if regime == "TREND":
            return cfg.delta_trend
        return cfg.delta_middle

    # start after enough history for SMA200 etc.; use iv + p75 availability
    start_idx = int(np.where(df["iv"].notna() & df["p75"].notna())[0][0])

    last_reb_idx = None

    for i in range(start_idx, len(df)):
        row = df.iloc[i]
        dt = row["dt"]
        S = float(row["close"])
        sigma = float(row["iv"])
        p = float(row["p"])
        p25 = float(row["p25"])
        p75 = float(row["p75"])
        regime = classify_regime(p, p25, p75)

        # 1) enter LEAP if none
        if leap is None:
            days = cfg.leap_days
            K = find_strike_for_delta_call(S, cfg.leap_delta, days/cfg.trading_days, cfg.r, cfg.q_div, sigma)
            entry_per_share = price_call(S, K, days, sigma)
            cost = entry_per_share * cfg.contract_multiplier * contracts + cfg.fee_per_trade
            cash -= cost
            leap = {"K": K, "entry": entry_per_share, "entry_dt": dt, "days_left": days, "contracts": contracts}
            trades.append({
                "dt": dt, 
                "type": "BUY_LEAP", 
                "K": K, 
                "price": entry_per_share, 
                "notional": entry_per_share * cfg.contract_multiplier * contracts,
                "cashflow": -cost, 
                "regime": regime
            })

            last_reb_idx = i  # initialize rebalance clock

        # Decrease days left on leap / short
        if leap is not None:
            leap["days_left"] = max(0, leap["days_left"] - 1)

        if short is not None:
            short["days_left"] = max(0, short["days_left"] - 1)

        # 2) handle short expiry settlement (cash settled approximation)
        if short is not None and short["days_left"] == 0:
            payoff_per_share = max(0.0, S - short["K"])  # call payoff per share
            total_payoff = payoff_per_share * cfg.contract_multiplier * short["contracts"]
            # we are short, so we PAY payoff
            cash -= total_payoff + cfg.fee_per_trade
            trades.append({
                "dt": dt, 
                "type": "SHORT_EXPIRE", 
                "K": short["K"], 
                "price": payoff_per_share, 
                "notional": total_payoff,
                "cashflow": -total_payoff - cfg.fee_per_trade, 
                "regime": regime
            })
            short = None

        # 3) rebalance: sell new short call every N days if none outstanding
        if last_reb_idx is None:
            last_reb_idx = i

        if (i - last_reb_idx) >= cfg.rebalance_every and short is None:
            if regime == "TREND" and cfg.skip_trend:
                trades.append({
                    "dt": dt, 
                    "type": "SKIP_SHORT", 
                    "K": None, 
                    "price": None, 
                    "notional": 0.0,
                    "cashflow": 0.0, 
                    "regime": regime
                })
            else:
                days = cfg.short_days
                target_delta = choose_short_delta(regime)
                K = find_strike_for_delta_call(S, target_delta, days/cfg.trading_days, cfg.r, cfg.q_div, sigma)
                premium_per_share = price_call(S, K, days, sigma)
                premium = premium_per_share * cfg.contract_multiplier * contracts - cfg.fee_per_trade
                cash += premium
                short = {"K": K, "entry": premium_per_share, "entry_dt": dt, "days_left": days, "contracts": contracts}
                trades.append({
                    "dt": dt, 
                    "type": "SELL_SHORT", 
                    "K": K, 
                    "price": premium_per_share, 
                    "notional": premium_per_share * cfg.contract_multiplier * contracts,
                    "cashflow": premium, 
                    "regime": regime
                })
            last_reb_idx = i

        # 4) daily mark-to-market equity
        leap_mtm = 0.0
        if leap is not None:
            leap_mtm_per_share = price_call(S, leap["K"], leap["days_left"], sigma)
            leap_mtm = leap_mtm_per_share * cfg.contract_multiplier * leap["contracts"]

        short_mtm = 0.0
        if short is not None:
            short_mtm_per_share = price_call(S, short["K"], short["days_left"], sigma)
            short_mtm = short_mtm_per_share * cfg.contract_multiplier * short["contracts"]

        # equity = cash + leap value - short value (we're short the call)
        eq = cash + leap_mtm - short_mtm
        equity.append({
            "dt": dt,
            "equity": eq,
            "cash": cash,
            "S": S,
            "p": p,
            "p25": p25,
            "p75": p75,
            "regime": regime,
            "leap_K": None if leap is None else leap["K"],
            "short_K": None if short is None else short["K"],
            "iv": sigma
        })

        # 5) if leap nearly expires: roll (sell & buy new)
        if leap is not None and leap["days_left"] <= 21:
            # sell old leap at mtm
            leap_value = leap_mtm - cfg.fee_per_trade
            cash += leap_value
            trades.append({
                "dt": dt, 
                "type": "SELL_LEAP_ROLL", 
                "K": leap["K"], 
                "price": leap_mtm_per_share, 
                "notional": leap_mtm,
                "cashflow": leap_value, 
                "regime": regime
            })
            leap = None
            # short remains; in real life you'd manage this, but for proxy we keep it.

    trades_df = pd.DataFrame(trades)
    equity_df = pd.DataFrame(equity)
    return trades_df, equity_df


def summarize(equity_df: pd.DataFrame):
    eq = equity_df.set_index("dt")["equity"]
    ret = eq.pct_change().dropna()
    ann = 252
    cagr = (eq.iloc[-1] / eq.iloc[0]) ** (ann / len(eq)) - 1 if len(eq) > 10 else np.nan
    vol = ret.std() * math.sqrt(ann)
    sharpe = (ret.mean() * ann) / vol if vol > 0 else np.nan
    maxdd = ((eq / eq.cummax()) - 1).min()
    return {"cagr": cagr, "ann_vol": vol, "sharpe_like": sharpe, "max_drawdown": maxdd, "start_eq": eq.iloc[0], "end_eq": eq.iloc[-1]}


def main():
    cfg = Cfg()
    eng = make_engine()

    px = load_prices(eng, cfg)
    pr = load_predictions(eng, cfg)

    df = px.merge(pr, on="dt", how="inner").dropna()
    if len(df) < 400:
        raise RuntimeError("Too few rows after merge. Check date range + model_version + ticker.")

    trades_df, equity_df = backtest_pmcc(df, cfg)

    os.makedirs("ML", exist_ok=True)
    trades_df.to_csv(cfg.out_trades, index=False)
    equity_df.to_csv(cfg.out_equity, index=False)

    summ = summarize(equity_df)
    print("\n=== PMCC proxy backtest summary ===")
    for k, v in summ.items():
        print(f"{k}: {v}")

    # Regime breakdown (average daily return per regime)
    eq = equity_df.copy()
    eq["ret"] = eq["equity"].pct_change()
    reg = eq.dropna(subset=["ret"]).groupby("regime")["ret"].agg(["count", "mean", "std"])
    print("\n=== Regime breakdown (daily returns) ===")
    print(reg)

    print(f"\nSaved trades: {cfg.out_trades}")
    print(f"Saved equity: {cfg.out_equity}")


if __name__ == "__main__":
    main()
