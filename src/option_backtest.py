"""
Realistic Option Backtesting Framework
=======================================

Uses proper IV modeling instead of assuming IV = RV.

Key improvements:
1. Separate ATM IV model with mean reversion
2. Skew modeling for OTM options
3. Proper Greek-based P&L decomposition
4. Sticky-delta repricing
5. Transaction costs
6. Multiple strategy support (straddles, strangles, directional)
"""

import pandas as pd
import numpy as np
from typing import Literal, Optional, Tuple, List
from dataclasses import dataclass
from datetime import datetime, timedelta
import sys
import math

sys.path.append('/Users/koenmarijt/Documents/Projects/ahold-options')
from src.iv_model import (
    IVSimulator, IVModelParams,
    black_scholes_price, black_scholes_greeks,
    decompose_pnl
)


@dataclass
class OptionLeg:
    """Single option leg in a strategy"""
    option_type: Literal['call', 'put']
    strike: float
    quantity: float  # Continuous contracts (not int)
    entry_price: float
    entry_iv: float
    entry_greeks: dict
    

@dataclass
class TradeResult:
    """Result of a single trade"""
    entry_date: datetime
    exit_date: datetime
    strategy: str
    
    # Entry details
    entry_spot: float
    entry_iv_atm: float
    entry_cost: float
    entry_dte: int
    
    # Exit details
    exit_spot: float
    exit_iv_atm: float
    exit_value: float
    exit_dte: int
    
    # P&L
    gross_pnl: float
    gross_return: float
    net_pnl: float  # After transaction costs
    net_return: float
    
    # P&L decomposition
    vega_pnl: float
    gamma_pnl: float
    theta_pnl: float
    
    # Trade metadata
    signal_strength: float
    entry_rv: float
    exit_rv: float
    days_held: int
    hit_exit_rule: Optional[str] = None


class OptionBacktester:
    """
    Realistic option strategy backtester.
    
    Instead of using RV as IV, this:
    1. Simulates realistic ATM IV path
    2. Models skew for OTM strikes
    3. Reprices using sticky-delta
    4. Decomposes P&L by Greeks
    """
    
    def __init__(
        self,
        iv_params: Optional[IVModelParams] = None,
        risk_free_rate: float = 0.02,
        seed: int = 42
    ):
        self.iv_sim = IVSimulator(iv_params, seed=seed)
        self.risk_free_rate = risk_free_rate
        self.trades: List[TradeResult] = []
        
    def prepare_iv_surface(
        self,
        df: pd.DataFrame,
        rv_col: str = 'rv_20',
        maturity_days: int = 120
    ) -> pd.DataFrame:
        """
        Generate realistic IV surface for backtesting.
        
        Args:
            df: DataFrame with OHLCV + RV data
            rv_col: Column name for realized volatility
            maturity_days: Constant maturity to model (e.g., 120 days)
        
        Returns:
            DataFrame with additional iv_atm column
        """
        df = df.copy()
        
        # Calibrate IV model to historical RV
        print("Calibrating IV model to historical data...")
        self.iv_sim.fit_to_historical_rv(df, rv_col)
        
        # Generate ATM IV path
        print(f"Generating {maturity_days}D ATM IV path...")
        df[f'iv_atm_{maturity_days}d'] = self.iv_sim.simulate_atm_iv_path(
            df, rv_col, maturity_days
        )
        
        # Add VRP column for analysis
        df['vrp'] = df[f'iv_atm_{maturity_days}d'] - df[rv_col]
        
        return df
    
    def create_straddle(
        self,
        spot: float,
        atm_iv: float,
        dte: int,
        position_size: float,
        spot_return_recent: float = 0.0
    ) -> Tuple[List[OptionLeg], float]:
        """
        Create ATM straddle position.
        
        Args:
            spot: Current spot price
            atm_iv: ATM implied volatility (%)
            dte: Days to expiration
            position_size: Dollar amount to invest
            spot_return_recent: Recent spot return for skew adjustment
        
        Returns:
            (legs, total_cost)
        """
        T = dte / 365
        q = self.iv_sim.params.dividend_yield
        
        # Use forward for ATM strike (better for dividend-paying stocks)
        forward = spot * math.exp((self.risk_free_rate - q) * T)
        strike = forward  # ATM-forward
        
        iv_decimal = atm_iv / 100
        
        # Price call and put
        call_price = black_scholes_price(
            spot, strike, T, self.risk_free_rate, iv_decimal, 'call', q
        )
        put_price = black_scholes_price(
            spot, strike, T, self.risk_free_rate, iv_decimal, 'put', q
        )
        
        total_cost = call_price + put_price
        contracts = position_size / total_cost
        
        # Calculate Greeks
        call_greeks = black_scholes_greeks(
            spot, strike, T, self.risk_free_rate, iv_decimal, 'call', q
        )
        put_greeks = black_scholes_greeks(
            spot, strike, T, self.risk_free_rate, iv_decimal, 'put', q
        )
        
        legs = [
            OptionLeg('call', strike, contracts, call_price, atm_iv, call_greeks),
            OptionLeg('put', strike, contracts, put_price, atm_iv, put_greeks)
        ]
        
        return legs, total_cost * contracts
    
    def create_strangle(
        self,
        spot: float,
        atm_iv: float,
        dte: int,
        position_size: float,
        otm_pct: float = 0.05,  # 5% OTM
        spot_return_recent: float = 0.0
    ) -> Tuple[List[OptionLeg], float]:
        """
        Create OTM strangle position.
        
        Args:
            otm_pct: How far OTM (e.g., 0.05 = 5%)
            spot_return_recent: Recent spot return for skew adjustment
        
        Returns:
            (legs, total_cost)
        """
        T = dte / 365
        q = self.iv_sim.params.dividend_yield
        
        # Set strikes
        call_strike = spot * (1 + otm_pct)
        put_strike = spot * (1 - otm_pct)
        
        # Get skewed IVs with dynamic adjustment
        call_moneyness = call_strike / spot
        put_moneyness = put_strike / spot
        
        call_iv = self.iv_sim.get_skewed_iv(atm_iv, call_moneyness, 'call', spot_return_recent)
        put_iv = self.iv_sim.get_skewed_iv(atm_iv, put_moneyness, 'put', spot_return_recent)
        
        # Price options
        call_price = black_scholes_price(
            spot, call_strike, T, self.risk_free_rate, call_iv / 100, 'call', q
        )
        put_price = black_scholes_price(
            spot, put_strike, T, self.risk_free_rate, put_iv / 100, 'put', q
        )
        
        total_cost = call_price + put_price
        contracts = position_size / total_cost
        
        # Calculate Greeks
        call_greeks = black_scholes_greeks(
            spot, call_strike, T, self.risk_free_rate, call_iv / 100, 'call', q
        )
        put_greeks = black_scholes_greeks(
            spot, put_strike, T, self.risk_free_rate, put_iv / 100, 'put', q
        )
        
        legs = [
            OptionLeg('call', call_strike, contracts, call_price, call_iv, call_greeks),
            OptionLeg('put', put_strike, contracts, put_price, put_iv, put_greeks)
        ]
        
        return legs, total_cost * contracts

    def create_single_leg(
        self,
        *,
        spot: float,
        atm_iv: float,
        dte: int,
        position_size: float,
        option_type: Literal['call', 'put'],
        side: Literal['long', 'short'] = 'long',
        otm_pct: float = 0.0,
        spot_return_recent: float = 0.0,
    ) -> Tuple[List[OptionLeg], float]:
        """Create a single-leg option position (buy/sell call/put).

        Notes:
        - Strike is defined off spot: call strike = S*(1+otm), put strike = S*(1-otm)
        - Skewed IV is applied using the existing IV skew model.
        - For short positions, quantity is negative; entry_cost is net premium (negative).
        """
        T = dte / 365
        q = self.iv_sim.params.dividend_yield

        if option_type == 'call':
            strike = spot * (1 + otm_pct)
        else:
            strike = spot * (1 - otm_pct)

        moneyness = strike / spot
        leg_iv = self.iv_sim.get_skewed_iv(atm_iv, moneyness, option_type, spot_return_recent)

        price = black_scholes_price(
            spot, strike, T, self.risk_free_rate, leg_iv / 100, option_type, q
        )

        # shares/contracts are continuous via quantity; we use position_size dollars notionally.
        contracts = position_size / price if price > 0 else 0.0
        qty = contracts if side == 'long' else -contracts

        greeks = black_scholes_greeks(
            spot, strike, T, self.risk_free_rate, leg_iv / 100, option_type, q
        )

        leg = OptionLeg(option_type, strike, qty, price, leg_iv, greeks)
        entry_cost = price * qty  # negative for shorts (premium received)
        return [leg], entry_cost

    def create_vertical_spread(
        self,
        *,
        spot: float,
        atm_iv: float,
        dte: int,
        position_size: float,
        option_type: Literal['call', 'put'],
        width_pct: float = 0.05,
        otm_pct_short: float = 0.05,
        spread_type: Literal['debit', 'credit'] = 'debit',
        spot_return_recent: float = 0.0,
    ) -> Tuple[List[OptionLeg], float]:
        """Create a simple vertical spread.

        Call spreads:
          - debit:  long lower strike (less OTM), short higher strike (more OTM)
          - credit: short lower strike, long higher strike

        Put spreads (bear put / bull put):
          - debit:  long higher strike, short lower strike
          - credit: short higher strike, long lower strike

        Parameters are expressed in spot percentages for simplicity.
        """
        if width_pct <= 0:
            raise ValueError('width_pct must be > 0')

        T = dte / 365
        q = self.iv_sim.params.dividend_yield

        # Define strikes in terms of moneyness.
        if option_type == 'call':
            k_short = spot * (1 + otm_pct_short)
            k_long = spot * (1 + max(0.0, otm_pct_short - width_pct))
            k_far = spot * (1 + otm_pct_short + width_pct)
            near, far = (k_long, k_far)
        else:
            k_short = spot * (1 - otm_pct_short)
            k_long = spot * (1 - max(0.0, otm_pct_short - width_pct))
            k_far = spot * (1 - (otm_pct_short + width_pct))
            # For puts, "higher strike" is closer to spot.
            near, far = (k_long, k_far)

        # For debit spread we want net cost > 0; for credit net cost < 0.
        if spread_type == 'debit':
            strike_long = near
            strike_short = far
            qty_long_sign = 1.0
            qty_short_sign = -1.0
        else:  # credit
            strike_long = far
            strike_short = near
            qty_long_sign = 1.0
            qty_short_sign = -1.0

        def price_leg(strike: float) -> tuple[float, float, dict]:
            m = strike / spot
            iv_leg = self.iv_sim.get_skewed_iv(atm_iv, m, option_type, spot_return_recent)
            px = black_scholes_price(spot, strike, T, self.risk_free_rate, iv_leg / 100, option_type, q)
            gr = black_scholes_greeks(spot, strike, T, self.risk_free_rate, iv_leg / 100, option_type, q)
            return px, iv_leg, gr

        px_long, iv_long, gr_long = price_leg(strike_long)
        px_short, iv_short, gr_short = price_leg(strike_short)

        net_premium_per_spread = px_long * qty_long_sign + px_short * qty_short_sign
        if net_premium_per_spread == 0:
            spreads = 0.0
        else:
            spreads = position_size / abs(net_premium_per_spread)
            # If caller asked for position_size dollars 'invested', keep sign by spread type
            # (debit spends cash, credit receives cash)
            if spread_type == 'credit':
                spreads = spreads  # premium received is implicit in sign below

        legs = [
            OptionLeg(option_type, strike_long, spreads * qty_long_sign, px_long, iv_long, gr_long),
            OptionLeg(option_type, strike_short, spreads * qty_short_sign, px_short, iv_short, gr_short),
        ]
        entry_cost = net_premium_per_spread * spreads
        return legs, entry_cost
    
    def reprice_position(
        self,
        legs: List[OptionLeg],
        entry_spot: float,
        exit_spot: float,
        exit_atm_iv: float,
        exit_dte: int,
        days_held: int,
        spot_return_recent: float = 0.0
    ) -> Tuple[float, dict]:
        """
        Reprice position at exit using realistic IV dynamics.
        
        Args:
            legs: List of option legs
            entry_spot: Spot at entry
            exit_spot: Spot at exit
            exit_atm_iv: ATM IV at exit
            exit_dte: Days to expiration at exit
            days_held: Actual number of days held
            spot_return_recent: Recent spot return for skew adjustment
        
        Returns:
            (total_exit_value, pnl_decomposition)
        """
        T_exit = exit_dte / 365
        q = self.iv_sim.params.dividend_yield
        total_value = 0.0
        
        # Aggregate P&L decomposition
        total_vega_pnl = 0.0
        total_gamma_pnl = 0.0
        total_theta_pnl = 0.0
        
        for leg in legs:
            # Sticky-delta approximation: keep entry moneyness
            entry_moneyness = leg.strike / entry_spot
            equiv_strike = entry_moneyness * exit_spot  # "Move" strike with spot
            effective_moneyness = equiv_strike / exit_spot  # = entry_moneyness
            
            # Get exit IV using skew model with dynamic adjustment
            exit_iv = self.iv_sim.get_skewed_iv(
                exit_atm_iv, effective_moneyness, leg.option_type, spot_return_recent
            )
            
            # Reprice option at actual strike
            if exit_dte <= 0:
                # At expiration: intrinsic value only
                if leg.option_type == 'call':
                    exit_price = max(0.0, exit_spot - leg.strike)
                else:
                    exit_price = max(0.0, leg.strike - exit_spot)
            else:
                exit_price = black_scholes_price(
                    exit_spot, leg.strike, T_exit,
                    self.risk_free_rate, exit_iv / 100, leg.option_type, q
                )
            
            # Calculate this leg's value
            leg_value = exit_price * leg.quantity
            total_value += leg_value
            
            # Decompose P&L for this leg
            entry_value = leg.entry_price * leg.quantity
            leg_pnl = leg_value - entry_value
            
            # Greek contributions (using actual days_held from caller)
            delta_iv = exit_iv - leg.entry_iv
            delta_spot = exit_spot - entry_spot
            
            total_vega_pnl += leg.entry_greeks['vega'] * delta_iv * leg.quantity
            total_gamma_pnl += 0.5 * leg.entry_greeks['gamma'] * (delta_spot ** 2) * leg.quantity
            total_theta_pnl += leg.entry_greeks['theta'] * days_held * leg.quantity
        
        pnl_decomp = {
            'vega_pnl': total_vega_pnl,
            'gamma_pnl': total_gamma_pnl,
            'theta_pnl': total_theta_pnl
        }
        
        return total_value, pnl_decomp
    
    def backtest_strategy(
        self,
        df: pd.DataFrame,
        signals: pd.DataFrame,
        strategy: Literal['straddle', 'strangle'] = 'straddle',
        dte_target: int = 120,
        hold_days: int = 30,
        position_size: float = 10000,
        otm_pct: float = 0.05,
        max_loss_pct: Optional[float] = None,  # Stop loss (e.g., 0.5 = 50% loss)
        take_profit_pct: Optional[float] = None,  # Take profit (e.g., 0.5 = 50% gain)
        iv_col: str = 'iv_atm_120d'
    ) -> pd.DataFrame:
        """
        Backtest option strategy with realistic IV.
        
        Args:
            df: DataFrame with OHLCV, RV, and IV data
            signals: DataFrame with entry signals (must have 'date' column)
            strategy: 'straddle' or 'strangle'
            dte_target: Target days to expiration at entry
            hold_days: Maximum days to hold
            position_size: Dollar size per trade
            otm_pct: For strangles, how far OTM (e.g., 0.05 = 5%)
            max_loss_pct: Optional stop loss
            take_profit_pct: Optional take profit
            iv_col: Column name for ATM IV
        
        Returns:
            DataFrame of trade results
        """
        print(f"\nBacktesting {strategy.upper()} strategy...")
        print(f"  DTE target: {dte_target}")
        print(f"  Hold period: {hold_days} days")
        print(f"  Position size: ${position_size:,.0f}")
        if strategy == 'strangle':
            print(f"  OTM level: {otm_pct*100:.1f}%")
        print()
        
        df = df.set_index('date')
        trades = []
        
        for _, signal in signals.iterrows():
            entry_date = signal['date']
            
            if entry_date not in df.index:
                continue
            
            entry_row = df.loc[entry_date]
            entry_spot = entry_row['close']
            entry_atm_iv = entry_row[iv_col]
            entry_rv = entry_row.get('rv_20', np.nan)
            
            # Calculate recent spot return for skew adjustment (5-day return)
            recent_prices = df[df.index <= entry_date]['close'].tail(6)
            if len(recent_prices) >= 2:
                spot_return_recent = (recent_prices.iloc[-1] / recent_prices.iloc[0] - 1)
            else:
                spot_return_recent = 0.0
            
            # Create position
            if strategy == 'straddle':
                legs, entry_cost = self.create_straddle(
                    entry_spot, entry_atm_iv, dte_target, position_size, spot_return_recent
                )
            else:  # strangle
                legs, entry_cost = self.create_strangle(
                    entry_spot, entry_atm_iv, dte_target, position_size, otm_pct, spot_return_recent
                )
            
            # Find exit date
            future_dates = df[df.index > entry_date].head(hold_days)
            
            if len(future_dates) == 0:
                continue  # Not enough data
            
            # Check each day for exit conditions
            exit_date = None
            exit_rule = 'time'
            
            for check_date in future_dates.index:
                check_row = df.loc[check_date]
                days_held = (check_date - entry_date).days
                dte_remaining = max(0, dte_target - days_held)
                
                # Calculate recent return for skew
                recent_prices = df[df.index <= check_date]['close'].tail(6)
                if len(recent_prices) >= 2:
                    spot_return_check = (recent_prices.iloc[-1] / recent_prices.iloc[0] - 1)
                else:
                    spot_return_check = 0.0
                
                # Reprice position (pass days_held!)
                exit_value, pnl_decomp = self.reprice_position(
                    legs, entry_spot, check_row['close'],
                    check_row[iv_col], dte_remaining, days_held, spot_return_check
                )
                
                current_pnl = exit_value - entry_cost
                current_return = current_pnl / entry_cost
                
                # Check exit rules
                if max_loss_pct and current_return <= -max_loss_pct:
                    exit_date = check_date
                    exit_rule = 'stop_loss'
                    break
                
                if take_profit_pct and current_return >= take_profit_pct:
                    exit_date = check_date
                    exit_rule = 'take_profit'
                    break
            
            # If no early exit, use end of hold period
            if exit_date is None:
                exit_date = future_dates.index[-1]
            
            # Final exit calculation
            exit_row = df.loc[exit_date]
            days_held = (exit_date - entry_date).days
            exit_dte = max(0, dte_target - days_held)
            
            # Calculate recent return for final skew
            recent_prices = df[df.index <= exit_date]['close'].tail(6)
            if len(recent_prices) >= 2:
                spot_return_exit = (recent_prices.iloc[-1] / recent_prices.iloc[0] - 1)
            else:
                spot_return_exit = 0.0
            
            exit_value, pnl_decomp = self.reprice_position(
                legs, entry_spot, exit_row['close'],
                exit_row[iv_col], exit_dte, days_held, spot_return_exit
            )
            
            # Calculate returns
            gross_pnl = exit_value - entry_cost
            gross_return = gross_pnl / entry_cost
            
            # Transaction costs (bid-ask spread on entry and exit)
            tc_rate = self.iv_sim.params.bid_ask_spread_pct
            transaction_costs = entry_cost * tc_rate * 2  # Entry + exit
            
            net_pnl = gross_pnl - transaction_costs
            net_return = net_pnl / entry_cost
            
            # Create trade result
            trade = TradeResult(
                entry_date=entry_date,
                exit_date=exit_date,
                strategy=strategy,
                entry_spot=entry_spot,
                entry_iv_atm=entry_atm_iv,
                entry_cost=entry_cost,
                entry_dte=dte_target,
                exit_spot=exit_row['close'],
                exit_iv_atm=exit_row[iv_col],
                exit_value=exit_value,
                exit_dte=exit_dte,
                gross_pnl=gross_pnl,
                gross_return=gross_return,
                net_pnl=net_pnl,
                net_return=net_return,
                vega_pnl=pnl_decomp['vega_pnl'],
                gamma_pnl=pnl_decomp['gamma_pnl'],
                theta_pnl=pnl_decomp['theta_pnl'],
                signal_strength=signal.get('pred_proba', 1.0),
                entry_rv=entry_rv,
                exit_rv=exit_row.get('rv_20', np.nan),
                days_held=days_held,
                hit_exit_rule=exit_rule if exit_rule != 'time' else None
            )
            
            trades.append(trade)
        
        # Convert to DataFrame
        results_df = pd.DataFrame([vars(t) for t in trades])
        self.trades = trades
        
        return results_df
    
    def print_summary(self, results_df: pd.DataFrame):
        """Print backtest summary statistics"""
        if len(results_df) == 0:
            print("No trades to summarize!")
            return
        
        print("\n" + "="*80)
        print("BACKTEST SUMMARY - REALISTIC IV MODEL")
        print("="*80)
        
        print(f"\nTotal trades: {len(results_df)}")
        print(f"Date range: {results_df['entry_date'].min()} to {results_df['exit_date'].max()}")
        
        print(f"\n{'Performance Metrics':-^80}")
        
        # Returns
        mean_net_return = results_df['net_return'].mean()
        median_net_return = results_df['net_return'].median()
        win_rate = (results_df['net_pnl'] > 0).mean()
        
        print(f"Mean return: {mean_net_return:+.2%}")
        print(f"Median return: {median_net_return:+.2%}")
        print(f"Win rate: {win_rate:.1%}")
        
        # Risk metrics
        sharpe = mean_net_return / results_df['net_return'].std() if results_df['net_return'].std() > 0 else 0
        max_loss = results_df['net_return'].min()
        max_win = results_df['net_return'].max()
        
        print(f"\nSharpe ratio: {sharpe:.2f}")
        print(f"Max win: {max_win:+.2%}")
        print(f"Max loss: {max_loss:+.2%}")
        
        # P&L decomposition
        print(f"\n{'P&L Attribution (Average per trade)':-^80}")
        total_pnl = results_df[['vega_pnl', 'gamma_pnl', 'theta_pnl']].sum(axis=1)
        
        vega_contribution = results_df['vega_pnl'].sum() / total_pnl.sum()
        gamma_contribution = results_df['gamma_pnl'].sum() / total_pnl.sum()
        theta_contribution = results_df['theta_pnl'].sum() / total_pnl.sum()
        
        print(f"Vega (IV changes): {vega_contribution:+.1%}")
        print(f"Gamma (spot moves): {gamma_contribution:+.1%}")
        print(f"Theta (time decay): {theta_contribution:+.1%}")
        
        # IV statistics
        print(f"\n{'IV vs RV Analysis':-^80}")
        entry_vrp = (results_df['entry_iv_atm'] - results_df['entry_rv']).mean()
        iv_change = (results_df['exit_iv_atm'] - results_df['entry_iv_atm']).mean()
        rv_change = (results_df['exit_rv'] - results_df['entry_rv']).mean()
        
        print(f"Average entry VRP (IV - RV): {entry_vrp:+.2f}%")
        print(f"Average IV change: {iv_change:+.2f}%")
        print(f"Average RV change: {rv_change:+.2f}%")
        
        # Exit rules
        if 'hit_exit_rule' in results_df.columns:
            exit_counts = results_df['hit_exit_rule'].value_counts()
            print(f"\n{'Exit Rules':-^80}")
            for rule, count in exit_counts.items():
                print(f"{rule or 'Time exit'}: {count} ({count/len(results_df):.1%})")
        
        print("="*80)


# Example usage
if __name__ == '__main__':
    from sqlalchemy import create_engine
    from src.config import settings
    
    print("Loading data...")
    engine = create_engine(settings.database_url)
    
    query = """
    SELECT 
        bo.trade_date as date,
        bo.close,
        fti.realized_volatility_20 as rv_20
    FROM bronze_ohlcv bo
    JOIN fact_technical_indicators fti ON bo.trade_date = fti.trade_date AND bo.ticker = fti.ticker
    WHERE bo.ticker = 'AD.AS' AND bo.trade_date >= '2020-01-01'
    ORDER BY bo.trade_date
    """
    
    df = pd.read_sql(query, engine)
    df['date'] = pd.to_datetime(df['date'])
    engine.dispose()
    
    # Initialize backtester
    backtester = OptionBacktester()
    
    # Prepare IV surface
    df = backtester.prepare_iv_surface(df)
    
    # Create simple signals (every 60 days for demo)
    signals = df[::60][['date']].copy()
    
    # Backtest straddle
    results = backtester.backtest_strategy(
        df, signals,
        strategy='straddle',
        dte_target=120,
        hold_days=30,
        position_size=10000
    )
    
    backtester.print_summary(results)
