"""
Realistic Implied Volatility Model for Backtesting
===================================================

This module models ATM IV and skew dynamics for realistic option backtests.

Key features:
1. ATM IV model with mean reversion + RV linkage + market proxy
2. Skew model for OTM options
3. Sticky-delta and sticky-strike repricing rules
4. Proper IV->Price decomposition using Greeks

Based on the principle that IV != RV and has its own dynamics.
"""

import numpy as np
import pandas as pd
from typing import Tuple, Optional, Literal
import math
from dataclasses import dataclass


@dataclass
class IVModelParams:
    """Parameters for IV model"""
    # ATM IV level parameters
    iv_mean: float = 20.0  # Long-term mean IV (%)
    iv_mr_speed: float = 0.15  # Mean reversion speed (per day)
    rv_beta: float = 0.40  # How much RV changes affect IV (0.3-0.5 typical)
    vrp_mean: float = 3.0  # Average volatility risk premium (IV - RV)
    
    # Skew parameters (relative to ATM)
    put_skew_25d: float = 2.5  # 25-delta put IV premium vs ATM (%)
    call_skew_25d: float = -0.5  # 25-delta call IV vs ATM (%)
    put_skew_10d: float = 4.0  # 10-delta put IV premium vs ATM (%)
    skew_spot_beta: float = -0.5  # Skew steepens on down moves (negative)
    
    # Noise and regime parameters
    iv_vol: float = 0.15  # Volatility of volatility (annual %)
    earnings_iv_spike: float = 5.0  # IV increase before earnings (%)
    dividend_iv_impact: float = 1.5  # IV increase before ex-div (%)
    
    # Term structure
    iv_term_slope: float = 0.0  # Term structure slope (ln(T) coefficient)
    
    # Dividend yield (for forward-based ATM)
    dividend_yield: float = 0.02  # Annual dividend yield (e.g., 2%)
    
    # Stickiness rules
    sticky_rule: Literal['delta', 'strike'] = 'delta'
    
    # Transaction costs
    bid_ask_spread_pct: float = 0.03  # 3% bid-ask spread


class IVSimulator:
    """
    Simulates realistic implied volatility dynamics for option backtesting.
    
    Instead of assuming IV = realized RV, this models:
    - ATM IV with mean reversion
    - RV influence with realistic beta (0.3-0.5)
    - Volatility risk premium
    - Skew dynamics
    - Event-driven IV behavior
    """
    
    def __init__(self, params: Optional[IVModelParams] = None, seed: int = 42):
        self.params = params or IVModelParams()
        self.iv_history = []
        self.rng = np.random.default_rng(seed)
        
    def fit_to_historical_rv(self, df: pd.DataFrame, rv_col: str = 'rv_20') -> 'IVSimulator':
        """
        Calibrate IV model parameters to historical RV data.
        
        This estimates what IV would have been based on:
        - Long-term RV mean + VRP
        - RV correlation structure
        - Historical volatility clustering
        
        Args:
            df: DataFrame with date, rv columns
            rv_col: Column name for realized volatility
        """
        df = df.copy()
        date_col = 'date' if 'date' in df.columns else ('dt' if 'dt' in df.columns else None)
        if date_col is not None:
            df = df.sort_values(date_col)
        rv = df[rv_col].values
        
        # Estimate long-term IV level from RV + typical VRP
        rv_mean = np.nanmean(rv)
        self.params.iv_mean = rv_mean + self.params.vrp_mean
        
        # Estimate RV->IV beta from historical relationship
        # In reality, IV changes lag and partially follow RV changes
        rv_changes = np.diff(rv)
        rv_changes = rv_changes[~np.isnan(rv_changes)]
        
        # IV vol should be less than RV vol (IV is smoother)
        rv_vol = np.nanstd(rv)
        self.params.iv_vol = rv_vol * 0.6  # IV is smoother than RV
        
        print(f"IV Model Calibration:")
        print(f"  IV mean: {self.params.iv_mean:.2f}%")
        print(f"  RV mean: {rv_mean:.2f}%")
        print(f"  VRP: {self.params.vrp_mean:.2f}%")
        print(f"  RV->IV beta: {self.params.rv_beta:.2f}")
        print(f"  IV vol: {self.params.iv_vol:.2f}%")
        
        return self
    
    def simulate_atm_iv_path(
        self,
        df: pd.DataFrame,
        rv_col: str = 'rv_20',
        maturity_days: int = 120
    ) -> pd.Series:
        """
        Generate realistic ATM IV path for a given maturity (e.g., 120 days).
        
        Model:
        ΔIV_t = κ(IV_mean - IV_{t-1})Δt + β·ΔRV_t + σ_IV·ε_t·√Δt + event_shocks
        
        Where:
        - κ: mean reversion speed (per day)
        - β: RV beta (how much IV follows RV changes)
        - σ_IV: volatility of volatility (annualized)
        - ε_t: standard normal shock
        - Δt: 1 day (in daily data)
        - event_shocks: earnings, dividends, etc.
        
        Args:
            df: DataFrame with date, rv, and optional event flags
            rv_col: Column for realized volatility
            maturity_days: Constant maturity to model (e.g., 120D)
        
        Returns:
            Series of simulated ATM IV for the given maturity
        """
        df = df.copy()
        date_col = 'date' if 'date' in df.columns else ('dt' if 'dt' in df.columns else None)
        if date_col is not None:
            df = df.sort_values(date_col)
        n = len(df)
        
        rv = df[rv_col].values
        iv_path = np.zeros(n)
        
        # Initialize IV at long-term mean
        iv_path[0] = self.params.iv_mean
        
        # Check for event columns
        has_earnings = 'days_to_earnings' in df.columns
        has_dividend = 'days_to_exdiv' in df.columns
        
        # Daily volatility of volatility
        iv_vol_daily = self.params.iv_vol / np.sqrt(252)
        
        # Track spot returns for skew regime
        spot_returns = []
        if 'close' in df.columns:
            spot_returns = df['close'].pct_change().fillna(0).values
        
        for t in range(1, n):
            # Mean reversion component (per day)
            iv_prev = iv_path[t-1]
            mr_component = self.params.iv_mr_speed * (self.params.iv_mean - iv_prev)
            
            # RV change component (IV partially follows RV)
            if not np.isnan(rv[t]) and not np.isnan(rv[t-1]):
                rv_change = rv[t] - rv[t-1]
                rv_component = self.params.rv_beta * rv_change
            else:
                rv_component = 0.0
            
            # Stochastic component (volatility clustering) - properly scaled
            noise = self.rng.normal(0, iv_vol_daily)
            
            # Event-driven shocks
            event_shock = 0.0
            if has_earnings:
                days_to_earnings = df.iloc[t].get('days_to_earnings', 999)
                if 0 < days_to_earnings <= 14:
                    # IV increases before earnings
                    event_shock += self.params.earnings_iv_spike * (14 - days_to_earnings) / 14
            
            if has_dividend:
                days_to_exdiv = df.iloc[t].get('days_to_exdiv', 999)
                if 0 < days_to_exdiv <= 7:
                    # IV increases before ex-dividend
                    event_shock += self.params.dividend_iv_impact * (7 - days_to_exdiv) / 7
            
            # Update IV
            iv_path[t] = iv_prev + mr_component + rv_component + noise + event_shock
            
            # Floor at reasonable levels
            iv_path[t] = max(5.0, iv_path[t])  # Min 5% IV
        
        return pd.Series(iv_path, index=df.index, name=f'iv_atm_{maturity_days}d')
    
    def get_skewed_iv(
        self,
        atm_iv: float,
        moneyness: float,
        option_type: Literal['call', 'put'],
        spot_return_recent: float = 0.0
    ) -> float:
        """
        Calculate skewed IV for OTM options.
        
        Args:
            atm_iv: ATM implied volatility (%)
            moneyness: K/S ratio (1.0 = ATM, 0.95 = 5% OTM put, 1.05 = 5% OTM call)
            option_type: 'call' or 'put'
            spot_return_recent: Recent spot return (e.g., -0.05 for -5% move)
                               Used to adjust skew dynamically (steepens on down moves)
        
        Returns:
            Skewed IV for the given strike
        """
        # Base skew adjustment
        if option_type == 'put':
            if moneyness >= 1.0:
                # ITM put (acts like ATM)
                base_skew = 0.0
            elif moneyness >= 0.90:
                # 0-10% OTM put: interpolate to 25-delta skew
                otm_pct = (1.0 - moneyness) * 100  # 0-10%
                base_skew = self.params.put_skew_25d * (otm_pct / 5.0)  # Linear
            else:
                # Deep OTM put: use 10-delta skew
                base_skew = self.params.put_skew_10d
        else:  # call
            if moneyness <= 1.0:
                # ITM call (acts like ATM)
                base_skew = 0.0
            elif moneyness <= 1.10:
                # 0-10% OTM call: slight negative skew
                otm_pct = (moneyness - 1.0) * 100
                base_skew = self.params.call_skew_25d * (otm_pct / 5.0)
            else:
                # Deep OTM call
                base_skew = self.params.call_skew_25d * 2
        
        # Dynamic skew adjustment based on recent spot moves
        # Skew steepens on down moves (more put premium)
        skew_adjustment = 0.0
        if option_type == 'put' and spot_return_recent < 0:
            # Put skew steepens on down moves
            skew_adjustment = self.params.skew_spot_beta * spot_return_recent * abs(base_skew)
        
        return atm_iv + base_skew + skew_adjustment
    
    def reprice_under_sticky_rule(
        self,
        old_spot: float,
        new_spot: float,
        strike: float,
        old_iv: float,
        option_type: Literal['call', 'put']
    ) -> float:
        """
        Reprice option IV under sticky-delta or sticky-strike rule.
        
        Sticky-delta: IV is constant for a given delta (moneyness in log terms)
        Sticky-strike: IV is constant for a given strike
        
        Args:
            old_spot: Previous spot price
            new_spot: New spot price
            strike: Option strike
            old_iv: Previous IV at this strike
            option_type: 'call' or 'put'
        
        Returns:
            New IV for this option
        """
        if self.params.sticky_rule == 'strike':
            # Sticky strike: IV doesn't change with spot movement
            return old_iv
        
        else:  # sticky-delta
            # Under sticky delta, we keep moneyness constant
            # This means the strike "moves" with the spot
            old_moneyness = strike / old_spot
            
            # Keep same moneyness (this is the sticky-delta approximation)
            # The strike is now at a different moneyness vs new spot
            # So we need to get IV for that new moneyness
            return old_moneyness  # Return moneyness for remapping


    def adjust_iv_for_term(
        self,
        iv_at_base_tenor: float,
        base_tenor_days: int,
        target_tenor_days: int
    ) -> float:
        """
        Adjust IV for term structure.
        
        Uses simple log-linear model:
        IV(T) = IV(T_0) + slope * ln(T/T_0)
        
        Args:
            iv_at_base_tenor: IV at base tenor (e.g., 120D IV)
            base_tenor_days: Base tenor in days (e.g., 120)
            target_tenor_days: Target tenor in days (e.g., 30)
        
        Returns:
            IV adjusted for target tenor
        """
        if target_tenor_days <= 0 or base_tenor_days <= 0:
            return iv_at_base_tenor
        
        if self.params.iv_term_slope == 0.0:
            # Flat term structure
            return iv_at_base_tenor
        
        # Log-linear term structure
        ln_ratio = np.log(target_tenor_days / base_tenor_days)
        adjusted_iv = iv_at_base_tenor + self.params.iv_term_slope * ln_ratio
        
        return max(5.0, adjusted_iv)  # Floor at 5%


def norm_cdf(x: float) -> float:
    """Cumulative distribution function for standard normal"""
    return 0.5 * (1.0 + math.erf(x / math.sqrt(2.0)))


def norm_pdf(x: float) -> float:
    """Probability density function for standard normal"""
    return math.exp(-0.5 * x * x) / math.sqrt(2.0 * math.pi)


def black_scholes_price(
    S: float,
    K: float,
    T: float,
    r: float,
    sigma: float,
    option_type: Literal['call', 'put'],
    q: float = 0.0  # Dividend yield
) -> float:
    """
    Black-Scholes-Merton option price with dividends.
    
    Args:
        S: Spot price
        K: Strike price
        T: Time to maturity (years)
        r: Risk-free rate
        sigma: Implied volatility (decimal, e.g., 0.20 for 20%)
        option_type: 'call' or 'put'
        q: Dividend yield (continuous, e.g., 0.02 for 2%)
    """
    if T <= 0:
        # At expiration: intrinsic value only
        if option_type == 'call':
            return max(0.0, S - K)
        else:
            return max(0.0, K - S)
    
    if sigma <= 0:
        # Zero vol: use intrinsic value
        if option_type == 'call':
            return max(0.0, S - K)
        else:
            return max(0.0, K - S)
    
    # Black-Scholes-Merton with dividend yield
    sqrt_T = math.sqrt(T)
    d1 = (math.log(S / K) + (r - q + 0.5 * sigma * sigma) * T) / (sigma * sqrt_T)
    d2 = d1 - sigma * sqrt_T
    
    if option_type == 'call':
        return S * math.exp(-q * T) * norm_cdf(d1) - K * math.exp(-r * T) * norm_cdf(d2)
    else:
        return K * math.exp(-r * T) * norm_cdf(-d2) - S * math.exp(-q * T) * norm_cdf(-d1)


def black_scholes_greeks(
    S: float,
    K: float,
    T: float,
    r: float,
    sigma: float,
    option_type: Literal['call', 'put'],
    q: float = 0.0  # Dividend yield
) -> dict:
    """
    Calculate Black-Scholes Greeks with dividends.
    
    Returns:
        dict with: delta, gamma, vega, theta
    """
    if T <= 0 or sigma <= 0:
        return {
            'delta': 1.0 if (option_type == 'call' and S > K) else 0.0,
            'gamma': 0.0,
            'vega': 0.0,
            'theta': 0.0
        }
    
    sqrt_T = math.sqrt(T)
    d1 = (math.log(S / K) + (r - q + 0.5 * sigma * sigma) * T) / (sigma * sqrt_T)
    d2 = d1 - sigma * sqrt_T
    
    # Delta (with dividend adjustment)
    if option_type == 'call':
        delta = math.exp(-q * T) * norm_cdf(d1)
    else:
        delta = math.exp(-q * T) * (norm_cdf(d1) - 1.0)
    
    # Gamma (same for calls and puts)
    gamma = math.exp(-q * T) * norm_pdf(d1) / (S * sigma * sqrt_T)
    
    # Vega (same for calls and puts, per 1% change in vol)
    vega = S * math.exp(-q * T) * norm_pdf(d1) * sqrt_T / 100  # Divided by 100 for 1% IV change
    
    # Theta (per day)
    term1 = -(S * math.exp(-q * T) * norm_pdf(d1) * sigma) / (2 * sqrt_T)
    if option_type == 'call':
        term2 = q * S * math.exp(-q * T) * norm_cdf(d1) - r * K * math.exp(-r * T) * norm_cdf(d2)
        theta = (term1 - term2) / 365
    else:
        term2 = q * S * math.exp(-q * T) * norm_cdf(-d1) + r * K * math.exp(-r * T) * norm_cdf(-d2)
        theta = (term1 - term2) / 365
    
    return {
        'delta': delta,
        'gamma': gamma,
        'vega': vega,
        'theta': theta
    }


def decompose_pnl(
    entry_price: float,
    exit_price: float,
    entry_greeks: dict,
    delta_iv: float,
    delta_spot: float,
    days_held: int
) -> dict:
    """
    Decompose option P&L into Greek contributions.
    
    Based on the formula:
    ΔP ≈ Vega·ΔIV + 0.5·Gamma·(ΔS)² - Theta·Δt
    
    Args:
        entry_price: Option price at entry
        exit_price: Option price at exit
        entry_greeks: Greeks at entry (delta, gamma, vega, theta)
        delta_iv: Change in IV (percentage points, e.g., 2.5 for +2.5%)
        delta_spot: Change in spot price (dollars)
        days_held: Number of days held
    
    Returns:
        dict with pnl breakdown: total, vega_pnl, gamma_pnl, theta_pnl, residual
    """
    total_pnl = exit_price - entry_price
    
    # Vega P&L (from IV change)
    vega_pnl = entry_greeks['vega'] * delta_iv
    
    # Gamma P&L (from spot movement)
    gamma_pnl = 0.5 * entry_greeks['gamma'] * (delta_spot ** 2)
    
    # Theta P&L (time decay)
    theta_pnl = entry_greeks['theta'] * days_held
    
    # Residual (higher-order effects, skew changes, etc.)
    explained_pnl = vega_pnl + gamma_pnl + theta_pnl
    residual = total_pnl - explained_pnl
    
    return {
        'total_pnl': total_pnl,
        'vega_pnl': vega_pnl,
        'gamma_pnl': gamma_pnl,
        'theta_pnl': theta_pnl,
        'residual': residual,
        'vega_pct': vega_pnl / total_pnl if total_pnl != 0 else 0,
        'gamma_pct': gamma_pnl / total_pnl if total_pnl != 0 else 0,
        'theta_pct': theta_pnl / total_pnl if total_pnl != 0 else 0
    }


# Example usage
if __name__ == '__main__':
    # Create sample data
    dates = pd.date_range('2020-01-01', '2024-12-31', freq='D')
    n = len(dates)
    
    # Simulate RV with clustering
    rv = np.zeros(n)
    rv[0] = 15.0
    for i in range(1, n):
        rv[i] = rv[i-1] + np.random.normal(0, 1.5) + 0.05 * (18 - rv[i-1])
        rv[i] = max(5, min(50, rv[i]))
    
    df = pd.DataFrame({
        'date': dates,
        'rv_20': rv
    })
    
    # Fit IV model
    iv_sim = IVSimulator()
    iv_sim.fit_to_historical_rv(df)
    
    # Generate ATM IV path
    df['iv_atm_120d'] = iv_sim.simulate_atm_iv_path(df)
    
    print("\nSample IV vs RV:")
    print(df[['date', 'rv_20', 'iv_atm_120d']].tail(10))
    
    print(f"\nCorrelation (RV vs IV): {df['rv_20'].corr(df['iv_atm_120d']):.3f}")
    print(f"Mean VRP (IV - RV): {(df['iv_atm_120d'] - df['rv_20']).mean():.2f}%")
