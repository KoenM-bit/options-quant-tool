"""
Silver Layer: Options with Greeks Calculation
Calculates implied volatility and Greeks using Black-Scholes model.
This runs in dbt Python model for proper separation of concerns.
"""

import pandas as pd
import numpy as np
from scipy.stats import norm
from scipy.optimize import brentq


def black_scholes_price(S, K, T, r, sigma, option_type='call'):
    """
    Calculate Black-Scholes option price.
    
    Args:
        S: Underlying price
        K: Strike price
        T: Time to expiration (years)
        r: Risk-free rate
        sigma: Volatility (annualized)
        option_type: 'call' or 'put'
    
    Returns:
        Option price
    """
    if T <= 0 or sigma <= 0:
        return None
    
    d1 = (np.log(S / K) + (r + 0.5 * sigma**2) * T) / (sigma * np.sqrt(T))
    d2 = d1 - sigma * np.sqrt(T)
    
    if option_type.lower() == 'call':
        price = S * norm.cdf(d1) - K * np.exp(-r * T) * norm.cdf(d2)
    else:  # put
        price = K * np.exp(-r * T) * norm.cdf(-d2) - S * norm.cdf(-d1)
    
    return price


def calculate_implied_volatility(option_price, S, K, T, r, option_type='call', max_iterations=100):
    """
    Calculate implied volatility using Brent's method.
    
    Args:
        option_price: Market price of option
        S: Underlying price
        K: Strike price
        T: Time to expiration (years)
        r: Risk-free rate
        option_type: 'call' or 'put'
    
    Returns:
        Implied volatility (or None if calculation fails)
    """
    if option_price <= 0 or S <= 0 or K <= 0 or T <= 0:
        return None
    
    # Check if option is too far out of the money
    intrinsic = max(S - K, 0) if option_type.lower() == 'call' else max(K - S, 0)
    if option_price < intrinsic * 0.9:  # Price below intrinsic value (likely stale data)
        return None
    
    try:
        # Define objective function
        def objective(sigma):
            return black_scholes_price(S, K, T, r, sigma, option_type) - option_price
        
        # Brent's method to find root (IV where BS price = market price)
        iv = brentq(objective, 0.001, 5.0, maxiter=max_iterations)
        
        # Sanity check
        if iv < 0.01 or iv > 5.0:
            return None
        
        return iv
    
    except (ValueError, RuntimeError):
        return None


def calculate_greeks(S, K, T, r, sigma, option_type='call'):
    """
    Calculate option Greeks.
    
    Args:
        S: Underlying price
        K: Strike price
        T: Time to expiration (years)
        r: Risk-free rate
        sigma: Implied volatility
        option_type: 'call' or 'put'
    
    Returns:
        Dictionary with delta, gamma, theta, vega
    """
    if T <= 0 or sigma <= 0 or sigma is None:
        return {'delta': None, 'gamma': None, 'theta': None, 'vega': None}
    
    try:
        d1 = (np.log(S / K) + (r + 0.5 * sigma**2) * T) / (sigma * np.sqrt(T))
        d2 = d1 - sigma * np.sqrt(T)
        
        # Delta
        if option_type.lower() == 'call':
            delta = norm.cdf(d1)
        else:  # put
            delta = norm.cdf(d1) - 1
        
        # Gamma (same for calls and puts)
        gamma = norm.pdf(d1) / (S * sigma * np.sqrt(T))
        
        # Theta (daily)
        if option_type.lower() == 'call':
            theta = (-(S * norm.pdf(d1) * sigma) / (2 * np.sqrt(T)) 
                    - r * K * np.exp(-r * T) * norm.cdf(d2)) / 365
        else:  # put
            theta = (-(S * norm.pdf(d1) * sigma) / (2 * np.sqrt(T)) 
                    + r * K * np.exp(-r * T) * norm.cdf(-d2)) / 365
        
        # Vega (per 1% change in volatility)
        vega = S * norm.pdf(d1) * np.sqrt(T) / 100
        
        return {
            'delta': delta,
            'gamma': gamma,
            'theta': theta,
            'vega': vega
        }
    
    except (ValueError, ZeroDivisionError, OverflowError):
        return {'delta': None, 'gamma': None, 'theta': None, 'vega': None}


def model(dbt, session):
    """
    dbt Python model to calculate Greeks for options.
    This model enriches silver_options with Black-Scholes calculations.
    Gold models should reference THIS model instead of silver_options.
    """
    # Configuration
    dbt.config(
        materialized='table',
        tags=['silver', 'options', 'greeks'],
        alias='silver_options_enriched'
    )
    
    # Load silver options (without Greeks yet)
    silver_df = dbt.ref('silver_options').to_pandas()
    
    # Risk-free rate (use ECB rate or constant for now)
    RISK_FREE_RATE = 0.03  # 3% annual
    
    # Calculate Greeks for each option
    results = []
    
    for idx, row in silver_df.iterrows():
        # Skip if missing critical data
        if pd.isna(row['underlying_price']) or pd.isna(row['days_to_expiry']) or pd.isna(row['mid_price']):
            results.append({
                'option_key': row['option_key'],
                'implied_volatility': None,
                'delta': None,
                'gamma': None,
                'theta': None,
                'vega': None
            })
            continue
        
        # Convert days to years
        T = row['days_to_expiry'] / 365.0
        
        if T <= 0:
            results.append({
                'option_key': row['option_key'],
                'implied_volatility': None,
                'delta': None,
                'gamma': None,
                'theta': None,
                'vega': None
            })
            continue
        
        # Calculate IV first
        iv = calculate_implied_volatility(
            option_price=row['mid_price'],
            S=row['underlying_price'],
            K=row['strike'],
            T=T,
            r=RISK_FREE_RATE,
            option_type=row['option_type']
        )
        
        # Calculate Greeks if we have valid IV
        if iv is not None:
            greeks = calculate_greeks(
                S=row['underlying_price'],
                K=row['strike'],
                T=T,
                r=RISK_FREE_RATE,
                sigma=iv,
                option_type=row['option_type']
            )
        else:
            greeks = {'delta': None, 'gamma': None, 'theta': None, 'vega': None}
        
        results.append({
            'option_key': row['option_key'],
            'implied_volatility': iv,
            'delta': greeks['delta'],
            'gamma': greeks['gamma'],
            'theta': greeks['theta'],
            'vega': greeks['vega']
        })
    
    # Convert to DataFrame
    greeks_df = pd.DataFrame(results)
    
    # Join back with original silver data
    final_df = silver_df.merge(greeks_df, on='option_key', how='left')
    
    # Add rho column (not calculated, kept as NULL for consistency)
    final_df['rho'] = None
    
    return final_df
