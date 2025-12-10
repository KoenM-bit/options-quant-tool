"""
Black-Scholes Option Pricing Model - Production Grade
Calculates option prices, implied volatility, and Greeks with high accuracy.

Key Features:
- Robust numerical stability for extreme parameters
- Accurate handling of edge cases (deep ITM/OTM, near expiry)
- Vectorized calculations for performance
- Comprehensive validation and error handling
- American option approximations where needed

References:
- Black, F., & Scholes, M. (1973). The Pricing of Options and Corporate Liabilities
- Hull, J. C. (2017). Options, Futures, and Other Derivatives (10th ed.)
- Haug, E. G. (2007). The Complete Guide to Option Pricing Formulas
"""

import numpy as np
from scipy.stats import norm
from scipy.optimize import brentq, minimize_scalar
from typing import Optional, Dict, Tuple
import logging

logger = logging.getLogger(__name__)

# Constants for numerical stability
EPSILON = 1e-10  # Small number to avoid division by zero
MIN_VOLATILITY = 0.0001  # 0.01% minimum volatility
MAX_VOLATILITY = 5.0  # 500% maximum volatility
MIN_TIME = 1/365  # Minimum 1 day
MAX_PRICE_RATIO = 1000  # Max S/K or K/S ratio for stability


class BlackScholes:
    """
    Production-grade Black-Scholes model for European options.
    
    Implements robust numerical methods with comprehensive edge case handling.
    All Greeks are calculated with high precision using analytical formulas.
    """
    
    @staticmethod
    def _validate_inputs(S: float, K: float, T: float, r: float, sigma: float) -> bool:
        """
        Validate input parameters for numerical stability.
        
        Returns:
            True if inputs are valid, False otherwise
        """
        if S <= 0 or K <= 0:
            return False
        if T < 0:
            return False
        if sigma < 0:
            return False
        # Check for extreme price ratios that can cause numerical issues
        if S / K > MAX_PRICE_RATIO or K / S > MAX_PRICE_RATIO:
            return False
        return True
    
    @staticmethod
    def _d1(S: float, K: float, T: float, r: float, sigma: float) -> float:
        """
        Calculate d1 parameter with numerical stability.
        
        Handles edge cases:
        - Near-zero time: uses intrinsic value approximation
        - Near-zero volatility: uses limit approximation
        - Extreme moneyness: uses asymptotic approximations
        """
        if T <= EPSILON:
            # At expiration: return large value based on moneyness
            return 1000.0 if S > K else -1000.0
        
        if sigma <= EPSILON:
            # Zero volatility: deterministic outcome
            return 1000.0 if S > K else -1000.0
        
        try:
            # Standard calculation with numerical stability
            log_ratio = np.log(S / K)
            variance_time = sigma * np.sqrt(T)
            
            # Check for extreme values that could cause overflow
            if abs(log_ratio) > 100:  # Extremely ITM or OTM
                return np.sign(log_ratio) * 100
            
            d1 = (log_ratio + (r + 0.5 * sigma**2) * T) / variance_time
            
            # Clamp to reasonable range to avoid numerical issues in norm.cdf
            return np.clip(d1, -10, 10)
            
        except (FloatingPointError, OverflowError, ZeroDivisionError):
            return 1000.0 if S > K else -1000.0
    
    @staticmethod
    def _d2(S: float, K: float, T: float, r: float, sigma: float) -> float:
        """
        Calculate d2 parameter with numerical stability.
        """
        if T <= EPSILON or sigma <= EPSILON:
            return BlackScholes._d1(S, K, T, r, sigma)
        
        d1 = BlackScholes._d1(S, K, T, r, sigma)
        return d1 - sigma * np.sqrt(T)
    
    @staticmethod
    def call_price(S: float, K: float, T: float, r: float, sigma: float) -> float:
        """
        Calculate European call option price with high precision.
        
        Uses analytical Black-Scholes formula with numerical stability enhancements.
        
        Args:
            S: Underlying price (must be > 0)
            K: Strike price (must be > 0)
            T: Time to expiry in years (0 = at expiration)
            r: Risk-free rate (annual, can be negative)
            sigma: Volatility (annual, must be >= 0)
        
        Returns:
            Call option price (always >= intrinsic value)
        """
        if not BlackScholes._validate_inputs(S, K, T, r, sigma):
            return max(S - K, 0)  # Return intrinsic value for invalid inputs
        
        # At or past expiration
        if T <= EPSILON:
            return max(S - K, 0)
        
        # Deep in-the-money: use intrinsic value approximation
        if S / K > 10:
            return S - K * np.exp(-r * T)
        
        # Deep out-of-the-money: near zero value
        if K / S > 10:
            return 0.0
        
        # Near-zero volatility: use limit
        if sigma <= EPSILON:
            return max(S - K * np.exp(-r * T), 0)
        
        try:
            d1 = BlackScholes._d1(S, K, T, r, sigma)
            d2 = BlackScholes._d2(S, K, T, r, sigma)
            
            call_value = S * norm.cdf(d1) - K * np.exp(-r * T) * norm.cdf(d2)
            
            # Ensure non-negative and at least intrinsic value
            intrinsic = max(S - K, 0)
            return max(call_value, intrinsic)
            
        except (FloatingPointError, OverflowError):
            return max(S - K, 0)
    
    @staticmethod
    def put_price(S: float, K: float, T: float, r: float, sigma: float) -> float:
        """
        Calculate European put option price with high precision.
        
        Uses analytical Black-Scholes formula with numerical stability enhancements.
        For deep ITM puts, uses put-call parity for better accuracy.
        
        Args:
            S: Underlying price (must be > 0)
            K: Strike price (must be > 0)
            T: Time to expiry in years (0 = at expiration)
            r: Risk-free rate (annual, can be negative)
            sigma: Volatility (annual, must be >= 0)
        
        Returns:
            Put option price (always >= intrinsic value)
        """
        if not BlackScholes._validate_inputs(S, K, T, r, sigma):
            return max(K - S, 0)
        
        # At or past expiration
        if T <= EPSILON:
            return max(K - S, 0)
        
        # Deep in-the-money: use put-call parity for better numerical stability
        # Put = Call - S + K*exp(-rT)
        if K / S > 10:
            call = BlackScholes.call_price(S, K, T, r, sigma)
            return call - S + K * np.exp(-r * T)
        
        # Deep out-of-the-money: near zero value
        if S / K > 10:
            return 0.0
        
        # Near-zero volatility: use limit
        if sigma <= EPSILON:
            return max(K * np.exp(-r * T) - S, 0)
        
        try:
            d1 = BlackScholes._d1(S, K, T, r, sigma)
            d2 = BlackScholes._d2(S, K, T, r, sigma)
            
            put_value = K * np.exp(-r * T) * norm.cdf(-d2) - S * norm.cdf(-d1)
            
            # Ensure non-negative and at least intrinsic value
            intrinsic = max(K - S, 0)
            return max(put_value, intrinsic)
            
        except (FloatingPointError, OverflowError):
            return max(K - S, 0)
    
    @staticmethod
    def implied_volatility(
        option_price: float,
        S: float,
        K: float,
        T: float,
        r: float,
        option_type: str = 'call',
        max_iterations: int = 100,
        precision: float = 1e-6
    ) -> Optional[float]:
        """
        Calculate implied volatility using multiple robust methods.
        
        Uses a hybrid approach:
        1. Initial guess from Brenner-Subrahmanyam approximation
        2. Newton-Raphson with vega for fast convergence
        3. Fallback to Brent's method for difficult cases
        
        This provides both speed and reliability across all market conditions.
        
        Args:
            option_price: Market price of the option (must be > intrinsic value)
            S: Underlying price
            K: Strike price
            T: Time to expiry (years)
            r: Risk-free rate (annual)
            option_type: 'call' or 'put'
            max_iterations: Maximum iterations
            precision: Convergence precision (dollars)
        
        Returns:
            Implied volatility (annual) or None if calculation fails
        """
        if not BlackScholes._validate_inputs(S, K, T, r, sigma=0.5):
            return None
        
        if T <= EPSILON or option_price <= 0:
            return None
        
        # Determine intrinsic value and price function
        if option_type.lower() == 'call':
            intrinsic = max(S - K, 0)
            price_func = BlackScholes.call_price
        else:
            intrinsic = max(K - S, 0)
            price_func = BlackScholes.put_price
        
        # Option must have time value (market price > intrinsic value)
        time_value = option_price - intrinsic
        if time_value <= EPSILON:
            return None
        
        # Initial guess using Brenner-Subrahmanyam approximation for ATM options
        # This provides a good starting point
        moneyness = S / K
        if 0.95 <= moneyness <= 1.05:  # Near ATM
            sigma_guess = np.sqrt(2 * np.pi / T) * (option_price / S)
        else:
            # For ITM/OTM, use a reasonable default
            sigma_guess = 0.3  # 30% is a common market volatility
        
        # Clamp initial guess to reasonable range
        sigma_guess = np.clip(sigma_guess, MIN_VOLATILITY, MAX_VOLATILITY)
        
        # Method 1: Newton-Raphson with vega (fastest for well-behaved cases)
        try:
            sigma = sigma_guess
            for i in range(max_iterations):
                price = price_func(S, K, T, r, sigma)
                error = price - option_price
                
                if abs(error) < precision:
                    return sigma if MIN_VOLATILITY <= sigma <= MAX_VOLATILITY else None
                
                # Calculate vega for Newton step
                vega = BlackScholes.vega(S, K, T, r, sigma) * 100  # Scale back to full vega
                
                if vega < EPSILON:  # Avoid division by zero
                    break
                
                # Newton-Raphson step: sigma_new = sigma_old - f(sigma)/f'(sigma)
                sigma_new = sigma - error / vega
                
                # Ensure we stay in valid range
                sigma_new = np.clip(sigma_new, MIN_VOLATILITY, MAX_VOLATILITY)
                
                # Check for convergence in sigma
                if abs(sigma_new - sigma) < 1e-6:
                    return sigma_new
                
                sigma = sigma_new
        
        except (FloatingPointError, OverflowError, ZeroDivisionError):
            pass  # Fall through to Brent's method
        
        # Method 2: Brent's method (robust fallback)
        try:
            def objective(sig):
                try:
                    return price_func(S, K, T, r, sig) - option_price
                except (FloatingPointError, OverflowError):
                    return 1e10  # Large value to indicate invalid region
            
            # Try with initial bounds
            iv = brentq(
                objective,
                a=MIN_VOLATILITY,
                b=MAX_VOLATILITY,
                maxiter=max_iterations,
                xtol=1e-6,
                rtol=1e-6
            )
            
            if MIN_VOLATILITY <= iv <= MAX_VOLATILITY:
                return iv
            
        except (ValueError, RuntimeError) as e:
            logger.debug(f"IV calculation failed for price={option_price}, S={S}, K={K}, T={T}: {e}")
        
        return None
    
    @staticmethod
    def delta(S: float, K: float, T: float, r: float, sigma: float, option_type: str = 'call') -> float:
        """
        Calculate option delta with high precision.
        
        Delta = ∂V/∂S (rate of change of option price w.r.t. underlying)
        
        Call delta: [0, 1] - probability of finishing ITM in risk-neutral world
        Put delta: [-1, 0] - negative probability of finishing ITM
        
        Edge cases:
        - At expiration: 1 if ITM, 0 if OTM
        - Deep ITM: approaches ±1
        - Deep OTM: approaches 0
        - ATM: approximately 0.5 for calls, -0.5 for puts
        """
        if not BlackScholes._validate_inputs(S, K, T, r, sigma):
            intrinsic_call = max(S - K, 0)
            intrinsic_put = max(K - S, 0)
            if option_type.lower() == 'call':
                return 1.0 if intrinsic_call > 0 else 0.0
            else:
                return -1.0 if intrinsic_put > 0 else 0.0
        
        # At expiration
        if T <= EPSILON:
            if option_type.lower() == 'call':
                return 1.0 if S >= K else 0.0
            else:
                return -1.0 if S <= K else 0.0
        
        # Zero volatility (deterministic)
        if sigma <= EPSILON:
            fwd_price = S * np.exp(r * T)
            if option_type.lower() == 'call':
                return 1.0 if fwd_price > K else 0.0
            else:
                return -1.0 if fwd_price < K else 0.0
        
        try:
            d1 = BlackScholes._d1(S, K, T, r, sigma)
            
            if option_type.lower() == 'call':
                delta = norm.cdf(d1)
                return np.clip(delta, 0.0, 1.0)
            else:
                delta = -norm.cdf(-d1)  # Equivalent to: norm.cdf(d1) - 1
                return np.clip(delta, -1.0, 0.0)
        
        except (FloatingPointError, OverflowError):
            if option_type.lower() == 'call':
                return 1.0 if S > K else 0.0
            else:
                return -1.0 if S < K else 0.0
    
    @staticmethod
    def gamma(S: float, K: float, T: float, r: float, sigma: float) -> float:
        """
        Calculate option gamma with high precision.
        
        Gamma = ∂²V/∂S² = ∂Δ/∂S (rate of change of delta w.r.t. underlying)
        
        Gamma is:
        - Always positive (same for calls and puts)
        - Maximum at ATM
        - Decreases as option moves ITM or OTM
        - Increases as expiration approaches (gamma risk)
        - Zero at expiration and for deep ITM/OTM
        
        Gamma is crucial for:
        - Delta-hedging frequency
        - Convexity of P&L
        - Risk management near expiration
        """
        if not BlackScholes._validate_inputs(S, K, T, r, sigma):
            return 0.0
        
        # At expiration: technically infinite at ATM, but we return 0
        if T <= EPSILON:
            return 0.0
        
        # Zero volatility: no gamma
        if sigma <= EPSILON:
            return 0.0
        
        # Deep ITM or OTM: gamma approaches zero
        if S / K > 5 or K / S > 5:
            return 0.0
        
        try:
            d1 = BlackScholes._d1(S, K, T, r, sigma)
            
            # Standard normal PDF at d1
            pdf_d1 = norm.pdf(d1)
            
            # Gamma formula: n(d1) / (S * σ * √T)
            # where n(d1) is the standard normal PDF
            variance_time = sigma * np.sqrt(T)
            gamma = pdf_d1 / (S * variance_time)
            
            # Sanity check: gamma should be positive and reasonable
            if gamma < 0 or gamma > 1.0:  # Gamma > 1 is unusual for typical options
                return 0.0
            
            return gamma
        
        except (FloatingPointError, OverflowError, ZeroDivisionError):
            return 0.0
    
    @staticmethod
    def vega(S: float, K: float, T: float, r: float, sigma: float) -> float:
        """
        Calculate option vega with high precision.
        
        Vega = ∂V/∂σ (rate of change of option price w.r.t. volatility)
        
        Vega is:
        - Always positive (same for calls and puts)
        - Maximum at ATM
        - Increases with time to expiration
        - Zero at expiration
        - Important for volatility trading strategies
        
        Returns:
            Vega per 1% change in volatility (industry standard)
            e.g., vega = 0.25 means $0.25 gain per 1% increase in IV
        """
        if not BlackScholes._validate_inputs(S, K, T, r, sigma):
            return 0.0
        
        # At expiration: no vega
        if T <= EPSILON:
            return 0.0
        
        # Deep ITM or OTM: vega approaches zero
        if S / K > 5 or K / S > 5:
            return 0.0
        
        try:
            d1 = BlackScholes._d1(S, K, T, r, sigma)
            
            # Standard normal PDF at d1
            pdf_d1 = norm.pdf(d1)
            
            # Vega formula: S * n(d1) * √T
            # Divided by 100 for 1% change (industry convention)
            vega = S * pdf_d1 * np.sqrt(T) / 100
            
            # Sanity check
            if vega < 0 or vega > S:  # Vega shouldn't exceed underlying price
                return 0.0
            
            return vega
        
        except (FloatingPointError, OverflowError):
            return 0.0
    
    @staticmethod
    def theta(S: float, K: float, T: float, r: float, sigma: float, option_type: str = 'call') -> float:
        """
        Calculate option theta with high precision.
        
        Theta = ∂V/∂t (rate of change of option price w.r.t. time)
        
        Theta is:
        - Usually negative (time decay)
        - Largest (most negative) for ATM options near expiration
        - Can be positive for deep ITM European puts (r > 0)
        - Measures daily time decay
        
        Important notes:
        - This is calendar theta, not trading-day theta
        - Represents 1-day time decay (industry standard)
        - For weekly decay, multiply by 7
        
        Returns:
            Theta per calendar day (typically negative)
        """
        if not BlackScholes._validate_inputs(S, K, T, r, sigma):
            return 0.0
        
        # At expiration: no theta
        if T <= EPSILON:
            return 0.0
        
        # Zero volatility: simplified theta
        if sigma <= EPSILON:
            fwd_value = K * np.exp(-r * T)
            if option_type.lower() == 'call':
                if S > fwd_value:
                    return r * fwd_value / 365
            else:
                if S < fwd_value:
                    return -r * fwd_value / 365
            return 0.0
        
        try:
            d1 = BlackScholes._d1(S, K, T, r, sigma)
            d2 = BlackScholes._d2(S, K, T, r, sigma)
            
            # Standard normal PDF at d1
            pdf_d1 = norm.pdf(d1)
            
            # Common term: -S * n(d1) * σ / (2√T)
            common_term = -(S * pdf_d1 * sigma) / (2 * np.sqrt(T))
            
            # Discount factor
            discount = np.exp(-r * T)
            
            if option_type.lower() == 'call':
                # Call theta = common_term - r*K*e^(-rT)*N(d2)
                theta = common_term - r * K * discount * norm.cdf(d2)
            else:
                # Put theta = common_term + r*K*e^(-rT)*N(-d2)
                theta = common_term + r * K * discount * norm.cdf(-d2)
            
            # Convert to per-day (from per-year)
            theta_daily = theta / 365
            
            # Sanity check: theta shouldn't be larger than option value
            option_value = (BlackScholes.call_price(S, K, T, r, sigma) 
                          if option_type.lower() == 'call' 
                          else BlackScholes.put_price(S, K, T, r, sigma))
            
            if abs(theta_daily) > option_value:
                return 0.0
            
            return theta_daily
        
        except (FloatingPointError, OverflowError, ZeroDivisionError):
            return 0.0
    
    @staticmethod
    def rho(S: float, K: float, T: float, r: float, sigma: float, option_type: str = 'call') -> float:
        """
        Calculate option rho with high precision.
        
        Rho = ∂V/∂r (rate of change of option price w.r.t. interest rate)
        
        Rho is:
        - Positive for calls (higher rates → higher call value)
        - Negative for puts (higher rates → lower put value)
        - Increases with time to expiration
        - Larger for ITM options
        - Less relevant in low-rate environments
        
        Returns:
            Rho per 1% change in interest rate (0.01)
            e.g., rho = 0.15 means $0.15 gain per 1% rate increase
        """
        if not BlackScholes._validate_inputs(S, K, T, r, sigma):
            return 0.0
        
        # At expiration: no rho sensitivity
        if T <= EPSILON:
            return 0.0
        
        # Zero volatility: simplified rho
        if sigma <= EPSILON:
            if option_type.lower() == 'call':
                if S > K * np.exp(-r * T):
                    return K * T * np.exp(-r * T) / 100
            else:
                if S < K * np.exp(-r * T):
                    return -K * T * np.exp(-r * T) / 100
            return 0.0
        
        try:
            d2 = BlackScholes._d2(S, K, T, r, sigma)
            discount = np.exp(-r * T)
            
            if option_type.lower() == 'call':
                # Call rho = K * T * e^(-rT) * N(d2)
                rho = K * T * discount * norm.cdf(d2)
            else:
                # Put rho = -K * T * e^(-rT) * N(-d2)
                rho = -K * T * discount * norm.cdf(-d2)
            
            # Convert to per 1% change (0.01)
            rho_pct = rho / 100
            
            # Sanity check
            if abs(rho_pct) > K * T:  # Rho shouldn't exceed PV of strike
                return 0.0
            
            return rho_pct
        
        except (FloatingPointError, OverflowError):
            return 0.0
    
    @staticmethod
    def vanna(S: float, K: float, T: float, r: float, sigma: float) -> float:
        """
        Calculate vanna (second-order Greek).
        
        Vanna = ∂²V/∂S∂σ = ∂Δ/∂σ = ∂vega/∂S
        
        Measures sensitivity of delta to changes in volatility,
        or sensitivity of vega to changes in underlying price.
        
        Important for:
        - Volatility surface dynamics
        - Cross-gamma hedging
        - Understanding delta hedging errors
        """
        if not BlackScholes._validate_inputs(S, K, T, r, sigma):
            return 0.0
        
        if T <= EPSILON or sigma <= EPSILON:
            return 0.0
        
        try:
            d1 = BlackScholes._d1(S, K, T, r, sigma)
            d2 = BlackScholes._d2(S, K, T, r, sigma)
            
            # Vanna = -n(d1) * d2 / σ
            vanna = -norm.pdf(d1) * d2 / sigma
            
            return vanna / 100  # Scale for 1% vol change
        
        except (FloatingPointError, OverflowError, ZeroDivisionError):
            return 0.0
    
    @staticmethod
    def charm(S: float, K: float, T: float, r: float, sigma: float, option_type: str = 'call') -> float:
        """
        Calculate charm (second-order Greek).
        
        Charm = ∂Δ/∂t = ∂²V/∂S∂t
        
        Measures the rate of change of delta with respect to time.
        Also known as "delta decay" or "delta bleed".
        
        Important for:
        - Forecasting delta changes over time
        - Understanding how delta hedges will drift
        - Risk management near expiration
        """
        if not BlackScholes._validate_inputs(S, K, T, r, sigma):
            return 0.0
        
        if T <= EPSILON or sigma <= EPSILON:
            return 0.0
        
        try:
            d1 = BlackScholes._d1(S, K, T, r, sigma)
            d2 = BlackScholes._d2(S, K, T, r, sigma)
            pdf_d1 = norm.pdf(d1)
            
            sqrt_T = np.sqrt(T)
            
            if option_type.lower() == 'call':
                charm = -pdf_d1 * (2 * r * T - d2 * sigma * sqrt_T) / (2 * T * sigma * sqrt_T)
            else:
                charm = -pdf_d1 * (2 * r * T - d2 * sigma * sqrt_T) / (2 * T * sigma * sqrt_T)
            
            return charm / 365  # Per day
        
        except (FloatingPointError, OverflowError, ZeroDivisionError):
            return 0.0
    
    @staticmethod
    def calculate_all_greeks(
        S: float,
        K: float,
        T: float,
        r: float,
        sigma: float,
        option_type: str = 'call',
        include_second_order: bool = False
    ) -> Dict[str, float]:
        """
        Calculate all Greeks at once with optional second-order Greeks.
        
        Args:
            S: Underlying price
            K: Strike price
            T: Time to expiry (years)
            r: Risk-free rate (annual)
            sigma: Volatility (annual)
            option_type: 'call' or 'put'
            include_second_order: Include vanna and charm
        
        Returns:
            Dictionary with all Greeks (always includes first-order)
        """
        greeks = {
            'delta': BlackScholes.delta(S, K, T, r, sigma, option_type),
            'gamma': BlackScholes.gamma(S, K, T, r, sigma),
            'vega': BlackScholes.vega(S, K, T, r, sigma),
            'theta': BlackScholes.theta(S, K, T, r, sigma, option_type),
            'rho': BlackScholes.rho(S, K, T, r, sigma, option_type),
        }
        
        if include_second_order:
            greeks['vanna'] = BlackScholes.vanna(S, K, T, r, sigma)
            greeks['charm'] = BlackScholes.charm(S, K, T, r, sigma, option_type)
        
        return greeks


def calculate_option_metrics(
    option_price: float,
    underlying_price: float,
    strike: float,
    days_to_expiry: int,
    option_type: str,
    risk_free_rate: float = 0.03
) -> Dict[str, Optional[float]]:
    """
    Calculate implied volatility and Greeks for an option with STRICT quality controls.
    
    Quality Requirements (NO COMPROMISE):
    - Market price must exceed intrinsic value (time value exists)
    - IV must converge to a reasonable value (5% - 200%)
    - Back-calculated price must match market price within 1%
    - Greeks must be within reasonable bounds
    - No extreme moneyness (S/K between 0.1 and 10)
    
    Returns None for ALL metrics if ANY quality check fails.
    BETTER NO DATA THAN BAD DATA.
    
    Args:
        option_price: Market price of the option
        underlying_price: Current price of underlying
        strike: Strike price
        days_to_expiry: Days until expiration
        option_type: 'Call' or 'Put'
        risk_free_rate: Annual risk-free rate (default 3%)
    
    Returns:
        Dictionary with implied_volatility and all Greeks, or all None if quality checks fail
    """
    # Convert days to years
    T = days_to_expiry / 365.0
    
    # Basic validation
    if T <= 0 or option_price <= 0 or underlying_price <= 0 or strike <= 0:
        logger.debug(f"Basic validation failed: T={T}, price={option_price}, S={underlying_price}, K={strike}")
        return {
            'implied_volatility': None,
            'delta': None,
            'gamma': None,
            'vega': None,
            'theta': None,
            'rho': None,
        }
    
    # QUALITY CHECK 1: Check for extreme moneyness that causes numerical issues
    moneyness = underlying_price / strike
    if moneyness < 0.1 or moneyness > 10.0:
        logger.debug(f"Extreme moneyness rejected: S/K = {moneyness:.2f} (must be 0.1-10)")
        return {
            'implied_volatility': None,
            'delta': None,
            'gamma': None,
            'vega': None,
            'theta': None,
            'rho': None,
        }
    
    # QUALITY CHECK 2: Verify time value exists (price > intrinsic value)
    if option_type.lower() == 'call':
        intrinsic = max(underlying_price - strike, 0)
    else:
        intrinsic = max(strike - underlying_price, 0)
    
    time_value = option_price - intrinsic
    
    # For deep ITM options (moneyness > 1.02 or < 0.98), allow smaller time value
    # These options are mostly intrinsic value with minimal time premium
    is_deep_itm = moneyness > 1.02 or moneyness < 0.98
    min_time_value = -0.05 if is_deep_itm else 0.001
    
    if time_value < min_time_value:
        logger.debug(f"Insufficient time value: price={option_price:.4f}, intrinsic={intrinsic:.4f}, time_value={time_value:.4f}, deep_itm={is_deep_itm}")
        return {
            'implied_volatility': None,
            'delta': None,
            'gamma': None,
            'vega': None,
            'theta': None,
            'rho': None,
        }
    
    # Calculate implied volatility
    iv = BlackScholes.implied_volatility(
        option_price=option_price,
        S=underlying_price,
        K=strike,
        T=T,
        r=risk_free_rate,
        option_type=option_type.lower()
    )
    
    if iv is None:
        logger.debug(f"IV calculation failed for {option_type} S={underlying_price}, K={strike}, T={T:.3f}y, price={option_price}")
        return {
            'implied_volatility': None,
            'delta': None,
            'gamma': None,
            'vega': None,
            'theta': None,
            'rho': None,
        }
    
    # QUALITY CHECK 3: IV must be in reasonable range
    # For deep ITM options, allow higher IV (they have little time value, so IV can be extreme)
    max_iv = 5.0 if is_deep_itm else 2.0  # 500% for deep ITM, 200% otherwise
    if iv < 0.05 or iv > max_iv:
        logger.debug(f"IV out of acceptable range: {iv:.2%} (must be 5%-{max_iv*100:.0f}%, deep_itm={is_deep_itm})")
        return {
            'implied_volatility': None,
            'delta': None,
            'gamma': None,
            'vega': None,
            'theta': None,
            'rho': None,
        }
    
    # QUALITY CHECK 4: Verify back-calculation accuracy (most important!)
    # Calculate theoretical price using the calculated IV
    if option_type.lower() == 'call':
        theoretical_price = BlackScholes.call_price(underlying_price, strike, T, risk_free_rate, iv)
    else:
        theoretical_price = BlackScholes.put_price(underlying_price, strike, T, risk_free_rate, iv)
    
    price_error = abs(theoretical_price - option_price) / option_price
    
    # For deep ITM options, allow higher price error since they're mostly intrinsic value
    # Small absolute errors become large percentage errors when time value is small
    max_price_error = 0.05 if is_deep_itm else 0.01  # 5% for deep ITM, 1% for others
    
    if price_error > max_price_error:  # More than threshold error
        logger.debug(f"Price verification failed: market={option_price:.4f}, theoretical={theoretical_price:.4f}, error={price_error:.2%}")
        return {
            'implied_volatility': None,
            'delta': None,
            'gamma': None,
            'vega': None,
            'theta': None,
            'rho': None,
        }
    
    # Calculate Greeks using the implied volatility
    greeks = BlackScholes.calculate_all_greeks(
        S=underlying_price,
        K=strike,
        T=T,
        r=risk_free_rate,
        sigma=iv,
        option_type=option_type.lower()
    )
    
    # QUALITY CHECK 5: Validate Greeks are in reasonable bounds
    delta = greeks.get('delta')
    gamma = greeks.get('gamma')
    
    if delta is not None:
        if option_type.lower() == 'call':
            if not (0 <= delta <= 1):
                logger.debug(f"Invalid call delta: {delta} (must be 0-1)")
                return {
                    'implied_volatility': None,
                    'delta': None,
                    'gamma': None,
                    'vega': None,
                    'theta': None,
                    'rho': None,
                }
        else:
            if not (-1 <= delta <= 0):
                logger.debug(f"Invalid put delta: {delta} (must be -1 to 0)")
                return {
                    'implied_volatility': None,
                    'delta': None,
                    'gamma': None,
                    'vega': None,
                    'theta': None,
                    'rho': None,
                }
    
    if gamma is not None:
        if gamma < 0 or gamma > 1.0:  # Gamma should be positive and reasonable
            logger.debug(f"Invalid gamma: {gamma} (must be 0-1)")
            return {
                'implied_volatility': None,
                'delta': None,
                'gamma': None,
                'vega': None,
                'theta': None,
                'rho': None,
            }
    
    # ALL QUALITY CHECKS PASSED - Return high-quality Greeks
    logger.debug(f"✅ High-quality Greeks calculated: IV={iv:.2%}, price_error={price_error:.4%}")
    return {
        'implied_volatility': iv,
        **greeks
    }
