"""
Fetch and cache risk-free rates for options pricing.
Uses ECB yield curve data for term-matched risk-free rates.

For accurate options pricing, we use:
- ECB €STR (overnight rate) for very short maturities
- ECB AAA-rated euro area government bond yields for longer maturities
- Linear interpolation between points on the yield curve

This provides term-structure matched rates for each option's time-to-expiry.
"""

import logging
from datetime import datetime, timedelta
from typing import Optional, Dict, List, Tuple
import requests
from functools import lru_cache
import numpy as np

logger = logging.getLogger(__name__)

# ECB Statistical Data Warehouse API
ECB_API_BASE = "https://data-api.ecb.europa.eu/service/data"

# ECB Yield Curve Series (AAA-rated euro area government bonds)
# These represent the risk-free rates for different maturities
ECB_YIELD_CURVE_SERIES = {
    'overnight': 'FM/D.U2.EUR.4F.KR.DFR.LEV',  # €STR (overnight)
    '1m': 'YC/B.U2.EUR.4F.G_N_A.SV_C_YM.SR_1M',  # 1-month
    '3m': 'YC/B.U2.EUR.4F.G_N_A.SV_C_YM.SR_3M',  # 3-month
    '6m': 'YC/B.U2.EUR.4F.G_N_A.SV_C_YM.SR_6M',  # 6-month
    '1y': 'YC/B.U2.EUR.4F.G_N_A.SV_C_YM.SR_1Y',  # 1-year
    '2y': 'YC/B.U2.EUR.4F.G_N_A.SV_C_YM.SR_2Y',  # 2-year
    '5y': 'YC/B.U2.EUR.4F.G_N_A.SV_C_YM.SR_5Y',  # 5-year
    '10y': 'YC/B.U2.EUR.4F.G_N_A.SV_C_YM.SR_10Y',  # 10-year
}

# Maturity mapping (years)
MATURITY_YEARS = {
    'overnight': 1/365,
    '1m': 1/12,
    '3m': 3/12,
    '6m': 6/12,
    '1y': 1.0,
    '2y': 2.0,
    '5y': 5.0,
    '10y': 10.0,
}


def fetch_yield_curve_point(series_code: str, target_date: datetime.date) -> Optional[float]:
    """
    Fetch a single point on the yield curve for a specific date.
    
    Args:
        series_code: ECB series code for the yield curve point
        target_date: The date to fetch data for
    
    Returns:
        Rate as decimal (e.g., 0.0200 for 2%) or None if not available
    """
    try:
        start_date = target_date.strftime('%Y-%m-%d')
        end_date = (target_date + timedelta(days=5)).strftime('%Y-%m-%d')
        
        url = f"{ECB_API_BASE}/{series_code}"
        params = {
            'startPeriod': start_date,
            'endPeriod': end_date,
            'format': 'jsondata'
        }
        
        response = requests.get(url, params=params, timeout=10)
        response.raise_for_status()
        
        data = response.json()
        
        if 'dataSets' in data and len(data['dataSets']) > 0:
            observations = data['dataSets'][0].get('series', {}).get('0:0:0:0:0:0:0', {}).get('observations', {})
            
            if observations:
                first_obs_key = sorted(observations.keys())[0]
                rate_percent = float(observations[first_obs_key][0])
                return rate_percent / 100.0
        
        return None
        
    except Exception as e:
        logger.debug(f"Failed to fetch {series_code} for {target_date}: {e}")
        return None


@lru_cache(maxsize=100)
def get_yield_curve_for_date(target_date: datetime.date) -> Dict[str, float]:
    """
    Fetch the complete yield curve for a specific date.
    
    Returns a dictionary mapping maturity labels to rates (as decimals).
    Falls back to reasonable defaults if ECB data is unavailable.
    
    Args:
        target_date: The date to fetch the yield curve for
    
    Returns:
        Dict mapping maturity labels to rates (e.g., {'overnight': 0.02, '1y': 0.025})
    """
    logger.info(f"Fetching yield curve for {target_date}...")
    
    yield_curve = {}
    
    # Fetch each point on the curve
    for maturity, series_code in ECB_YIELD_CURVE_SERIES.items():
        rate = fetch_yield_curve_point(series_code, target_date)
        
        if rate is not None:
            yield_curve[maturity] = rate
            logger.debug(f"  {maturity}: {rate:.4%}")
        else:
            logger.warning(f"  {maturity}: No data available")
    
    # If we couldn't fetch any data, use fallback rates
    if not yield_curve:
        logger.warning("Could not fetch any yield curve data, using fallback rates")
        yield_curve = {
            'overnight': 0.020,
            '1m': 0.021,
            '3m': 0.022,
            '6m': 0.023,
            '1y': 0.025,
            '2y': 0.027,
            '5y': 0.030,
            '10y': 0.032,
        }
    
    # Fill in missing points with interpolation from available data
    if len(yield_curve) < len(ECB_YIELD_CURVE_SERIES):
        yield_curve = interpolate_missing_points(yield_curve)
    
    logger.info(f"✅ Yield curve fetched with {len(yield_curve)} points")
    
    return yield_curve


def interpolate_missing_points(partial_curve: Dict[str, float]) -> Dict[str, float]:
    """
    Fill in missing yield curve points using linear interpolation.
    
    Args:
        partial_curve: Dict with some yield curve points
    
    Returns:
        Complete yield curve with all maturities
    """
    # Get available points sorted by maturity
    available = [(MATURITY_YEARS[mat], rate) for mat, rate in partial_curve.items()]
    available.sort()
    
    if len(available) < 2:
        # Not enough points to interpolate, return as-is
        return partial_curve
    
    maturities = [m for m, r in available]
    rates = [r for m, r in available]
    
    # Interpolate for all required maturities
    result = {}
    for maturity_label, maturity_years in MATURITY_YEARS.items():
        if maturity_label in partial_curve:
            # Already have this point
            result[maturity_label] = partial_curve[maturity_label]
        else:
            # Interpolate
            interpolated_rate = np.interp(maturity_years, maturities, rates)
            result[maturity_label] = interpolated_rate
            logger.debug(f"  Interpolated {maturity_label}: {interpolated_rate:.4%}")
    
    return result


def get_rate_for_time_to_expiry(time_to_expiry: float, target_date: datetime.date) -> float:
    """
    Get the appropriate risk-free rate for a specific time-to-expiry.
    
    Uses the yield curve to find the term-matched rate, with linear interpolation
    between curve points.
    
    Args:
        time_to_expiry: Time to option expiry in years (e.g., 0.5 for 6 months)
        target_date: The date for which to get the rate
    
    Returns:
        Term-matched risk-free rate as decimal
    """
    # Get the yield curve for this date
    yield_curve = get_yield_curve_for_date(target_date)
    
    # Convert yield curve to arrays for interpolation
    maturities = []
    rates = []
    for maturity_label in sorted(yield_curve.keys(), key=lambda x: MATURITY_YEARS[x]):
        maturities.append(MATURITY_YEARS[maturity_label])
        rates.append(yield_curve[maturity_label])
    
    # Interpolate to get the rate for exact time-to-expiry
    if time_to_expiry <= maturities[0]:
        # Use shortest rate for very short maturities
        return rates[0]
    elif time_to_expiry >= maturities[-1]:
        # Use longest rate for very long maturities
        return rates[-1]
    else:
        # Interpolate
        interpolated_rate = np.interp(time_to_expiry, maturities, rates)
        return interpolated_rate


@lru_cache(maxsize=365)
def get_risk_free_rate_for_date(target_date: datetime.date) -> float:
    """
    Get the risk-free rate (€STR) for a specific date.
    
    For EUR options on Amsterdam stocks, we use the ECB's Euro Short-Term Rate (€STR)
    which is the overnight rate for the euro area.
    
    Args:
        target_date: The date to get the risk-free rate for
    
    Returns:
        Annual risk-free rate as a decimal (e.g., 0.0350 for 3.5%)
        Falls back to 3.5% if API fails
    """
    # Use the new yield curve approach - just return overnight rate
    yield_curve = get_yield_curve_for_date(target_date)
    return yield_curve.get('overnight', 0.020)  # Default to 2% if not available


def get_current_risk_free_rate() -> float:
    """
    Get the current risk-free rate (most recent €STR).
    
    Returns:
        Current annual risk-free rate as a decimal
    """
    today = datetime.now().date()
    # Go back a few days to ensure we get the most recent published rate
    # (€STR is published with T+1 lag)
    target_date = today - timedelta(days=2)
    return get_risk_free_rate_for_date(target_date)


def get_historical_rates_batch(start_date: datetime.date, end_date: datetime.date) -> dict:
    """
    Fetch overnight risk-free rates for a date range.
    Returns a dictionary mapping dates to overnight rates (€STR).
    
    For term-matched rates, use get_rate_for_time_to_expiry() instead.
    
    Args:
        start_date: Start date (inclusive)
        end_date: End date (inclusive)
    
    Returns:
        Dict mapping date strings (YYYY-MM-DD) to overnight risk-free rates
    """
    try:
        start_str = start_date.strftime('%Y-%m-%d')
        end_str = end_date.strftime('%Y-%m-%d')
        
        # Fetch overnight rate (€STR) for the period
        series_code = ECB_YIELD_CURVE_SERIES['overnight']
        url = f"{ECB_API_BASE}/{series_code}"
        params = {
            'startPeriod': start_str,
            'endPeriod': end_str,
            'format': 'jsondata'
        }
        
        logger.info(f"Fetching overnight rates for period {start_str} to {end_str}...")
        response = requests.get(url, params=params, timeout=15)
        response.raise_for_status()
        
        data = response.json()
        rates = {}
        
        if 'dataSets' in data and len(data['dataSets']) > 0:
            observations = data['dataSets'][0].get('series', {}).get('0:0:0:0:0:0:0', {}).get('observations', {})
            
            # Get time dimension for date mapping
            structure = data.get('structure', {})
            time_values = structure.get('dimensions', {}).get('observation', [])
            if time_values and len(time_values) > 0:
                time_dimension = time_values[0].get('values', [])
                
                for idx, obs_value in observations.items():
                    idx_int = int(idx)
                    if idx_int < len(time_dimension):
                        date_str = time_dimension[idx_int]['id']
                        rate_percent = float(obs_value[0])
                        rates[date_str] = rate_percent / 100.0
        
        logger.info(f"✅ Fetched {len(rates)} overnight rates from ECB")
        return rates
        
    except Exception as e:
        logger.error(f"Failed to fetch historical rates: {e}")
        return {}


if __name__ == '__main__':
    # Test the yield curve functionality
    logging.basicConfig(level=logging.INFO)
    
    print("\n" + "=" * 60)
    print("TESTING YIELD CURVE FETCHER")
    print("=" * 60)
    
    # Test 1: Current overnight rate
    current_rate = get_current_risk_free_rate()
    print(f"\n1. Current overnight rate (€STR): {current_rate:.4%}")
    
    # Test 2: Specific date overnight rate
    test_date = datetime(2025, 10, 24).date()
    rate = get_risk_free_rate_for_date(test_date)
    print(f"\n2. Overnight rate for {test_date}: {rate:.4%}")
    
    # Test 3: Complete yield curve for a specific date
    print(f"\n3. Complete yield curve for {test_date}:")
    yield_curve = get_yield_curve_for_date(test_date)
    for maturity in ['overnight', '1m', '3m', '6m', '1y', '2y', '5y', '10y']:
        if maturity in yield_curve:
            maturity_years = MATURITY_YEARS[maturity]
            print(f"   {maturity:10s} ({maturity_years:5.2f} years): {yield_curve[maturity]:.4%}")
    
    # Test 4: Term-matched rates for different expiries
    print(f"\n4. Term-matched rates for different option expiries (on {test_date}):")
    test_expiries = [
        (7/365, "1 week"),
        (30/365, "1 month"),
        (90/365, "3 months"),
        (180/365, "6 months"),
        (365/365, "1 year"),
        (730/365, "2 years"),
    ]
    
    for time_to_expiry, label in test_expiries:
        rate = get_rate_for_time_to_expiry(time_to_expiry, test_date)
        print(f"   {label:12s} ({time_to_expiry:.4f} years): {rate:.4%}")
    
    # Test 5: Batch fetch historical overnight rates
    start = datetime(2025, 10, 24).date()
    end = datetime(2025, 12, 5).date()
    rates = get_historical_rates_batch(start, end)
    print(f"\n5. Fetched {len(rates)} overnight rates for period {start} to {end}")
    if rates:
        print("   Sample rates (first 5):")
        for date_str in sorted(rates.keys())[:5]:
            print(f"     {date_str}: {rates[date_str]:.4%}")
    
    print("\n" + "=" * 60)
    print("✅ YIELD CURVE TESTS COMPLETE")
    print("=" * 60)
