/*
Gold Layer: Volatility Term Structure
Analyzes implied volatility across different time-to-expiries.
Shows contango/backwardation and identifies calendar spread opportunities.

Business Value:
- Spot volatility regime changes (contango vs backwardation)
- Identify calendar spread opportunities
- Track term premium evolution over time
*/

{{ config(
    materialized='incremental',
    unique_key='term_structure_key',
    tags=['gold', 'volatility', 'term_structure']
) }}

WITH options_with_tte AS (
    SELECT
        trade_date,
        expiry_date,
        option_type,
        strike,
        implied_volatility,
        delta,
        volume,
        open_interest,
        underlying_price,
        -- Calculate time-to-expiry in years
        (expiry_date - trade_date) / 365.0 AS time_to_expiry_years,
        -- Calculate days to expiry
        (expiry_date - trade_date) AS days_to_expiry,
        -- Moneyness
        CASE 
            WHEN option_type = 'Call' THEN underlying_price / strike
            WHEN option_type = 'Put' THEN strike / underlying_price
        END AS moneyness
    FROM {{ ref('silver_options') }}
    WHERE 
        greeks_valid = TRUE  -- ✅ Only validated Greeks for term structure
        AND implied_volatility IS NOT NULL
        AND implied_volatility > 0
        AND expiry_date > trade_date
        {% if is_incremental() %}
        AND created_at > COALESCE((SELECT MAX(created_at) FROM {{ this }}), '1900-01-01'::timestamp)
        {% endif %}
),

-- ATM options only (moneyness between 0.95 and 1.05)
atm_options AS (
    SELECT
        trade_date,
        expiry_date,
        option_type,
        time_to_expiry_years,
        days_to_expiry,
        AVG(implied_volatility) AS avg_iv,
        PERCENTILE_CONT(0.5) WITHIN GROUP (ORDER BY implied_volatility) AS median_iv,
        MIN(implied_volatility) AS min_iv,
        MAX(implied_volatility) AS max_iv,
        COUNT(*) AS option_count,
        SUM(volume) AS total_volume,
        SUM(open_interest) AS total_oi,
        AVG(underlying_price) AS avg_underlying_price
    FROM options_with_tte
    WHERE moneyness BETWEEN 0.95 AND 1.05  -- Near ATM
    GROUP BY 
        trade_date,
        expiry_date,
        option_type,
        time_to_expiry_years,
        days_to_expiry
),

-- Calculate term structure metrics
term_structure_with_metrics AS (
    SELECT
        trade_date,
        expiry_date,
        option_type,
        time_to_expiry_years,
        days_to_expiry,
        median_iv AS atm_iv,
        avg_iv,
        min_iv,
        max_iv,
        option_count,
        total_volume,
        total_oi,
        avg_underlying_price,
        
        -- Term structure slope (compare to next expiry)
        LEAD(median_iv) OVER (
            PARTITION BY trade_date, option_type 
            ORDER BY expiry_date
        ) - median_iv AS iv_slope_to_next,
        
        -- Compare to previous day same expiry
        LAG(median_iv) OVER (
            PARTITION BY expiry_date, option_type 
            ORDER BY trade_date
        ) AS prev_day_iv,
        
        median_iv - LAG(median_iv) OVER (
            PARTITION BY expiry_date, option_type 
            ORDER BY trade_date
        ) AS iv_change_1d,
        
        -- Term premium vs front month
        median_iv - FIRST_VALUE(median_iv) OVER (
            PARTITION BY trade_date, option_type 
            ORDER BY expiry_date
            ROWS BETWEEN UNBOUNDED PRECEDING AND UNBOUNDED FOLLOWING
        ) AS term_premium,
        
        -- Rank by time to expiry
        ROW_NUMBER() OVER (
            PARTITION BY trade_date, option_type 
            ORDER BY expiry_date
        ) AS expiry_rank
        
    FROM atm_options
),

-- Identify term structure regime
with_regime AS (
    SELECT
        *,
        CASE 
            WHEN expiry_rank = 1 THEN NULL  -- Front month has no comparison
            WHEN iv_slope_to_next > 0 THEN 'Contango'  -- IV rises with time
            WHEN iv_slope_to_next < 0 THEN 'Backwardation'  -- IV falls with time
            ELSE 'Flat'
        END AS term_structure_regime,
        
        -- Quality flags
        CASE 
            WHEN option_count >= 5 AND total_volume > 0 THEN TRUE
            ELSE FALSE
        END AS is_reliable,
        
        CASE
            WHEN total_volume > 50 AND total_oi > 100 THEN 'High'
            WHEN total_volume > 10 AND total_oi > 50 THEN 'Medium'
            ELSE 'Low'
        END AS liquidity_tier
        
    FROM term_structure_with_metrics
)

SELECT
    trade_date || '_' || expiry_date::text || '_' || option_type AS term_structure_key,
    trade_date,
    expiry_date,
    option_type,
    days_to_expiry,
    time_to_expiry_years,
    expiry_rank AS term_rank,
    
    -- IV Metrics
    atm_iv AS atm_implied_volatility,
    avg_iv AS avg_implied_volatility,
    min_iv AS min_implied_volatility,
    max_iv AS max_implied_volatility,
    iv_change_1d AS iv_change_1day,
    
    -- Term Structure
    term_premium,
    iv_slope_to_next AS iv_slope_to_next_expiry,
    term_structure_regime,
    
    -- Volume & OI
    total_volume,
    total_oi AS total_open_interest,
    option_count,
    
    -- Market Context
    avg_underlying_price AS underlying_price,
    liquidity_tier,
    is_reliable,
    
    -- Metadata
    CURRENT_TIMESTAMP AS created_at,
    CURRENT_TIMESTAMP AS updated_at
    
FROM with_regime
WHERE is_reliable = TRUE  -- Only include reliable data points
