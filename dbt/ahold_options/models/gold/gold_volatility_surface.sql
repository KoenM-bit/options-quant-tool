/*
Gold Layer: Volatility Surface
IV surface data by strike and expiry for visualization.
*/

{{ config(
    materialized='incremental',
    unique_key=['vol_surface_key', 'calculation_timestamp'],
    tags=['gold', 'volatility', 'surface'],
    incremental_strategy='append'
) }}

WITH options_data AS (
    -- ✅ QUALITY GATE: Only use validated Greeks and IV from Silver
    SELECT
        ticker,
        trade_date,
        expiry_date,
        strike,
        option_type,
        days_to_expiry,
        moneyness,
        implied_volatility,
        volume,
        open_interest,
        mid_price,
        underlying_price,
        risk_free_rate_used
    FROM {{ ref('silver_options') }}
    WHERE 
        greeks_valid = TRUE  -- ✅ Only high-quality validated Greeks
        AND implied_volatility IS NOT NULL
        AND days_to_expiry > 0
        {% if is_incremental() %}
        AND trade_date > COALESCE((SELECT MAX(trade_date) FROM {{ this }}), '1900-01-01'::date)
        {% endif %}
),

aggregated AS (
    SELECT
        ticker,
        trade_date,
        expiry_date,
        strike,
        days_to_expiry,
        moneyness,
        
        -- Separate call/put IV
        AVG(CASE WHEN option_type = 'Call' THEN implied_volatility END) AS call_iv,
        AVG(CASE WHEN option_type = 'Put' THEN implied_volatility END) AS put_iv,
        AVG(implied_volatility) AS avg_iv,
        
        -- Volume and OI
        SUM(CASE WHEN option_type = 'Call' THEN volume ELSE 0 END) AS call_volume,
        SUM(CASE WHEN option_type = 'Put' THEN volume ELSE 0 END) AS put_volume,
        SUM(CASE WHEN option_type = 'Call' THEN open_interest ELSE 0 END) AS call_oi,
        SUM(CASE WHEN option_type = 'Put' THEN open_interest ELSE 0 END) AS put_oi,
        
        -- Price data
        MAX(underlying_price) AS underlying_price,
        AVG(CASE WHEN option_type = 'Call' THEN mid_price END) AS call_mid,
        AVG(CASE WHEN option_type = 'Put' THEN mid_price END) AS put_mid,
        
        CURRENT_TIMESTAMP AS created_at,
        CURRENT_TIMESTAMP AS updated_at
        
    FROM options_data
    GROUP BY 
        ticker, 
        trade_date, 
        expiry_date, 
        strike, 
        days_to_expiry, 
        moneyness
)

SELECT
    ticker || '_' || trade_date || '_' || expiry_date || '_' || strike AS vol_surface_key,
    *
FROM aggregated
ORDER BY trade_date DESC, expiry_date, strike
