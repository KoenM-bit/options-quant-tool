/*
Silver Layer: Options Data
Cleans options data and calculates derived metrics.
*/

{{ config(
    materialized='incremental',
    unique_key=['ticker', 'option_type', 'strike', 'expiry_date', 'trade_date'],
    tags=['silver', 'options']
) }}

WITH source_data AS (
    SELECT
        ticker,
        isin,
        option_type,
        strike,
        expiry_date,
        -- Derive trade_date: Weekend scrapes contain Friday's data
        CASE 
            WHEN EXTRACT(DOW FROM scraped_at) = 6 THEN DATE(scraped_at) - INTERVAL '1 day'  -- Saturday scrape = Friday data
            WHEN EXTRACT(DOW FROM scraped_at) = 0 THEN DATE(scraped_at) - INTERVAL '2 days' -- Sunday scrape = Friday data
            ELSE DATE(scraped_at) - INTERVAL '1 day'  -- Weekday evening scrape = previous day data
        END AS trade_date,
        bid,
        ask,
        laatste AS last_price,
        volume,
        open_interest,
        underlying_price,
        -- Greeks removed - calculated in silver_options_with_greeks Python model
        -- delta,
        -- gamma,
        -- theta,
        -- vega,
        -- implied_volatility,
        scraped_at,
        id AS source_id
    FROM {{ source('bronze', 'bronze_fd_options') }}
    WHERE 
        expiry_date IS NOT NULL
        AND strike IS NOT NULL
        AND strike > 0
        {% if is_incremental() %}
        AND scraped_at > COALESCE((SELECT MAX(created_at) FROM {{ this }}), '1900-01-01'::timestamp)
        {% endif %}
),

deduplicated AS (
    SELECT
        *,
        ROW_NUMBER() OVER (
            PARTITION BY ticker, option_type, strike, expiry_date, trade_date
            ORDER BY scraped_at DESC
        ) AS rn
    FROM source_data
),

calculated AS (
    SELECT
        ticker,
        isin,
        option_type,
        strike,
        expiry_date,
        trade_date,
        bid,
        ask,
        CASE 
            WHEN bid IS NOT NULL AND ask IS NOT NULL 
            THEN (bid + ask) / 2.0
            ELSE last_price
        END AS mid_price,
        last_price,
        volume,
        open_interest,
        
        -- OI change (requires previous day data)
        open_interest - LAG(open_interest, 1) OVER (
            PARTITION BY ticker, option_type, strike, expiry_date
            ORDER BY trade_date
        ) AS oi_change,
        
        underlying_price,
        
        -- Moneyness
        CASE 
            WHEN option_type = 'Call' AND underlying_price > 0 
            THEN underlying_price / strike
            WHEN option_type = 'Put' AND strike > 0
            THEN strike / underlying_price
            ELSE NULL
        END AS moneyness,
        
        -- Intrinsic value
        CASE 
            WHEN option_type = 'Call' 
            THEN GREATEST(underlying_price - strike, 0)
            WHEN option_type = 'Put'
            THEN GREATEST(strike - underlying_price, 0)
            ELSE 0
        END AS intrinsic_value,
        
        -- Days to expiry (extract days from interval)
        EXTRACT(DAY FROM (expiry_date - trade_date))::INTEGER AS days_to_expiry,
        
        -- Greeks (calculated by Airflow AFTER this model runs)
        -- Initially NULL, enriched by enrich_silver_with_greeks()
        NULL::FLOAT AS delta,
        NULL::FLOAT AS gamma,
        NULL::FLOAT AS theta,
        NULL::FLOAT AS vega,
        NULL::FLOAT AS rho,
    NULL::FLOAT AS implied_volatility,
    -- Greeks metadata (populated by enrichment step)
    NULL::FLOAT AS risk_free_rate_used,
    NULL::BOOLEAN AS greeks_valid,
    NULL::VARCHAR(50) AS greeks_status,
        
        -- Spread metrics
        CASE 
            WHEN bid IS NOT NULL AND ask IS NOT NULL 
            THEN ask - bid
            ELSE NULL
        END AS bid_ask_spread,
        
        CASE 
            WHEN bid IS NOT NULL AND ask IS NOT NULL AND ask > 0
            THEN ((ask - bid) / ask) * 100
            ELSE NULL
        END AS bid_ask_spread_pct,
        
        -- Data quality flags
        CASE 
            WHEN bid IS NOT NULL AND ask IS NOT NULL AND bid <= ask
            THEN TRUE
            ELSE FALSE
        END AS is_validated,
        
        CASE 
            WHEN volume IS NOT NULL AND volume > 0 
            THEN TRUE
            ELSE FALSE
        END AS has_volume,
        
        CASE 
            WHEN volume > 10 AND open_interest > 50
            THEN TRUE
            ELSE FALSE
        END AS is_liquid,
        
        source_id,
        CURRENT_TIMESTAMP AS created_at,
        CURRENT_TIMESTAMP AS updated_at
    FROM deduplicated
    WHERE rn = 1
),

with_time_value AS (
    SELECT
        *,
        CASE 
            WHEN mid_price IS NOT NULL 
            THEN mid_price - intrinsic_value
            ELSE NULL
        END AS time_value
    FROM calculated
)

SELECT
    *
FROM with_time_value
