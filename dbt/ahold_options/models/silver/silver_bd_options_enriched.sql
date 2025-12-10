/*
Silver Layer: BD Options Enriched with Greeks
Clean BD options data ready for Greeks calculation.

Data Flow:
1. Take BD bronze data (best bid/ask coverage)
2. Add underlying price from BD underlying table
3. Calculate days_to_expiry
4. Greeks will be added by Python enrichment AFTER this model runs

Quality: BD has 88-99% bid/ask coverage, providing reliable pricing data.
*/

{{ config(
    materialized='incremental',
    unique_key=['ticker', 'trade_date', 'option_type', 'strike', 'expiry_date'],
    tags=['silver', 'options', 'bd']
) }}

WITH bd_options_cleaned AS (
    -- Clean and standardize BD options data
    SELECT
        o.ticker,
        o.trade_date,
        o.option_type,
        o.strike,
        o.expiry_date,
        o.symbol_code,
        o.issue_id,
        o.bid,
        o.ask,
        o.last_price,
        o.volume,
        o.last_timestamp,
        o.source_url,
        o.scraped_at,
        -- Calculate mid price
        CASE 
            WHEN o.bid IS NOT NULL AND o.ask IS NOT NULL 
            THEN (o.bid + o.ask) / 2.0
            ELSE o.last_price
        END AS mid_price
    FROM {{ source('bronze', 'bronze_bd_options') }} o
    WHERE 
        o.expiry_date IS NOT NULL
        AND o.strike IS NOT NULL
        AND o.strike > 0
        {% if is_incremental() %}
        AND o.trade_date > COALESCE((SELECT MAX(trade_date) FROM {{ this }}), '1900-01-01'::date)
        {% endif %}
),

bd_options_deduped AS (
    -- Handle any duplicates by keeping most recent scrape
    SELECT DISTINCT ON (ticker, trade_date, option_type, strike, expiry_date)
        *
    FROM bd_options_cleaned
    ORDER BY ticker, trade_date, option_type, strike, expiry_date, scraped_at DESC
),

bd_underlying AS (
    -- Get underlying price for each trade date
    SELECT DISTINCT ON (ticker, trade_date)
        ticker,
        trade_date,
        last_price AS underlying_price,
        volume AS underlying_volume
    FROM {{ source('bronze', 'bronze_bd_underlying') }}
    WHERE last_price IS NOT NULL
    ORDER BY ticker, trade_date, scraped_at DESC
),

final AS (
    SELECT
        o.ticker,
        o.trade_date,
        o.option_type,
        o.strike,
        o.expiry_date,
        o.symbol_code,
        o.issue_id,
        
        -- Pricing
        o.bid,
        o.ask,
        o.mid_price,
        o.last_price,
        u.underlying_price,
        
        -- Trading activity
        o.volume,
        u.underlying_volume,
        o.last_timestamp,
        
        -- Calculated fields
        (o.expiry_date - o.trade_date) AS days_to_expiry,
        
        -- Moneyness indicator
        CASE 
            WHEN o.option_type = 'call' THEN 
                CASE 
                    WHEN u.underlying_price > o.strike * 1.05 THEN 'ITM'
                    WHEN u.underlying_price < o.strike * 0.95 THEN 'OTM'
                    ELSE 'ATM'
                END
            WHEN o.option_type = 'put' THEN 
                CASE 
                    WHEN u.underlying_price < o.strike * 0.95 THEN 'ITM'
                    WHEN u.underlying_price > o.strike * 1.05 THEN 'OTM'
                    ELSE 'ATM'
                END
        END AS moneyness,
        
        -- Greeks (NULL initially, populated by Python enrichment)
        NULL::DECIMAL(10,6) AS delta,
        NULL::DECIMAL(10,6) AS gamma,
        NULL::DECIMAL(10,6) AS theta,
        NULL::DECIMAL(10,6) AS vega,
        NULL::DECIMAL(10,6) AS rho,
        NULL::DECIMAL(8,4) AS implied_volatility,
        
        -- Metadata
        o.source_url,
        o.scraped_at,
        CURRENT_TIMESTAMP AS transformed_at
        
    FROM bd_options_deduped o
    LEFT JOIN bd_underlying u 
        ON o.ticker = u.ticker 
        AND o.trade_date = u.trade_date
    WHERE u.underlying_price IS NOT NULL  -- Require underlying price for Greeks calculation
)

SELECT * FROM final
