{{
  config(
    materialized='incremental',
    unique_key=['option_id', 'trade_date'],
    tags=['silver', 'fact']
  )
}}

-- depends_on: {{ ref('dim_option_contract') }}

-- fact_option_timeseries: Intraday pricing from Beursduivel ONLY
-- This is real-time bid/ask data scraped during trading hours

WITH bronze_data AS (
    SELECT
        bo.trade_date,
        bo.ticker,
        bo.expiry_date,
        bo.strike,
        CASE WHEN LOWER(bo.option_type) = 'call' THEN 'C' ELSE 'P' END as call_put,
        bo.bid,
        bo.ask,
        (bo.bid + bo.ask) / 2.0 as mid_price,
        bo.last_price,
        bo.volume,
        NULL::INTEGER as open_interest,  -- BD doesn't provide OI
        bu.last_price::NUMERIC(10,4) as underlying_price,
        bu.bid::NUMERIC(10,4) as underlying_bid,
        bu.ask::NUMERIC(10,4) as underlying_ask,
        'beursduivel' as source
    FROM {{ source('bronze', 'bronze_bd_options') }} bo
    LEFT JOIN {{ source('bronze', 'bronze_bd_underlying') }} bu
        ON bo.ticker = bu.ticker
        AND bo.trade_date = bu.trade_date::DATE
    WHERE bo.ticker IS NOT NULL
        AND bo.option_type IS NOT NULL
    
    {% if is_incremental() %}
        -- Multi-ticker support: Check max trade_date PER TICKER
        AND NOT EXISTS (
            SELECT 1 FROM {{ this }} existing
            JOIN {{ ref('dim_option_contract') }} c ON existing.option_id = c.option_id
            WHERE c.ticker = bo.ticker 
            AND existing.trade_date = bo.trade_date
        )
    {% endif %}
)

SELECT
    ROW_NUMBER() OVER (ORDER BY b.trade_date, b.ticker, b.expiry_date, b.strike, b.call_put) as ts_id,
    b.trade_date,
    CURRENT_TIMESTAMP as ts,
    MD5(b.ticker || b.expiry_date::TEXT || b.strike::TEXT || b.call_put) as option_id,
    MD5(b.ticker) as underlying_id,
    b.underlying_price,
    b.underlying_bid,
    b.underlying_ask,
    b.bid,
    b.ask,
    b.mid_price,
    b.last_price,
    NULL::NUMERIC(10,6) as iv,
    NULL::NUMERIC(10,6) as delta,
    NULL::NUMERIC(10,6) as gamma,
    NULL::NUMERIC(10,6) as vega,
    NULL::NUMERIC(10,6) as theta,
    NULL::NUMERIC(10,6) as rho,
    b.volume,
    b.open_interest,
    CASE 
        WHEN b.call_put = 'C' THEN GREATEST(b.underlying_price - b.strike, 0)
        ELSE GREATEST(b.strike - b.underlying_price, 0)
    END as intrinsic_value,
    b.mid_price - CASE 
        WHEN b.call_put = 'C' THEN GREATEST(b.underlying_price - b.strike, 0)
        ELSE GREATEST(b.strike - b.underlying_price, 0)
    END as time_value,
    b.underlying_price / NULLIF(b.strike, 0) as moneyness,
    b.expiry_date - b.trade_date as days_to_expiry,
    b.source,
    CURRENT_TIMESTAMP as created_at
FROM bronze_data b
