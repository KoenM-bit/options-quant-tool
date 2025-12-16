{{
  config(
    materialized='incremental',
    unique_key=['option_id', 'trade_date'],
    tags=['silver', 'fact', 'eod']
  )
}}

-- depends_on: {{ ref('dim_option_contract') }}

-- fact_option_eod: End-of-day settlement data from FD.nl
-- This is the official closing prices, volume, and open interest from previous trading day

WITH bronze_fd AS (
    SELECT
        fo.trade_date,
        fo.ticker,
        fo.expiry_date,
        fo.strike,
        CASE WHEN LOWER(fo.option_type) = 'call' THEN 'C' ELSE 'P' END as call_put,
        fo.bid,
        fo.ask,
        (fo.bid + fo.ask) / 2.0 as mid_price,
        fo.laatste as last_price,
        fo.volume,
        fo.open_interest,
        fo.underlying_price::NUMERIC(10,4) as underlying_price,
        fo.scraped_at,
        'fd.nl' as source
    FROM {{ source('bronze', 'bronze_fd_options') }} fo
    WHERE fo.ticker IS NOT NULL
        AND fo.option_type IS NOT NULL
    
    {% if is_incremental() %}
        -- Multi-ticker support: Check max trade_date PER TICKER
        AND NOT EXISTS (
            SELECT 1 FROM {{ this }} existing
            JOIN {{ ref('dim_option_contract') }} c ON existing.option_id = c.option_id
            WHERE c.ticker = fo.ticker 
            AND existing.trade_date = fo.trade_date
        )
    {% endif %}
),

deduplicated AS (
    -- Take latest scrape per option/date (in case of re-scrapes)
    SELECT *,
        ROW_NUMBER() OVER (
            PARTITION BY ticker, expiry_date, strike, call_put, trade_date 
            ORDER BY scraped_at DESC
        ) AS rn
    FROM bronze_fd
)

SELECT
    ROW_NUMBER() OVER (ORDER BY d.trade_date, d.ticker, d.expiry_date, d.strike, d.call_put) as eod_id,
    d.trade_date,
    CURRENT_TIMESTAMP as ts,
    MD5(d.ticker || d.expiry_date::TEXT || d.strike::TEXT || d.call_put) as option_id,
    MD5(d.ticker) as underlying_id,
    
    -- EOD settlement price
    d.last_price,
    
    -- Market activity (OFFICIAL end-of-day numbers - the critical data!)
    d.volume,
    d.open_interest,
    
    -- Metadata
    d.source,
    CURRENT_TIMESTAMP as created_at

FROM deduplicated d
WHERE d.rn = 1
