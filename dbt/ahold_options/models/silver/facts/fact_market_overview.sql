{{
    config(
        materialized='incremental',
        unique_key=['underlying_id', 'trade_date'],
        on_schema_change='append_new_columns'
    )
}}

WITH bronze_overview AS (
    SELECT
        ticker,
        symbol_code,
        trade_date,
        
        -- Underlying price data
        koers AS underlying_price,
        vorige AS underlying_open,
        hoog AS underlying_high,
        laag AS underlying_low,
        volume_underlying AS underlying_volume,
        delta AS underlying_change,
        delta_pct AS underlying_change_pct,
        
        -- Total volume
        totaal_volume AS total_volume,
        totaal_volume_calls AS total_call_volume,
        totaal_volume_puts AS total_put_volume,
        
        -- Total open interest
        totaal_oi AS total_oi,
        totaal_oi_calls AS total_call_oi,
        totaal_oi_puts AS total_put_oi,
        
        -- Ratios
        call_put_ratio,
        
        -- Metadata
        tijd AS market_time,
        scraped_at,
        
        ROW_NUMBER() OVER (
            PARTITION BY ticker, trade_date 
            ORDER BY scraped_at DESC
        ) AS rn
    FROM {{ source('bronze', 'bronze_fd_overview') }}
    
    {% if is_incremental() %}
    -- Only process new data
    WHERE trade_date > (SELECT MAX(trade_date) FROM {{ this }})
    {% endif %}
),

latest_overview AS (
    SELECT * FROM bronze_overview
    WHERE rn = 1  -- Take most recent scrape per ticker/date
),

underlying_dim AS (
    SELECT 
        underlying_id,
        ticker
    FROM {{ ref('dim_underlying') }}
)

SELECT
    -- Generate surrogate key using MD5 hash
    MD5(u.underlying_id || '_' || o.trade_date::TEXT) AS overview_id,
    
    -- Dimensions
    o.trade_date,
    u.underlying_id,
    
    -- Underlying price metrics
    o.underlying_price,
    o.underlying_open,
    o.underlying_high,
    o.underlying_low,
    o.underlying_volume,
    o.underlying_change,
    o.underlying_change_pct,
    
    -- Volume metrics
    o.total_volume,
    o.total_call_volume,
    o.total_put_volume,
    
    -- Open interest metrics
    o.total_oi,
    o.total_call_oi,
    o.total_put_oi,
    
    -- Calculated ratios
    CASE 
        WHEN o.total_put_volume > 0 
        THEN o.total_call_volume::FLOAT / o.total_put_volume::FLOAT
        ELSE NULL 
    END AS call_put_volume_ratio,
    
    CASE 
        WHEN o.total_put_oi > 0 
        THEN o.total_call_oi::FLOAT / o.total_put_oi::FLOAT
        ELSE NULL 
    END AS call_put_oi_ratio,
    
    -- Metadata
    o.market_time,
    'fd.nl' AS source,
    CURRENT_TIMESTAMP AS created_at

FROM latest_overview o
INNER JOIN underlying_dim u
    ON o.ticker = u.ticker
