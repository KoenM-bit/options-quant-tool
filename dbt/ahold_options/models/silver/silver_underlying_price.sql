/*
Silver Layer: Underlying Price History
Cleans and deduplicates underlying price data from bronze layer.
Calculates daily returns and validates data quality.
*/

{{ config(
    materialized='incremental',
    unique_key=['ticker', 'trade_date'],
    tags=['silver', 'underlying']
) }}

WITH source_data AS (
    SELECT
        ticker,
        peildatum AS trade_date,
        koers AS close_price,
        hoog AS high_price,
        laag AS low_price,
        vorige AS previous_close,
        volume_underlying AS volume,
        delta AS price_change,
        delta_pct AS price_change_pct,
        scraped_at,
        id AS source_id
    FROM {{ source('bronze', 'bronze_fd_overview') }}
    WHERE 
        peildatum IS NOT NULL
        AND koers IS NOT NULL
        AND koers > 0
        {% if is_incremental() %}
        AND scraped_at > COALESCE((SELECT MAX(created_at) FROM {{ this }}), '1900-01-01'::timestamp)
        {% endif %}
),

deduplicated AS (
    SELECT
        *,
        ROW_NUMBER() OVER (
            PARTITION BY ticker, trade_date 
            ORDER BY scraped_at DESC
        ) AS rn
    FROM source_data
),

calculated AS (
    SELECT
        ticker,
        trade_date,
        close_price,
        high_price,
        low_price,
        previous_close AS open_price,
        volume,
        
        -- Calculate returns
        CASE 
            WHEN previous_close > 0 
            THEN close_price - previous_close
            ELSE NULL
        END AS daily_return,
        
        CASE 
            WHEN previous_close > 0 
            THEN ((close_price - previous_close) / previous_close) * 100
            ELSE NULL
        END AS daily_return_pct,
        
        -- Data quality flags
        CASE 
            WHEN close_price > 0 
            AND high_price >= close_price 
            AND low_price <= close_price
            THEN TRUE
            ELSE FALSE
        END AS is_validated,
        
        source_id,
        CURRENT_TIMESTAMP AS created_at,
        CURRENT_TIMESTAMP AS updated_at
    FROM deduplicated
    WHERE rn = 1
)

SELECT
    *
FROM calculated
