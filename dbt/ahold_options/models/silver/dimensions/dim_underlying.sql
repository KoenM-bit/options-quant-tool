{{
  config(
    materialized='table',
    tags=['silver', 'dimension']
  )
}}

-- dim_underlying: Extract unique underlyings from bronze

SELECT DISTINCT
    MD5(ticker) as underlying_id,
    ticker,
    MAX(name) as name,
    'Stock' as asset_class,
    NULL as sector,
    'Euronext Amsterdam' as exchange,
    'EUR' as currency,
    MAX(isin) as isin,
    CURRENT_TIMESTAMP as created_at,
    CURRENT_TIMESTAMP as updated_at
FROM {{ ref('bronze_bd_underlying') }}
WHERE ticker IS NOT NULL
GROUP BY ticker
