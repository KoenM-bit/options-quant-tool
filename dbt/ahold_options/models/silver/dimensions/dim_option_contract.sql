{{
  config(
    materialized='table',
    tags=['silver', 'dimension']
  )
}}

-- dim_option_contract: Extract unique option contracts from bronze

WITH all_contracts AS (
    -- From Beursduivel
    SELECT DISTINCT
        ticker,
        expiry_date as expiration_date,
        strike,
        CASE WHEN LOWER(option_type) = 'call' THEN 'C' ELSE 'P' END as call_put,
        symbol_code,
        issue_id,
        NULL as isin
    FROM {{ ref('bronze_bd_options') }}
    WHERE ticker IS NOT NULL
        AND option_type IS NOT NULL
        AND strike IS NOT NULL
        AND expiry_date IS NOT NULL
    
    UNION
    
    -- From FD
    SELECT DISTINCT
        ticker,
        expiry_date as expiration_date,
        strike,
        CASE WHEN LOWER(option_type) = 'call' THEN 'C' ELSE 'P' END as call_put,
        NULL as symbol_code,
        NULL as issue_id,
        isin
    FROM {{ ref('bronze_fd_options') }}
    WHERE ticker IS NOT NULL
        AND option_type IS NOT NULL
        AND strike IS NOT NULL
        AND expiry_date IS NOT NULL
),

merged AS (
    SELECT
        ticker,
        expiration_date,
        strike,
        call_put,
        MAX(symbol_code) as symbol_code,
        MAX(issue_id) as issue_id,
        MAX(isin) as isin
    FROM all_contracts
    GROUP BY ticker, expiration_date, strike, call_put
)

SELECT
    MD5(ticker || expiration_date::TEXT || strike::TEXT || call_put) as option_id,
    MD5(ticker) as underlying_id,
    ticker,
    expiration_date,
    strike,
    call_put,
    100 as contract_size,
    'European' as style,
    symbol_code,
    issue_id,
    isin,
    CURRENT_TIMESTAMP as created_at,
    CURRENT_TIMESTAMP as updated_at
FROM merged
