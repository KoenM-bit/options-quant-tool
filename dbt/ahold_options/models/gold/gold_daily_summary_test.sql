/*
Gold Layer: Daily Options Summary (from Merged Silver)
Simple daily aggregation to test the full pipeline: Bronze → Silver Merged → Gold.

This is a TEST model to validate the complete data flow with BD + FD merged data.
*/

{{ config(
    materialized='table',
    tags=['gold', 'summary', 'test']
) }}

WITH daily_summary AS (
    SELECT
        ticker,
        trade_date,
        
        -- Contract counts
        COUNT(*) AS total_contracts,
        COUNT(DISTINCT strike) AS unique_strikes,
        COUNT(DISTINCT expiry_date) AS unique_expiries,
        SUM(CASE WHEN option_type = 'Call' THEN 1 ELSE 0 END) AS call_contracts,
        SUM(CASE WHEN option_type = 'Put' THEN 1 ELSE 0 END) AS put_contracts,
        
        -- Data source coverage
        SUM(CASE WHEN has_bd_data THEN 1 ELSE 0 END) AS bd_contracts,
        SUM(CASE WHEN has_fd_data THEN 1 ELSE 0 END) AS fd_contracts,
        SUM(CASE WHEN has_bd_data AND has_fd_data THEN 1 ELSE 0 END) AS both_contracts,
        
        -- Pricing quality
        COUNT(bid) AS contracts_with_bid,
        COUNT(ask) AS contracts_with_ask,
        COUNT(CASE WHEN bid IS NOT NULL AND ask IS NOT NULL THEN 1 END) AS contracts_with_spread,
        AVG(bid_ask_spread_pct) AS avg_spread_pct,
        
        -- Volume metrics
        SUM(volume) AS total_volume,
        SUM(CASE WHEN option_type = 'Call' THEN volume ELSE 0 END) AS call_volume,
        SUM(CASE WHEN option_type = 'Put' THEN volume ELSE 0 END) AS put_volume,
        
        -- Open interest (FD only)
        SUM(open_interest) AS total_oi,
        SUM(CASE WHEN option_type = 'Call' THEN open_interest ELSE 0 END) AS call_oi,
        SUM(CASE WHEN option_type = 'Put' THEN open_interest ELSE 0 END) AS put_oi,
        
        -- Put/Call ratios
        CASE 
            WHEN SUM(CASE WHEN option_type = 'Call' THEN volume ELSE 0 END) > 0
            THEN SUM(CASE WHEN option_type = 'Put' THEN volume ELSE 0 END)::FLOAT / 
                 SUM(CASE WHEN option_type = 'Call' THEN volume ELSE 0 END)
            ELSE NULL
        END AS put_call_volume_ratio,
        
        CASE 
            WHEN SUM(CASE WHEN option_type = 'Call' THEN open_interest ELSE 0 END) > 0
            THEN SUM(CASE WHEN option_type = 'Put' THEN open_interest ELSE 0 END)::FLOAT / 
                 SUM(CASE WHEN option_type = 'Call' THEN open_interest ELSE 0 END)
            ELSE NULL
        END AS put_call_oi_ratio,
        
        -- Underlying price (use BD if available, else FD)
        MAX(underlying_price) AS underlying_price,
        MAX(bd_underlying_price) AS bd_underlying_price,
        MAX(fd_underlying_price) AS fd_underlying_price,
        
        -- Strike range
        MIN(strike) AS min_strike,
        MAX(strike) AS max_strike,
        
        -- Moneyness distribution
        SUM(CASE WHEN moneyness IS NOT NULL AND moneyness > 1.02 THEN 1 ELSE 0 END) AS itm_contracts,
        SUM(CASE WHEN moneyness IS NOT NULL AND ABS(moneyness - 1.0) <= 0.02 THEN 1 ELSE 0 END) AS atm_contracts,
        SUM(CASE WHEN moneyness IS NOT NULL AND moneyness < 0.98 THEN 1 ELSE 0 END) AS otm_contracts,
        
        -- Quality flags
        SUM(CASE WHEN is_validated THEN 1 ELSE 0 END) AS validated_contracts,
        SUM(CASE WHEN has_volume THEN 1 ELSE 0 END) AS contracts_with_volume,
        SUM(CASE WHEN is_liquid THEN 1 ELSE 0 END) AS liquid_contracts,
        
        -- Timestamps
        MAX(as_of_ts) AS latest_scrape_time,
        MAX(created_at) AS silver_created_at
        
    FROM {{ ref('silver_options_chain_merged') }}
    GROUP BY ticker, trade_date
)

SELECT
    -- Generate unique key
    ticker || '_' || trade_date::TEXT AS summary_key,
    
    -- All summary fields
    *,
    
    -- Calculate data quality score (0-100)
    CASE 
        WHEN total_contracts > 0 THEN
            (
                (contracts_with_spread::FLOAT / total_contracts * 40) +  -- 40% weight on spread availability
                (validated_contracts::FLOAT / total_contracts * 30) +     -- 30% weight on validation
                (bd_contracts::FLOAT / total_contracts * 30)              -- 30% weight on BD coverage
            )
        ELSE 0
    END AS data_quality_score,
    
    -- Pipeline metadata
    CURRENT_TIMESTAMP AS created_at
    
FROM daily_summary
ORDER BY trade_date DESC, ticker
