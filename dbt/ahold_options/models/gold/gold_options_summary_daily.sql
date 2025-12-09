/*
Gold Layer: Daily Options Summary
Daily aggregated metrics for high-level dashboards.
*/

{{ config(
    materialized='incremental',
    unique_key='summary_key',
    tags=['gold', 'summary', 'daily']
) }}

WITH underlying AS (
    SELECT
        ticker,
        trade_date,
        close_price AS underlying_close,
        daily_return_pct AS underlying_return_pct
    FROM {{ ref('silver_underlying_price') }}
    {% if is_incremental() %}
    WHERE trade_date > COALESCE((SELECT MAX(trade_date) FROM {{ this }}), '1900-01-01'::date)
    {% endif %}
),

options_agg AS (
    SELECT
        ticker,
        trade_date,
        
        -- Volume metrics
        SUM(volume) AS total_volume,
        SUM(CASE WHEN option_type = 'Call' THEN volume ELSE 0 END) AS call_volume,
        SUM(CASE WHEN option_type = 'Put' THEN volume ELSE 0 END) AS put_volume,
        
        -- OI metrics
        SUM(open_interest) AS total_oi,
        SUM(CASE WHEN option_type = 'Call' THEN open_interest ELSE 0 END) AS call_oi,
        SUM(CASE WHEN option_type = 'Put' THEN open_interest ELSE 0 END) AS put_oi,
        
        -- OI changes
        SUM(oi_change) AS total_oi_change,
        SUM(CASE WHEN option_type = 'Call' THEN oi_change ELSE 0 END) AS call_oi_change,
        SUM(CASE WHEN option_type = 'Put' THEN oi_change ELSE 0 END) AS put_oi_change,
        
        -- Strike metrics
        MIN(strike) AS min_strike,
        MAX(strike) AS max_strike,
        COUNT(DISTINCT strike) AS num_strikes,
        
        -- Volatility metrics
        AVG(implied_volatility) AS avg_implied_vol,
        AVG(CASE WHEN ABS(moneyness - 1.0) < 0.05 THEN implied_volatility END) AS atm_implied_vol,
        
        -- Greeks (only from validated records)
        SUM(CASE WHEN option_type = 'Call' AND greeks_valid = TRUE THEN delta * open_interest END) AS total_call_delta,
        SUM(CASE WHEN option_type = 'Put' AND greeks_valid = TRUE THEN delta * open_interest END) AS total_put_delta,
        SUM(CASE WHEN greeks_valid = TRUE THEN gamma * open_interest END) AS total_gamma,
        SUM(CASE WHEN greeks_valid = TRUE THEN vega * open_interest END) AS total_vega
        
    FROM {{ ref('silver_options') }}
    WHERE 1=1
        {% if is_incremental() %}
        AND trade_date > COALESCE((SELECT MAX(trade_date) FROM {{ this }}), '1900-01-01'::date)
        {% endif %}
        -- Note: We still include all options for volume/OI, but only validated Greeks for exposure calculations
    GROUP BY ticker, trade_date
),

joined AS (
    SELECT
        u.ticker,
        u.trade_date,
        u.underlying_close,
        u.underlying_return_pct,
        
        o.total_volume,
        o.call_volume,
        o.put_volume,
        CASE 
            WHEN o.put_volume > 0 
            THEN o.call_volume::FLOAT / o.put_volume
            ELSE NULL
        END AS call_put_volume_ratio,
        
        o.total_oi,
        o.call_oi,
        o.put_oi,
        CASE 
            WHEN o.put_oi > 0 
            THEN o.call_oi::FLOAT / o.put_oi
            ELSE NULL
        END AS call_put_oi_ratio,
        
        o.total_oi_change,
        o.call_oi_change,
        o.put_oi_change,
        
        -- Find ATM strike
        (SELECT strike 
         FROM {{ ref('silver_options') }} s
         WHERE s.ticker = u.ticker 
         AND s.trade_date = u.trade_date
         ORDER BY ABS(s.strike - u.underlying_close)
         LIMIT 1
        ) AS atm_strike,
        
        o.min_strike,
        o.max_strike,
        o.num_strikes,
        
        o.avg_implied_vol,
        o.atm_implied_vol,
        
        -- Volatility skew (put IV - call IV for OTM options) - only validated Greeks
        (SELECT AVG(CASE WHEN option_type = 'Put' THEN implied_volatility END) -
                AVG(CASE WHEN option_type = 'Call' THEN implied_volatility END)
         FROM {{ ref('silver_options') }} s
         WHERE s.ticker = u.ticker
         AND s.trade_date = u.trade_date
         AND s.greeks_valid = TRUE  -- ✅ Only validated Greeks
         AND s.moneyness < 0.95
        ) AS vol_skew,
        
        o.total_call_delta,
        o.total_put_delta,
        o.total_call_delta + o.total_put_delta AS net_delta,
        o.total_gamma,
        o.total_vega,
        
        CURRENT_TIMESTAMP AS created_at,
        CURRENT_TIMESTAMP AS updated_at
        
    FROM underlying u
    LEFT JOIN options_agg o
        ON u.ticker = o.ticker
        AND u.trade_date = o.trade_date
)

SELECT
    ticker || '_' || trade_date AS summary_key,
    *
FROM joined
