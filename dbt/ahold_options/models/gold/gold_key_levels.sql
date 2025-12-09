{{
    config(
        materialized='incremental',
        unique_key=['ticker', 'trade_date', 'expiry_date', 'calculation_timestamp'],
        on_schema_change='sync_all_columns',
        incremental_strategy='append'
    )
}}

/*
KEY PRICE LEVELS - INSTITUTIONAL SUPPORT/RESISTANCE
====================================================
Consolidates all option-derived price levels into actionable trading levels:
1. Gamma walls (highest GEX strikes)
2. Zero gamma level (GEX flip point)
3. Max pain strike (pinning target)
4. High OI strikes (hedging concentrations)
5. Vanna trigger points (IV-sensitive levels)

These levels act as magnetic price zones where:
- Dealers concentrate hedging activity
- Price action slows or reverses
- Volume and volatility cluster
*/

WITH gamma_levels AS (
    SELECT
        ticker,
        trade_date,
        expiry_date,
        underlying_price,
        days_to_expiry,
        
        -- Top 3 gamma walls
        MAX(CASE WHEN gex_rank = 1 THEN strike END) AS gamma_wall_1,
        MAX(CASE WHEN gex_rank = 1 THEN net_dealer_gex END) AS gamma_wall_1_gex,
        MAX(CASE WHEN gex_rank = 1 THEN gex_classification END) AS gamma_wall_1_type,
        
        MAX(CASE WHEN gex_rank = 2 THEN strike END) AS gamma_wall_2,
        MAX(CASE WHEN gex_rank = 2 THEN net_dealer_gex END) AS gamma_wall_2_gex,
        
        MAX(CASE WHEN gex_rank = 3 THEN strike END) AS gamma_wall_3,
        MAX(CASE WHEN gex_rank = 3 THEN net_dealer_gex END) AS gamma_wall_3_gex,
        
        -- Zero gamma level (where GEX crosses zero)
        -- Approximate as strike closest to zero GEX
        MIN(CASE 
            WHEN ABS(net_dealer_gex) < expiry_total_gex * 0.05 
            THEN strike 
            ELSE NULL 
        END) AS zero_gamma_strike,
        
        -- Aggregate metrics
        SUM(net_dealer_gex) AS total_net_gex,
        SUM(ABS(net_dealer_gex)) AS total_abs_gex,
        
        -- Put/Call concentrations
        MAX(CASE WHEN call_dealer_gex < 0 THEN strike END) AS max_call_gex_strike,
        MIN(call_dealer_gex) AS max_call_gex_value,
        
        MAX(CASE WHEN put_dealer_gex > 0 THEN strike END) AS max_put_gex_strike,
        MAX(put_dealer_gex) AS max_put_gex_value
        
    FROM {{ ref('gold_gamma_exposure') }}
    WHERE 1=1
        {% if is_incremental() %}
        AND trade_date > COALESCE((SELECT MAX(trade_date) FROM {{ this }}), '1900-01-01'::date)
        {% endif %}
    GROUP BY ticker, trade_date, expiry_date, underlying_price, days_to_expiry
),

max_pain_levels AS (
    SELECT
        ticker,
        trade_date,
        expiry_date,
        max_pain_strike,
        max_pain_distance_pct,
        pin_risk_level,
        total_oi,
        put_call_oi_ratio
    FROM {{ ref('gold_max_pain') }}
    WHERE 1=1
        {% if is_incremental() %}
        AND trade_date > COALESCE((SELECT MAX(trade_date) FROM {{ this }}), '1900-01-01'::date)
        {% endif %}
),

oi_ranked AS (
    SELECT
        ticker,
        trade_date,
        expiry_date,
        strike,
        total_oi,
        ROW_NUMBER() OVER (PARTITION BY ticker, trade_date, expiry_date ORDER BY total_oi DESC) AS oi_rank
    FROM {{ ref('gold_gamma_exposure') }}
    WHERE 1=1
        {% if is_incremental() %}
        AND trade_date > COALESCE((SELECT MAX(trade_date) FROM {{ this }}), '1900-01-01'::date)
        {% endif %}
),

oi_concentrations AS (
    SELECT
        ticker,
        trade_date,
        expiry_date,
        
        -- Top 3 OI strikes
        MAX(CASE WHEN oi_rank = 1 THEN strike END) AS max_oi_strike_1,
        MAX(CASE WHEN oi_rank = 1 THEN total_oi END) AS max_oi_value_1,
        MAX(CASE WHEN oi_rank = 2 THEN strike END) AS max_oi_strike_2,
        MAX(CASE WHEN oi_rank = 3 THEN strike END) AS max_oi_strike_3
        
    FROM oi_ranked
    GROUP BY ticker, trade_date, expiry_date
),

vanna_ranked AS (
    SELECT
        ticker,
        trade_date,
        expiry_date,
        strike,
        net_vega_exposure,
        ROW_NUMBER() OVER (PARTITION BY ticker, trade_date, expiry_date ORDER BY ABS(net_vega_exposure) DESC) AS vanna_rank
    FROM {{ ref('gold_gamma_exposure') }}
    WHERE net_vega_exposure IS NOT NULL
        {% if is_incremental() %}
        AND trade_date > COALESCE((SELECT MAX(trade_date) FROM {{ this }}), '1900-01-01'::date)
        {% endif %}
),

vanna_levels AS (
    -- Strikes with high vanna (delta sensitivity to IV changes)
    SELECT
        ticker,
        trade_date,
        expiry_date,
        
        -- High vanna strikes (most sensitive to IV)
        MAX(CASE WHEN vanna_rank = 1 THEN strike END) AS high_vanna_strike_1,
        MAX(CASE WHEN vanna_rank = 1 THEN net_vega_exposure END) AS high_vanna_value_1,
        MAX(CASE WHEN vanna_rank = 2 THEN strike END) AS high_vanna_strike_2
        
    FROM vanna_ranked
    GROUP BY ticker, trade_date, expiry_date
),

level_consolidation AS (
    SELECT
        gl.ticker,
        gl.trade_date,
        gl.expiry_date,
        gl.underlying_price,
        gl.days_to_expiry,
        
        -- GAMMA LEVELS (KEY)
        gl.gamma_wall_1,
        gl.gamma_wall_1_gex,
        gl.gamma_wall_1_type,
        gl.gamma_wall_2,
        gl.gamma_wall_2_gex,
        gl.gamma_wall_3,
        gl.gamma_wall_3_gex,
        gl.zero_gamma_strike,
        
        -- MAX PAIN (KEY)
        mp.max_pain_strike,
        mp.max_pain_distance_pct,
        mp.pin_risk_level,
        
        -- OI CONCENTRATIONS
        oi.max_oi_strike_1,
        oi.max_oi_value_1,
        oi.max_oi_strike_2,
        oi.max_oi_strike_3,
        
        -- VANNA LEVELS
        va.high_vanna_strike_1,
        va.high_vanna_value_1,
        va.high_vanna_strike_2,
        
        -- AGGREGATE METRICS
        gl.total_net_gex,
        gl.total_abs_gex,
        mp.total_oi,
        mp.put_call_oi_ratio,
        
        -- GEX concentrations
        gl.max_call_gex_strike,
        gl.max_call_gex_value,
        gl.max_put_gex_strike,
        gl.max_put_gex_value
        
    FROM gamma_levels gl
    LEFT JOIN max_pain_levels mp
        ON gl.ticker = mp.ticker
        AND gl.trade_date = mp.trade_date
        AND gl.expiry_date = mp.expiry_date
    LEFT JOIN oi_concentrations oi
        ON gl.ticker = oi.ticker
        AND gl.trade_date = oi.trade_date
        AND gl.expiry_date = oi.expiry_date
    LEFT JOIN vanna_levels va
        ON gl.ticker = va.ticker
        AND gl.trade_date = va.trade_date
        AND gl.expiry_date = va.expiry_date
)

SELECT
    ticker,
    trade_date,
    expiry_date,
    underlying_price,
    days_to_expiry,
    
    -- PRIMARY SUPPORT/RESISTANCE LEVELS
    gamma_wall_1 AS resistance_1,
    gamma_wall_1_gex AS resistance_1_strength,
    'Gamma Wall' AS resistance_1_type,
    
    gamma_wall_2 AS resistance_2,
    gamma_wall_2_gex AS resistance_2_strength,
    
    gamma_wall_3 AS resistance_3,
    gamma_wall_3_gex AS resistance_3_strength,
    
    -- MAX PAIN (magnetic level)
    max_pain_strike,
    max_pain_distance_pct AS max_pain_distance,
    pin_risk_level,
    
    -- ZERO GAMMA (critical flip point)
    zero_gamma_strike AS gex_flip_level,
    CASE 
        WHEN underlying_price > zero_gamma_strike THEN 'Positive GEX Zone'
        WHEN underlying_price < zero_gamma_strike THEN 'Negative GEX Zone'
        ELSE 'At Zero Gamma'
    END AS current_gex_zone,
    
    -- OI CONCENTRATIONS (hedging levels)
    max_oi_strike_1 AS key_oi_level_1,
    max_oi_value_1 AS key_oi_level_1_contracts,
    max_oi_strike_2 AS key_oi_level_2,
    max_oi_strike_3 AS key_oi_level_3,
    
    -- VANNA LEVELS (IV trigger points)
    high_vanna_strike_1 AS vanna_trigger_1,
    high_vanna_value_1 AS vanna_sensitivity_1,
    high_vanna_strike_2 AS vanna_trigger_2,
    
    -- CALL/PUT WALLS
    max_call_gex_strike AS call_wall,
    max_call_gex_value AS call_wall_strength,
    max_put_gex_strike AS put_wall,
    max_put_gex_value AS put_wall_strength,
    
    -- MARKET REGIME INDICATORS
    CASE 
        WHEN total_net_gex > 0 THEN 'Dealer Long Gamma (Stabilizing)'
        WHEN total_net_gex < 0 THEN 'Dealer Short Gamma (Volatility Amplifying)'
        ELSE 'Neutral'
    END AS market_regime,
    
    CASE 
        WHEN put_call_oi_ratio > 1.5 THEN 'Very Bearish'
        WHEN put_call_oi_ratio > 1.0 THEN 'Bearish'
        WHEN put_call_oi_ratio < 0.7 THEN 'Bullish'
        ELSE 'Neutral'
    END AS sentiment_from_oi,
    
    -- ACTIONABLE FLAGS
    CASE 
        WHEN ABS(max_pain_strike - underlying_price) / NULLIF(underlying_price, 0) < 0.02 
            AND days_to_expiry <= 3 
        THEN TRUE ELSE FALSE 
    END AS high_pin_probability,
    
    CASE 
        WHEN underlying_price BETWEEN gamma_wall_1 * 0.99 AND gamma_wall_1 * 1.01 
        THEN TRUE ELSE FALSE 
    END AS at_gamma_wall,
    
    CASE 
        WHEN underlying_price BETWEEN zero_gamma_strike * 0.99 AND zero_gamma_strike * 1.01 
        THEN TRUE ELSE FALSE 
    END AS at_gex_flip,
    
    -- AGGREGATE METRICS
    total_net_gex,
    total_abs_gex,
    total_oi,
    put_call_oi_ratio,
    
    -- TIME BUCKET
    CASE 
        WHEN days_to_expiry = 0 THEN 'Expiration Day'
        WHEN days_to_expiry <= 3 THEN 'Week of Expiry'
        WHEN days_to_expiry <= 7 THEN 'Within 1 Week'
        WHEN days_to_expiry <= 14 THEN 'Within 2 Weeks'
        WHEN days_to_expiry <= 30 THEN 'Within 1 Month'
        ELSE 'More than 1 Month'
    END AS time_to_expiry,
    
    -- Metadata
    CURRENT_TIMESTAMP AS created_at

FROM level_consolidation

ORDER BY ticker, trade_date, days_to_expiry
