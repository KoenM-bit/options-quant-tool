/*
WEEKLY GAMMA EXPOSURE (GEX) ANALYSIS
=====================================
FRIDAY DATA ONLY - OI is only accurate on weekends (Saturday scrapes of Friday close)

Calculates dealer gamma exposure to identify:
- Gamma walls (high concentration levels)
- Zero gamma flip point
- Hedging pressure zones
- Pinning levels for upcoming week

FORMULA: Dealer GEX = OI × Gamma × 100 × S²
- Dealers are SHORT options (opposite to buyers)
- Call GEX: Negative (dealers hedge by BUYING stock as price rises)
- Put GEX: Positive (dealers hedge by SELLING stock as price falls)

GEX Regime:
- Positive Net GEX: Dealers stabilize (sell rallies, buy dips) → Low volatility
- Negative Net GEX: Dealers amplify (buy rallies, sell dips) → High volatility
- Zero GEX: Critical flip level
*/

{{ config(
    materialized='incremental',
    unique_key='gex_key',
    tags=['gold', 'gamma', 'gex', 'weekly']
) }}

WITH friday_data_only AS (
    -- Only use Friday trading data (or Thursday if Friday is holiday)
    -- Saturday scrapes contain Friday's accurate closing OI
    SELECT 
        trade_date,
        EXTRACT(DOW FROM trade_date) AS day_of_week,
        -- Get the Friday of this week
        trade_date + INTERVAL '1 day' * (5 - EXTRACT(DOW FROM trade_date))::INTEGER AS week_ending,
        ticker,
        option_type,
        strike,
        expiry_date,
        open_interest,
        gamma,
        delta,
        vega,
        implied_volatility,
        underlying_price,
        (expiry_date - trade_date) AS days_to_expiry,
        last_price
    FROM {{ ref('silver_options') }}
    WHERE 
        open_interest IS NOT NULL
        AND open_interest > 0
        AND gamma IS NOT NULL
        AND gamma > 0
        AND underlying_price > 0
        -- Only Friday or if no Friday, take Thursday/Saturday
        AND EXTRACT(DOW FROM trade_date) IN (5, 4, 6)  -- Friday=5, Thurs=4, Sat=6
        {% if is_incremental() %}
        AND created_at > COALESCE((SELECT MAX(created_at) FROM {{ this }}), '1900-01-01'::timestamp)
        {% endif %}
),

-- Get most recent Friday per week
weekly_data AS (
    SELECT
        *,
        ROW_NUMBER() OVER (
            PARTITION BY week_ending, ticker, option_type, strike, expiry_date
            ORDER BY trade_date DESC
        ) AS rn
    FROM friday_data_only
),

current_week AS (
    SELECT * 
    FROM weekly_data 
    WHERE rn = 1  -- Most recent Friday data for each week
),

-- Calculate GEX per option contract
option_gex AS (
    SELECT
        week_ending,
        trade_date,
        ticker,
        option_type,
        strike,
        expiry_date,
        underlying_price,
        open_interest,
        gamma,
        delta,
        vega,
        implied_volatility,
        EXTRACT(DAY FROM days_to_expiry) AS days_to_expiry,
        last_price,
        
        -- Time decay weight (near-term has more hedging impact)
        CASE 
            WHEN EXTRACT(DAY FROM days_to_expiry) <= 7 THEN 2.0    -- Weekly options - highest impact
            WHEN EXTRACT(DAY FROM days_to_expiry) <= 30 THEN 1.5   -- Near-term monthlies
            WHEN EXTRACT(DAY FROM days_to_expiry) <= 60 THEN 1.0   -- Medium-term
            WHEN EXTRACT(DAY FROM days_to_expiry) <= 120 THEN 0.6  -- Quarterly
            ELSE 0.3                              -- LEAPS - minimal impact
        END AS time_weight,
        
        -- Moneyness
        CASE 
            WHEN option_type = 'Call' THEN underlying_price / strike
            WHEN option_type = 'Put' THEN strike / underlying_price
        END AS moneyness,
        
        -- GROSS Gamma Exposure (absolute value per contract)
        -- Formula: OI × Gamma × 100 shares × S²
        open_interest * gamma * 100 * POWER(underlying_price, 2) AS gross_gex,
        
        -- DEALER Gamma Exposure (what dealers must hedge)
        -- Dealers are SHORT, so opposite sign
        CASE 
            WHEN option_type = 'Call' THEN 
                -1 * open_interest * gamma * 100 * POWER(underlying_price, 2)
            WHEN option_type = 'Put' THEN 
                open_interest * gamma * 100 * POWER(underlying_price, 2)
            ELSE 0
        END AS dealer_gex,
        
        -- Delta exposure (directional hedge)
        open_interest * delta * 100 AS delta_exposure,
        
        -- Notional value
        open_interest * last_price * 100 AS notional_value
        
    FROM current_week
),

-- Aggregate by strike level (one row per week/strike/expiry)
strike_gex AS (
    SELECT
        week_ending,
        MAX(trade_date) AS trade_date,  -- Keep most recent trade date for reference
        ticker,
        strike,
        expiry_date,
        AVG(underlying_price) AS underlying_price,
        MAX(days_to_expiry) AS days_to_expiry,
        
        -- Open Interest by type
        SUM(CASE WHEN option_type = 'Call' THEN open_interest ELSE 0 END) AS call_oi,
        SUM(CASE WHEN option_type = 'Put' THEN open_interest ELSE 0 END) AS put_oi,
        SUM(open_interest) AS total_oi,
        
        -- Gross GEX (absolute exposure regardless of sign)
        SUM(CASE WHEN option_type = 'Call' THEN gross_gex ELSE 0 END) AS call_gross_gex,
        SUM(CASE WHEN option_type = 'Put' THEN gross_gex ELSE 0 END) AS put_gross_gex,
        SUM(gross_gex) AS total_gross_gex,
        
        -- Net Dealer GEX (THE KEY METRIC)
        SUM(dealer_gex) AS net_dealer_gex,
        SUM(CASE WHEN option_type = 'Call' THEN dealer_gex ELSE 0 END) AS call_dealer_gex,
        SUM(CASE WHEN option_type = 'Put' THEN dealer_gex ELSE 0 END) AS put_dealer_gex,
        
        -- Time-weighted GEX (near-term matters more)
        SUM(dealer_gex * time_weight) AS weighted_dealer_gex,
        
        -- Delta exposure (spot equivalent)
        SUM(delta_exposure) AS net_delta_exposure,
        
        -- Notional values
        SUM(notional_value) AS total_notional,
        
        -- IV metrics (separate by option type)
        AVG(implied_volatility) AS avg_iv,
        AVG(CASE WHEN option_type = 'Call' THEN implied_volatility END) AS call_iv,
        AVG(CASE WHEN option_type = 'Put' THEN implied_volatility END) AS put_iv
        
    FROM option_gex
    GROUP BY 
        week_ending,
        ticker,
        strike,
        expiry_date
),

-- Add distance calculation separately
strike_gex_with_distance AS (
    SELECT
        *,
        ABS(strike - underlying_price) AS distance_from_spot
    FROM strike_gex
),

-- Calculate expiry-level totals and find gamma flip
expiry_aggregates AS (
    SELECT
        week_ending,
        ticker,
        expiry_date,
        SUM(ABS(net_dealer_gex)) AS total_gex_magnitude,
        SUM(net_dealer_gex) AS expiry_net_gex,
        SUM(total_oi) AS expiry_total_oi,
        SUM(weighted_dealer_gex) AS expiry_weighted_gex
    FROM strike_gex_with_distance
    GROUP BY week_ending, ticker, expiry_date
),

-- Overall portfolio aggregates
portfolio_aggregates AS (
    SELECT
        week_ending,
        ticker,
        SUM(ABS(net_dealer_gex)) AS portfolio_total_gex,
        SUM(net_dealer_gex) AS portfolio_net_gex,
        SUM(weighted_dealer_gex) AS portfolio_weighted_gex,
        AVG(underlying_price) AS avg_underlying_price
    FROM strike_gex_with_distance
    GROUP BY week_ending, ticker
),

-- Identify gamma flip level (zero crossing)
gamma_flip AS (
    SELECT DISTINCT ON (sg.week_ending, sg.ticker, sg.expiry_date)
        sg.week_ending,
        sg.ticker,
        sg.expiry_date,
        sg.strike AS gamma_flip_strike,
        sg.net_dealer_gex AS flip_gex_value,
        sg.underlying_price
    FROM strike_gex_with_distance sg
    CROSS JOIN LATERAL (
        SELECT strike, net_dealer_gex
        FROM strike_gex_with_distance sg2
        WHERE sg2.week_ending = sg.week_ending
          AND sg2.ticker = sg.ticker
          AND sg2.expiry_date = sg.expiry_date
          AND sg2.strike != sg.strike
        ORDER BY ABS(sg2.strike - sg.underlying_price)
        LIMIT 1
    ) neighbor
    WHERE SIGN(sg.net_dealer_gex) != SIGN(neighbor.net_dealer_gex)
    ORDER BY sg.week_ending, sg.ticker, sg.expiry_date, distance_from_spot
),

-- Rank and classify strikes
ranked_strikes AS (
    SELECT
        sg.*,
        ea.total_gex_magnitude,
        ea.expiry_net_gex,
        ea.expiry_weighted_gex,
        pa.portfolio_net_gex,
        pa.portfolio_weighted_gex,
        gf.gamma_flip_strike,
        
        -- Concentration at this strike
        (ABS(sg.net_dealer_gex) / NULLIF(ea.total_gex_magnitude, 0)) * 100 AS gex_concentration_pct,
        
        -- Rank strikes by absolute GEX
        ROW_NUMBER() OVER (
            PARTITION BY sg.week_ending, sg.ticker, sg.expiry_date 
            ORDER BY ABS(sg.net_dealer_gex) DESC
        ) AS gex_rank,
        
        -- Classify strike importance
        CASE 
            WHEN ABS(sg.net_dealer_gex) > ea.total_gex_magnitude * 0.20 THEN 'Gamma Wall'
            WHEN ABS(sg.net_dealer_gex) > ea.total_gex_magnitude * 0.10 THEN 'High GEX'
            WHEN ABS(sg.net_dealer_gex) > ea.total_gex_magnitude * 0.03 THEN 'Medium GEX'
            ELSE 'Low GEX'
        END AS gex_tier,
        
        -- Moneyness classification
        CASE 
            WHEN ABS(sg.strike - sg.underlying_price) / sg.underlying_price < 0.025 THEN 'ATM'
            WHEN sg.strike > sg.underlying_price THEN 'OTM Calls / ITM Puts'
            ELSE 'ITM Calls / OTM Puts'
        END AS moneyness_zone,
        
        -- Expected behavior
        CASE 
            WHEN sg.net_dealer_gex > 0 THEN 'Stabilizing (dealers sell rallies, buy dips)'
            WHEN sg.net_dealer_gex < 0 THEN 'Amplifying (dealers buy rallies, sell dips)'
            ELSE 'Neutral'
        END AS hedging_behavior,
        
        -- Pin risk (strikes with high positive GEX attract price)
        CASE 
            WHEN sg.net_dealer_gex > ea.total_gex_magnitude * 0.15 
                 AND sg.days_to_expiry <= 7 
            THEN TRUE
            ELSE FALSE
        END AS has_pin_risk
        
    FROM strike_gex_with_distance sg
    JOIN expiry_aggregates ea 
        ON sg.week_ending = ea.week_ending
        AND sg.ticker = ea.ticker 
        AND sg.expiry_date = ea.expiry_date
    JOIN portfolio_aggregates pa
        ON sg.week_ending = pa.week_ending
        AND sg.ticker = pa.ticker
    LEFT JOIN gamma_flip gf
        ON sg.week_ending = gf.week_ending
        AND sg.ticker = gf.ticker
        AND sg.expiry_date = gf.expiry_date
)

SELECT
    week_ending || '_' || ticker || '_' || strike::text || '_' || expiry_date::text AS gex_key,
    week_ending,
    trade_date AS friday_trade_date,
    ticker,
    strike,
    expiry_date,
    days_to_expiry,
    underlying_price,
    
    -- Open Interest
    call_oi,
    put_oi,
    total_oi,
    CASE WHEN call_oi > 0 THEN put_oi::NUMERIC / call_oi ELSE NULL END AS put_call_oi_ratio,
    
    -- GAMMA EXPOSURE (KEY METRICS FOR DEALERS)
    ROUND(net_dealer_gex::NUMERIC, 0) AS net_dealer_gex,
    ROUND(call_dealer_gex::NUMERIC, 0) AS call_dealer_gex,
    ROUND(put_dealer_gex::NUMERIC, 0) AS put_dealer_gex,
    ROUND(weighted_dealer_gex::NUMERIC, 0) AS time_weighted_gex,
    
    -- Gross exposure (absolute)
    ROUND(total_gross_gex::NUMERIC, 0) AS total_gross_gex,
    
    -- Delta exposure (spot equivalent)
    ROUND(net_delta_exposure::NUMERIC, 0) AS net_delta_exposure,
    
    -- Context
    ROUND(gex_concentration_pct::NUMERIC, 2) AS gex_concentration_pct,
    gex_rank,
    gex_tier,
    moneyness_zone,
    hedging_behavior,
    has_pin_risk,
    
    -- Portfolio level metrics
    ROUND(portfolio_net_gex::NUMERIC, 0) AS portfolio_net_gex,
    ROUND(portfolio_weighted_gex::NUMERIC, 0) AS portfolio_weighted_gex,
    
    -- Expiry level
    ROUND(expiry_net_gex::NUMERIC, 0) AS expiry_net_gex,
    ROUND(expiry_weighted_gex::NUMERIC, 0) AS expiry_weighted_gex,
    
    -- Gamma flip strike
    gamma_flip_strike,
    CASE 
        WHEN gamma_flip_strike IS NOT NULL THEN
            CASE 
                WHEN underlying_price > gamma_flip_strike THEN 'Above Flip (Negative GEX Zone)'
                WHEN underlying_price < gamma_flip_strike THEN 'Below Flip (Positive GEX Zone)'
                ELSE 'At Flip Level'
            END
        ELSE NULL
    END AS gamma_regime,
    
    -- Market metrics
    ROUND(avg_iv::NUMERIC, 4) AS avg_implied_volatility,
    ROUND(call_iv::NUMERIC, 4) AS call_iv,
    ROUND(put_iv::NUMERIC, 4) AS put_iv,
    ROUND(total_notional::NUMERIC, 0) AS total_notional_eur,
    
    -- Metadata
    CURRENT_TIMESTAMP AS created_at,
    CURRENT_TIMESTAMP AS updated_at
    
FROM ranked_strikes
WHERE gex_rank <= 50  -- Top 50 strikes per expiry (focus on important levels)
ORDER BY week_ending DESC, expiry_date, ABS(net_dealer_gex) DESC
