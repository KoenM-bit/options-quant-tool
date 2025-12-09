/*
Gold Layer: Open Interest Flow Analysis
Tracks weekly OI changes to identify institutional positioning.
WEEKLY ONLY - OI data is only accurate when comparing Friday to Friday.

Business Value:
- Identify "smart money" accumulation/distribution
- Spot dealer positioning changes
- Track institutional hedging activity
- Find support/resistance from large positions

Note: This model only processes Friday data (or last trading day of week)
*/

{{ config(
    materialized='incremental',
    unique_key='oi_flow_key',
    tags=['gold', 'open_interest', 'flow', 'weekly']
) }}

WITH fridays_only AS (
    -- Only process Friday trading data (or last trading day if Friday is holiday)
    SELECT 
        trade_date,
        EXTRACT(DOW FROM trade_date) AS day_of_week,
        -- Get the Friday of this week (cast to interval first)
        trade_date + ((5 - EXTRACT(DOW FROM trade_date))::INTEGER || ' days')::INTERVAL AS week_ending,
        ticker,
        option_type,
        strike,
        expiry_date,
        open_interest,
        volume,
        last_price,
        underlying_price,
        delta,
        gamma,
        implied_volatility
    FROM {{ ref('silver_options') }}
    WHERE 
        open_interest IS NOT NULL
        AND open_interest > 0
        -- Only Friday or if no Friday data, take last day of week
        AND EXTRACT(DOW FROM trade_date) IN (5, 4, 6)  -- Friday=5, Thurs=4, Sat=6
        {% if is_incremental() %}
        AND created_at > COALESCE((SELECT MAX(created_at) FROM {{ this }}), '1900-01-01'::timestamp)
        {% endif %}
),

-- Get the most recent Friday per week
weekly_data AS (
    SELECT
        week_ending,
        trade_date,
        ticker,
        option_type,
        strike,
        expiry_date,
        open_interest,
        volume,
        last_price,
        underlying_price,
        delta,
        gamma,
        implied_volatility,
        ROW_NUMBER() OVER (
            PARTITION BY week_ending, ticker, option_type, strike, expiry_date
            ORDER BY trade_date DESC
        ) AS rn
    FROM fridays_only
),

current_week AS (
    SELECT * 
    FROM weekly_data 
    WHERE rn = 1  -- Most recent Friday data for each week
),

-- Calculate OI changes week-over-week
oi_changes AS (
    SELECT
        curr.week_ending,
        curr.trade_date,
        curr.ticker,
        curr.option_type,
        curr.strike,
        curr.expiry_date,
        curr.underlying_price,
        
        -- Current week data
        curr.open_interest AS current_oi,
        curr.volume AS current_volume,
        curr.last_price AS current_price,
        curr.delta AS current_delta,
        curr.gamma AS current_gamma,
        curr.implied_volatility AS current_iv,
        
        -- Previous week data
        LAG(curr.open_interest) OVER (
            PARTITION BY curr.ticker, curr.option_type, curr.strike, curr.expiry_date
            ORDER BY curr.week_ending
        ) AS prev_oi,
        
        LAG(curr.last_price) OVER (
            PARTITION BY curr.ticker, curr.option_type, curr.strike, curr.expiry_date
            ORDER BY curr.week_ending
        ) AS prev_price,
        
        LAG(curr.implied_volatility) OVER (
            PARTITION BY curr.ticker, curr.option_type, curr.strike, curr.expiry_date
            ORDER BY curr.week_ending
        ) AS prev_iv,
        
        -- Calculate changes
        curr.open_interest - LAG(curr.open_interest) OVER (
            PARTITION BY curr.ticker, curr.option_type, curr.strike, curr.expiry_date
            ORDER BY curr.week_ending
        ) AS oi_change,
        
        CASE 
            WHEN LAG(curr.open_interest) OVER (
                PARTITION BY curr.ticker, curr.option_type, curr.strike, curr.expiry_date
                ORDER BY curr.week_ending
            ) > 0 THEN
                (curr.open_interest - LAG(curr.open_interest) OVER (
                    PARTITION BY curr.ticker, curr.option_type, curr.strike, curr.expiry_date
                    ORDER BY curr.week_ending
                ))::NUMERIC / LAG(curr.open_interest) OVER (
                    PARTITION BY curr.ticker, curr.option_type, curr.strike, curr.expiry_date
                    ORDER BY curr.week_ending
                ) * 100
            ELSE NULL
        END AS oi_change_pct,
        
        -- Notional value changes
        curr.open_interest * curr.last_price * 100 AS current_notional,
        (curr.open_interest - LAG(curr.open_interest) OVER (
            PARTITION BY curr.ticker, curr.option_type, curr.strike, curr.expiry_date
            ORDER BY curr.week_ending
        )) * curr.last_price * 100 AS notional_flow
        
    FROM current_week curr
),

-- Add classification and metrics
with_classification AS (
    SELECT
        *,
        -- Classify flow direction
        CASE 
            WHEN oi_change > 100 THEN 'Heavy Accumulation'
            WHEN oi_change > 50 THEN 'Accumulation'
            WHEN oi_change > 0 THEN 'Mild Buildup'
            WHEN oi_change = 0 THEN 'Flat'
            WHEN oi_change > -50 THEN 'Mild Liquidation'
            WHEN oi_change > -100 THEN 'Liquidation'
            ELSE 'Heavy Liquidation'
        END AS flow_type,
        
        -- Moneyness
        CASE 
            WHEN option_type = 'Call' THEN underlying_price / strike
            WHEN option_type = 'Put' THEN strike / underlying_price
        END AS moneyness,
        
        -- Days to expiry (extract days from interval)
        EXTRACT(DAY FROM (expiry_date - trade_date))::INTEGER AS days_to_expiry,
        
        -- Quality flags
        CASE 
            WHEN current_oi >= 100 THEN TRUE
            ELSE FALSE
        END AS is_significant,
        
        CASE
            WHEN ABS(oi_change) >= 100 THEN 'Large'
            WHEN ABS(oi_change) >= 50 THEN 'Medium'
            WHEN ABS(oi_change) >= 10 THEN 'Small'
            ELSE 'Minimal'
        END AS change_magnitude
        
    FROM oi_changes
    WHERE prev_oi IS NOT NULL  -- Must have previous week for comparison
),

-- Add context about where the flow is happening
with_context AS (
    SELECT
        *,
        CASE 
            WHEN moneyness BETWEEN 0.95 AND 1.05 THEN 'ATM'
            WHEN (option_type = 'Call' AND moneyness > 1.05) OR 
                 (option_type = 'Put' AND moneyness > 1.05) THEN 'ITM'
            ELSE 'OTM'
        END AS moneyness_category,
        
        CASE
            WHEN days_to_expiry <= 7 THEN 'Weekly'
            WHEN days_to_expiry <= 30 THEN 'Near-term'
            WHEN days_to_expiry <= 90 THEN 'Medium-term'
            ELSE 'Long-term'
        END AS expiry_category,
        
        -- Directional bias (based on delta-weighted OI change)
        oi_change * COALESCE(current_delta, 0) AS delta_weighted_oi_change
        
    FROM with_classification
)

SELECT
    week_ending || '_' || ticker || '_' || option_type || '_' || strike::text || '_' || expiry_date::text AS oi_flow_key,
    week_ending,
    trade_date,
    ticker,
    option_type,
    strike,
    expiry_date,
    days_to_expiry,
    expiry_category,
    moneyness_category,
    
    -- OI Metrics
    current_oi AS open_interest,
    prev_oi AS previous_week_oi,
    oi_change AS oi_change_contracts,
    oi_change_pct AS oi_change_percent,
    flow_type,
    change_magnitude,
    
    -- Notional Flow (EUR)
    current_notional AS notional_value_eur,
    notional_flow AS notional_flow_eur,
    
    -- Price & Vol Context
    current_price AS option_price,
    prev_price AS previous_week_price,
    current_iv AS implied_volatility,
    prev_iv AS previous_week_iv,
    
    -- Greeks
    current_delta AS delta,
    current_gamma AS gamma,
    delta_weighted_oi_change,
    
    -- Market Context
    underlying_price,
    current_volume AS weekly_volume,
    
    -- Quality Flags
    is_significant,
    
    -- Metadata
    CURRENT_TIMESTAMP AS created_at,
    CURRENT_TIMESTAMP AS updated_at
    
FROM with_context
WHERE is_significant = TRUE  -- Only track significant positions (OI >= 100)
ORDER BY week_ending DESC, ABS(notional_flow) DESC
