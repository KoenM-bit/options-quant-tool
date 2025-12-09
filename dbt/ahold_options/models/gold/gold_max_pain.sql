{{
    config(
        materialized='incremental',
        unique_key=['ticker', 'trade_date', 'expiry_date', 'calculation_timestamp'],
        on_schema_change='sync_all_columns',
        incremental_strategy='append'
    )
}}

/*
MAX PAIN ANALYSIS
=================
Calculates the "max pain" strike where the most option value would expire worthless.
This is where market makers would experience maximum profit (or minimum loss) at expiration.

Theory: Price tends to gravitate toward max pain as expiration approaches due to:
1. Delta hedging by market makers
2. Gamma scalping
3. Pin risk management

The max pain strike is where: SUM(Call_Value + Put_Value) is MINIMIZED

For each potential strike price, calculate:
- Call value = MAX(0, Strike - Spot) × OI (for all strikes below spot)
- Put value = MAX(0, Spot - Strike) × OI (for all strikes above spot)
*/

WITH option_positions AS (
    SELECT
        ticker,
        trade_date,
        expiry_date,
        strike,
        underlying_price,
        option_type,
        open_interest,
        days_to_expiry,
        last_price,
        bid,
        ask,
        
        -- Current intrinsic value
        CASE 
            WHEN option_type = 'Call' THEN 
                GREATEST(0, underlying_price - strike)
            WHEN option_type = 'Put' THEN 
                GREATEST(0, strike - underlying_price)
            ELSE 0
        END AS intrinsic_value,
        
        -- Current time value
        CASE 
            WHEN option_type = 'Call' THEN 
                GREATEST(0, last_price - GREATEST(0, underlying_price - strike))
            WHEN option_type = 'Put' THEN 
                GREATEST(0, last_price - GREATEST(0, strike - underlying_price))
            ELSE 0
        END AS time_value
        
    FROM {{ ref('silver_options') }}
    WHERE 
        greeks_valid = TRUE  -- ✅ Only validated Greeks for max pain calculation
        AND open_interest > 0
        AND strike > 0
        {% if is_incremental() %}
        AND trade_date > COALESCE((SELECT MAX(trade_date) FROM {{ this }}), '1900-01-01'::date)
        {% endif %}
),

strike_range AS (
    -- Generate all possible expiration prices to test
    -- We'll test every strike that exists in the chain
    SELECT DISTINCT
        ticker,
        trade_date,
        expiry_date,
        underlying_price,
        days_to_expiry,
        strike AS potential_expiry_price
    FROM option_positions
),

pain_calculation AS (
    -- For each potential expiry price, calculate total loss to option holders
    SELECT
        sr.ticker,
        sr.trade_date,
        sr.expiry_date,
        sr.underlying_price AS current_price,
        sr.days_to_expiry,
        sr.potential_expiry_price,
        
        -- Calculate value at expiration for calls
        SUM(
            CASE 
                WHEN op.option_type = 'Call' THEN
                    GREATEST(0, sr.potential_expiry_price - op.strike) * op.open_interest
                ELSE 0
            END
        ) AS total_call_value,
        
        -- Calculate value at expiration for puts
        SUM(
            CASE 
                WHEN op.option_type = 'Put' THEN
                    GREATEST(0, op.strike - sr.potential_expiry_price) * op.open_interest
                ELSE 0
            END
        ) AS total_put_value,
        
        -- Count of positions
        SUM(CASE WHEN op.option_type = 'Call' THEN op.open_interest ELSE 0 END) AS total_call_oi,
        SUM(CASE WHEN op.option_type = 'Put' THEN op.open_interest ELSE 0 END) AS total_put_oi
        
    FROM strike_range sr
    JOIN option_positions op
        ON sr.ticker = op.ticker
        AND sr.trade_date = op.trade_date
        AND sr.expiry_date = op.expiry_date
    GROUP BY 
        sr.ticker, 
        sr.trade_date, 
        sr.expiry_date, 
        sr.underlying_price,
        sr.days_to_expiry,
        sr.potential_expiry_price
),

pain_with_totals AS (
    SELECT
        *,
        -- Total value that would be paid out to option holders
        (total_call_value + total_put_value) AS total_option_value,
        
        -- Total open interest
        (total_call_oi + total_put_oi) AS total_oi,
        
        -- Distance from current price
        (potential_expiry_price - current_price) AS distance_from_current,
        ABS(potential_expiry_price - current_price) / NULLIF(current_price, 0) AS pct_from_current
        
    FROM pain_calculation
),

max_pain_strike AS (
    -- Find the strike with minimum total option value (max pain)
    SELECT
        ticker,
        trade_date,
        expiry_date,
        current_price,
        days_to_expiry,
        
        -- Max pain is where total_option_value is MINIMUM
        FIRST_VALUE(potential_expiry_price) OVER (
            PARTITION BY ticker, trade_date, expiry_date 
            ORDER BY total_option_value ASC, ABS(potential_expiry_price - current_price) ASC
        ) AS max_pain_strike,
        
        FIRST_VALUE(total_option_value) OVER (
            PARTITION BY ticker, trade_date, expiry_date 
            ORDER BY total_option_value ASC, ABS(potential_expiry_price - current_price) ASC
        ) AS max_pain_value,
        
        FIRST_VALUE(total_call_value) OVER (
            PARTITION BY ticker, trade_date, expiry_date 
            ORDER BY total_option_value ASC, ABS(potential_expiry_price - current_price) ASC
        ) AS max_pain_call_value,
        
        FIRST_VALUE(total_put_value) OVER (
            PARTITION BY ticker, trade_date, expiry_date 
            ORDER BY total_option_value ASC, ABS(potential_expiry_price - current_price) ASC
        ) AS max_pain_put_value,
        
        -- Value at current price
        FIRST_VALUE(total_option_value) OVER (
            PARTITION BY ticker, trade_date, expiry_date 
            ORDER BY ABS(potential_expiry_price - current_price) ASC
        ) AS current_price_value,
        
        -- Max and min values across all strikes
        MAX(total_option_value) OVER (
            PARTITION BY ticker, trade_date, expiry_date
        ) AS max_option_value,
        
        MIN(total_option_value) OVER (
            PARTITION BY ticker, trade_date, expiry_date
        ) AS min_option_value,
        
        -- Total OI
        FIRST_VALUE(total_oi) OVER (
            PARTITION BY ticker, trade_date, expiry_date 
            ORDER BY total_option_value ASC
        ) AS total_oi,
        
        FIRST_VALUE(total_call_oi) OVER (
            PARTITION BY ticker, trade_date, expiry_date 
            ORDER BY total_option_value ASC
        ) AS total_call_oi,
        
        FIRST_VALUE(total_put_oi) OVER (
            PARTITION BY ticker, trade_date, expiry_date 
            ORDER BY total_option_value ASC
        ) AS total_put_oi
        
    FROM pain_with_totals
),

deduped AS (
    SELECT DISTINCT
        ticker,
        trade_date,
        expiry_date,
        current_price,
        days_to_expiry,
        max_pain_strike,
        max_pain_value,
        max_pain_call_value,
        max_pain_put_value,
        current_price_value,
        max_option_value,
        min_option_value,
        total_oi,
        total_call_oi,
        total_put_oi
    FROM max_pain_strike
)

SELECT
    ticker,
    trade_date,
    expiry_date,
    current_price AS underlying_price,
    days_to_expiry,
    
    -- MAX PAIN METRICS (KEY)
    max_pain_strike,
    (max_pain_strike - current_price) AS max_pain_distance,
    ((max_pain_strike - current_price) / NULLIF(current_price, 0)) * 100 AS max_pain_distance_pct,
    
    -- Value metrics
    max_pain_value AS min_total_value,              -- Minimum value to option holders
    current_price_value AS current_total_value,      -- Value if expired at current price
    max_option_value AS max_total_value,             -- Maximum value to option holders
    (current_price_value - max_pain_value) AS potential_loss_to_holders,
    
    -- Breakdown
    max_pain_call_value,
    max_pain_put_value,
    
    -- Open interest
    total_oi,
    total_call_oi,
    total_put_oi,
    (total_put_oi::FLOAT / NULLIF(total_call_oi, 0)) AS put_call_oi_ratio,
    
    -- Pain intensity (how much value is at stake)
    ((max_option_value - min_option_value) / NULLIF(min_option_value, 0)) * 100 AS pain_range_pct,
    
    -- Pinning probability indicators
    CASE 
        WHEN days_to_expiry <= 3 AND ABS(max_pain_strike - current_price) / NULLIF(current_price, 0) < 0.02 
            THEN 'High Pin Risk'
        WHEN days_to_expiry <= 7 AND ABS(max_pain_strike - current_price) / NULLIF(current_price, 0) < 0.05 
            THEN 'Moderate Pin Risk'
        ELSE 'Low Pin Risk'
    END AS pin_risk_level,
    
    CASE 
        WHEN days_to_expiry = 0 THEN 'Expiration Day'
        WHEN days_to_expiry <= 3 THEN 'Week of Expiry'
        WHEN days_to_expiry <= 7 THEN 'Within 1 Week'
        WHEN days_to_expiry <= 14 THEN 'Within 2 Weeks'
        WHEN days_to_expiry <= 30 THEN 'Within 1 Month'
        ELSE 'More than 1 Month'
    END AS time_to_expiry_bucket,
    
    -- Flags
    CASE WHEN ABS(max_pain_strike - current_price) < current_price * 0.01 THEN TRUE ELSE FALSE END AS is_near_max_pain,
    CASE WHEN max_pain_strike > current_price THEN TRUE ELSE FALSE END AS max_pain_above_price,
    
    -- Metadata
    CURRENT_TIMESTAMP AS created_at

FROM deduped

ORDER BY ticker, trade_date, days_to_expiry
