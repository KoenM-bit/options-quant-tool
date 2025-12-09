{{
    config(
        materialized='incremental',
        unique_key=['ticker', 'trade_date', 'expiry_date', 'strike', 'calculation_timestamp'],
        on_schema_change='sync_all_columns',
        incremental_strategy='append'
    )
}}

/*
GAMMA EXPOSURE (GEX) ANALYSIS
==============================
⚠️  WEEKLY DATA ONLY - OI is only accurate on Fridays (scraped Saturdays)
Daily OI data is stale/unreliable, so we only calculate GEX from Friday snapshots.

Calculates dealer gamma exposure by strike to identify:
- Gamma walls (high positive/negative GEX levels)
- Zero gamma level (flip point between positive/negative gamma)
- Hedging pressure zones
- Pinning levels

Dealer GEX = Open Interest × Gamma × 100 × Underlying Price²
- Dealers are SHORT options (opposite sign to buyers)
- Call GEX: Negative for dealers (they sell calls, hedge by buying stock)
- Put GEX: Positive for dealers (they sell puts, hedge by selling stock)

Key Concepts:
- Positive GEX zone: Dealers suppress volatility (sell high, buy low)
- Negative GEX zone: Dealers amplify volatility (buy high, sell low)
- Zero GEX: Critical level where dealer hedging flips
*/

WITH fridays_only AS (
    -- Only process Friday trading data (or last trading day if Friday is holiday)
    -- ✅ QUALITY GATE: Only use validated Greeks from Silver
    SELECT 
        ticker,
        trade_date,
        EXTRACT(DOW FROM trade_date) AS day_of_week,
        expiry_date,
        strike,
        option_type,
        underlying_price,
        open_interest,
        gamma,
        delta,
        vega,
        theta,
        implied_volatility,
        days_to_expiry,
        risk_free_rate_used
    FROM {{ ref('silver_options') }}
    WHERE 
        greeks_valid = TRUE  -- ✅ Only use high-quality validated Greeks
        AND gamma IS NOT NULL
        AND open_interest > 0
        AND underlying_price > 0
        -- Only Friday or if no Friday data, take Thursday or Saturday
        AND EXTRACT(DOW FROM trade_date) IN (5, 4, 6)  -- Friday=5, Thurs=4, Sat=6
        {% if is_incremental() %}
        AND trade_date > COALESCE((SELECT MAX(trade_date) FROM {{ this }}), '1900-01-01'::date)
        {% endif %}
),

-- Get the most recent Friday per week
weekly_data AS (
    SELECT
        ticker,
        trade_date,
        expiry_date,
        strike,
        option_type,
        underlying_price,
        open_interest,
        gamma,
        delta,
        vega,
        theta,
        implied_volatility,
        days_to_expiry,
        ROW_NUMBER() OVER (
            PARTITION BY DATE_TRUNC('week', trade_date), ticker, expiry_date, strike, option_type
            ORDER BY 
                CASE WHEN EXTRACT(DOW FROM trade_date) = 5 THEN 0 ELSE 1 END,  -- Prefer Friday
                trade_date DESC
        ) AS rn
    FROM fridays_only
),

option_exposure AS (
    SELECT
        ticker,
        trade_date,
        expiry_date,
        strike,
        option_type,
        underlying_price,
        
        -- Open interest and Greeks
        open_interest,
        gamma,
        delta,
        vega,
        theta,
        implied_volatility,
        
        -- Days to expiry weight (near-term has more impact)
        days_to_expiry,
        CASE 
            WHEN days_to_expiry <= 7 THEN 1.5
            WHEN days_to_expiry <= 30 THEN 1.0
            WHEN days_to_expiry <= 60 THEN 0.7
            ELSE 0.4
        END AS time_weight,
        
        -- Moneyness
        (strike / NULLIF(underlying_price, 0)) AS moneyness,
        
        -- Calculate per-contract exposure
        -- GEX formula: OI × Gamma × 100 × S²
        -- 100 = shares per contract, S² = dollar gamma
        open_interest * gamma * 100 * POWER(underlying_price, 2) AS gross_gamma_exposure,
        
        -- Dealer exposure (opposite sign for sellers)
        CASE 
            WHEN option_type = 'Call' THEN 
                -1 * open_interest * gamma * 100 * POWER(underlying_price, 2)
            WHEN option_type = 'Put' THEN 
                open_interest * gamma * 100 * POWER(underlying_price, 2)
            ELSE 0
        END AS dealer_gamma_exposure,
        
        -- Delta exposure
        open_interest * delta * 100 AS delta_exposure,
        
        -- Vega exposure (IV sensitivity)
        open_interest * vega * 100 AS vega_exposure
        
    FROM weekly_data
    WHERE rn = 1  -- Only use the most recent Friday data per week
),

strike_aggregates AS (
    SELECT
        ticker,
        trade_date,
        expiry_date,
        strike,
        underlying_price,
        
        -- Aggregate open interest
        SUM(CASE WHEN option_type = 'Call' THEN open_interest ELSE 0 END) AS call_oi,
        SUM(CASE WHEN option_type = 'Put' THEN open_interest ELSE 0 END) AS put_oi,
        SUM(open_interest) AS total_oi,
        
        -- Aggregate gamma exposure by type
        SUM(CASE WHEN option_type = 'Call' THEN gross_gamma_exposure ELSE 0 END) AS call_gamma_notional,
        SUM(CASE WHEN option_type = 'Put' THEN gross_gamma_exposure ELSE 0 END) AS put_gamma_notional,
        SUM(gross_gamma_exposure) AS total_gamma_notional,
        
        -- Net dealer exposure (this is the key metric)
        SUM(dealer_gamma_exposure) AS net_dealer_gex,
        SUM(CASE WHEN option_type = 'Call' THEN dealer_gamma_exposure ELSE 0 END) AS call_dealer_gex,
        SUM(CASE WHEN option_type = 'Put' THEN dealer_gamma_exposure ELSE 0 END) AS put_dealer_gex,
        
        -- Time-weighted exposure (near-term matters more)
        SUM(dealer_gamma_exposure * time_weight) AS weighted_dealer_gex,
        
        -- Other Greek exposures
        SUM(delta_exposure) AS net_delta_exposure,
        SUM(vega_exposure) AS net_vega_exposure,
        
        -- IV metrics
        AVG(implied_volatility) AS avg_iv,
        MAX(implied_volatility) AS max_iv,
        MIN(implied_volatility) AS min_iv,
        
        -- Distance from underlying
        MIN(ABS(strike - underlying_price)) AS distance_from_spot,
        
        -- Days to expiry (should be same for all in group)
        MAX(days_to_expiry) AS days_to_expiry
        
    FROM option_exposure
    GROUP BY ticker, trade_date, expiry_date, strike, underlying_price
),

expiry_totals AS (
    SELECT
        ticker,
        trade_date,
        expiry_date,
        SUM(ABS(net_dealer_gex)) AS total_gex_magnitude,
        SUM(net_dealer_gex) AS total_net_gex
    FROM strike_aggregates
    GROUP BY ticker, trade_date, expiry_date
),

ranked_strikes AS (
    SELECT
        sa.*,
        et.total_gex_magnitude,
        et.total_net_gex,
        
        -- Concentration metrics
        (ABS(sa.net_dealer_gex) / NULLIF(et.total_gex_magnitude, 0)) * 100 AS gex_concentration_pct,
        
        -- Rank by exposure
        ROW_NUMBER() OVER (
            PARTITION BY sa.ticker, sa.trade_date, sa.expiry_date 
            ORDER BY ABS(sa.net_dealer_gex) DESC
        ) AS gex_rank,
        
        -- Identify key levels
        CASE 
            WHEN ABS(sa.net_dealer_gex) > et.total_gex_magnitude * 0.15 THEN 'Gamma Wall'
            WHEN ABS(sa.net_dealer_gex) > et.total_gex_magnitude * 0.05 THEN 'High GEX'
            WHEN ABS(sa.net_dealer_gex) < et.total_gex_magnitude * 0.01 THEN 'Low GEX'
            ELSE 'Medium GEX'
        END AS gex_classification,
        
        -- Distance buckets
        CASE 
            WHEN ABS(sa.strike - sa.underlying_price) / NULLIF(sa.underlying_price, 0) < 0.02 THEN 'ATM'
            WHEN sa.strike > sa.underlying_price THEN 'OTM Call / ITM Put'
            ELSE 'ITM Call / OTM Put'
        END AS moneyness_bucket
        
    FROM strike_aggregates sa
    JOIN expiry_totals et 
        ON sa.ticker = et.ticker 
        AND sa.trade_date = et.trade_date
        AND sa.expiry_date = et.expiry_date
)

SELECT
    ticker,
    trade_date,
    expiry_date,
    strike,
    underlying_price,
    days_to_expiry,
    
    -- Open Interest
    call_oi,
    put_oi,
    total_oi,
    (put_oi::FLOAT / NULLIF(call_oi, 0)) AS put_call_oi_ratio,
    
    -- Gamma Exposure (KEY METRICS)
    net_dealer_gex,                    -- Most important: net dealer exposure
    call_dealer_gex,                   -- Dealer call gamma (negative)
    put_dealer_gex,                    -- Dealer put gamma (positive)
    weighted_dealer_gex,               -- Time-weighted exposure
    
    -- Notional values
    call_gamma_notional,
    put_gamma_notional,
    total_gamma_notional,
    
    -- Other Greek exposures
    net_delta_exposure,
    net_vega_exposure,
    
    -- IV metrics
    avg_iv,
    max_iv,
    min_iv,
    (max_iv - min_iv) AS iv_spread,
    
    -- Classification
    gex_rank,
    gex_classification,
    gex_concentration_pct,
    moneyness_bucket,
    
    -- Distance metrics
    distance_from_spot,
    (strike - underlying_price) AS strike_distance,
    ((strike - underlying_price) / NULLIF(underlying_price, 0)) * 100 AS strike_distance_pct,
    
    -- Context
    total_gex_magnitude AS expiry_total_gex,
    total_net_gex AS expiry_net_gex,
    
    -- Flags for key levels
    CASE WHEN gex_rank <= 3 THEN TRUE ELSE FALSE END AS is_top_3_gex,
    CASE WHEN gex_classification = 'Gamma Wall' THEN TRUE ELSE FALSE END AS is_gamma_wall,
    CASE WHEN ABS(strike - underlying_price) < underlying_price * 0.01 THEN TRUE ELSE FALSE END AS is_near_spot,
    
    -- Metadata
    CURRENT_TIMESTAMP AS created_at

FROM ranked_strikes

ORDER BY ticker, trade_date, expiry_date, ABS(net_dealer_gex) DESC
