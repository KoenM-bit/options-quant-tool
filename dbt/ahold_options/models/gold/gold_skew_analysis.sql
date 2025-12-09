{{
    config(
        materialized='incremental',
        unique_key=['ticker', 'trade_date', 'expiry_date', 'calculation_timestamp'],
        on_schema_change='sync_all_columns',
        incremental_strategy='append'
    )
}}

/*
VOLATILITY SKEW & TERM STRUCTURE ANALYSIS
==========================================
Analyzes the implied volatility surface to identify:
1. Volatility skew (smile/smirk) - IV by moneyness
2. Term structure - IV across expiration dates
3. Risk reversals - Call IV vs Put IV (sentiment indicator)
4. Butterfly spreads - OTM vs ATM volatility
5. Put/Call IV disparity - Fear index

Key Concepts:
- Negative skew (typical for equities): OTM puts > ATM > OTM calls
- Positive skew: OTM calls > ATM > OTM puts (rare, bullish)
- Flat skew: All strikes similar IV (low fear)
- Term structure: Near-term > far-term (normal), or inverted (stress)
*/

WITH iv_by_moneyness AS (
    SELECT
        ticker,
        trade_date,
        expiry_date,
        days_to_expiry,
        underlying_price,
        strike,
        option_type,
        implied_volatility,
        open_interest,
        volume,
        delta,
        gamma,
        vega,
        
        -- Moneyness (standardized)
        (strike / NULLIF(underlying_price, 0)) AS moneyness_ratio,
        ((strike - underlying_price) / NULLIF(underlying_price, 0)) * 100 AS moneyness_pct,
        
        -- Moneyness buckets
        CASE 
            WHEN ABS((strike - underlying_price) / NULLIF(underlying_price, 0)) < 0.01 THEN 'ATM'
            WHEN strike < underlying_price * 0.90 THEN 'Deep OTM Put'
            WHEN strike < underlying_price * 0.95 THEN 'OTM Put'
            WHEN strike < underlying_price * 0.99 THEN 'Near ATM Put'
            WHEN strike > underlying_price * 1.10 THEN 'Deep OTM Call'
            WHEN strike > underlying_price * 1.05 THEN 'OTM Call'
            WHEN strike > underlying_price * 1.01 THEN 'Near ATM Call'
            ELSE 'ATM'
        END AS moneyness_bucket,
        
        -- Delta buckets (another way to classify)
        CASE 
            WHEN option_type = 'Call' THEN
                CASE 
                    WHEN delta >= 0.45 AND delta <= 0.55 THEN '50 Delta'
                    WHEN delta >= 0.20 AND delta < 0.45 THEN '25 Delta'
                    WHEN delta >= 0.08 AND delta < 0.20 THEN '10 Delta'
                    WHEN delta < 0.08 THEN 'Far OTM'
                    ELSE 'ITM'
                END
            WHEN option_type = 'Put' THEN
                CASE 
                    WHEN delta <= -0.45 AND delta >= -0.55 THEN '50 Delta'
                    WHEN delta <= -0.20 AND delta > -0.45 THEN '25 Delta'
                    WHEN delta <= -0.08 AND delta > -0.20 THEN '10 Delta'
                    WHEN delta > -0.08 THEN 'Far OTM'
                    ELSE 'ITM'
                END
            ELSE 'Unknown'
        END AS delta_bucket
        
    FROM {{ ref('silver_options') }}
    WHERE 
        greeks_valid = TRUE  -- ✅ Only use validated Greeks for accurate skew analysis
        AND implied_volatility IS NOT NULL
        AND implied_volatility > 0
        AND implied_volatility < 5.0  -- Filter outliers (500% IV)
        {% if is_incremental() %}
        AND trade_date > COALESCE((SELECT MAX(trade_date) FROM {{ this }}), '1900-01-01'::date)
        {% endif %}
),

skew_metrics AS (
    SELECT
        ticker,
        trade_date,
        expiry_date,
        days_to_expiry,
        underlying_price,
        
        -- ATM volatility
        AVG(CASE WHEN moneyness_bucket = 'ATM' THEN implied_volatility ELSE NULL END) AS atm_iv,
        
        -- Call side
        AVG(CASE WHEN moneyness_bucket = 'Near ATM Call' THEN implied_volatility ELSE NULL END) AS near_atm_call_iv,
        AVG(CASE WHEN moneyness_bucket = 'OTM Call' THEN implied_volatility ELSE NULL END) AS otm_call_iv,
        AVG(CASE WHEN moneyness_bucket = 'Deep OTM Call' THEN implied_volatility ELSE NULL END) AS deep_otm_call_iv,
        
        -- Put side
        AVG(CASE WHEN moneyness_bucket = 'Near ATM Put' THEN implied_volatility ELSE NULL END) AS near_atm_put_iv,
        AVG(CASE WHEN moneyness_bucket = 'OTM Put' THEN implied_volatility ELSE NULL END) AS otm_put_iv,
        AVG(CASE WHEN moneyness_bucket = 'Deep OTM Put' THEN implied_volatility ELSE NULL END) AS deep_otm_put_iv,
        
        -- By delta
        AVG(CASE WHEN delta_bucket = '50 Delta' THEN implied_volatility ELSE NULL END) AS delta_50_iv,
        AVG(CASE WHEN delta_bucket = '25 Delta' AND option_type = 'Call' THEN implied_volatility ELSE NULL END) AS call_25d_iv,
        AVG(CASE WHEN delta_bucket = '25 Delta' AND option_type = 'Put' THEN implied_volatility ELSE NULL END) AS put_25d_iv,
        AVG(CASE WHEN delta_bucket = '10 Delta' AND option_type = 'Call' THEN implied_volatility ELSE NULL END) AS call_10d_iv,
        AVG(CASE WHEN delta_bucket = '10 Delta' AND option_type = 'Put' THEN implied_volatility ELSE NULL END) AS put_10d_iv,
        
        -- Overall statistics
        AVG(implied_volatility) AS avg_iv,
        MIN(implied_volatility) AS min_iv,
        MAX(implied_volatility) AS max_iv,
        STDDEV(implied_volatility) AS iv_stddev,
        
        -- Weighted by OI
        SUM(implied_volatility * open_interest) / NULLIF(SUM(open_interest), 0) AS oi_weighted_iv,
        
        -- Weighted by volume
        SUM(implied_volatility * COALESCE(volume, 0)) / NULLIF(SUM(COALESCE(volume, 0)), 0) AS volume_weighted_iv,
        
        -- Total OI and volume
        SUM(open_interest) AS total_oi,
        SUM(COALESCE(volume, 0)) AS total_volume,
        
        -- Vega-weighted IV (institutional standard)
        SUM(implied_volatility * ABS(vega) * open_interest) / NULLIF(SUM(ABS(vega) * open_interest), 0) AS vega_weighted_iv,
        
        -- Count of options
        COUNT(*) AS num_options,
        SUM(CASE WHEN option_type = 'Call' THEN 1 ELSE 0 END) AS num_calls,
        SUM(CASE WHEN option_type = 'Put' THEN 1 ELSE 0 END) AS num_puts
        
    FROM iv_by_moneyness
    GROUP BY ticker, trade_date, expiry_date, days_to_expiry, underlying_price
),

skew_calculations AS (
    SELECT
        *,
        
        -- RISK REVERSAL (25 delta): sentiment indicator
        -- Positive = Calls more expensive (bullish)
        -- Negative = Puts more expensive (bearish/fear)
        (call_25d_iv - put_25d_iv) AS risk_reversal_25d,
        
        -- BUTTERFLY (25 delta): convexity measure
        -- High = fat tails, crash protection expensive
        ((call_25d_iv + put_25d_iv) / 2.0 - atm_iv) AS butterfly_25d,
        
        -- SKEW SLOPE (put side - most important for equities)
        (otm_put_iv - atm_iv) AS put_skew,
        (deep_otm_put_iv - atm_iv) AS deep_put_skew,
        
        -- SKEW SLOPE (call side)
        (otm_call_iv - atm_iv) AS call_skew,
        (deep_otm_call_iv - atm_iv) AS deep_call_skew,
        
        -- SKEW ASYMMETRY
        (otm_put_iv - otm_call_iv) AS put_call_skew_diff,
        ((otm_put_iv - otm_call_iv) / NULLIF(atm_iv, 0)) * 100 AS skew_asymmetry_pct,
        
        -- IV RANGE (max - min)
        (max_iv - min_iv) AS iv_range,
        ((max_iv - min_iv) / NULLIF(min_iv, 0)) * 100 AS iv_range_pct
        
    FROM skew_metrics
),

expiry_comparison AS (
    -- Compare near-term vs far-term for term structure
    SELECT
        ticker,
        trade_date,
        
        -- Near-term (< 30 days)
        AVG(CASE WHEN days_to_expiry < 30 THEN atm_iv ELSE NULL END) AS near_term_iv,
        AVG(CASE WHEN days_to_expiry < 30 THEN vega_weighted_iv ELSE NULL END) AS near_term_vega_iv,
        
        -- Mid-term (30-90 days)
        AVG(CASE WHEN days_to_expiry BETWEEN 30 AND 90 THEN atm_iv ELSE NULL END) AS mid_term_iv,
        
        -- Far-term (> 90 days)
        AVG(CASE WHEN days_to_expiry > 90 THEN atm_iv ELSE NULL END) AS far_term_iv
        
    FROM skew_calculations
    GROUP BY ticker, trade_date
)

SELECT
    sc.ticker,
    sc.trade_date,
    sc.expiry_date,
    sc.days_to_expiry,
    sc.underlying_price,
    
    -- ATM IV (benchmark)
    sc.atm_iv,
    sc.delta_50_iv,
    sc.vega_weighted_iv,
    sc.oi_weighted_iv,
    sc.volume_weighted_iv,
    
    -- IV SKEW METRICS (KEY)
    sc.risk_reversal_25d,          -- Sentiment: positive = bullish, negative = bearish
    sc.butterfly_25d,              -- Tail risk: higher = more expensive wings
    sc.put_skew,                   -- Put protection premium
    sc.call_skew,                  -- Call premium
    sc.skew_asymmetry_pct,         -- Overall skew direction
    
    -- By moneyness
    sc.near_atm_call_iv,
    sc.otm_call_iv,
    sc.deep_otm_call_iv,
    sc.near_atm_put_iv,
    sc.otm_put_iv,
    sc.deep_otm_put_iv,
    
    -- By delta
    sc.call_25d_iv,
    sc.put_25d_iv,
    sc.call_10d_iv,
    sc.put_10d_iv,
    
    -- Range
    sc.min_iv,
    sc.max_iv,
    sc.iv_range,
    sc.iv_range_pct,
    sc.iv_stddev,
    
    -- Statistics
    sc.avg_iv,
    sc.total_oi,
    sc.total_volume,
    sc.num_options,
    sc.num_calls,
    sc.num_puts,
    
    -- Term structure
    ec.near_term_iv,
    ec.mid_term_iv,
    ec.far_term_iv,
    (ec.near_term_iv - ec.far_term_iv) AS term_structure_slope,
    
    CASE 
        WHEN ec.near_term_iv > ec.far_term_iv THEN 'Normal (Contango)'
        WHEN ec.near_term_iv < ec.far_term_iv THEN 'Inverted (Backwardation)'
        ELSE 'Flat'
    END AS term_structure_shape,
    
    -- Skew classification
    CASE 
        WHEN sc.put_skew > 0.05 THEN 'Steep Put Skew'
        WHEN sc.put_skew > 0.02 THEN 'Moderate Put Skew'
        WHEN sc.put_skew < -0.02 THEN 'Negative Put Skew'
        ELSE 'Flat'
    END AS put_skew_classification,
    
    CASE 
        WHEN sc.risk_reversal_25d < -0.03 THEN 'Very Bearish'
        WHEN sc.risk_reversal_25d < -0.01 THEN 'Bearish'
        WHEN sc.risk_reversal_25d > 0.03 THEN 'Very Bullish'
        WHEN sc.risk_reversal_25d > 0.01 THEN 'Bullish'
        ELSE 'Neutral'
    END AS sentiment_from_rr,
    
    CASE 
        WHEN sc.butterfly_25d > 0.03 THEN 'High Tail Risk Premium'
        WHEN sc.butterfly_25d > 0.01 THEN 'Moderate Tail Risk'
        ELSE 'Low Tail Risk'
    END AS tail_risk_classification,
    
    -- Time bucket
    CASE 
        WHEN sc.days_to_expiry <= 7 THEN 'Weekly'
        WHEN sc.days_to_expiry <= 30 THEN 'Monthly'
        WHEN sc.days_to_expiry <= 90 THEN 'Quarterly'
        ELSE 'LEAPS'
    END AS expiry_bucket,
    
    -- Metadata
    CURRENT_TIMESTAMP AS created_at

FROM skew_calculations sc
LEFT JOIN expiry_comparison ec
    ON sc.ticker = ec.ticker
    AND sc.trade_date = ec.trade_date

ORDER BY sc.ticker, sc.trade_date, sc.days_to_expiry
