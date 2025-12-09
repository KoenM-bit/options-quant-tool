/*
Gold Layer: Put/Call Ratio Metrics
Tracks put/call ratios across volume, OI, and notional to gauge market sentiment.

Business Value:
- Market sentiment indicator (fear vs greed)
- Contrarian signals (extreme ratios often precede reversals)
- Hedging demand analysis
- Positioning shifts over time

Ratios:
- > 1.0 = More puts (bearish/hedging)
- < 1.0 = More calls (bullish/speculation)
- Extreme readings (>1.5 or <0.5) = Potential reversal signals
*/

{{ config(
    materialized='incremental',
    unique_key='pc_metrics_key',
    tags=['gold', 'sentiment', 'put_call_ratio']
) }}

WITH daily_options AS (
    SELECT
        trade_date,
        ticker,
        option_type,
        strike,
        expiry_date,
        volume,
        open_interest,
        last_price,
        underlying_price,
        implied_volatility,
        delta,
        gamma,
        EXTRACT(DAY FROM (expiry_date - trade_date))::INTEGER AS days_to_expiry
    FROM {{ ref('silver_options') }}
    WHERE 
        greeks_valid = TRUE  -- ✅ Only validated Greeks for put/call analysis
        AND expiry_date > trade_date
        {% if is_incremental() %}
        AND created_at > COALESCE((SELECT MAX(created_at) FROM {{ this }}), '1900-01-01'::timestamp)
        {% endif %}
),

-- Calculate notional values (contract value in EUR)
with_notional AS (
    SELECT
        *,
        COALESCE(volume, 0) * last_price * 100 AS volume_notional,  -- 100 shares per contract
        COALESCE(open_interest, 0) * last_price * 100 AS oi_notional,
        -- Moneyness category
        CASE 
            WHEN option_type = 'Call' AND underlying_price / strike BETWEEN 0.95 AND 1.05 THEN 'ATM'
            WHEN option_type = 'Put' AND strike / underlying_price BETWEEN 0.95 AND 1.05 THEN 'ATM'
            WHEN (option_type = 'Call' AND underlying_price > strike * 1.05) OR 
                 (option_type = 'Put' AND strike > underlying_price * 1.05) THEN 'ITM'
            ELSE 'OTM'
        END AS moneyness_category,
        -- Expiry category
        CASE
            WHEN days_to_expiry <= 7 THEN 'Weekly'
            WHEN days_to_expiry <= 30 THEN 'Near-term'
            WHEN days_to_expiry <= 90 THEN 'Medium-term'
            ELSE 'Long-term'
        END AS expiry_category
    FROM daily_options
),

-- Aggregate by date, expiry, and moneyness
aggregated AS (
    SELECT
        trade_date,
        ticker,
        expiry_category,
        moneyness_category,
        option_type,
        
        -- Volume metrics
        SUM(volume) AS total_volume,
        SUM(volume_notional) AS total_volume_notional,
        COUNT(CASE WHEN volume > 0 THEN 1 END) AS contracts_with_volume,
        
        -- OI metrics
        SUM(open_interest) AS total_oi,
        SUM(oi_notional) AS total_oi_notional,
        COUNT(CASE WHEN open_interest > 0 THEN 1 END) AS contracts_with_oi,
        
        -- Greeks (delta-weighted exposure)
        SUM(COALESCE(open_interest, 0) * COALESCE(delta, 0)) AS delta_weighted_oi,
        SUM(COALESCE(open_interest, 0) * COALESCE(gamma, 0)) AS gamma_weighted_oi,
        
        -- IV metrics
        AVG(implied_volatility) AS avg_iv,
        
        -- Market context
        AVG(underlying_price) AS avg_underlying_price
        
    FROM with_notional
    GROUP BY 
        trade_date,
        ticker,
        expiry_category,
        moneyness_category,
        option_type
),

-- Pivot to get put and call metrics side by side
pivoted AS (
    SELECT
        trade_date,
        ticker,
        expiry_category,
        moneyness_category,
        
        -- Put metrics
        MAX(CASE WHEN option_type = 'Put' THEN total_volume END) AS put_volume,
        MAX(CASE WHEN option_type = 'Put' THEN total_volume_notional END) AS put_volume_notional,
        MAX(CASE WHEN option_type = 'Put' THEN total_oi END) AS put_oi,
        MAX(CASE WHEN option_type = 'Put' THEN total_oi_notional END) AS put_oi_notional,
        MAX(CASE WHEN option_type = 'Put' THEN delta_weighted_oi END) AS put_delta_oi,
        MAX(CASE WHEN option_type = 'Put' THEN gamma_weighted_oi END) AS put_gamma_oi,
        MAX(CASE WHEN option_type = 'Put' THEN avg_iv END) AS put_avg_iv,
        MAX(CASE WHEN option_type = 'Put' THEN contracts_with_volume END) AS put_contract_count_vol,
        MAX(CASE WHEN option_type = 'Put' THEN contracts_with_oi END) AS put_contract_count_oi,
        
        -- Call metrics
        MAX(CASE WHEN option_type = 'Call' THEN total_volume END) AS call_volume,
        MAX(CASE WHEN option_type = 'Call' THEN total_volume_notional END) AS call_volume_notional,
        MAX(CASE WHEN option_type = 'Call' THEN total_oi END) AS call_oi,
        MAX(CASE WHEN option_type = 'Call' THEN total_oi_notional END) AS call_oi_notional,
        MAX(CASE WHEN option_type = 'Call' THEN delta_weighted_oi END) AS call_delta_oi,
        MAX(CASE WHEN option_type = 'Call' THEN gamma_weighted_oi END) AS call_gamma_oi,
        MAX(CASE WHEN option_type = 'Call' THEN avg_iv END) AS call_avg_iv,
        MAX(CASE WHEN option_type = 'Call' THEN contracts_with_volume END) AS call_contract_count_vol,
        MAX(CASE WHEN option_type = 'Call' THEN contracts_with_oi END) AS call_contract_count_oi,
        
        -- Market context
        AVG(avg_underlying_price) AS underlying_price
        
    FROM aggregated
    GROUP BY 
        trade_date,
        ticker,
        expiry_category,
        moneyness_category
),

-- Calculate ratios and add interpretations
with_ratios AS (
    SELECT
        *,
        -- Put/Call Ratios
        CASE 
            WHEN COALESCE(call_volume, 0) > 0 
            THEN COALESCE(put_volume, 0)::NUMERIC / call_volume
            ELSE NULL
        END AS pc_ratio_volume,
        
        CASE 
            WHEN COALESCE(call_oi, 0) > 0 
            THEN COALESCE(put_oi, 0)::NUMERIC / call_oi
            ELSE NULL
        END AS pc_ratio_oi,
        
        CASE 
            WHEN COALESCE(call_volume_notional, 0) > 0 
            THEN COALESCE(put_volume_notional, 0)::NUMERIC / call_volume_notional
            ELSE NULL
        END AS pc_ratio_notional,
        
        -- Net positioning (delta-adjusted)
        COALESCE(call_delta_oi, 0) - COALESCE(put_delta_oi, 0) AS net_delta_exposure,
        
        -- IV skew (put IV premium over calls)
        COALESCE(put_avg_iv, 0) - COALESCE(call_avg_iv, 0) AS put_call_iv_skew,
        
        -- Total activity
        COALESCE(put_volume, 0) + COALESCE(call_volume, 0) AS total_volume,
        COALESCE(put_oi, 0) + COALESCE(call_oi, 0) AS total_oi,
        COALESCE(put_volume_notional, 0) + COALESCE(call_volume_notional, 0) AS total_notional
        
    FROM pivoted
),

-- Add sentiment classification and moving averages
with_sentiment AS (
    SELECT
        *,
        -- Sentiment based on volume P/C ratio
        CASE 
            WHEN pc_ratio_volume > 1.5 THEN 'Very Bearish'
            WHEN pc_ratio_volume > 1.2 THEN 'Bearish'
            WHEN pc_ratio_volume > 0.8 THEN 'Neutral'
            WHEN pc_ratio_volume > 0.5 THEN 'Bullish'
            WHEN pc_ratio_volume IS NOT NULL THEN 'Very Bullish'
            ELSE 'No Signal'
        END AS sentiment_signal,
        
        -- Quality flag
        CASE 
            WHEN total_volume > 100 AND total_oi > 500 THEN 'High'
            WHEN total_volume > 50 AND total_oi > 200 THEN 'Medium'
            ELSE 'Low'
        END AS data_quality,
        
        -- 5-day moving average of P/C ratio
        AVG(CASE 
            WHEN COALESCE(call_volume, 0) > 0 
            THEN COALESCE(put_volume, 0)::NUMERIC / call_volume
            ELSE NULL
        END) OVER (
            PARTITION BY ticker, expiry_category, moneyness_category
            ORDER BY trade_date
            ROWS BETWEEN 4 PRECEDING AND CURRENT ROW
        ) AS pc_ratio_volume_5d_ma,
        
        -- Deviation from moving average (contrarian signal)
        CASE 
            WHEN COALESCE(call_volume, 0) > 0 THEN
                (COALESCE(put_volume, 0)::NUMERIC / call_volume) - 
                AVG(CASE 
                    WHEN COALESCE(call_volume, 0) > 0 
                    THEN COALESCE(put_volume, 0)::NUMERIC / call_volume
                    ELSE NULL
                END) OVER (
                    PARTITION BY ticker, expiry_category, moneyness_category
                    ORDER BY trade_date
                    ROWS BETWEEN 4 PRECEDING AND CURRENT ROW
                )
            ELSE NULL
        END AS pc_ratio_deviation
        
    FROM with_ratios
)

SELECT
    trade_date || '_' || ticker || '_' || expiry_category || '_' || moneyness_category AS pc_metrics_key,
    trade_date,
    ticker,
    expiry_category,
    moneyness_category,
    
    -- Put/Call Ratios
    ROUND(pc_ratio_volume::NUMERIC, 3) AS pc_ratio_volume,
    ROUND(pc_ratio_oi::NUMERIC, 3) AS pc_ratio_oi,
    ROUND(pc_ratio_notional::NUMERIC, 3) AS pc_ratio_notional,
    ROUND(pc_ratio_volume_5d_ma::NUMERIC, 3) AS pc_ratio_volume_ma5,
    ROUND(pc_ratio_deviation::NUMERIC, 3) AS pc_ratio_deviation_from_ma,
    
    -- Put Metrics
    put_volume,
    put_oi AS put_open_interest,
    ROUND(put_volume_notional::NUMERIC, 0) AS put_notional_eur,
    put_contract_count_vol AS put_contracts_traded,
    put_contract_count_oi AS put_contracts_open,
    ROUND(put_avg_iv::NUMERIC, 4) AS put_avg_iv,
    
    -- Call Metrics
    call_volume,
    call_oi AS call_open_interest,
    ROUND(call_volume_notional::NUMERIC, 0) AS call_notional_eur,
    call_contract_count_vol AS call_contracts_traded,
    call_contract_count_oi AS call_contracts_open,
    ROUND(call_avg_iv::NUMERIC, 4) AS call_avg_iv,
    
    -- Combined Metrics
    total_volume,
    total_oi AS total_open_interest,
    ROUND(total_notional::NUMERIC, 0) AS total_notional_eur,
    ROUND(net_delta_exposure::NUMERIC, 2) AS net_delta_exposure,
    ROUND(put_call_iv_skew::NUMERIC, 4) AS put_call_iv_skew,
    
    -- Sentiment
    sentiment_signal,
    data_quality,
    
    -- Market Context
    ROUND(underlying_price::NUMERIC, 2) AS underlying_price,
    
    -- Metadata
    CURRENT_TIMESTAMP AS created_at,
    CURRENT_TIMESTAMP AS updated_at
    
FROM with_sentiment
WHERE data_quality IN ('High', 'Medium')  -- Only include meaningful data
ORDER BY trade_date DESC, total_notional DESC
