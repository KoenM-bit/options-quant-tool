/*
GAMMA EXPOSURE POSITIONING TRENDS
==================================
Tracks how GEX positioning evolves over time for each strike/expiry combination.

This layer enables:
- Identify gamma wall shifts (support/resistance changes)
- Track conviction building (OI and GEX magnitude changes)
- Detect regime shifts (negative → positive GEX transitions)
- Monitor put/call ratio evolution
- Predict pinning levels and market behavior

Key Metrics:
- OI trends (growing/declining positions)
- GEX magnitude changes (conviction strength)
- Put/Call ratio evolution (sentiment shifts)
- Week-over-week deltas (momentum)
*/

{{ config(
    materialized='table',
    tags=['gold', 'gex', 'trends', 'positioning']
) }}

-- Realized volatility & trend from underlying price history
-- RV_10d: 10-day realized volatility, based on log returns and annualized with sqrt(252)
WITH underlying_rv AS (
    WITH base AS (
        SELECT
            ticker,
            trade_date,
            close_price,
            LAG(close_price) OVER (
                PARTITION BY ticker
                ORDER BY trade_date
            ) AS prev_close
        FROM {{ ref('silver_underlying_price') }}
    ),
    returns AS (
        SELECT
            ticker,
            trade_date,
            close_price,
            CASE
                WHEN prev_close > 0
                THEN LN(close_price / prev_close)
                ELSE NULL
            END AS log_return
        FROM base
    )
    SELECT
        ticker,
        trade_date,
        close_price,
        -- 10-day realized volatility (annualized, in decimal terms)
        STDDEV(log_return) OVER (
            PARTITION BY ticker
            ORDER BY trade_date
            ROWS BETWEEN 9 PRECEDING AND CURRENT ROW
        ) * SQRT(252) AS rv_10d,
        -- 20-day realized volatility (annualized, in decimal terms)
        STDDEV(log_return) OVER (
            PARTITION BY ticker
            ORDER BY trade_date
            ROWS BETWEEN 19 PRECEDING AND CURRENT ROW
        ) * SQRT(252) AS rv_20d,
        -- 5-day price trend (percentage move)
        ((close_price - LAG(close_price, 5) OVER (PARTITION BY ticker ORDER BY trade_date))
         / NULLIF(LAG(close_price, 5) OVER (PARTITION BY ticker ORDER BY trade_date), 0) * 100) AS trend_5d,
        -- 5-day log-return momentum (sum of log returns)
        SUM(log_return) OVER (
            PARTITION BY ticker
            ORDER BY trade_date
            ROWS BETWEEN 4 PRECEDING AND CURRENT ROW
        ) AS momentum_5d
    FROM returns
),

weekly_gex AS (
    SELECT
        week_ending,
        friday_trade_date,
        strike,
        expiry_date,
        call_oi,
        put_oi,
        call_dealer_gex,
        put_dealer_gex,
        net_dealer_gex,
        call_iv,
        put_iv,
        avg_implied_volatility AS avg_iv,
        underlying_price,
        (expiry_date - friday_trade_date) AS days_to_expiry
    FROM {{ ref('gold_gamma_exposure_weekly') }}
),

-- Enrich with ATM IV and realized volatility
enriched_gex AS (
    SELECT
        wg.*,
        -- Get ATM IV for this expiry/week (closest strike to underlying)
        (SELECT avg_iv 
         FROM weekly_gex wg2
         WHERE wg2.week_ending = wg.week_ending
         AND wg2.expiry_date = wg.expiry_date
         ORDER BY ABS(wg2.strike - wg2.underlying_price)
         LIMIT 1
        ) AS iv_atm,
    -- Get realized volatility for the Friday of this week
    rv.rv_10d,
    rv.rv_20d,
        rv.trend_5d,
        rv.momentum_5d
    FROM weekly_gex wg
    LEFT JOIN underlying_rv rv 
        ON rv.trade_date = wg.friday_trade_date
),

-- Calculate put/call ratios and positioning metrics
positioning_metrics AS (
    SELECT
        *,
        -- Explicit gamma magnitudes
        (ABS(call_dealer_gex) + ABS(put_dealer_gex)) AS abs_oi_gamma,
        (put_dealer_gex - call_dealer_gex)           AS net_oi_gamma,
        CASE 
            WHEN call_oi > 0 THEN put_oi::FLOAT / call_oi::FLOAT
            ELSE NULL
        END AS put_call_ratio,
        
        -- Determine strike role based on net GEX
        CASE
            WHEN net_dealer_gex > 50000000 THEN 'Strong Support'
            WHEN net_dealer_gex > 20000000 THEN 'Support'
            WHEN net_dealer_gex BETWEEN -20000000 AND 20000000 THEN 'Flip Point'
            WHEN net_dealer_gex < -50000000 THEN 'Strong Resistance'
            WHEN net_dealer_gex < -20000000 THEN 'Resistance'
            ELSE 'Neutral'
        END AS strike_role,
        
        -- Market regime
        CASE
            WHEN net_dealer_gex > 0 THEN 'Positive (Stabilizing)'
            WHEN net_dealer_gex < 0 THEN 'Negative (Amplifying)'
            ELSE 'Neutral'
        END AS market_regime
        
    FROM enriched_gex
),

-- Calculate week-over-week changes
week_over_week_changes AS (
    SELECT
        pm.*,
        
        -- Previous week metrics (for same strike/expiry)
        LAG(call_oi) OVER (PARTITION BY strike, expiry_date ORDER BY week_ending) AS prev_call_oi,
        LAG(put_oi) OVER (PARTITION BY strike, expiry_date ORDER BY week_ending) AS prev_put_oi,
        LAG(net_dealer_gex) OVER (PARTITION BY strike, expiry_date ORDER BY week_ending) AS prev_net_gex,
        LAG(put_call_ratio) OVER (PARTITION BY strike, expiry_date ORDER BY week_ending) AS prev_pc_ratio,
        LAG(strike_role) OVER (PARTITION BY strike, expiry_date ORDER BY week_ending) AS prev_strike_role,
        
        -- Previous week IV metrics
        LAG(call_iv) OVER (PARTITION BY strike, expiry_date ORDER BY week_ending) AS prev_call_iv,
        LAG(put_iv) OVER (PARTITION BY strike, expiry_date ORDER BY week_ending) AS prev_put_iv,
        LAG(avg_iv) OVER (PARTITION BY strike, expiry_date ORDER BY week_ending) AS prev_avg_iv,
        
        -- Calculate deltas
        call_oi - LAG(call_oi) OVER (PARTITION BY strike, expiry_date ORDER BY week_ending) AS call_oi_change,
        put_oi - LAG(put_oi) OVER (PARTITION BY strike, expiry_date ORDER BY week_ending) AS put_oi_change,
        net_dealer_gex - LAG(net_dealer_gex) OVER (PARTITION BY strike, expiry_date ORDER BY week_ending) AS net_gex_change,
        
        -- Approximate per-contract gamma (GEX per unit of OI) for calls and puts
        CASE 
            WHEN call_oi > 0 THEN call_dealer_gex::FLOAT / call_oi::FLOAT
            ELSE NULL
        END AS gamma_per_oi_call,
        CASE 
            WHEN put_oi > 0 THEN put_dealer_gex::FLOAT / put_oi::FLOAT
            ELSE NULL
        END AS gamma_per_oi_put,
        
        -- Previous week's per-contract gamma
        CASE 
            WHEN LAG(call_oi) OVER (PARTITION BY strike, expiry_date ORDER BY week_ending) > 0
            THEN LAG(call_dealer_gex) OVER (PARTITION BY strike, expiry_date ORDER BY week_ending)::FLOAT
                 / LAG(call_oi) OVER (PARTITION BY strike, expiry_date ORDER BY week_ending)::FLOAT
            ELSE NULL
        END AS prev_gamma_per_oi_call,
        CASE 
            WHEN LAG(put_oi) OVER (PARTITION BY strike, expiry_date ORDER BY week_ending) > 0
            THEN LAG(put_dealer_gex) OVER (PARTITION BY strike, expiry_date ORDER BY week_ending)::FLOAT
                 / LAG(put_oi) OVER (PARTITION BY strike, expiry_date ORDER BY week_ending)::FLOAT
            ELSE NULL
        END AS prev_gamma_per_oi_put,
        
        -- Decompose week-over-week GEX change into OI vs gamma components
        -- ΔGEX_call_OI ≈ ΔOI_call * previous gamma_per_oi_call
        (call_oi - LAG(call_oi) OVER (PARTITION BY strike, expiry_date ORDER BY week_ending))
            * COALESCE(
                CASE 
                    WHEN LAG(call_oi) OVER (PARTITION BY strike, expiry_date ORDER BY week_ending) > 0
                    THEN LAG(call_dealer_gex) OVER (PARTITION BY strike, expiry_date ORDER BY week_ending)::FLOAT
                         / LAG(call_oi) OVER (PARTITION BY strike, expiry_date ORDER BY week_ending)::FLOAT
                    ELSE NULL
                END,
                0
            ) AS call_gex_change_from_oi,
        
        -- ΔGEX_put_OI ≈ ΔOI_put * previous gamma_per_oi_put
        (put_oi - LAG(put_oi) OVER (PARTITION BY strike, expiry_date ORDER BY week_ending))
            * COALESCE(
                CASE 
                    WHEN LAG(put_oi) OVER (PARTITION BY strike, expiry_date ORDER BY week_ending) > 0
                    THEN LAG(put_dealer_gex) OVER (PARTITION BY strike, expiry_date ORDER BY week_ending)::FLOAT
                         / LAG(put_oi) OVER (PARTITION BY strike, expiry_date ORDER BY week_ending)::FLOAT
                    ELSE NULL
                END,
                0
            ) AS put_gex_change_from_oi,
        
        -- Total ΔGEX from OI changes (calls + puts)
        COALESCE(
            (call_oi - LAG(call_oi) OVER (PARTITION BY strike, expiry_date ORDER BY week_ending))
                * COALESCE(
                    CASE 
                        WHEN LAG(call_oi) OVER (PARTITION BY strike, expiry_date ORDER BY week_ending) > 0
                        THEN LAG(call_dealer_gex) OVER (PARTITION BY strike, expiry_date ORDER BY week_ending)::FLOAT
                             / LAG(call_oi) OVER (PARTITION BY strike, expiry_date ORDER BY week_ending)::FLOAT
                        ELSE NULL
                    END,
                    0
                ),
            0
        )
        + COALESCE(
            (put_oi - LAG(put_oi) OVER (PARTITION BY strike, expiry_date ORDER BY week_ending))
                * COALESCE(
                    CASE 
                        WHEN LAG(put_oi) OVER (PARTITION BY strike, expiry_date ORDER BY week_ending) > 0
                        THEN LAG(put_dealer_gex) OVER (PARTITION BY strike, expiry_date ORDER BY week_ending)::FLOAT
                             / LAG(put_oi) OVER (PARTITION BY strike, expiry_date ORDER BY week_ending)::FLOAT
                        ELSE NULL
                    END,
                    0
                ),
            0
        ) AS net_gex_change_from_oi,
        
        -- Residual ΔGEX attributed to changes in per-contract gamma (time decay, delta curvature, vol moves)
        (net_dealer_gex - LAG(net_dealer_gex) OVER (PARTITION BY strike, expiry_date ORDER BY week_ending))
        - COALESCE(
            (call_oi - LAG(call_oi) OVER (PARTITION BY strike, expiry_date ORDER BY week_ending))
                * COALESCE(
                    CASE 
                        WHEN LAG(call_oi) OVER (PARTITION BY strike, expiry_date ORDER BY week_ending) > 0
                        THEN LAG(call_dealer_gex) OVER (PARTITION BY strike, expiry_date ORDER BY week_ending)::FLOAT
                             / LAG(call_oi) OVER (PARTITION BY strike, expiry_date ORDER BY week_ending)::FLOAT
                        ELSE NULL
                    END,
                    0
                ),
            0
        )
        - COALESCE(
            (put_oi - LAG(put_oi) OVER (PARTITION BY strike, expiry_date ORDER BY week_ending))
                * COALESCE(
                    CASE 
                        WHEN LAG(put_oi) OVER (PARTITION BY strike, expiry_date ORDER BY week_ending) > 0
                        THEN LAG(put_dealer_gex) OVER (PARTITION BY strike, expiry_date ORDER BY week_ending)::FLOAT
                             / LAG(put_oi) OVER (PARTITION BY strike, expiry_date ORDER BY week_ending)::FLOAT
                        ELSE NULL
                    END,
                    0
                ),
            0
        ) AS net_gex_change_from_gamma,
        
        -- Percentage changes
        CASE 
            WHEN LAG(call_oi) OVER (PARTITION BY strike, expiry_date ORDER BY week_ending) > 0 
            THEN ((call_oi::FLOAT - LAG(call_oi) OVER (PARTITION BY strike, expiry_date ORDER BY week_ending)::FLOAT) 
                  / LAG(call_oi) OVER (PARTITION BY strike, expiry_date ORDER BY week_ending)::FLOAT * 100)
            ELSE NULL
        END AS call_oi_pct_change,
        
        CASE 
            WHEN LAG(put_oi) OVER (PARTITION BY strike, expiry_date ORDER BY week_ending) > 0 
            THEN ((put_oi::FLOAT - LAG(put_oi) OVER (PARTITION BY strike, expiry_date ORDER BY week_ending)::FLOAT) 
                  / LAG(put_oi) OVER (PARTITION BY strike, expiry_date ORDER BY week_ending)::FLOAT * 100)
            ELSE NULL
        END AS put_oi_pct_change,
        
        CASE 
            WHEN LAG(net_dealer_gex) OVER (PARTITION BY strike, expiry_date ORDER BY week_ending) != 0 
            THEN ((net_dealer_gex::FLOAT - LAG(net_dealer_gex) OVER (PARTITION BY strike, expiry_date ORDER BY week_ending)::FLOAT) 
                  / ABS(LAG(net_dealer_gex) OVER (PARTITION BY strike, expiry_date ORDER BY week_ending)::FLOAT) * 100)
            ELSE NULL
        END AS net_gex_pct_change,
        
        -- IV changes (absolute change in volatility points)
        call_iv - LAG(call_iv) OVER (PARTITION BY strike, expiry_date ORDER BY week_ending) AS call_iv_change,
        put_iv - LAG(put_iv) OVER (PARTITION BY strike, expiry_date ORDER BY week_ending) AS put_iv_change,
        avg_iv - LAG(avg_iv) OVER (PARTITION BY strike, expiry_date ORDER BY week_ending) AS avg_iv_change,
        
        -- IV percentage changes
        CASE 
            WHEN LAG(call_iv) OVER (PARTITION BY strike, expiry_date ORDER BY week_ending) > 0 
            THEN ((call_iv::FLOAT - LAG(call_iv) OVER (PARTITION BY strike, expiry_date ORDER BY week_ending)::FLOAT) 
                  / LAG(call_iv) OVER (PARTITION BY strike, expiry_date ORDER BY week_ending)::FLOAT * 100)
            ELSE NULL
        END AS call_iv_pct_change,
        
        CASE 
            WHEN LAG(put_iv) OVER (PARTITION BY strike, expiry_date ORDER BY week_ending) > 0 
            THEN ((put_iv::FLOAT - LAG(put_iv) OVER (PARTITION BY strike, expiry_date ORDER BY week_ending)::FLOAT) 
                  / LAG(put_iv) OVER (PARTITION BY strike, expiry_date ORDER BY week_ending)::FLOAT * 100)
            ELSE NULL
        END AS put_iv_pct_change
        
    FROM positioning_metrics pm
),

-- Flag significant changes and events
trend_analysis AS (
    SELECT
        *,
        
        -- Detect role shifts
        CASE 
            WHEN strike_role != prev_strike_role AND prev_strike_role IS NOT NULL 
            THEN TRUE 
            ELSE FALSE 
        END AS role_shifted,
        
        -- Detect conviction building (OI increasing significantly)
        CASE
            WHEN call_oi_pct_change > 20 OR put_oi_pct_change > 20 THEN 'Building'
            WHEN call_oi_pct_change < -20 OR put_oi_pct_change < -20 THEN 'Unwinding'
            ELSE 'Stable'
        END AS conviction_trend,
        
        -- Detect GEX magnitude changes
        CASE
            WHEN net_gex_pct_change > 50 THEN 'Accelerating'
            WHEN net_gex_pct_change < -50 THEN 'Weakening'
            ELSE 'Steady'
        END AS gex_trend,
        
        -- FLOW DETECTION: Determine if OI changes are buyer or seller driven
        -- Call Flow: OI up + IV up = Buyers aggressive | OI up + IV down = Sellers aggressive
        CASE
            WHEN call_oi_change > 0 AND call_iv_change > 0 THEN 'Buyer Driven'
            WHEN call_oi_change > 0 AND call_iv_change < 0 THEN 'Seller Driven'
            WHEN call_oi_change < 0 AND call_iv_change > 0 THEN 'Closing Shorts (Demand)'
            WHEN call_oi_change < 0 AND call_iv_change < 0 THEN 'Closing Longs (Supply)'
            WHEN call_oi_change > 0 AND call_iv_change = 0 THEN 'Neutral Flow'
            ELSE NULL
        END AS call_flow_type,
        
        -- Put Flow: OI up + IV up = Buyers aggressive | OI up + IV down = Sellers aggressive  
        CASE
            WHEN put_oi_change > 0 AND put_iv_change > 0 THEN 'Buyer Driven'
            WHEN put_oi_change > 0 AND put_iv_change < 0 THEN 'Seller Driven'
            WHEN put_oi_change < 0 AND put_iv_change > 0 THEN 'Closing Shorts (Demand)'
            WHEN put_oi_change < 0 AND put_iv_change < 0 THEN 'Closing Longs (Supply)'
            WHEN put_oi_change > 0 AND put_iv_change = 0 THEN 'Neutral Flow'
            ELSE NULL
        END AS put_flow_type,
        
        -- Dominant flow (what matters more based on OI magnitude)
        CASE
            WHEN ABS(call_oi_change) > ABS(put_oi_change) THEN
                CASE
                    WHEN call_oi_change > 0 AND call_iv_change > 0 THEN 'Call Buying'
                    WHEN call_oi_change > 0 AND call_iv_change < 0 THEN 'Call Selling'
                    ELSE 'Call Closing'
                END
            WHEN ABS(put_oi_change) > ABS(call_oi_change) THEN
                CASE
                    WHEN put_oi_change > 0 AND put_iv_change > 0 THEN 'Put Buying'
                    WHEN put_oi_change > 0 AND put_iv_change < 0 THEN 'Put Selling'
                    ELSE 'Put Closing'
                END
            ELSE 'Balanced'
        END AS dominant_flow,
        
        -- Flag gamma walls (highest positive GEX for the expiry/week)
        ROW_NUMBER() OVER (
            PARTITION BY expiry_date, week_ending 
            ORDER BY net_dealer_gex DESC
        ) = 1 AND net_dealer_gex > 0 AS is_gamma_wall,
        
        -- Flag flip points (near-zero GEX)
        ABS(net_dealer_gex) < 20000000 AS is_flip_point
        
    FROM week_over_week_changes
),

-- Calculate bias scores (0-100 scale) and calibrated variants
bias_scores AS (
    SELECT
        *,

        -- Relative change in absolute gamma based on explicit abs_oi_gamma
        CASE
            WHEN LAG(abs_oi_gamma) OVER (PARTITION BY strike, expiry_date ORDER BY week_ending) > 0
            THEN ((abs_oi_gamma - LAG(abs_oi_gamma) OVER (PARTITION BY strike, expiry_date ORDER BY week_ending))
                  / LAG(abs_oi_gamma) OVER (PARTITION BY strike, expiry_date ORDER BY week_ending))
            ELSE NULL
        END AS d_abs_oi_gamma_rel,

        -- IV week-over-week change (alias for clarity)
        avg_iv_change AS d_iv_wow,

    -- Raw HOLD BIAS SCORE recipe (0-100 heuristic)
        GREATEST(0, LEAST(100,
            50 +
            (CASE WHEN net_dealer_gex > 0 THEN LEAST(25, net_dealer_gex / 2000000.0) ELSE 0 END) +
            (CASE WHEN put_call_ratio > 1.5 THEN 10 ELSE 0 END) +
            (CASE WHEN strike_role IN ('Strong Support', 'Support') THEN 10 ELSE 0 END) +
            (CASE WHEN ABS(COALESCE(trend_5d, 0)) < 2 THEN 5 ELSE 0 END) +
            (CASE WHEN COALESCE(rv_10d, 0) < COALESCE(iv_atm, avg_iv) * 0.8 THEN 3 ELSE 0 END) +
            (CASE WHEN COALESCE(rv_20d, 0) < COALESCE(iv_atm, avg_iv) * 0.8 THEN 2 ELSE 0 END) -
            (CASE WHEN ABS(COALESCE(net_gex_pct_change, 0)) > 30 THEN 15 ELSE 0 END) -
            (CASE WHEN dominant_flow LIKE '%Closing%' THEN 10 ELSE 0 END) -
            (CASE WHEN EXTRACT(DAY FROM days_to_expiry) < 7 THEN 10 ELSE 0 END)
        )) AS hold_bias_score,

    -- Raw BREAK BIAS SCORE recipe (0-100 heuristic)
        GREATEST(0, LEAST(100,
            50 +
            (CASE WHEN net_dealer_gex < 0 THEN LEAST(25, ABS(net_dealer_gex) / 2000000.0) ELSE 0 END) +
            (CASE WHEN strike_role IN ('Strong Resistance', 'Resistance') THEN 10 ELSE 0 END) +
            (CASE WHEN ABS(COALESCE(trend_5d, 0)) > 5 THEN 15 ELSE 0 END) +
            (CASE WHEN COALESCE(rv_10d, 0) > COALESCE(iv_atm, avg_iv) * 1.2 THEN 6 ELSE 0 END) +
            (CASE WHEN COALESCE(rv_20d, 0) > COALESCE(iv_atm, avg_iv) * 1.1 THEN 4 ELSE 0 END) +
            (CASE WHEN dominant_flow IN ('Call Buying', 'Put Buying') THEN 10 ELSE 0 END) +
            (CASE WHEN ABS(COALESCE(net_gex_pct_change, 0)) > 30 THEN 10 ELSE 0 END) -
            (CASE WHEN net_dealer_gex > 0 THEN LEAST(20, net_dealer_gex / 2500000.0) ELSE 0 END) -
            (CASE WHEN put_call_ratio > 1.5 AND net_dealer_gex > 0 THEN 10 ELSE 0 END) -
            (CASE WHEN is_gamma_wall THEN 15 ELSE 0 END)
        )) AS break_bias_score

    FROM trend_analysis
),

-- Calibrate bias scores across the cross-section: z-scores and percentiles per week
calibrated_bias AS (
    SELECT
        bs.*,
        
                -- BUYER-FLOW BIAS: scalar indicator of how buyer- vs seller-driven the recent flow is
                -- Starts at 50 (neutral). Positive values tilt towards buyer-driven flows, negative towards seller-driven.
                -- Interprets call/put flow labels and IV/OI dynamics into a single number (0-100).
                GREATEST(0, LEAST(100,
                        50
                        + CASE 
                                -- Strongly buyer-driven regimes
                                WHEN bs.dominant_flow = 'Call Buying' THEN 20
                                WHEN bs.dominant_flow = 'Put Buying' THEN 20
                                WHEN bs.call_flow_type = 'Buyer Driven' THEN 10
                                WHEN bs.put_flow_type = 'Buyer Driven' THEN 10
                                ELSE 0
                            END
                        + CASE 
                                -- Strongly seller-driven regimes
                                WHEN bs.dominant_flow = 'Call Selling' THEN -20
                                WHEN bs.dominant_flow = 'Put Selling' THEN -20
                                WHEN bs.call_flow_type = 'Seller Driven' THEN -10
                                WHEN bs.put_flow_type = 'Seller Driven' THEN -10
                                ELSE 0
                            END
                        + CASE 
                                -- Closing shorts in an up IV environment is effectively demand/buyer pressure
                                WHEN bs.call_flow_type = 'Closing Shorts (Demand)' THEN 5
                                WHEN bs.put_flow_type = 'Closing Shorts (Demand)' THEN 5
                                ELSE 0
                            END
                        + CASE 
                                -- Closing longs in a down IV environment looks like supply/seller pressure
                                WHEN bs.call_flow_type = 'Closing Longs (Supply)' THEN -5
                                WHEN bs.put_flow_type = 'Closing Longs (Supply)' THEN -5
                                ELSE 0
                            END
                        + CASE
                                -- If both call and put IV are rising with OI, more broad-based demand
                                WHEN bs.call_oi_change > 0 AND bs.put_oi_change > 0 AND bs.avg_iv_change > 0 THEN 5
                                -- If both OI are rising while IV falls, broad supply
                                WHEN bs.call_oi_change > 0 AND bs.put_oi_change > 0 AND bs.avg_iv_change < 0 THEN -5
                                ELSE 0
                            END
                )) AS buyer_flow_bias,

        -- Per-week cross-sectional stats for normalization
        AVG(hold_bias_score) OVER (PARTITION BY week_ending)  AS hold_bias_mean_week,
        STDDEV_SAMP(hold_bias_score) OVER (PARTITION BY week_ending) AS hold_bias_std_week,
        AVG(break_bias_score) OVER (PARTITION BY week_ending) AS break_bias_mean_week,
        STDDEV_SAMP(break_bias_score) OVER (PARTITION BY week_ending) AS break_bias_std_week,

        -- Z-scores: how many std devs above/below weekly mean
        CASE
            WHEN STDDEV_SAMP(hold_bias_score) OVER (PARTITION BY week_ending) IS NULL
                 OR STDDEV_SAMP(hold_bias_score) OVER (PARTITION BY week_ending) = 0
            THEN NULL
            ELSE (hold_bias_score - AVG(hold_bias_score) OVER (PARTITION BY week_ending))
                 / NULLIF(STDDEV_SAMP(hold_bias_score) OVER (PARTITION BY week_ending), 0)
        END AS hold_bias_z_week,

        CASE
            WHEN STDDEV_SAMP(break_bias_score) OVER (PARTITION BY week_ending) IS NULL
                 OR STDDEV_SAMP(break_bias_score) OVER (PARTITION BY week_ending) = 0
            THEN NULL
            ELSE (break_bias_score - AVG(break_bias_score) OVER (PARTITION BY week_ending))
                 / NULLIF(STDDEV_SAMP(break_bias_score) OVER (PARTITION BY week_ending), 0)
        END AS break_bias_z_week,

        -- Simple percentile ranks within each week (0-100)
        NTILE(100) OVER (PARTITION BY week_ending ORDER BY hold_bias_score) - 1 AS hold_bias_percentile_week,
        NTILE(100) OVER (PARTITION BY week_ending ORDER BY break_bias_score) - 1 AS break_bias_percentile_week

    FROM bias_scores bs
)

SELECT
    -- Identifiers
    week_ending,
    friday_trade_date,
    strike,
    expiry_date,
    days_to_expiry,
    
    -- Current positioning
    call_oi,
    put_oi,
    put_call_ratio,
    call_dealer_gex,
    put_dealer_gex,
    net_dealer_gex,
    
    -- Classification
    strike_role,
    market_regime,
    is_gamma_wall,
    is_flip_point,
    
    -- Previous week (for comparison)
    prev_call_oi,
    prev_put_oi,
    prev_net_gex,
    prev_pc_ratio,
    prev_strike_role,
    
    -- Week-over-week changes
    call_oi_change,
    put_oi_change,
    net_gex_change,
    call_oi_pct_change,
    put_oi_pct_change,
    net_gex_pct_change,
    gamma_per_oi_call,
    gamma_per_oi_put,
    prev_gamma_per_oi_call,
    prev_gamma_per_oi_put,
    call_gex_change_from_oi,
    put_gex_change_from_oi,
    net_gex_change_from_oi,
    net_gex_change_from_gamma,
    
    -- IV metrics (current)
    call_iv,
    put_iv,
    avg_iv,
    
    -- IV previous week
    prev_call_iv,
    prev_put_iv,
    prev_avg_iv,
    
    -- IV changes
    call_iv_change,
    put_iv_change,
    avg_iv_change,
    call_iv_pct_change,
    put_iv_pct_change,
    
    -- Flow detection (CRITICAL: buyer vs seller driven)
    call_flow_type,
    put_flow_type,
    dominant_flow,
    
    -- Trend signals
    role_shifted,
    conviction_trend,
    gex_trend,
    
    -- NEW: Advanced metrics for analysis
    abs_oi_gamma,              -- Absolute gamma exposure magnitude
    net_oi_gamma,              -- Net gamma exposure (same as net_dealer_gex)
    d_abs_oi_gamma_rel,        -- Relative change in absolute gamma (week-over-week %)
    iv_atm,                    -- ATM implied volatility for this expiry
    d_iv_wow,                  -- IV week-over-week change (same as avg_iv_change)
    rv_10d,                    -- 10-day realized volatility (annualized)
    rv_20d,                    -- 20-day realized volatility (annualized)
    trend_5d,                  -- 5-day price trend (percentage)
    buyer_flow_bias,           -- Buyer vs seller flow bias (0-100, >50 buyer-tilted)
    hold_bias_score,           -- Heuristic hold bias score (0-100)
    break_bias_score,          -- Heuristic break bias score (0-100)
    hold_bias_z_week,          -- Hold score z-score within week
    break_bias_z_week,         -- Break score z-score within week
    hold_bias_percentile_week, -- Hold score percentile (0-99) within week
    break_bias_percentile_week,-- Break score percentile (0-99) within week
    
    -- Metadata
    CURRENT_TIMESTAMP AS created_at

FROM calibrated_bias

ORDER BY 
    expiry_date,
    week_ending DESC,
    ABS(net_dealer_gex) DESC
