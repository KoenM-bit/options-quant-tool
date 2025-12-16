{{
    config(
        materialized='table',
        indexes=[
            {'columns': ['ticker', 'trade_date']},
        ]
    )
}}

WITH base_data AS (
    SELECT
        ticker,
        trade_date,
        close,
        atr_14,
        adx_14,
        realized_volatility_20
    FROM {{ source('bronze', 'fact_technical_indicators') }}
),

earnings_calendar AS (
    SELECT
        ticker,
        earnings_date,
        quarter,
        fiscal_year
    FROM calendar_earnings
),

dividend_calendar AS (
    SELECT
        ticker,
        ex_dividend_date,
        payment_date,
        amount,
        dividend_type
    FROM calendar_dividends
),

-- Calculate next and previous earnings for each trading day
earnings_features AS (
    SELECT
        b.ticker,
        b.trade_date,
        -- Next earnings
        (
            SELECT MIN(e.earnings_date)
            FROM earnings_calendar e
            WHERE e.ticker = b.ticker AND e.earnings_date >= b.trade_date
        ) AS next_earnings_date,
        -- Previous earnings
        (
            SELECT MAX(e.earnings_date)
            FROM earnings_calendar e
            WHERE e.ticker = b.ticker AND e.earnings_date < b.trade_date
        ) AS prev_earnings_date
    FROM base_data b
),

earnings_enriched AS (
    SELECT
        ef.*,
        ec.quarter AS next_earnings_quarter,
        -- Days to/from earnings
        CASE
            WHEN ef.next_earnings_date IS NOT NULL 
            THEN LEAST(ef.next_earnings_date - ef.trade_date, 90)
            ELSE NULL
        END AS days_to_earnings,
        CASE
            WHEN ef.prev_earnings_date IS NOT NULL 
            THEN ef.trade_date - ef.prev_earnings_date
            ELSE NULL
        END AS days_since_earnings
    FROM earnings_features ef
    LEFT JOIN earnings_calendar ec ON ef.next_earnings_date = ec.earnings_date AND ef.ticker = ec.ticker
),

-- Calculate next and previous dividends for each trading day
dividend_features AS (
    SELECT
        b.ticker,
        b.trade_date,
        -- Next ex-dividend
        (
            SELECT MIN(d.ex_dividend_date)
            FROM dividend_calendar d
            WHERE d.ticker = b.ticker AND d.ex_dividend_date >= b.trade_date
        ) AS next_exdiv_date,
        -- Previous ex-dividend
        (
            SELECT MAX(d.ex_dividend_date)
            FROM dividend_calendar d
            WHERE d.ticker = b.ticker AND d.ex_dividend_date < b.trade_date
        ) AS prev_exdiv_date
    FROM base_data b
),

dividend_enriched AS (
    SELECT
        df.*,
        dc.amount AS next_div_amount,
        dc.dividend_type AS next_div_type,
        -- Days to/from ex-dividend
        CASE
            WHEN df.next_exdiv_date IS NOT NULL 
            THEN LEAST(df.next_exdiv_date - df.trade_date, 90)
            ELSE NULL
        END AS days_to_exdiv,
        CASE
            WHEN df.prev_exdiv_date IS NOT NULL 
            THEN df.trade_date - df.prev_exdiv_date
            ELSE NULL
        END AS days_since_exdiv
    FROM dividend_features df
    LEFT JOIN dividend_calendar dc ON df.next_exdiv_date = dc.ex_dividend_date AND df.ticker = dc.ticker
),

-- Calculate OPEX (3rd Friday of each month)
opex_features AS (
    SELECT
        ticker,
        trade_date,
        -- Get next OPEX (3rd Friday)
        (
            WITH month_opex AS (
                -- Try current month first
                SELECT DATE_TRUNC('month', trade_date)::date + 
                       ((19 - EXTRACT(DOW FROM DATE_TRUNC('month', trade_date)::date)::int % 7) % 7 + 14) AS opex
            )
            SELECT CASE
                WHEN (SELECT opex FROM month_opex) >= trade_date THEN (SELECT opex FROM month_opex)
                ELSE DATE_TRUNC('month', trade_date + INTERVAL '1 month')::date +
                     ((19 - EXTRACT(DOW FROM DATE_TRUNC('month', trade_date + INTERVAL '1 month')::date)::int % 7) % 7 + 14)
            END
        ) AS next_opex_date
    FROM base_data
),

opex_enriched AS (
    SELECT
        *,
        next_opex_date - trade_date AS days_to_opex
    FROM opex_features
),

-- Calendar features (month-end, quarter-end)
calendar_features AS (
    SELECT
        ticker,
        trade_date,
        -- Days to month end
        (DATE_TRUNC('month', trade_date) + INTERVAL '1 month' - INTERVAL '1 day')::date - trade_date AS days_to_month_end,
        -- Days to quarter end
        (DATE_TRUNC('quarter', trade_date) + INTERVAL '3 months' - INTERVAL '1 day')::date - trade_date AS days_to_quarter_end,
        -- Quarter
        EXTRACT(QUARTER FROM trade_date)::int AS quarter
    FROM base_data
),

-- Combine all features
final AS (
    SELECT
        b.ticker,
        b.trade_date,
        
        -- Earnings features
        ee.next_earnings_date,
        ee.prev_earnings_date,
        ee.days_to_earnings,
        ee.days_since_earnings,
        (ee.days_to_earnings BETWEEN 0 AND 5)::int AS is_earnings_week,
        (ee.days_to_earnings = 1)::int AS is_earnings_tomorrow,
        (ee.days_to_earnings = 0)::int AS is_earnings_today,
        (ee.days_since_earnings = 1)::int AS is_post_earnings_1d,
        (ee.days_since_earnings BETWEEN 1 AND 5)::int AS is_post_earnings_5d,
        ee.next_earnings_quarter,
        (ee.next_earnings_quarter = 'Q1')::int AS earnings_q1,
        (ee.next_earnings_quarter = 'Q2')::int AS earnings_q2,
        (ee.next_earnings_quarter = 'Q3')::int AS earnings_q3,
        (ee.next_earnings_quarter = 'Q4')::int AS earnings_q4,
        
        -- Dividend features
        de.next_exdiv_date,
        de.prev_exdiv_date,
        de.days_to_exdiv,
        de.days_since_exdiv,
        (de.days_to_exdiv BETWEEN 0 AND 5)::int AS is_exdiv_week,
        (de.days_to_exdiv = 1)::int AS is_exdiv_tomorrow,
        (de.days_to_exdiv = 0)::int AS is_exdiv_today,
        (de.days_since_exdiv = 1)::int AS is_post_exdiv_1d,
        de.next_div_amount,
        de.next_div_type,
        (de.next_div_type = 'interim')::int AS is_dividend_interim,
        (de.next_div_type = 'final')::int AS is_dividend_final,
        CASE WHEN de.next_div_amount IS NOT NULL AND b.close > 0 
            THEN (de.next_div_amount * 2) / b.close 
            ELSE 0 
        END AS div_yield_annual,
        CASE WHEN de.next_div_amount IS NOT NULL AND b.atr_14 > 0 
            THEN de.next_div_amount / b.atr_14 
            ELSE 0 
        END AS div_as_atr,
        
        -- OPEX features
        oe.next_opex_date,
        oe.days_to_opex,
        (oe.days_to_opex BETWEEN 0 AND 5)::int AS is_opex_week,
        (oe.days_to_opex = 0)::int AS is_opex_friday,
        (oe.days_to_opex BETWEEN -5 AND -1)::int AS is_post_opex_week,
        
        -- Calendar features
        cf.days_to_month_end,
        cf.days_to_quarter_end,
        (cf.days_to_month_end BETWEEN 0 AND 5)::int AS is_month_end_week,
        (cf.days_to_quarter_end BETWEEN 0 AND 5)::int AS is_quarter_end_week,
        cf.quarter,
        
        -- Interaction features (conditional event effects)
        CASE WHEN ee.days_to_earnings BETWEEN 0 AND 5 THEN b.atr_14 ELSE 0 END AS earnings_week_x_atr,
        CASE WHEN oe.days_to_opex BETWEEN 0 AND 5 THEN b.adx_14 ELSE 0 END AS opex_week_x_adx,
        CASE WHEN ee.days_to_earnings BETWEEN 0 AND 5 THEN b.realized_volatility_20 ELSE 0 END AS earnings_week_x_rv,
        
        -- Metadata
        CURRENT_TIMESTAMP AS created_at,
        CURRENT_TIMESTAMP AS updated_at
        
    FROM base_data b
    LEFT JOIN earnings_enriched ee ON b.ticker = ee.ticker AND b.trade_date = ee.trade_date
    LEFT JOIN dividend_enriched de ON b.ticker = de.ticker AND b.trade_date = de.trade_date
    LEFT JOIN opex_enriched oe ON b.ticker = oe.ticker AND b.trade_date = oe.trade_date
    LEFT JOIN calendar_features cf ON b.ticker = cf.ticker AND b.trade_date = cf.trade_date
)

SELECT * FROM final
ORDER BY ticker, trade_date
