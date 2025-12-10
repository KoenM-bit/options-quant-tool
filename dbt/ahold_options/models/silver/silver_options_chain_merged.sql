/*
Silver Layer: Merged Options Chain (BD Primary + FD Secondary)
Merges Beursduivel (bid/ask/underlying) with FD (open_interest) for complete options data.

Data Flow:
1. BD provides: bid, ask, last_price, underlying_price (synchronized, high coverage)
2. FD provides: open_interest, supplementary pricing
3. Merge on: ticker + trade_date + option_type + strike + expiry_date
4. Greeks calculated by Python enrichment AFTER this model runs

Architecture Decision:
- BD data is PRIMARY (better bid/ask coverage: 88-99% vs FD's 1-5%)
- FD data is SECONDARY (provides open_interest which BD lacks)
- trade_date allows merging data scraped at different times for same trading day
*/

{{ config(
    materialized='incremental',
    unique_key=['ticker', 'trade_date', 'option_type', 'strike', 'expiry_date'],
    tags=['silver', 'options', 'merged']
) }}

WITH bd_options AS (
    -- Beursduivel data (PRIMARY source for pricing)
    SELECT
        o.ticker,
        o.trade_date,
        o.option_type,
        o.strike,
        o.expiry_date,
        o.symbol_code AS bd_symbol_code,
        o.issue_id AS bd_issue_id,
        o.bid AS bd_bid,
        o.ask AS bd_ask,
        o.last_price AS bd_last_price,
        o.volume AS bd_volume,
        o.last_timestamp AS bd_last_timestamp,
        o.scraped_at AS bd_scraped_at,
        TRUE AS has_bd_data
    FROM {{ source('bronze', 'bronze_bd_options') }} o
    WHERE 
        o.expiry_date IS NOT NULL
        AND o.strike IS NOT NULL
        AND o.strike > 0
        {% if is_incremental() %}
        AND o.trade_date > COALESCE((SELECT MAX(trade_date) FROM {{ this }}), '1900-01-01'::date)
        {% endif %}
),

bd_options_deduped AS (
    -- Deduplicate BD data (keep latest scrape for each contract)
    SELECT
        *,
        ROW_NUMBER() OVER (
            PARTITION BY ticker, trade_date, option_type, strike, expiry_date
            ORDER BY bd_scraped_at DESC
        ) AS rn
    FROM bd_options
),

bd_underlying AS (
    -- Beursduivel underlying price (synchronized with options scrape)
    SELECT
        ticker,
        trade_date,
        last_price AS bd_underlying_price,
        bid AS bd_underlying_bid,
        ask AS bd_underlying_ask,
        volume AS bd_underlying_volume,
        scraped_at AS bd_underlying_scraped_at
    FROM {{ source('bronze', 'bronze_bd_underlying') }}
    {% if is_incremental() %}
    WHERE trade_date > COALESCE((SELECT MAX(trade_date) FROM {{ this }}), '1900-01-01'::date)
    {% endif %}
),

bd_underlying_deduped AS (
    -- Deduplicate underlying (keep latest scrape for each trade_date)
    SELECT
        *,
        ROW_NUMBER() OVER (
            PARTITION BY ticker, trade_date
            ORDER BY bd_underlying_scraped_at DESC
        ) AS rn
    FROM bd_underlying
),

fd_options AS (
    -- FD data (SECONDARY source - provides open_interest)
    SELECT
        o.ticker,
        -- Derive trade_date from scraped_at (same logic as existing silver_options)
        CASE 
            WHEN EXTRACT(DOW FROM o.scraped_at) = 6 THEN DATE(o.scraped_at) - INTERVAL '1 day'
            WHEN EXTRACT(DOW FROM o.scraped_at) = 0 THEN DATE(o.scraped_at) - INTERVAL '2 days'
            ELSE DATE(o.scraped_at)
        END AS trade_date,
        o.option_type,
        o.strike,
        o.expiry_date,
        o.isin AS fd_isin,
        o.bid AS fd_bid,
        o.ask AS fd_ask,
        o.laatste AS fd_last_price,
        o.volume AS fd_volume,
        o.open_interest AS fd_open_interest,
        o.scraped_at AS fd_scraped_at,
        TRUE AS has_fd_data
    FROM {{ source('bronze', 'bronze_fd_options') }} o
    WHERE 
        o.expiry_date IS NOT NULL
        AND o.strike IS NOT NULL
        AND o.strike > 0
        {% if is_incremental() %}
        AND DATE(o.scraped_at) > COALESCE((SELECT MAX(trade_date) FROM {{ this }}), '1900-01-01'::date)
        {% endif %}
),

fd_options_deduped AS (
    -- Deduplicate FD data
    SELECT
        *,
        ROW_NUMBER() OVER (
            PARTITION BY ticker, trade_date, option_type, strike, expiry_date
            ORDER BY fd_scraped_at DESC
        ) AS rn
    FROM fd_options
),

fd_underlying AS (
    -- FD underlying price (from overview table)
    SELECT
        ticker,
        peildatum AS trade_date,
        koers AS fd_underlying_price,
        volume_underlying AS fd_underlying_volume,
        scraped_at AS fd_underlying_scraped_at
    FROM {{ source('bronze', 'bronze_fd_overview') }}
    {% if is_incremental() %}
    WHERE peildatum > COALESCE((SELECT MAX(trade_date) FROM {{ this }}), '1900-01-01'::date)
    {% endif %}
),

fd_underlying_deduped AS (
    SELECT
        *,
        ROW_NUMBER() OVER (
            PARTITION BY ticker, trade_date
            ORDER BY fd_underlying_scraped_at DESC
        ) AS rn
    FROM fd_underlying
),

merged AS (
    -- LEFT JOIN: BD is left table (primary), FD is right (secondary)
    SELECT
        -- Identifiers (from BD as primary)
        COALESCE(bd.ticker, fd.ticker) AS ticker,
        COALESCE(bd.trade_date, fd.trade_date) AS trade_date,
        COALESCE(bd.option_type, fd.option_type) AS option_type,
        COALESCE(bd.strike, fd.strike) AS strike,
        COALESCE(bd.expiry_date, fd.expiry_date) AS expiry_date,
        
        -- BD fields
        bd.bd_symbol_code,
        bd.bd_issue_id,
        bd.bd_bid,
        bd.bd_ask,
        bd.bd_last_price,
        bd.bd_volume,
        bd.bd_last_timestamp,
        bd.bd_scraped_at,
        
        -- FD fields
        fd.fd_isin,
        fd.fd_bid,
        fd.fd_ask,
        fd.fd_last_price,
        fd.fd_volume,
        fd.fd_open_interest,
        fd.fd_scraped_at,
        
        -- Source tracking
        COALESCE(bd.has_bd_data, FALSE) AS has_bd_data,
        COALESCE(fd.has_fd_data, FALSE) AS has_fd_data
        
    FROM bd_options_deduped bd
    FULL OUTER JOIN fd_options_deduped fd
        ON bd.ticker = fd.ticker
        AND bd.trade_date = fd.trade_date
        AND bd.option_type = fd.option_type
        AND bd.strike = fd.strike
        AND bd.expiry_date = fd.expiry_date
        AND bd.rn = 1
        AND fd.rn = 1
    WHERE bd.rn = 1 OR fd.rn = 1
),

with_underlying AS (
    -- Add underlying prices (BD primary, FD fallback)
    SELECT
        m.*,
        
        -- Underlying price (BD primary, FD fallback)
        COALESCE(bd_u.bd_underlying_price, fd_u.fd_underlying_price) AS underlying_price,
        bd_u.bd_underlying_price,
        bd_u.bd_underlying_bid,
        bd_u.bd_underlying_ask,
        bd_u.bd_underlying_volume,
        fd_u.fd_underlying_price,
        fd_u.fd_underlying_volume,
        
        -- Scrape timestamps
        bd_u.bd_underlying_scraped_at,
        fd_u.fd_underlying_scraped_at,
        COALESCE(bd_u.bd_underlying_scraped_at, fd_u.fd_underlying_scraped_at) AS as_of_ts
        
    FROM merged m
    LEFT JOIN bd_underlying_deduped bd_u
        ON m.ticker = bd_u.ticker
        AND m.trade_date = bd_u.trade_date
        AND bd_u.rn = 1
    LEFT JOIN fd_underlying_deduped fd_u
        ON m.ticker = fd_u.ticker
        AND m.trade_date = fd_u.trade_date
        AND fd_u.rn = 1
),

calculated AS (
    SELECT
        -- Identifiers
        ticker,
        trade_date,
        option_type,
        strike,
        expiry_date,
        
        -- Merged pricing (BD primary)
        COALESCE(bd_bid, fd_bid) AS bid,
        COALESCE(bd_ask, fd_ask) AS ask,
        COALESCE(bd_last_price, fd_last_price) AS last_price,
        
        -- Mid price calculation
        CASE 
            WHEN COALESCE(bd_bid, fd_bid) IS NOT NULL AND COALESCE(bd_ask, fd_ask) IS NOT NULL 
            THEN (COALESCE(bd_bid, fd_bid) + COALESCE(bd_ask, fd_ask)) / 2.0
            ELSE COALESCE(bd_last_price, fd_last_price)
        END AS mid_price,
        
        -- Volume (BD primary)
        COALESCE(bd_volume, fd_volume) AS volume,
        
        -- Open interest (FD only - BD doesn't provide this)
        fd_open_interest AS open_interest,
        
        -- OI change (requires previous day)
        fd_open_interest - LAG(fd_open_interest, 1) OVER (
            PARTITION BY ticker, option_type, strike, expiry_date
            ORDER BY trade_date
        ) AS oi_change,
        
        -- Underlying price
        underlying_price,
        
        -- Moneyness
        CASE 
            WHEN option_type = 'Call' AND underlying_price > 0 
            THEN underlying_price / strike
            WHEN option_type = 'Put' AND strike > 0
            THEN strike / underlying_price
            ELSE NULL
        END AS moneyness,
        
        -- Intrinsic value
        CASE 
            WHEN option_type = 'Call' 
            THEN GREATEST(underlying_price - strike, 0)
            WHEN option_type = 'Put'
            THEN GREATEST(strike - underlying_price, 0)
            ELSE 0
        END AS intrinsic_value,
        
        -- Days to expiry
        EXTRACT(DAY FROM (expiry_date - trade_date))::INTEGER AS days_to_expiry,
        
        -- Time value (calculated from mid_price)
        CASE 
            WHEN COALESCE(bd_bid, fd_bid) IS NOT NULL AND COALESCE(bd_ask, fd_ask) IS NOT NULL
            THEN (COALESCE(bd_bid, fd_bid) + COALESCE(bd_ask, fd_ask)) / 2.0 - 
                 CASE 
                     WHEN option_type = 'Call' THEN GREATEST(underlying_price - strike, 0)
                     WHEN option_type = 'Put' THEN GREATEST(strike - underlying_price, 0)
                     ELSE 0
                 END
            ELSE NULL
        END AS time_value,
        
        -- Spread metrics
        CASE 
            WHEN COALESCE(bd_bid, fd_bid) IS NOT NULL AND COALESCE(bd_ask, fd_ask) IS NOT NULL 
            THEN COALESCE(bd_ask, fd_ask) - COALESCE(bd_bid, fd_bid)
            ELSE NULL
        END AS bid_ask_spread,
        
        CASE 
            WHEN COALESCE(bd_bid, fd_bid) IS NOT NULL AND COALESCE(bd_ask, fd_ask) IS NOT NULL 
                AND COALESCE(bd_ask, fd_ask) > 0
            THEN ((COALESCE(bd_ask, fd_ask) - COALESCE(bd_bid, fd_bid)) / COALESCE(bd_ask, fd_ask)) * 100
            ELSE NULL
        END AS bid_ask_spread_pct,
        
        -- Greeks (populated by Python enrichment AFTER DBT)
        NULL::FLOAT AS implied_volatility,
        NULL::FLOAT AS delta,
        NULL::FLOAT AS gamma,
        NULL::FLOAT AS theta,
        NULL::FLOAT AS vega,
        NULL::FLOAT AS rho,
        NULL::FLOAT AS risk_free_rate_used,
        NULL::BOOLEAN AS greeks_valid,
        NULL::VARCHAR(50) AS greeks_status,
        
        -- Data quality flags
        CASE 
            WHEN COALESCE(bd_bid, fd_bid) IS NOT NULL 
                AND COALESCE(bd_ask, fd_ask) IS NOT NULL 
                AND COALESCE(bd_bid, fd_bid) <= COALESCE(bd_ask, fd_ask)
            THEN TRUE
            ELSE FALSE
        END AS is_validated,
        
        CASE 
            WHEN COALESCE(bd_volume, fd_volume) IS NOT NULL 
                AND COALESCE(bd_volume, fd_volume) > 0 
            THEN TRUE
            ELSE FALSE
        END AS has_volume,
        
        CASE 
            WHEN COALESCE(bd_volume, fd_volume) > 10 
                AND fd_open_interest > 50
            THEN TRUE
            ELSE FALSE
        END AS is_liquid,
        
        -- Source tracking
        has_bd_data,
        has_fd_data,
        
        -- BD raw fields (for debugging)
        bd_symbol_code,
        bd_issue_id,
        bd_bid,
        bd_ask,
        bd_last_price,
        bd_volume,
        
        -- FD raw fields (for debugging)
        fd_isin,
        fd_bid,
        fd_ask,
        fd_last_price,
        fd_volume,
        fd_open_interest,
        
        -- Underlying details
        bd_underlying_price,
        fd_underlying_price,
        bd_underlying_volume,
        fd_underlying_volume,
        
        -- Timestamps
        as_of_ts,
        trade_date AS as_of_date,
        bd_scraped_at,
        fd_scraped_at,
        bd_underlying_scraped_at,
        fd_underlying_scraped_at,
        
        -- Audit
        CURRENT_TIMESTAMP AS created_at,
        CURRENT_TIMESTAMP AS updated_at
        
    FROM with_underlying
)

SELECT * FROM calculated
WHERE underlying_price IS NOT NULL  -- Only include contracts where we have underlying price
