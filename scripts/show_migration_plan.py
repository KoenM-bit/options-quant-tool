"""
MIGRATION PLAN: option_prices_live (MariaDB) → bronze_bd_options (PostgreSQL)

================================================================================
DATA SOURCE ANALYSIS
================================================================================

MariaDB Table: option_prices_live
- Total records: 147,254
- Date range: 2025-11-10 to 2025-12-10 (18 days)
- Distinct ticker: 1 (AD.AS)
- Scraping pattern: Multiple scrapes per day (9 hours of data per day)
- Last scrape time: ~17:30-17:45 each day

STRATEGY: Extract ONLY the LAST complete snapshot from each day
- This represents the end-of-day state
- Avoids duplicates (each day has 8000+ records from multiple scrapes)
- Gives us 1 clean snapshot per trading day with ~130-260 contracts

================================================================================
COLUMN MAPPING
================================================================================

MariaDB (option_prices_live)          →  PostgreSQL (bronze_bd_options)
----------------------------------------  ----------------------------------------
ticker                  (varchar)      →  ticker                (String)
'AD.AS'                                   'AD.AS'

issue_id                (varchar)      →  issue_id              (String)
'522941539'                               '522941539' (Beursduivel internal ID)

type                    (enum)         →  option_type           (String)
'Call' / 'Put'                            'Call' / 'Put'

expiry                  (varchar)      →  expiry_date           (Date)
'December 2029'                           NEEDS PARSING → 2029-12-??
NOTE: expiry is text like 'December 2029', 'Juni 2026'
      We need to parse this to actual Date

strike                  (decimal)      →  strike                (Float)
80.000                                    80.0

bid                     (decimal)      →  bid                   (Float)
0.1000                                    0.10

ask                     (decimal)      →  ask                   (Float)
0.2000                                    0.20

price                   (decimal)      →  mid_price             (Float?)
0.1500                                    NOT IN SCHEMA - skip or use (bid+ask)/2

last_price              (decimal)      →  last_price            (Float)
NULL / 0.1500                             0.15 (can be NULL)

volume                  (int)          →  volume                (Integer)
NULL / 10                                 10 (can be NULL)

spot_price              (decimal)      →  underlying_price      (Float?)
34.3200                                   NOT IN SCHEMA - but needed!
                                          This is the AD.AS stock price

created_at              (timestamp)    →  scraped_at            (DateTime)
'2025-12-10 17:45:27'                     '2025-12-10 17:45:27'

created_at              (timestamp)    →  trade_date            (Date)
'2025-12-10 17:45:27'                     '2025-12-10' (extract date)

last_time               (datetime)     →  last_timestamp        (DateTime)
NULL / '2025-12-09 00:00:00'              Can be NULL

fetched_at              (datetime)     →  SKIP (same as created_at)

----------------------------------------  ----------------------------------------
DERIVED/FIXED VALUES:
----------------------------------------  ----------------------------------------
                                       →  symbol_code           (String)
                                          'AH' (fixed - we only have AD.AS)

                                       →  expiry_text           (String)
expiry                                    'December 2029' (preserve original)

                                       →  source                (String)
                                          'mariadb_migration' or 'beursduivel'

                                       →  source_url            (Text)
                                          NULL (no URLs in old data)

                                       →  last_date_text        (String)
last_time                                 String representation

----------------------------------------  ----------------------------------------
COLUMNS WE HAVE BUT WON'T MIGRATE (Greeks - belong in Silver):
----------------------------------------  ----------------------------------------
iv, delta, gamma, theta, vega             → SKIP (calculated in Silver layer)
iv_bid, iv_ask, iv_mid, iv_spread         → SKIP
delta_exposure, gamma_exposure            → SKIP
moneyness, bidask_spread_pct              → SKIP
vpi, iv_delta_15m, size_imbalance         → SKIP

================================================================================
CRITICAL ISSUES TO RESOLVE
================================================================================

1. ❌ EXPIRY DATE PARSING
   - Source has text: 'December 2029', 'Juni 2026', 'Maart 2026'
   - Target needs Date: 2029-12-??, 2026-06-??
   - PROBLEM: No exact day information!
   - SOLUTION OPTIONS:
     a) Use 3rd Friday of month (standard options expiry)
     b) Use last day of month as fallback
     c) Store as expiry_text and calculate later

2. ❌ UNDERLYING_PRICE NOT IN BRONZE_BD_OPTIONS SCHEMA
   - MariaDB has: spot_price (34.32)
   - bronze_bd_options has NO underlying_price column!
   - SOLUTION: Need to add underlying_price column to bronze_bd_options
     OR create companion bronze_bd_underlying table

3. ✅ DEDUPLICATION STRATEGY
   - Use MAX(created_at) per DATE(created_at) to get last snapshot
   - This gives us 1 clean set per day (~130-260 contracts)

4. ✅ SYMBOL_CODE MAPPING
   - All data is AD.AS (Ahold Delhaize)
   - Symbol code: 'AH' (from Euronext)

================================================================================
MIGRATION APPROACH
================================================================================

1. PRE-MIGRATION FIXES:
   - Add underlying_price column to bronze_bd_options (or create separate table)
   - Create expiry date parser for Dutch month names

2. MIGRATION QUERY:
   ```sql
   WITH daily_latest AS (
       SELECT DATE(created_at) as trade_date,
              MAX(created_at) as last_scrape
       FROM option_prices_live
       GROUP BY DATE(created_at)
   )
   SELECT 
       opl.ticker,
       'AH' as symbol_code,
       opl.issue_id,
       DATE(opl.created_at) as trade_date,
       opl.type as option_type,
       PARSE_EXPIRY(opl.expiry) as expiry_date,  -- Custom function
       opl.expiry as expiry_text,
       opl.strike,
       opl.bid,
       opl.ask,
       opl.last_price,
       opl.volume,
       opl.last_time as last_timestamp,
       opl.last_time as last_date_text,
       'beursduivel' as source,
       NULL as source_url,
       opl.created_at as scraped_at,
       opl.spot_price as underlying_price  -- NEEDS COLUMN ADDITION
   FROM option_prices_live opl
   INNER JOIN daily_latest dl 
       ON DATE(opl.created_at) = dl.trade_date 
       AND opl.created_at = dl.last_scrape
   ORDER BY trade_date DESC, option_type, strike
   ```

3. POST-MIGRATION:
   - Verify record counts match
   - Check for NULL values in critical columns
   - Validate expiry date parsing

================================================================================
EXPECTED RESULTS
================================================================================

- Source: 147,254 total records (with duplicates)
- Target: ~4,400 records (18 days × 130-260 contracts/day)
- Date range: 2025-11-10 to 2025-12-10
- All AD.AS options with complete bid/ask/strike/expiry data

================================================================================
NEXT STEPS FOR REVIEW
================================================================================

1. ✅ APPROVE column mapping above
2. ⚠️  DECIDE on expiry parsing strategy (3rd Friday? last day?)
3. ⚠️  DECIDE on underlying_price storage (add column? separate table?)
4. 🔧 CREATE expiry date parser
5. 🔧 UPDATE bronze_bd_options schema if needed
6. ▶️  RUN migration script
"""

print(__doc__)
