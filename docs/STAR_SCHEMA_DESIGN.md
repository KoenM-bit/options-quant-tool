# Star Schema Design for Silver Layer

## Overview
Redesign the silver layer from a denormalized wide table (`silver_bd_options_enriched`) into a **star schema** with separated dimension and fact tables. This improves query performance, reduces redundancy, and makes analytics clearer.

---

## Current State (Denormalized)

### Current Table: `silver_bd_options_enriched`
- **Purpose**: Wide table with contract details + pricing + Greeks
- **Issues**:
  - Contract attributes (ticker, symbol_code, option_type, strike, expiry) repeated for every trade date
  - Underlying price repeated across all contracts for same date
  - High redundancy (same strike/expiry repeated across dates)
  - Hard to query "all contracts for a given underlying" or "all dates for a contract"
  - Greeks and prices mixed with dimensional attributes

---

## Target State (Star Schema)

### Architecture
```
                    ┌─────────────────┐
                    │  dim_contract   │
                    │  (Contract SCD) │
                    └────────┬────────┘
                             │
                             │ contract_key
                             │
┌──────────────┐    ┌────────▼────────────┐    ┌──────────────┐
│   dim_date   │────│ fact_options_daily  │────│ dim_underlying│
│  (Date Dim)  │    │   (Measures/Facts)  │    │ (Stock Info) │
└──────────────┘    └────────┬────────────┘    └──────────────┘
                             │
                             │ source_key
                             │
                    ┌────────▼────────┐
                    │   dim_source    │
                    │  (Data Source)  │
                    └─────────────────┘
```

---

## Dimension Tables

### 1. `dim_contract` - Contract Dimension (Type 2 SCD)
**Purpose**: Store unique option contract definitions

```sql
CREATE TABLE dim_contract (
    contract_key INTEGER PRIMARY KEY AUTOINCREMENT,  -- Surrogate key
    
    -- Natural key (business key)
    ticker VARCHAR(20) NOT NULL,
    symbol_code VARCHAR(20),
    issue_id VARCHAR(50),
    option_type VARCHAR(10) NOT NULL,  -- 'call' or 'put'
    strike DECIMAL(10,2) NOT NULL,
    expiry_date DATE NOT NULL,
    
    -- Descriptive attributes
    expiry_text VARCHAR(100),
    isin VARCHAR(20),
    
    -- SCD Type 2 fields (track changes over time)
    valid_from DATE NOT NULL,
    valid_to DATE,  -- NULL = current version
    is_current BOOLEAN DEFAULT TRUE,
    
    -- Metadata
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    
    -- Indexes
    UNIQUE (ticker, option_type, strike, expiry_date, valid_from),
    INDEX idx_contract_natural_key (ticker, strike, expiry_date),
    INDEX idx_contract_current (is_current, ticker)
);
```

**Rationale**:
- Separates contract definition from daily measures
- SCD Type 2 allows tracking if contract attributes change (e.g., symbol_code update)
- Most queries: JOIN on `contract_key` (integer FK) instead of multiple columns

---

### 2. `dim_date` - Date Dimension
**Purpose**: Store date attributes for time-based analysis

```sql
CREATE TABLE dim_date (
    date_key INTEGER PRIMARY KEY,  -- YYYYMMDD format (20251212)
    
    -- Date attributes
    date DATE NOT NULL UNIQUE,
    year INTEGER NOT NULL,
    quarter INTEGER NOT NULL,
    month INTEGER NOT NULL,
    week INTEGER NOT NULL,
    day_of_week INTEGER NOT NULL,  -- 1=Monday, 7=Sunday
    day_of_month INTEGER NOT NULL,
    day_of_year INTEGER NOT NULL,
    
    -- Business calendar
    is_weekday BOOLEAN,
    is_month_end BOOLEAN,
    is_quarter_end BOOLEAN,
    is_year_end BOOLEAN,
    
    -- Descriptive
    month_name VARCHAR(20),
    day_name VARCHAR(20),
    
    -- Metadata
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    
    INDEX idx_date_natural (date)
);
```

**Rationale**:
- Pre-computed date attributes speed up time-based aggregations
- Enables easy filtering: "Show me all Fridays", "Month-end positions"
- `date_key` as integer (YYYYMMDD) is faster to join than DATE type

---

### 3. `dim_underlying` - Underlying Asset Dimension (Type 2 SCD)
**Purpose**: Store underlying stock/asset information

```sql
CREATE TABLE dim_underlying (
    underlying_key INTEGER PRIMARY KEY AUTOINCREMENT,  -- Surrogate key
    
    -- Natural key
    ticker VARCHAR(20) NOT NULL,
    
    -- Attributes
    isin VARCHAR(20),
    name VARCHAR(200),
    exchange VARCHAR(50),
    currency VARCHAR(10) DEFAULT 'EUR',
    
    -- SCD Type 2 fields
    valid_from DATE NOT NULL,
    valid_to DATE,  -- NULL = current version
    is_current BOOLEAN DEFAULT TRUE,
    
    -- Metadata
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    
    UNIQUE (ticker, valid_from),
    INDEX idx_underlying_current (is_current, ticker)
);
```

**Rationale**:
- Separates ticker metadata from daily prices
- SCD Type 2 tracks name/ISIN changes over time
- Multiple tickers (AD.AS, AH.AS) can be easily distinguished

---

### 4. `dim_source` - Data Source Dimension
**Purpose**: Track which scraper/source provided the data

```sql
CREATE TABLE dim_source (
    source_key INTEGER PRIMARY KEY AUTOINCREMENT,
    
    -- Source attributes
    source_name VARCHAR(50) NOT NULL UNIQUE,  -- 'beursduivel', 'fd', 'manual'
    source_description VARCHAR(200),
    source_url_pattern VARCHAR(500),
    
    -- Quality metadata
    typical_latency_minutes INTEGER,  -- How delayed is this source
    has_greeks BOOLEAN DEFAULT FALSE,
    has_open_interest BOOLEAN DEFAULT FALSE,
    
    -- Metadata
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);
```

**Rationale**:
- Track data lineage (BD vs FD data)
- Enables quality metrics per source
- Easy to filter: "Show only FD data" or "Show only sources with OI"

---

## Fact Table

### `fact_options_daily` - Daily Options Fact Table
**Purpose**: Store time-varying measures (prices, Greeks, volume) with references to dimensions

```sql
CREATE TABLE fact_options_daily (
    fact_key INTEGER PRIMARY KEY AUTOINCREMENT,
    
    -- Foreign keys to dimensions
    contract_key INTEGER NOT NULL,  -- FK to dim_contract
    date_key INTEGER NOT NULL,      -- FK to dim_date
    underlying_key INTEGER NOT NULL, -- FK to dim_underlying
    source_key INTEGER NOT NULL,    -- FK to dim_source
    
    -- Pricing measures (from bronze)
    bid DECIMAL(10,4),
    ask DECIMAL(10,4),
    mid_price DECIMAL(10,4),  -- (bid + ask) / 2
    last_price DECIMAL(10,4),
    
    -- Market activity measures
    volume INTEGER,
    open_interest INTEGER,  -- NULL for BD (no OI)
    
    -- Underlying price measures
    underlying_price DECIMAL(10,4),
    underlying_bid DECIMAL(10,4),
    underlying_ask DECIMAL(10,4),
    
    -- Derived measures
    moneyness DECIMAL(10,6),  -- underlying_price / strike
    intrinsic_value DECIMAL(10,4),
    time_value DECIMAL(10,4),
    days_to_expiry INTEGER,
    
    -- Greeks (calculated measures)
    delta DECIMAL(10,6),
    gamma DECIMAL(10,6),
    theta DECIMAL(10,6),
    vega DECIMAL(10,6),
    rho DECIMAL(10,6),
    implied_volatility DECIMAL(10,6),
    
    -- Quality indicators
    greeks_calculated BOOLEAN DEFAULT FALSE,
    iv_converged BOOLEAN DEFAULT FALSE,
    data_quality_score DECIMAL(3,2),  -- 0.0 to 1.0
    
    -- Metadata
    scraped_at TIMESTAMP NOT NULL,
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    
    -- Constraints
    UNIQUE (contract_key, date_key, source_key),  -- One row per contract/date/source
    
    -- Indexes for common queries
    INDEX idx_fact_contract (contract_key, date_key),
    INDEX idx_fact_date (date_key, contract_key),
    INDEX idx_fact_underlying (underlying_key, date_key),
    INDEX idx_fact_source (source_key, date_key),
    INDEX idx_fact_moneyness (date_key, moneyness)
);
```

**Rationale**:
- Only stores **time-varying measures** (prices, Greeks, volume)
- All **descriptive attributes** moved to dimensions
- Small integer FKs instead of repeated strings (saves space)
- Enables fast analytics: "Sum all delta by date", "Avg IV by strike"

---

## Benefits of Star Schema

### 1. Query Performance
- **Fewer JOINs**: 4-5 small dimension tables vs 1 huge wide table
- **Faster aggregations**: No need to GROUP BY repeated strings (ticker, option_type)
- **Better indexes**: Can index dimensions separately from facts

### 2. Storage Efficiency
- **Reduced redundancy**: Contract details stored once, not repeated per date
- **Smaller fact table**: Only integers (FKs) + measures = narrow rows
- **Example**: 10,000 contracts × 252 trading days = 2.5M rows
  - Current: ~200 bytes/row × 2.5M = 500 MB
  - Star: ~80 bytes/row × 2.5M = 200 MB (60% smaller)

### 3. Analytical Clarity
- **Dimensions answer "who/what/where/when"**: Which contract? Which date? Which source?
- **Facts answer "how much"**: What price? What delta? What volume?
- **Easy to add dimensions**: Add `dim_market_regime` (bull/bear), `dim_volatility_regime` (high/low)
- **BI tool friendly**: Power BI, Tableau love star schemas

### 4. Data Quality
- **Referential integrity**: FKs ensure valid contracts/dates
- **SCD Type 2**: Track contract metadata changes over time
- **Centralized lookups**: Update contract name once in dimension, not in every fact row

---

## Migration Strategy

### Phase 1: Build Star Schema (Parallel to Existing)
1. **Create new dimension/fact tables** (SQLAlchemy models)
2. **Create dbt models** to populate from bronze
3. **Test locally** with `test_pipeline_e2e.py`
4. **Validate counts**: fact rows = current silver rows

### Phase 2: Populate Historical Data
1. **Backfill dimensions** from existing bronze data
2. **Backfill fact table** from bronze + silver (Greeks)
3. **Validate**: counts, nulls, FK integrity

### Phase 3: Update Pipeline
1. **Update DAGs**: Run dbt to populate star schema
2. **Keep old silver table** temporarily (backward compatibility)
3. **Update exports**: Export from star schema OR create view

### Phase 4: Deprecate Old Silver (Optional)
1. **Create view**: `silver_bd_options_enriched_view` joins star schema
2. **Update downstream consumers** (dashboards, exports)
3. **Drop old table** after validation period

---

## dbt Model Structure

### Proposed dbt File Organization
```
dbt/ahold_options/models/silver/
├── dimensions/
│   ├── dim_contract.sql        -- Dedupe contracts from bronze
│   ├── dim_date.sql            -- Generate date dimension
│   ├── dim_underlying.sql      -- Dedupe tickers from bronze
│   └── dim_source.sql          -- Seed or manual insert
├── facts/
│   └── fact_options_daily.sql  -- Join bronze + dimensions
└── legacy/
    └── silver_bd_options_enriched_view.sql  -- Backward compat view
```

---

## Example Query Comparisons

### Current (Denormalized)
```sql
-- Get all ATM calls for last 30 days
SELECT 
    ticker, trade_date, strike, expiry_date,
    bid, ask, delta, implied_volatility
FROM silver_bd_options_enriched
WHERE option_type = 'call'
  AND ABS(moneyness - 1.0) < 0.05
  AND trade_date >= CURRENT_DATE - INTERVAL '30 days';
```

### Star Schema (Same Query)
```sql
-- Get all ATM calls for last 30 days
SELECT 
    d.date, c.ticker, c.strike, c.expiry_date,
    f.bid, f.ask, f.delta, f.implied_volatility
FROM fact_options_daily f
JOIN dim_contract c ON f.contract_key = c.contract_key
JOIN dim_date d ON f.date_key = d.date_key
WHERE c.option_type = 'call'
  AND c.is_current = TRUE
  AND ABS(f.moneyness - 1.0) < 0.05
  AND d.date >= CURRENT_DATE - INTERVAL '30 days';
```

**Star schema advantages**:
- Filter on `c.is_current` ensures correct SCD version
- `date_key` join faster than date comparison
- Can add `WHERE d.is_weekday = TRUE` easily
- Dimensions are small (indexed, cached)

---

## Implementation Timeline

### Week 1: Design & Models
- ✅ Create this design document
- ☐ Create SQLAlchemy models for all dimensions + fact
- ☐ Create dbt SQL for dimension population
- ☐ Create dbt SQL for fact population

### Week 2: Local Testing
- ☐ Update `test_pipeline_e2e.py` with star schema
- ☐ Validate data flows correctly
- ☐ Test Greeks enrichment with new schema
- ☐ Performance benchmarks (query speed, storage)

### Week 3: Backfill & Deploy
- ☐ Run backfill scripts for historical data
- ☐ Deploy to Synology (parallel to old schema)
- ☐ Monitor for data quality issues
- ☐ Validate ClickHouse exports

### Week 4: Cutover (Optional)
- ☐ Create view for backward compatibility
- ☐ Update dashboards to use star schema
- ☐ Deprecate old `silver_bd_options_enriched`

---

## Risks & Mitigations

### Risk 1: Complex JOINs slow down queries
- **Mitigation**: Dimension tables are small (< 10K rows each), heavily indexed, will be cached
- **Mitigation**: Create materialized view if needed for common query patterns

### Risk 2: Backfill takes too long
- **Mitigation**: Backfill in batches (by month)
- **Mitigation**: Run in parallel to existing pipeline (no downtime)

### Risk 3: Breaking downstream consumers
- **Mitigation**: Keep old table as view on top of star schema
- **Mitigation**: Phase cutover over 2-4 weeks with monitoring

### Risk 4: SCD Type 2 complexity
- **Mitigation**: Most contracts are immutable (strike/expiry don't change)
- **Mitigation**: dbt handles SCD logic cleanly with snapshots

---

## Next Steps

1. **Review & Approve Design**: Confirm this matches your vision
2. **Create SQLAlchemy Models**: I'll build the dimension/fact models
3. **Create dbt SQL**: I'll write the transformation logic
4. **Test Locally**: Run `test_pipeline_e2e.py` with star schema
5. **Deploy**: Roll out to Synology in phases

Would you like me to start implementing the SQLAlchemy models and dbt SQL now?
