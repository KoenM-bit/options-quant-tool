# Architecture Documentation

## System Architecture

The Ahold Options platform follows a modern data engineering architecture with clear separation of concerns:

```
┌─────────────────────────────────────────────────────────────┐
│                      Data Sources                            │
│                   FD.nl Options Data                         │
└────────────────────┬────────────────────────────────────────┘
                     │
                     ▼
┌─────────────────────────────────────────────────────────────┐
│                   Ingestion Layer                            │
│  ┌──────────────────────────────────────────────────────┐   │
│  │ Airflow DAG: ahold_options_daily                     │   │
│  │ - HTTP Client with retry logic                       │   │
│  │ - Rate limiting                                      │   │
│  │ - BeautifulSoup parsing                              │   │
│  └──────────────────────────────────────────────────────┘   │
└────────────────────┬────────────────────────────────────────┘
                     │
                     ▼
┌─────────────────────────────────────────────────────────────┐
│                    Bronze Layer (Raw)                        │
│  PostgreSQL Tables:                                          │
│  - bronze_fd_overview: Market summary data                  │
│  - bronze_fd_options: Raw options chain                     │
│  Storage: 90 days retention                                 │
└────────────────────┬────────────────────────────────────────┘
                     │
                     ▼
┌─────────────────────────────────────────────────────────────┐
│                 Transformation Layer                         │
│  ┌──────────────────────────────────────────────────────┐   │
│  │ DBT Models (Airflow DAG: ahold_dbt_transform)        │   │
│  │ - Data cleaning and validation                       │   │
│  │ - Deduplication                                      │   │
│  │ - Calculated fields (Greeks, moneyness)             │   │
│  │ - Incremental processing                             │   │
│  └──────────────────────────────────────────────────────┘   │
└────────────────────┬────────────────────────────────────────┘
                     │
                     ▼
┌─────────────────────────────────────────────────────────────┐
│                   Silver Layer (Cleaned)                     │
│  PostgreSQL Tables:                                          │
│  - silver_underlying_price: Clean price history             │
│  - silver_options: Validated options with Greeks            │
│  Storage: 365 days retention                                │
└────────────────────┬────────────────────────────────────────┘
                     │
                     ▼
┌─────────────────────────────────────────────────────────────┐
│                  Aggregation Layer                           │
│  DBT Models for business metrics                             │
└────────────────────┬────────────────────────────────────────┘
                     │
                     ▼
┌─────────────────────────────────────────────────────────────┐
│                   Gold Layer (Analytics)                     │
│  PostgreSQL Tables:                                          │
│  - gold_options_summary_daily: Daily aggregations           │
│  - gold_volatility_surface: IV surface data                 │
│  - gold_greek_analytics: Risk metrics by expiry             │
│  - gold_open_interest_flow: Position tracking               │
│  Storage: Permanent                                          │
└────────────────────┬────────────────────────────────────────┘
                     │
                     ▼
┌─────────────────────────────────────────────────────────────┐
│                  Consumption Layer                           │
│  ┌──────────────┐  ┌──────────────┐  ┌──────────────┐      │
│  │  Power BI    │  │   Jupyter    │  │   Custom     │      │
│  │  Dashboards  │  │   Notebooks  │  │     Apps     │      │
│  └──────────────┘  └──────────────┘  └──────────────┘      │
└─────────────────────────────────────────────────────────────┘
```

## Component Details

### 1. Scraping Layer

**Technology**: Python with BeautifulSoup, Requests

**Components**:
- `FDOverviewScraper`: Scrapes market summary
- `HTTPClient`: HTTP client with retry and rate limiting
- Parsers: Dutch number format conversion

**Design Patterns**:
- Factory pattern for scraper creation
- Retry with exponential backoff
- Rate limiting (10 requests/minute)
- User-agent rotation

### 2. Orchestration Layer

**Technology**: Apache Airflow 2.8

**DAGs**:

1. **ahold_options_daily**
   - Schedule: Daily at 18:00 CET
   - Tasks: Scrape → Validate → Load → Notify
   - Retries: 3 with 5min delay

2. **ahold_dbt_transform**
   - Schedule: Triggered by daily DAG
   - Tasks: DBT run → Test → Docs
   - Idempotent transformations

3. **data_quality_checks**
   - Schedule: Every 4 hours
   - Tasks: Freshness → Completeness → Anomalies

### 3. Storage Layer

**Technology**: PostgreSQL 15

**Schema Design**:
- Time-series optimized indexes
- Partitioning by date (future)
- Numeric types for precision
- Efficient composite indexes

**Retention Policy**:
- Bronze: 90 days
- Silver: 365 days
- Gold: Permanent

### 4. Transformation Layer

**Technology**: DBT 1.7

**Model Structure**:
```
models/
├── bronze/           # Source definitions
├── silver/           # Cleaning & validation
│   ├── silver_underlying_price.sql
│   └── silver_options.sql
└── gold/            # Business aggregations
    ├── gold_options_summary_daily.sql
    ├── gold_volatility_surface.sql
    ├── gold_greek_analytics.sql
    └── gold_open_interest_flow.sql
```

**Key Features**:
- Incremental materialization
- Data quality tests
- Documentation generation
- Dependency management

### 5. Monitoring Layer

**Components**:
- Airflow task monitoring
- Database health checks
- Data quality alerts
- Slack/Email notifications

## Data Flow

### Ingestion Flow
```
FD.nl → HTTP Request → HTML Parse → Bronze Table
  ↓
Rate Limit Check
  ↓
Retry on Failure (3x)
  ↓
Data Quality Check
  ↓
Alert on Issues
```

### Transformation Flow
```
Bronze → DBT Source → Silver Models → Gold Models
         ↓              ↓                ↓
      Validation    Deduplication   Aggregation
                       ↓
                  Calculated Fields
                       ↓
                  Quality Tests
```

## Deployment Architecture

### Local Development
```
Docker Compose
├── PostgreSQL (5432)
├── Airflow Webserver (8080)
├── Airflow Scheduler
└── Shared Volumes
```

### Synology NAS Production
```
Container Manager
├── PostgreSQL (SSD recommended)
├── Airflow Webserver
├── Airflow Scheduler
└── Volume1 mounts
```

## Scalability Considerations

### Current Capacity
- **Data Volume**: ~10K options contracts/day
- **Processing Time**: ~2 minutes scrape + 1 minute transform
- **Storage**: ~1GB/year
- **Concurrent Users**: 5-10

### Scale-Out Options
1. **More Symbols**: Add scraper tasks in parallel
2. **Larger Dataset**: Partition PostgreSQL tables
3. **Distributed Processing**: Switch to CeleryExecutor
4. **High Availability**: Add read replicas

## Security Architecture

### Data Security
- Encrypted connections (TLS)
- Fernet encryption for Airflow secrets
- Environment variable based config
- No credentials in code

### Network Security
- Internal Docker network
- Exposed ports: 8080 (Airflow), 5432 (optional)
- Firewall rules on Synology

### Access Control
- Airflow RBAC
- PostgreSQL role-based access
- Admin-only DAG modification

## Disaster Recovery

### Backup Strategy
1. **Database Dumps**: Daily pg_dump
2. **Configuration**: Git version control
3. **Logs**: 30 days retention

### Recovery Procedures
1. Restore PostgreSQL from backup
2. Redeploy containers
3. Replay failed DAG runs
4. Validate data integrity

## Performance Optimization

### Database
- Indexes on foreign keys and date columns
- VACUUM ANALYZE weekly
- Connection pooling (5-10 connections)

### Airflow
- LocalExecutor for Synology
- Task parallelism: 4
- Pool configuration for scrapers

### DBT
- Incremental models
- Materializations optimized
- Targeted runs via tags

## Monitoring & Observability

### Metrics
- DAG success rate
- Task duration
- Data freshness
- Row counts

### Alerts
- Failed DAG runs
- Stale data (>24h)
- Database connection issues
- Disk space warnings

### Logs
- Structured JSON logging
- 30-day retention
- Error aggregation

## Future Enhancements

### Phase 2
- [ ] Historical data backfill
- [ ] Real-time streaming (WebSocket)
- [ ] ML-based anomaly detection
- [ ] Multi-symbol support

### Phase 3
- [ ] REST API for data access
- [ ] GraphQL interface
- [ ] Mobile app
- [ ] Alerting on Greek thresholds

---

**Last Updated**: 2024-12-07
