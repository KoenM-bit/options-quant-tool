# Ahold Options Quantitative Analysis Tool 📊

Enterprise-grade options analytics platform for Ahold Delhaize (AD.AS) with automated data pipeline, Greeks calculation, and advanced risk metrics.

[![Docker Build](https://github.com/KoenM-bit/options-quant-tool/actions/workflows/docker-build.yml/badge.svg)](https://github.com/KoenM-bit/options-quant-tool/actions/workflows/docker-build.yml)
[![Python 3.10+](https://img.shields.io/badge/python-3.10+-blue.svg)](https://www.python.org/downloads/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

## 🚀 Features

### Data Pipeline (Bronze → Silver → Gold)
- **Bronze Layer**: Raw scraped options data from FD.nl (13,114+ records)
- **Silver Layer**: Deduplicated + enriched with Black-Scholes Greeks (6,607 validated)
- **Gold Layer**: Advanced analytics (GEX, max pain, volatility surface, skew)

### Analytics
- ✅ **Gamma Exposure (GEX)**: Market maker hedging pressure analysis
- ✅ **Max Pain**: Price level with maximum option seller profit
- ✅ **Volatility Surface**: Full IV surface by strike and expiry
- ✅ **Volatility Skew**: Put/Call IV disparity analysis
- ✅ **Term Structure**: IV across expiration dates
- ✅ **Greeks Calculation**: Delta, Gamma, Vega, Theta, Rho with 5-stage validation (83.6% success rate)

### Quality Monitoring
- 📊 Real-time Streamlit dashboard for Bronze/Silver/Gold layers
- 🔍 5-stage Greeks validation with quality gates
- 📈 ATM IV trends with neighboring strikes (OTM-2 → ATM → ITM+2)
- ✅ Enterprise-quality metadata tracking (risk-free rates, validation status)

## 🏗️ Architecture

```
┌─────────────────────────────────────────────────────────────────┐
│  Automated Daily Pipeline (22:00 CET, Mon-Fri)                  │
└─────────────────────────────────────────────────────────────────┘

1. Scrape (Bronze)
   └─ FD.nl options chain → PostgreSQL raw tables

2. DBT Silver
   └─ Deduplicate + standardize (option_key unique constraint)

3. Greeks Enrichment
   └─ Black-Scholes model with py_vollib
   └─ ECB €STR risk-free rates (per-date caching)
   └─ 5-stage validation gates
   └─ Metadata: risk_free_rate_used, greeks_valid, greeks_status

4. DBT Gold
   └─ 11 analytics models (only greeks_valid = TRUE records):
      • GEX (gamma exposure)           → 1,140 rows
      • Max Pain                       → 179 rows
      • Volatility Surface             → 6,607 rows
      • Skew Analysis                  → 380 rows
      • Term Structure                 → 13 rows
      • Put/Call Metrics               → 107 rows
      • Open Interest Flow             → 884 rows
      • Daily Summary                  → 25 rows
      • Key Levels                     → 79 rows
      • Weekly GEX                     → 1,140 rows
      • GEX Positioning Trends         → 1,140 rows

5. Export
   └─ Parquet files for Power BI integration
```

### Data Layers

- **Bronze** (13,114 records): Raw scraped data with immutable audit trail
- **Silver** (12,624 deduplicated, 6,607 validated Greeks): Enriched with Black-Scholes calculations, quality flags
- **Gold** (10,000+ analytics points): Business-ready metrics filtered for validated Greeks only

## 🚀 Quick Start

### Prerequisites

- Docker & Docker Compose
- Python 3.10+
- 8GB RAM minimum
- For Synology: DSM 7.0+ with Container Manager

### Option 1: Pull Pre-built Docker Image

```bash
# Pull from GitHub Container Registry
docker pull ghcr.io/koenm-bit/options-quant-tool:latest

# Run with docker-compose
git clone https://github.com/KoenM-bit/options-quant-tool.git
cd options-quant-tool
cp .env.example .env
docker compose up -d
```

### Option 2: Build from Source

1. **Clone and configure**
   ```bash
   git clone https://github.com/KoenM-bit/options-quant-tool.git
   cd options-quant-tool
   cp .env.example .env
   # Edit .env with your configuration
   ```

2. **Generate Fernet key for Airflow**
   ```bash
   python -c "from cryptography.fernet import Fernet; print(Fernet.generate_key().decode())"
   # Add the output to AIRFLOW__CORE__FERNET_KEY in .env
   ```

3. **Start services**
   ```bash
   docker compose up -d --build
   ```

4. **Access services**
   - **Airflow UI**: http://localhost:8081 (admin/admin)
   - **Data Quality Dashboard**: http://localhost:8501
   - **Adminer (DB admin)**: http://localhost:8082
   - **PostgreSQL**: localhost:5432 (airflow/airflow)

5. **Initialize database and run first DAG**
   ```bash
   # Wait for services to be healthy (~30 seconds)
   docker compose exec airflow-webserver airflow dags trigger ahold_options_daily
   ```

### Services Overview

| Service | Port | Description | Status |
|---------|------|-------------|--------|
| Airflow Webserver | 8081 | DAG monitoring and execution | ✅ Running |
| Airflow Scheduler | - | Background job scheduler | ✅ Running |
| PostgreSQL | 5432 | Primary database | ✅ Running |
| Streamlit Dashboard | 8501 | Quality monitoring | ✅ Running |
| Adminer | 8082 | Database administration | ✅ Running |

### Synology Deployment

1. **Pull image on Synology**
   ```bash
   docker pull ghcr.io/koenm-bit/options-quant-tool:latest
   ```

2. **Create project folder**: `/volume1/docker/ahold-options/`

3. **Upload files** via File Station:
   - docker-compose.yml
   - .env (configured for your environment)

4. **Import in Container Manager**:
   - Open Container Manager → Project
   - Create from docker-compose.yml
   - Set environment variables from .env

5. **Start project**

See [docs/SYNOLOGY_SETUP.md](docs/SYNOLOGY_SETUP.md) for detailed instructions.

## 📦 Project Structure

```
ahold-options/
├── dags/                          # Airflow DAGs
│   ├── ahold_options_daily.py     # Main daily scraping DAG
│   ├── ahold_dbt_transform.py     # DBT transformation DAG
│   └── data_quality_checks.py     # Data quality monitoring
├── src/                           # Python source code
│   ├── scrapers/                  # Scraper modules
│   │   ├── fd_overview_scraper.py
│   │   └── fd_options_scraper.py
│   ├── models/                    # SQLAlchemy models
│   │   ├── bronze.py
│   │   ├── silver.py
│   │   └── gold.py
│   ├── utils/                     # Utilities
│   │   ├── http_client.py
│   │   ├── parsers.py
│   │   ├── db.py
│   │   └── alerts.py
│   └── config.py                  # Configuration management
├── dbt/                           # DBT project
│   └── ahold_options/
│       ├── models/
│       │   ├── bronze/
│       │   ├── silver/
│       │   └── gold/
│       ├── tests/
│       ├── macros/
│       └── dbt_project.yml
├── tests/                         # Unit & integration tests
├── docs/                          # Documentation
├── scripts/                       # Utility scripts
│   ├── init_db.py
│   └── backfill.py
├── docker-compose.yml             # Local development
├── docker-compose.synology.yml    # Synology optimized
├── Dockerfile                     # Custom Airflow image
├── requirements.txt               # Python dependencies
└── .env.example                   # Environment template
```

## 🔄 Pipeline Overview

### Main DAG: `ahold_options_daily`

**Schedule**: 22:00 CET, Monday-Friday (after market close)

**Task Flow**:
```
scrape_data_task
    ↓
validate_data_task
    ↓
run_dbt_silver_task  (Bronze → Silver deduplication)
    ↓
calculate_greeks_task  (Greeks enrichment with ECB rates)
    ↓
run_dbt_gold_task  (Silver → Gold analytics, greeks_valid only)
    ↓
export_parquet_task  (Gold → Parquet for Power BI)
    ↓
copy_to_powerbi_task  (Move to Power BI directory)
    ↓
success_notification_task  (Slack/Email notification)
```

### Greeks Calculation Details

**Implementation**: Black-Scholes model with py_vollib
- **Implied Volatility**: Brent's method with bounded optimization [1%, 400%]
- **Risk-Free Rate**: ECB €STR overnight rate with per-date caching
- **Rate Clamping**: [0%, 10%] range with 2% fallback
- **Statistics**: 26 ECB API calls, 0 failures, 100% cache hit rate

**5-Stage Validation**:
1. **Moneyness Check**: S/K within [0.5, 2.0]
2. **Time Value Check**: Positive time to expiration
3. **IV Range Check**: 1% ≤ IV ≤ 400%
4. **Price Error Check**: |Theoretical - Market| < 10%
5. **Greeks Bounds Check**: Realistic ranges for all Greeks

**Success Rate**: 83.6% (6,607 valid / 7,903 processed)

### Data Quality Monitoring

**Streamlit Dashboard** (http://localhost:8501)

Features:
- **Bronze View**: Scrape volume trends, daily aggregates, latest records
- **Silver View**: Greeks quality metrics, IV distributions, time series
- **ATM IV Trends**: Neighboring strikes analysis (OTM-2 → ATM → ITM+2)
  - IV trends over time
  - Current IV smile/skew visualization
  - Put-call skew metrics
- **Comparison View**: Bronze → Silver funnel analysis

Launch dashboard:
```bash
./run_quality_dashboard.sh
# Or manually:
source .venv/bin/activate
streamlit run dashboards/data_quality_monitor.py --server.port=8501
```

## 🗄️ Database Schema

### Bronze Layer
- `bronze_fd_overview`: Raw market overview data (underlying price, totals)
- `bronze_fd_options`: Raw options chain data (all strikes/expirations)

### Silver Layer
- `silver_options`: Deduplicated options with validated Greeks
  - **Unique key**: `option_key` (type + strike + expiry)
  - **Greeks**: delta, gamma, vega, theta, rho, implied_volatility
  - **Metadata**: `risk_free_rate_used`, `greeks_valid`, `greeks_status`
  - **Quality**: 52.34% coverage with validated Greeks (6,607 / 12,624)

### Gold Layer (11 Analytics Models)

| Model | Description | Records |
|-------|-------------|---------|
| `gold_gamma_exposure` | Daily GEX by strike | 1,140 |
| `gold_gamma_exposure_weekly` | Weekly GEX snapshots (Fridays) | 1,140 |
| `gold_gex_positioning_trends` | GEX trend analysis | 1,140 |
| `gold_volatility_surface` | Full IV surface matrix | 6,607 |
| `gold_skew_analysis` | Put/Call skew by moneyness | 380 |
| `gold_volatility_term_structure` | IV term structure | 13 |
| `gold_max_pain` | Max pain price levels | 179 |
| `gold_put_call_metrics` | P/C ratios and volume | 107 |
| `gold_options_summary_daily` | Daily aggregates | 25 |
| `gold_open_interest_flow` | OI flow analysis | 884 |
| `gold_key_levels` | Support/resistance levels | 79 |

**Total Gold Layer**: 10,694 analytics data points

## 🔧 Configuration

### Environment Variables

Key variables in `.env`:

```bash
# Database
POSTGRES_USER=airflow
POSTGRES_PASSWORD=airflow
POSTGRES_DB=ahold_options
POSTGRES_HOST=localhost  # Use 127.0.0.1 for dashboard
POSTGRES_PORT=5432

# Ahold Settings
AHOLD_TICKER=AD.AS
AHOLD_SYMBOL_CODE=AEX.AH/O

# Scraper
SCRAPER_TIMEOUT=30
SCRAPER_RETRY_ATTEMPTS=3
SCRAPER_USER_AGENT=Mozilla/5.0...

# ECB Risk-Free Rate
ECB_RATE_API_URL=https://data.ecb.europa.eu/data-detail-api
ECB_RATE_DEFAULT=0.02  # 2% fallback
ECB_RATE_MIN=0.00  # Clamp minimum
ECB_RATE_MAX=0.10  # Clamp maximum

# Airflow
AIRFLOW__CORE__EXECUTOR=LocalExecutor
AIRFLOW__CORE__FERNET_KEY=<your-fernet-key>
AIRFLOW__WEBSERVER__SECRET_KEY=<your-secret-key>

# Logging
LOG_LEVEL=INFO
```

### Database Connection Notes

- **Airflow tasks**: Use `postgres` as hostname (Docker network)
- **Local scripts/dashboard**: Use `localhost` or `127.0.0.1`
- **Credentials**: airflow/airflow (default for development)
- **Port**: 5432 (standard PostgreSQL)

## 📊 Power BI Integration

### Option 1: Direct PostgreSQL Connection

1. **Install PostgreSQL connector** in Power BI Desktop

2. **Connect to database**
   ```
   Server: localhost:5432 (or Synology IP)
   Database: ahold_options
   Username: airflow
   Password: airflow
   ```

3. **Import Gold layer tables** (11 models available):
   - `gold_gamma_exposure` - Daily GEX by strike
   - `gold_max_pain` - Max pain price levels
   - `gold_volatility_surface` - Full IV surface
   - `gold_skew_analysis` - Put/Call skew metrics
   - `gold_put_call_metrics` - P/C ratios
   - `gold_options_summary_daily` - Daily aggregates
   - `gold_volatility_term_structure` - IV term structure
   - `gold_open_interest_flow` - OI flow analysis
   - `gold_key_levels` - Support/resistance
   - `gold_gamma_exposure_weekly` - Weekly GEX snapshots
   - `gold_gex_positioning_trends` - GEX trends

### Option 2: Parquet Files

Pipeline exports Gold models to parquet:
```
/path/to/powerbi/data/
  ├── gold_gamma_exposure.parquet
  ├── gold_max_pain.parquet
  ├── gold_volatility_surface.parquet
  └── ... (all 11 models)
```

Import parquet files in Power BI for faster refresh and offline analysis.

## 🧪 Testing

```bash
# Run all tests
docker-compose exec airflow-webserver pytest tests/

# Run specific test suite
pytest tests/test_scrapers.py
pytest tests/test_transformations.py

# Run with coverage
pytest --cov=src tests/
```

## 📈 Monitoring & Alerts

- **Airflow UI**: DAG runs, task logs, metrics
- **Database queries**: Data freshness, quality checks
- **Alerts**: Slack/Email notifications on failures
- **Logs**: Structured JSON logs in `logs/` directory

## 🛠️ Development & Maintenance

### Local Development Setup

```bash
# Install dependencies
pip install -r requirements.txt

# Configure Python environment
python -m venv .venv
source .venv/bin/activate  # On Windows: .venv\Scripts\activate

# Install development dependencies
pip install pytest pytest-cov black flake8 mypy

# Run DBT models locally
cd dbt/ahold_options
dbt run --models tag:silver
dbt run --models tag:gold

# Test DBT models
dbt test

# Generate DBT docs
dbt docs generate
dbt docs serve
```

### Manual Pipeline Triggers

```bash
# Trigger full daily pipeline
docker compose exec airflow-webserver airflow dags trigger ahold_options_daily

# Run specific task
docker compose exec airflow-webserver airflow tasks test ahold_options_daily calculate_greeks_task 2024-12-09

# Backfill Silver Greeks
docker compose exec airflow-webserver python src/analytics/enrich_silver_greeks.py

# Check pipeline status
docker compose exec airflow-webserver airflow dags list
docker compose exec airflow-webserver airflow dags state ahold_options_daily
```

### Database Operations

```bash
# Backup database
docker compose exec postgres pg_dump -U airflow ahold_options > backup_$(date +%Y%m%d).sql

# Restore database
docker compose exec -T postgres psql -U airflow ahold_options < backup_20241209.sql

# Connect to database
docker compose exec postgres psql -U airflow -d ahold_options

# Check data quality
docker compose exec postgres psql -U airflow -d ahold_options -c "
  SELECT 
    (SELECT COUNT(*) FROM bronze_fd_options) as bronze_count,
    (SELECT COUNT(*) FROM silver_options) as silver_count,
    (SELECT COUNT(*) FROM silver_options WHERE greeks_valid = TRUE) as valid_greeks,
    (SELECT COUNT(*) FROM gold_gamma_exposure) as gold_gex_count;
"
```

### Monitoring & Logs

```bash
# View Airflow logs
docker compose logs -f airflow-scheduler
docker compose logs -f airflow-webserver

# View specific task logs
docker compose exec airflow-webserver airflow tasks logs ahold_options_daily calculate_greeks_task 2024-12-09

# View database logs
docker compose logs -f postgres

# View dashboard logs
docker compose logs -f streamlit
```

### Update Dependencies

```bash
# Update Python packages
pip install -r requirements.txt --upgrade
pip freeze > requirements.txt

# Rebuild Docker images
docker compose build --no-cache

# Pull latest pre-built image
docker pull ghcr.io/koenm-bit/options-quant-tool:latest
```

## 🔐 Security Best Practices

- ✅ All secrets in `.env` (never commit)
- ✅ Database credentials rotated regularly
- ✅ Airflow Fernet key for connection encryption
- ✅ Rate limiting on scrapers
- ✅ User-agent rotation
- ✅ HTTPS only for external requests

## 📚 Documentation

- [Architecture Deep Dive](docs/ARCHITECTURE.md) - Detailed system design
- [Synology Setup Guide](docs/SYNOLOGY_SETUP.md) - Deploy to Synology NAS
- [DBT Models Documentation](docs/DBT_MODELS.md) - Data transformation logic
- [Greeks Calculation](docs/GREEKS.md) - Black-Scholes implementation details
- [Dashboard Guide](dashboards/README.md) - Quality monitoring dashboard
- [API Reference](docs/API.md) - Python API documentation
- [Troubleshooting](docs/TROUBLESHOOTING.md) - Common issues and solutions

## 🐳 Docker Images

### Automated Builds

Docker images are automatically built and published to GitHub Container Registry on every push to `main` branch.

**Pull latest image**:
```bash
docker pull ghcr.io/koenm-bit/options-quant-tool:latest
```

**Available tags**:
- `latest` - Latest build from main branch
- `v1.0.0` - Semantic version tags
- `sha-abc1234` - Specific commit SHA
- `pr-123` - Pull request builds

**Multi-platform support**:
- linux/amd64 (Intel/AMD x86_64)
- linux/arm64 (Apple Silicon, Raspberry Pi)

### Build Locally

```bash
# Build all services
docker compose build

# Build specific service
docker compose build airflow-webserver

# Build with no cache
docker compose build --no-cache
```

## 🤝 Contributing

Contributions are welcome! Here's how:

1. **Fork** the repository
2. **Create feature branch**: `git checkout -b feature/amazing-feature`
3. **Make changes** and test thoroughly
4. **Commit**: `git commit -m 'Add amazing feature'`
5. **Push**: `git push origin feature/amazing-feature`
6. **Open Pull Request** with clear description

### Development Guidelines

- Follow PEP 8 style guide
- Add tests for new features
- Update documentation
- Ensure all tests pass: `pytest tests/`
- Run linters: `black src/` and `flake8 src/`

## 📄 License

This project is licensed under the MIT License - see [LICENSE](LICENSE) file for details.

## 🙏 Acknowledgments

- **Data Source**: FD.nl (Financieele Dagblad) for Ahold Delhaize options data
- **Risk-Free Rates**: European Central Bank (ECB) €STR overnight rate
- **Technologies**: Apache Airflow, DBT, PostgreSQL, Streamlit, Docker
- **Options Pricing**: py_vollib for Black-Scholes Greeks

## 📧 Contact

**Koen Marijt** - [@KoenM-bit](https://github.com/KoenM-bit)

**Project Link**: [https://github.com/KoenM-bit/options-quant-tool](https://github.com/KoenM-bit/options-quant-tool)

**Docker Images**: [ghcr.io/koenm-bit/options-quant-tool](https://github.com/KoenM-bit/options-quant-tool/pkgs/container/options-quant-tool)

## 🆘 Support

- 🐛 **Issues**: [GitHub Issues](https://github.com/KoenM-bit/options-quant-tool/issues)
- 💬 **Discussions**: [GitHub Discussions](https://github.com/KoenM-bit/options-quant-tool/discussions)
- 📖 **Documentation**: [Wiki](https://github.com/KoenM-bit/options-quant-tool/wiki)

---

**Built with enterprise-quality data engineering best practices** 🚀

**Stack**: Python 🐍 | Airflow 🌊 | DBT 🔄 | PostgreSQL 🐘 | Docker 🐳 | Streamlit 📊
