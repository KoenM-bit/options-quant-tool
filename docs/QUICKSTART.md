# Quick Start Guide

Get the Ahold Options platform running in 5 minutes.

## Prerequisites

- Docker & Docker Compose installed
- Python 3.10+ (for local development)
- 8GB RAM minimum

## Local Development Setup

### 1. Clone & Configure

```bash
# Clone repository
git clone <repository-url>
cd ahold-options

# Create environment file
cp .env.example .env
```

### 2. Generate Fernet Key

```bash
# Run this command to generate encryption key
python -c "from cryptography.fernet import Fernet; print(Fernet.generate_key().decode())"

# Copy output to .env file
# AIRFLOW__CORE__FERNET_KEY=<paste-key-here>
```

### 3. Start Services

```bash
# Start all services
docker-compose up -d

# Wait for services to be healthy (~30 seconds)
docker-compose ps

# Check logs
docker-compose logs -f
```

### 4. Access Airflow

1. Open browser: http://localhost:8080
2. Login: admin / admin (from .env)
3. Enable DAGs by toggling them ON

### 5. Run First Scrape

```bash
# Trigger the daily scraping DAG manually
docker-compose exec airflow-webserver \
  airflow dags trigger ahold_options_daily

# Watch progress in Airflow UI
```

## Verify Setup

### Check Database

```bash
# Connect to PostgreSQL
docker-compose exec postgres psql -U airflow -d ahold_options

# Check tables
\dt

# Query data
SELECT * FROM bronze_fd_overview LIMIT 5;

# Exit
\q
```

### Run DBT Transformations

```bash
# Trigger DBT transformation DAG
docker-compose exec airflow-webserver \
  airflow dags trigger ahold_dbt_transform
```

### View Gold Layer Data

```sql
-- Connect to database and query gold tables
SELECT * FROM gold_options_summary_daily 
ORDER BY trade_date DESC 
LIMIT 10;
```

## Common Commands

### Docker

```bash
# Start services
docker-compose up -d

# Stop services
docker-compose down

# Restart services
docker-compose restart

# View logs
docker-compose logs -f [service-name]

# Rebuild images
docker-compose build --no-cache
```

### Airflow

```bash
# List DAGs
docker-compose exec airflow-webserver airflow dags list

# Trigger DAG
docker-compose exec airflow-webserver airflow dags trigger <dag-id>

# View DAG runs
docker-compose exec airflow-webserver airflow dags list-runs -d <dag-id>

# Test task
docker-compose exec airflow-webserver \
  airflow tasks test <dag-id> <task-id> 2024-01-01
```

### DBT

```bash
# Run DBT models
docker-compose exec airflow-webserver bash -c \
  "cd /opt/airflow/dbt/ahold_options && dbt run"

# Run tests
docker-compose exec airflow-webserver bash -c \
  "cd /opt/airflow/dbt/ahold_options && dbt test"

# Generate docs
docker-compose exec airflow-webserver bash -c \
  "cd /opt/airflow/dbt/ahold_options && dbt docs generate"
```

## Troubleshooting

### Services won't start

```bash
# Check Docker resources
docker system df

# Check logs for errors
docker-compose logs

# Clean up and restart
docker-compose down -v
docker-compose up -d
```

### Database connection errors

```bash
# Check PostgreSQL is running
docker-compose ps postgres

# Test connection
docker-compose exec postgres pg_isready -U airflow

# Reset database (CAUTION: deletes all data)
docker-compose down -v
docker-compose up -d
```

### Airflow webserver not accessible

```bash
# Check if service is healthy
docker-compose ps airflow-webserver

# View logs
docker-compose logs airflow-webserver

# Restart webserver
docker-compose restart airflow-webserver
```

### Import errors in DAGs

```bash
# Rebuild with latest code
docker-compose down
docker-compose build --no-cache
docker-compose up -d

# Check Python path
docker-compose exec airflow-webserver env | grep PYTHON
```

## Next Steps

1. **Customize Configuration**: Edit `.env` for your needs
2. **Schedule DAGs**: Adjust schedules in DAG files
3. **Add More Scrapers**: Extend `src/scrapers/`
4. **Create Custom Transformations**: Add DBT models
5. **Connect Power BI**: See `docs/POWERBI_SETUP.md`
6. **Deploy to Synology**: See `docs/SYNOLOGY_SETUP.md`

## Support

- 📚 Full documentation in `docs/`
- 🐛 Report issues on GitHub
- 💬 Community chat: [Link]

---

**Happy Options Analyzing! 📊**
