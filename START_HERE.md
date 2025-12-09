# 🚀 Quick Start - Step by Step

Follow these steps to get your platform running:

## ✅ Step 1: Environment is Configured
Your `.env` file is already set up with:
- ✅ Fernet key generated
- ✅ Database credentials
- ✅ Airflow admin credentials (admin/admin)

## 🏗️ Step 2: Build is Running
Docker is currently building your images. This takes ~5-10 minutes the first time.

## 📋 Step 3: Once Build Completes

### Start All Services
```bash
docker compose up -d
```

Wait ~30-60 seconds for services to initialize.

### Check Service Status
```bash
docker compose ps
```

All services should show as "healthy" or "running".

### Access Airflow UI
Open browser: http://localhost:8080

Login:
- Username: `admin`
- Password: `admin`

## 🎯 Step 4: Initialize Database

```bash
docker compose exec airflow-webserver python /opt/airflow/scripts/init_db.py
```

This creates all database tables (Bronze, Silver, Gold layers).

## 🎬 Step 5: Enable and Run Your First DAG

### In Airflow UI:
1. Find `ahold_options_daily` DAG
2. Toggle the switch to ON (enable)
3. Click the "Play" button (▶️) to trigger manually

### Or via Command Line:
```bash
docker compose exec airflow-webserver airflow dags trigger ahold_options_daily
```

### Watch Progress:
1. Click on the DAG name
2. Go to "Graph" view
3. Watch tasks turn green as they complete

## 📊 Step 6: View Your Data

### Via Database:
```bash
docker compose exec postgres psql -U airflow -d ahold_options
```

Then:
```sql
-- See scraped data
SELECT * FROM bronze_fd_overview ORDER BY scraped_at DESC LIMIT 5;

-- Check data exists
SELECT COUNT(*) FROM bronze_fd_overview;
```

Type `\q` to exit.

## 🔍 Step 7: Run Transformations

After data is scraped, run DBT:

```bash
docker compose exec airflow-webserver airflow dags trigger ahold_dbt_transform
```

Or enable automatic triggering by editing the DAG schedule.

## 📈 Step 8: Check Gold Layer

```bash
docker compose exec postgres psql -U airflow -d ahold_options
```

```sql
-- View daily summaries
SELECT * FROM gold_options_summary_daily 
ORDER BY trade_date DESC 
LIMIT 5;

-- View volatility surface
SELECT * FROM gold_volatility_surface 
ORDER BY trade_date DESC, expiry_date, strike 
LIMIT 10;
```

## 🎉 Success!

You now have:
- ✅ Options data scraped from FD.nl
- ✅ Data cleaned and transformed
- ✅ Analytics-ready tables in Gold layer
- ✅ Automated daily updates

## 🔧 Useful Commands

```bash
# View logs
docker compose logs -f

# View specific service logs
docker compose logs -f airflow-scheduler

# Restart services
docker compose restart

# Stop everything
docker compose down

# Clean restart
docker compose down -v
docker compose up -d
```

## 📚 Next Steps

1. **Connect Power BI**: Use Gold layer tables
2. **Customize Scrapers**: Edit `src/scrapers/`
3. **Add More Symbols**: Modify DAGs
4. **Build Dashboards**: Query Gold tables

## ❓ Troubleshooting

### Services won't start?
```bash
docker compose logs
```

### Database connection error?
```bash
docker compose exec postgres pg_isready -U airflow
```

### DAG import errors?
```bash
docker compose restart airflow-scheduler
docker compose logs airflow-scheduler
```

## 📖 Documentation

- Full docs: `docs/QUICKSTART.md`
- Architecture: `docs/ARCHITECTURE.md`
- Development: `docs/DEVELOPMENT.md`

---

**Your platform is being built now! Come back in ~10 minutes to continue.**
